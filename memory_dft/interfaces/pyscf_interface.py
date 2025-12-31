"""
PySCF Interface for Memory-DFT / DSE
====================================

Enables direct comparison between standard DFT and
history-dependent DSE calculations.

Key Features:
  - Standard DFT via PySCF
  - DSE with memory kernel corrections
  - Path-dependent energy comparisons
  - Publication-ready outputs

Example:
    >>> from memory_dft.interfaces.pyscf_interface import DSECalculator
    >>> calc = DSECalculator(basis='cc-pvdz', xc='B3LYP')
    >>> 
    >>> # Path 1: stretch then compress
    >>> E_dft_1, E_dse_1 = calc.compute_path(mol, path1)
    >>> 
    >>> # Path 2: compress then stretch  
    >>> E_dft_2, E_dse_2 = calc.compute_path(mol, path2)
    >>> 
    >>> # DFT: E_dft_1 == E_dft_2 (same final state)
    >>> # DSE: E_dse_1 != E_dse_2 (history matters!)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import warnings

# PySCF import (optional)
try:
    from pyscf import gto, dft, scf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    warnings.warn("PySCF not available. Install with: pip install pyscf")


@dataclass
class GeometryStep:
    """Single geometry in a reaction path."""
    atoms: str           # PySCF atom string format
    time: float          # Pseudo-time for memory kernel
    label: str = ""      # Optional label (e.g., "TS", "Product")
    
    def __post_init__(self):
        if self.label == "":
            self.label = f"t={self.time:.2f}"


@dataclass  
class PathResult:
    """Result of a path calculation."""
    # DFT results (no memory)
    E_dft: List[float]           # DFT energies along path
    E_dft_final: float           # Final DFT energy
    
    # DSE results (with memory)
    E_dse: List[float]           # DSE energies along path
    E_dse_final: float           # Final DSE energy
    
    # Memory contributions
    delta_memory: List[float]    # Memory correction at each step
    
    # Metadata
    path_label: str = ""
    n_steps: int = 0
    
    @property
    def memory_effect(self) -> float:
        """Total memory effect (DSE - DFT at final point)."""
        return self.E_dse_final - self.E_dft_final
    
    @property
    def integrated_memory(self) -> float:
        """Integrated memory contribution."""
        return sum(self.delta_memory)


@dataclass
class ComparisonResult:
    """Result comparing two paths."""
    path1: PathResult
    path2: PathResult
    
    @property
    def delta_dft(self) -> float:
        """DFT energy difference (should be ~0 for same final state)."""
        return abs(self.path1.E_dft_final - self.path2.E_dft_final)
    
    @property
    def delta_dse(self) -> float:
        """DSE energy difference (captures history!)."""
        return abs(self.path1.E_dse_final - self.path2.E_dse_final)
    
    @property
    def amplification(self) -> float:
        """How much DSE amplifies the difference."""
        if self.delta_dft < 1e-10:
            return float('inf')
        return self.delta_dse / self.delta_dft
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "DSE vs DFT Path Comparison",
            "=" * 60,
            "",
            f"Path 1: {self.path1.path_label}",
            f"  E_DFT (final):  {self.path1.E_dft_final:.6f} Ha",
            f"  E_DSE (final):  {self.path1.E_dse_final:.6f} Ha",
            f"  Memory effect:  {self.path1.memory_effect:.6f} Ha",
            "",
            f"Path 2: {self.path2.path_label}",
            f"  E_DFT (final):  {self.path2.E_dft_final:.6f} Ha",
            f"  E_DSE (final):  {self.path2.E_dse_final:.6f} Ha",
            f"  Memory effect:  {self.path2.memory_effect:.6f} Ha",
            "",
            "─" * 60,
            f"|ΔE| DFT:  {self.delta_dft:.8f} Ha  ({self.delta_dft * 27.211:.4f} eV)",
            f"|ΔE| DSE:  {self.delta_dse:.8f} Ha  ({self.delta_dse * 27.211:.4f} eV)",
            "",
        ]
        
        if self.delta_dft < 1e-8:
            lines.append("🎯 DFT: Cannot distinguish paths! (ΔE ≈ 0)")
            lines.append(f"🎯 DSE: REVEALS difference! (ΔE = {self.delta_dse:.6f} Ha)")
        else:
            lines.append(f"Amplification: {self.amplification:.1f}x")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class MemoryKernelDFT:
    """
    Memory kernel for DFT energy corrections.
    
    Computes history-dependent corrections to DFT energies
    based on the geometric path taken.
    """
    
    def __init__(self, 
                 eta: float = 0.1,
                 tau: float = 5.0,
                 gamma: float = 0.5):
        """
        Initialize memory kernel.
        
        Args:
            eta: Memory strength parameter
            tau: Memory decay time constant
            gamma: Power-law exponent
        """
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.history: List[Tuple[float, float, np.ndarray]] = []
    
    def add_state(self, t: float, E: float, coords: np.ndarray):
        """Record a state in history."""
        self.history.append((t, E, coords.copy()))
        # Keep last 100 states
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def compute_memory_contribution(self, t: float, E: float, coords: np.ndarray) -> float:
        """
        Compute memory correction to energy.
        
        The correction depends on:
          1. Time since previous states (exponential decay)
          2. Geometric similarity (overlap in configuration space)
          3. Energy differences (strain history)
        """
        if len(self.history) == 0:
            return 0.0
        
        memory = 0.0
        
        for t_hist, E_hist, coords_hist in self.history:
            dt = t - t_hist
            if dt <= 0:
                continue
            
            # Time kernel: exponential with power-law correction
            K_time = np.exp(-dt / self.tau) * (1 + dt) ** (-self.gamma)
            
            # Geometric kernel: based on coordinate change
            if coords_hist.shape == coords.shape:
                delta_r = np.linalg.norm(coords - coords_hist)
                K_geom = np.exp(-delta_r / 0.5)  # 0.5 Å characteristic length
            else:
                K_geom = 0.5  # Default if shapes don't match
            
            # Energy difference contribution
            delta_E = E - E_hist
            
            # Combined memory contribution
            memory += self.eta * K_time * K_geom * abs(delta_E)
        
        return memory
    
    def clear(self):
        """Clear history."""
        self.history = []


class DSECalculator:
    """
    Calculator for DSE (Direct Schrödinger Evolution) with PySCF.
    
    Enables comparison between standard DFT and history-dependent DSE.
    """
    
    def __init__(self,
                 basis: str = 'cc-pvdz',
                 xc: str = 'B3LYP',
                 memory_eta: float = 0.1,
                 memory_tau: float = 5.0,
                 memory_gamma: float = 0.5,
                 verbose: int = 0):
        """
        Initialize DSE calculator.
        
        Args:
            basis: Basis set for DFT calculation
            xc: Exchange-correlation functional
            memory_eta: Memory kernel strength
            memory_tau: Memory decay time constant
            memory_gamma: Power-law exponent
            verbose: PySCF verbosity level
        """
        if not HAS_PYSCF:
            raise ImportError("PySCF is required. Install with: pip install pyscf")
        
        self.basis = basis
        self.xc = xc
        self.verbose = verbose
        
        # Initialize memory kernel
        self.memory_kernel = MemoryKernelDFT(
            eta=memory_eta,
            tau=memory_tau,
            gamma=memory_gamma
        )
    
    def compute_dft(self, atoms: str, charge: int = 0, spin: int = 0) -> Tuple[float, np.ndarray]:
        """
        Compute standard DFT energy.
        
        Args:
            atoms: Atom string in PySCF format (e.g., "H 0 0 0; H 0 0 0.74")
            charge: Molecular charge
            spin: Spin multiplicity - 1
            
        Returns:
            (energy, coordinates)
        """
        mol = gto.M(
            atom=atoms,
            basis=self.basis,
            charge=charge,
            spin=spin,
            verbose=self.verbose
        )
        
        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        
        mf.xc = self.xc
        E = mf.kernel()
        
        # Extract coordinates
        coords = mol.atom_coords()
        
        return E, coords
    
    def compute_path(self, 
                     path: List[GeometryStep],
                     charge: int = 0,
                     spin: int = 0,
                     label: str = "") -> PathResult:
        """
        Compute energies along a reaction path.
        
        Args:
            path: List of GeometryStep objects defining the path
            charge: Molecular charge
            spin: Spin multiplicity - 1
            label: Path label for output
            
        Returns:
            PathResult with DFT and DSE energies
        """
        self.memory_kernel.clear()
        
        E_dft_list = []
        E_dse_list = []
        delta_mem_list = []
        
        for step in path:
            # Standard DFT
            E_dft, coords = self.compute_dft(step.atoms, charge, spin)
            E_dft_list.append(E_dft)
            
            # Memory contribution
            delta_mem = self.memory_kernel.compute_memory_contribution(
                step.time, E_dft, coords
            )
            delta_mem_list.append(delta_mem)
            
            # DSE energy
            E_dse = E_dft + delta_mem
            E_dse_list.append(E_dse)
            
            # Update history
            self.memory_kernel.add_state(step.time, E_dft, coords)
        
        return PathResult(
            E_dft=E_dft_list,
            E_dft_final=E_dft_list[-1],
            E_dse=E_dse_list,
            E_dse_final=E_dse_list[-1],
            delta_memory=delta_mem_list,
            path_label=label,
            n_steps=len(path)
        )
    
    def compare_paths(self,
                      path1: List[GeometryStep],
                      path2: List[GeometryStep],
                      charge: int = 0,
                      spin: int = 0,
                      label1: str = "Path 1",
                      label2: str = "Path 2") -> ComparisonResult:
        """
        Compare two reaction paths.
        
        This is the key demonstration of DSE:
        - Same final geometry
        - Different paths
        - DFT gives same energy
        - DSE gives different energies!
        
        Args:
            path1: First path
            path2: Second path  
            charge: Molecular charge
            spin: Spin multiplicity - 1
            label1: Label for path 1
            label2: Label for path 2
            
        Returns:
            ComparisonResult with both paths and differences
        """
        result1 = self.compute_path(path1, charge, spin, label1)
        result2 = self.compute_path(path2, charge, spin, label2)
        
        return ComparisonResult(path1=result1, path2=result2)


# =============================================================================
# Helper functions for common molecular paths
# =============================================================================

def create_h2_stretch_path(r_start: float = 0.74, 
                           r_end: float = 1.5,
                           r_return: float = 0.74,
                           n_steps: int = 10) -> List[GeometryStep]:
    """
    Create H2 stretch-return path.
    
    Args:
        r_start: Starting bond length (Å)
        r_end: Maximum stretch (Å)
        r_return: Return bond length (Å)
        n_steps: Steps per segment
        
    Returns:
        List of GeometryStep for the path
    """
    path = []
    t = 0.0
    dt = 1.0
    
    # Stretch: r_start -> r_end
    for r in np.linspace(r_start, r_end, n_steps):
        atoms = f"H 0 0 0; H 0 0 {r}"
        path.append(GeometryStep(atoms=atoms, time=t, label=f"stretch r={r:.2f}"))
        t += dt
    
    # Return: r_end -> r_return
    for r in np.linspace(r_end, r_return, n_steps):
        atoms = f"H 0 0 0; H 0 0 {r}"
        path.append(GeometryStep(atoms=atoms, time=t, label=f"return r={r:.2f}"))
        t += dt
    
    return path


def create_h2_compress_path(r_start: float = 0.74,
                            r_end: float = 0.5,
                            r_return: float = 0.74,
                            n_steps: int = 10) -> List[GeometryStep]:
    """
    Create H2 compress-return path.
    
    Args:
        r_start: Starting bond length (Å)
        r_end: Minimum compression (Å)
        r_return: Return bond length (Å)
        n_steps: Steps per segment
        
    Returns:
        List of GeometryStep for the path
    """
    path = []
    t = 0.0
    dt = 1.0
    
    # Compress: r_start -> r_end
    for r in np.linspace(r_start, r_end, n_steps):
        atoms = f"H 0 0 0; H 0 0 {r}"
        path.append(GeometryStep(atoms=atoms, time=t, label=f"compress r={r:.2f}"))
        t += dt
    
    # Return: r_end -> r_return
    for r in np.linspace(r_end, r_return, n_steps):
        atoms = f"H 0 0 0; H 0 0 {r}"
        path.append(GeometryStep(atoms=atoms, time=t, label=f"return r={r:.2f}"))
        t += dt
    
    return path


# =============================================================================
# Quick demo function
# =============================================================================

def demo_h2_comparison():
    """
    Demonstrate DSE vs DFT for H2 molecule.
    
    Shows that:
    - DFT gives same energy for stretch-return and compress-return paths
    - DSE gives DIFFERENT energies due to history dependence
    """
    print("=" * 60)
    print("DSE vs DFT Demonstration: H2 Molecule")
    print("=" * 60)
    print()
    
    # Create calculator
    calc = DSECalculator(
        basis='sto-3g',  # Small basis for speed
        xc='LDA',
        memory_eta=0.05,
        memory_tau=5.0,
        verbose=0
    )
    
    # Create paths
    print("Creating paths...")
    path_stretch = create_h2_stretch_path(r_start=0.74, r_end=1.2, r_return=0.74, n_steps=5)
    path_compress = create_h2_compress_path(r_start=0.74, r_end=0.5, r_return=0.74, n_steps=5)
    
    print(f"  Path 1 (stretch→return): {len(path_stretch)} steps")
    print(f"  Path 2 (compress→return): {len(path_compress)} steps")
    print()
    
    # Compare
    print("Running calculations...")
    result = calc.compare_paths(
        path_stretch, path_compress,
        label1="Stretch→Return",
        label2="Compress→Return"
    )
    
    print()
    print(result.summary())
    
    return result


if __name__ == "__main__":
    demo_h2_comparison()
