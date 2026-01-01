"""
PySCF Interface for DSE
=======================

Interface between PySCF DFT calculations and DSE memory framework.

IMPORTANT: This module now uses the unified memory kernel from
           memory_dft.core.memory_kernel (NOT a separate implementation!)

Key Classes:
  - DSECalculator: Main calculator combining DFT with memory effects
  - GeometryStep: Single step in a reaction path
  - PathResult: Results from path computation
  - ComparisonResult: Comparison of two paths

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Check PySCF availability
try:
    from pyscf import gto, dft
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeometryStep:
    """Single geometry step in a reaction path."""
    atoms: str       # Atom string in PySCF format
    time: float      # Pseudo-time for memory tracking
    label: str = ""  # Optional label


@dataclass
class PathResult:
    """Results from computing a reaction path."""
    E_dft: List[float]        # DFT energies
    E_dft_final: float        # Final DFT energy
    E_dse: List[float]        # DSE energies (DFT + memory)
    E_dse_final: float        # Final DSE energy
    delta_memory: List[float] # Memory contributions
    path_label: str           # Path identifier
    n_steps: int              # Number of steps
    
    @property
    def memory_effect(self) -> float:
        """Total memory effect."""
        return sum(self.delta_memory)


@dataclass
class ComparisonResult:
    """Results from comparing two paths."""
    path1: PathResult
    path2: PathResult
    
    @property
    def delta_dft(self) -> float:
        """DFT energy difference (should be ~0 for same final state)."""
        return abs(self.path1.E_dft_final - self.path2.E_dft_final)
    
    @property
    def delta_dse(self) -> float:
        """DSE energy difference (reveals history dependence!)."""
        return abs(self.path1.E_dse_final - self.path2.E_dse_final)


# =============================================================================
# Memory Kernel Wrapper (uses core.memory_kernel!)
# =============================================================================

class MemoryKernelDFTWrapper:
    """
    Wrapper around CompositeMemoryKernel for DFT path calculations.
    
    This class wraps the unified kernel from core.memory_kernel
    to provide a simple interface for DFT-based path calculations.
    
    NOTE: This is NOT a separate implementation! It uses core.memory_kernel.
    """
    
    def __init__(self,
                 eta: float = 0.1,
                 tau: float = 5.0,
                 gamma: float = 0.5):
        """
        Initialize memory kernel wrapper.
        
        Args:
            eta: Memory strength parameter
            tau: Memory decay time constant
            gamma: Power-law exponent
        """
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        
        # Use the unified kernel from core!
        try:
            from memory_dft.core.memory_kernel import (
                CompositeMemoryKernel, 
                KernelWeights
            )
            self._kernel = CompositeMemoryKernel(
                weights=KernelWeights(
                    field=gamma,  # Map gamma to field weight
                    phys=0.25,
                    chem=0.25,
                    exclusion=0.2
                ),
                gamma_field=gamma,
                tau0_phys=tau,
            )
        except ImportError:
            # Fallback for standalone testing
            self._kernel = None
        
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
        
        Uses the unified CompositeMemoryKernel from core.
        """
        if len(self.history) == 0:
            return 0.0
        
        memory = 0.0
        
        for t_hist, E_hist, coords_hist in self.history:
            dt = t - t_hist
            if dt <= 0:
                continue
            
            # Use unified kernel if available
            if self._kernel is not None:
                # Get weight from composite kernel
                history_times = np.array([h[0] for h in self.history])
                weights = self._kernel.integrate(t, history_times)
                idx = np.searchsorted(history_times, t_hist)
                if idx < len(weights):
                    K_time = weights[idx]
                else:
                    K_time = np.exp(-dt / self.tau)
            else:
                # Fallback: simple exponential
                K_time = np.exp(-dt / self.tau) * (1 + dt) ** (-self.gamma)
            
            # Geometric kernel
            if coords_hist.shape == coords.shape:
                delta_r = np.linalg.norm(coords - coords_hist)
                K_geom = np.exp(-delta_r / 0.5)
            else:
                K_geom = 0.5
            
            # Energy difference
            delta_E = E - E_hist
            
            # Combined
            memory += self.eta * K_time * K_geom * abs(delta_E)
        
        return memory
    
    def clear(self):
        """Clear history."""
        self.history = []


# =============================================================================
# DSE Calculator
# =============================================================================

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
        
        # Use wrapper (which uses core.memory_kernel!)
        self.memory_kernel = MemoryKernelDFTWrapper(
            eta=memory_eta,
            tau=memory_tau,
            gamma=memory_gamma
        )
    
    def compute_dft(self, atoms: str, charge: int = 0, spin: int = 0) -> Tuple[float, np.ndarray]:
        """Compute standard DFT energy."""
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
        coords = mol.atom_coords()
        
        return E, coords
    
    def compute_path(self, 
                     path: List[GeometryStep],
                     charge: int = 0,
                     spin: int = 0,
                     label: str = "") -> PathResult:
        """Compute energies along a reaction path."""
        self.memory_kernel.clear()
        
        E_dft_list = []
        E_dse_list = []
        delta_mem_list = []
        
        for step in path:
            E_dft, coords = self.compute_dft(step.atoms, charge, spin)
            E_dft_list.append(E_dft)
            
            delta_mem = self.memory_kernel.compute_memory_contribution(
                step.time, E_dft, coords
            )
            delta_mem_list.append(delta_mem)
            
            E_dse = E_dft + delta_mem
            E_dse_list.append(E_dse)
            
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
        """Compare two reaction paths."""
        result1 = self.compute_path(path1, charge, spin, label1)
        result2 = self.compute_path(path2, charge, spin, label2)
        
        return ComparisonResult(path1=result1, path2=result2)


# =============================================================================
# Helper Functions
# =============================================================================

def create_h2_stretch_path(r_start: float = 0.74, 
                           r_end: float = 1.5,
                           r_return: float = 0.74,
                           n_steps: int = 10) -> List[GeometryStep]:
    """Create H2 stretch-return path."""
    path = []
    
    # Stretch
    for i, r in enumerate(np.linspace(r_start, r_end, n_steps)):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(i),
            label=f"stretch_{i}"
        ))
    
    # Return
    for i, r in enumerate(np.linspace(r_end, r_return, n_steps)):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(n_steps + i),
            label=f"return_{i}"
        ))
    
    return path


def create_h2_compress_path(r_start: float = 0.74, 
                            r_end: float = 0.5,
                            r_return: float = 0.74,
                            n_steps: int = 10) -> List[GeometryStep]:
    """Create H2 compress-return path."""
    path = []
    
    # Compress
    for i, r in enumerate(np.linspace(r_start, r_end, n_steps)):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(i),
            label=f"compress_{i}"
        ))
    
    # Return
    for i, r in enumerate(np.linspace(r_end, r_return, n_steps)):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(n_steps + i),
            label=f"return_{i}"
        ))
    
    return path


def demo_h2_comparison():
    """Demo: Compare H2 stretch vs compress paths."""
    if not HAS_PYSCF:
        print("PySCF not available")
        return None
    
    calc = DSECalculator(basis='sto-3g', xc='LDA')
    
    path_stretch = create_h2_stretch_path()
    path_compress = create_h2_compress_path()
    
    result = calc.compare_paths(
        path_stretch, path_compress,
        label1="Stretch→Return",
        label2="Compress→Return"
    )
    
    print(f"DFT difference: {result.delta_dft:.6f} Ha (should be ~0)")
    print(f"DSE difference: {result.delta_dse:.6f} Ha (history effect!)")
    
    return result


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Old name (deprecated)
MemoryKernelDFT = MemoryKernelDFTWrapper


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DSECalculator',
    'PathResult',
    'ComparisonResult',
    'GeometryStep',
    'MemoryKernelDFTWrapper',
    'MemoryKernelDFT',  # deprecated alias
    'create_h2_stretch_path',
    'create_h2_compress_path',
    'demo_h2_comparison',
    'HAS_PYSCF',
]
