"""
DFT Compare Command
===================

Compare DFT vs DSE using PySCF (REAL DFT!).

This uses actual DFT calculations to demonstrate that:
- DFT gives identical energies for different paths to same final state
- DSE captures history dependence

Usage:
    memory-dft dft-compare --mol H2 --basis cc-pvdz --xc B3LYP

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

from ..utils import (
    print_banner, print_section, print_key_value, 
    save_json, error_exit
)


# =============================================================================
# DFT vs DSE Runner
# =============================================================================

@dataclass
class DFTPathResult:
    """Result from a single path calculation."""
    E_dft: List[float]
    E_dse: List[float]
    E_dft_final: float
    E_dse_final: float
    memory_effect: float
    path_label: str


class DFTCompareRunner:
    """
    Runs DFT vs DSE comparison using PySCF.
    
    Demonstrates that DFT is path-independent (history-blind)
    while DSE captures history dependence.
    """
    
    # Predefined molecules
    MOLECULES = {
        'H2': {'r_eq': 0.74, 'template': "H 0 0 0; H 0 0 {r}"},
        'LiH': {'r_eq': 1.60, 'template': "Li 0 0 0; H 0 0 {r}"},
    }
    
    def __init__(self, 
                 molecule: str = 'H2',
                 basis: str = 'sto-3g',
                 xc: str = 'LDA'):
        self.molecule = molecule.upper()
        self.basis = basis
        self.xc = xc
        
        # Get molecule info
        if self.molecule in self.MOLECULES:
            self.r_eq = self.MOLECULES[self.molecule]['r_eq']
            self.atom_template = self.MOLECULES[self.molecule]['template']
        else:
            self.r_eq = 1.0
            self.atom_template = f"{molecule}; H 0 0 {{r}}"
        
        # Import PySCF
        try:
            from pyscf import gto, dft
            self.gto = gto
            self.dft = dft
        except ImportError:
            raise ImportError("PySCF not installed. Run: pip install pyscf")
        
        # Import memory kernel from core
        try:
            from memory_dft.core.memory_kernel import CompositeMemoryKernel, KernelWeights
            self.CompositeMemoryKernel = CompositeMemoryKernel
            self.KernelWeights = KernelWeights
        except ImportError as e:
            raise ImportError(f"Error importing memory kernel: {e}")
    
    def compute_dft(self, r: float) -> float:
        """Compute DFT energy at given distance."""
        atom_str = self.atom_template.format(r=r)
        mol = self.gto.M(atom=atom_str, basis=self.basis, verbose=0)
        mf = self.dft.RKS(mol)
        mf.xc = self.xc
        return mf.kernel()
    
    def run_path(self, 
                 r_values: List[float], 
                 path_label: str) -> DFTPathResult:
        """Run DFT and DSE along a path."""
        # Initialize memory kernel
        kernel = self.CompositeMemoryKernel(
            weights=self.KernelWeights(field=0.3, phys=0.25, chem=0.25, exclusion=0.2)
        )
        
        E_dft_list = []
        E_dse_list = []
        memory_effects = []
        
        for i, r in enumerate(r_values):
            # DFT calculation
            E_dft = self.compute_dft(r)
            E_dft_list.append(E_dft)
            
            # Memory contribution
            if i > 0:
                history_times = np.arange(i, dtype=float)
                weights = kernel.integrate(float(i), history_times)
                memory_contrib = np.sum(weights * np.array(E_dft_list[:-1])) * 0.01
            else:
                memory_contrib = 0.0
            
            memory_effects.append(memory_contrib)
            E_dse_list.append(E_dft + memory_contrib)
        
        return DFTPathResult(
            E_dft=E_dft_list,
            E_dse=E_dse_list,
            E_dft_final=E_dft_list[-1],
            E_dse_final=E_dse_list[-1],
            memory_effect=sum(memory_effects),
            path_label=path_label
        )
    
    def compare_stretch_compress(self,
                                  r_stretch: float,
                                  r_compress: float,
                                  steps: int) -> Dict[str, Any]:
        """Compare stretch-return vs compress-return paths."""
        # Path 1: Equilibrium → Stretch → Return
        path1_r = (
            list(np.linspace(self.r_eq, r_stretch, steps)) +
            list(np.linspace(r_stretch, self.r_eq, steps))
        )
        
        # Path 2: Equilibrium → Compress → Return
        path2_r = (
            list(np.linspace(self.r_eq, r_compress, steps)) +
            list(np.linspace(r_compress, self.r_eq, steps))
        )
        
        result1 = self.run_path(path1_r, "Stretch→Return")
        result2 = self.run_path(path2_r, "Compress→Return")
        
        return {
            'path1': result1,
            'path2': result2,
            'diff_dft': abs(result1.E_dft_final - result2.E_dft_final),
            'diff_dse': abs(result1.E_dse_final - result2.E_dse_final),
        }


# =============================================================================
# CLI Command
# =============================================================================

def dft_compare(
    molecule: str = typer.Option("H2", "--mol", "-m", 
                                  help="Molecule: H2, LiH, or custom"),
    basis: str = typer.Option("sto-3g", "--basis", "-b", 
                               help="Basis set (sto-3g, cc-pvdz, etc.)"),
    xc: str = typer.Option("LDA", "--xc", 
                           help="XC functional (LDA, B3LYP, PBE, etc.)"),
    r_stretch: float = typer.Option(1.5, "--r-stretch", 
                                     help="Max stretch distance (Å)"),
    r_compress: float = typer.Option(0.5, "--r-compress", 
                                      help="Min compress distance (Å)"),
    steps: int = typer.Option(5, "-n", "--steps", 
                              help="Steps per path segment"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", 
                                          help="Output JSON file"),
):
    """
    Compare DFT vs DSE using PySCF (REAL DFT!).
    
    This uses actual DFT calculations to demonstrate that:
    - DFT gives identical energies for different paths to same final state
    - DSE captures history dependence
    
    Requires: pip install pyscf
    
    Example:
        memory-dft dft-compare --mol H2 --basis cc-pvdz --xc B3LYP
    """
    print_banner()
    
    typer.echo("🔬 DFT vs DSE Comparison (PySCF)")
    typer.echo("─" * 50)
    print_key_value("Molecule", molecule)
    print_key_value("Basis", basis)
    print_key_value("XC", xc)
    print_key_value("Stretch", f"{r_stretch} Å")
    print_key_value("Compress", f"{r_compress} Å")
    print_key_value("Steps", f"{steps} per segment")
    typer.echo()
    
    # Initialize runner
    try:
        runner = DFTCompareRunner(molecule, basis, xc)
    except ImportError as e:
        error_exit(str(e), "pip install pyscf")
    
    typer.echo(f"  r_eq: {runner.r_eq} Å")
    typer.echo()
    
    # Run comparison
    typer.echo("Running path calculations...")
    typer.echo("  Path 1: Stretch → Return")
    typer.echo("  Path 2: Compress → Return")
    
    results = runner.compare_stretch_compress(r_stretch, r_compress, steps)
    
    result1 = results['path1']
    result2 = results['path2']
    diff_dft = results['diff_dft']
    diff_dse = results['diff_dse']
    
    # Display results
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("📊 PATH COMPARISON RESULTS")
    typer.echo("=" * 60)
    
    typer.echo(f"\n  Path 1 ({result1.path_label}):")
    typer.echo(f"    E_DFT final:  {result1.E_dft_final:.6f} Ha")
    typer.echo(f"    E_DSE final:  {result1.E_dse_final:.6f} Ha")
    typer.echo(f"    Memory effect: {result1.memory_effect:.6f} Ha")
    
    typer.echo(f"\n  Path 2 ({result2.path_label}):")
    typer.echo(f"    E_DFT final:  {result2.E_dft_final:.6f} Ha")
    typer.echo(f"    E_DSE final:  {result2.E_dse_final:.6f} Ha")
    typer.echo(f"    Memory effect: {result2.memory_effect:.6f} Ha")
    
    typer.echo()
    typer.echo("─" * 60)
    typer.echo(f"  |ΔE| DFT:  {diff_dft:.8f} Ha  ({diff_dft * 27.211:.4f} eV)")
    typer.echo(f"  |ΔE| DSE:  {diff_dse:.8f} Ha  ({diff_dse * 27.211:.4f} eV)")
    
    if diff_dft < 1e-8:
        typer.echo(f"\n  🎯 DFT: Cannot distinguish paths! (ΔE ≈ 0)")
        typer.echo(f"  🎯 DSE: REVEALS difference! (ΔE = {diff_dse:.6f} Ha)")
    else:
        ratio = diff_dse / diff_dft if diff_dft > 0 else float('inf')
        typer.echo(f"\n  Amplification: {ratio:.1f}x")
    
    typer.echo("=" * 60)
    
    # Save output
    if output:
        data = {
            'config': {
                'molecule': molecule,
                'basis': basis,
                'xc': xc,
                'r_eq': runner.r_eq,
                'r_stretch': r_stretch,
                'r_compress': r_compress,
                'steps': steps,
            },
            'path1': {
                'label': result1.path_label,
                'E_dft': result1.E_dft,
                'E_dse': result1.E_dse,
                'E_dft_final': result1.E_dft_final,
                'E_dse_final': result1.E_dse_final,
                'memory_effect': result1.memory_effect,
            },
            'path2': {
                'label': result2.path_label,
                'E_dft': result2.E_dft,
                'E_dse': result2.E_dse,
                'E_dft_final': result2.E_dft_final,
                'E_dse_final': result2.E_dse_final,
                'memory_effect': result2.memory_effect,
            },
            'comparison': {
                'delta_dft_Ha': diff_dft,
                'delta_dft_eV': diff_dft * 27.211,
                'delta_dse_Ha': diff_dse,
                'delta_dse_eV': diff_dse * 27.211,
            }
        }
        save_json(data, output)
    
    typer.echo("\n✅ Done!")
    return results
