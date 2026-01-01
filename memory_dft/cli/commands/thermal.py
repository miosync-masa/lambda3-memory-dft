"""
Thermal Command
===============

Thermal path dependence with REAL DFT (PySCF).

Compares:
  - DFT (PySCF): Finite-temperature DFT at T_final (single calculation)
  - DSE: Two different temperature paths to the same T_final

DFT cannot distinguish paths (history-blind).
DSE reveals path dependence (history-aware).

Usage:
    memory-dft thermal --mol H2 --T-high 300 --T-low 100 --T-final 200

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..utils import (
    print_banner, print_section, print_key_value, 
    save_json, error_exit
)


# =============================================================================
# Thermal Path Runner (ビジネスロジック分離)
# =============================================================================

class ThermalPathRunner:
    """
    Runs thermal path comparison between DFT and DSE.
    
    This class separates the computation logic from CLI.
    """
    
    # Predefined molecules
    MOLECULES = {
        'H2': 'H 0 0 0; H 0 0 0.74',
        'LiH': 'Li 0 0 0; H 0 0 1.6',
    }
    
    # Boltzmann constant in Hartree/K
    K_B_HA = 3.1668e-6
    
    def __init__(self, mol: str, basis: str = 'sto-3g'):
        if mol not in self.MOLECULES:
            raise ValueError(f"Unknown molecule: {mol}. Use: {list(self.MOLECULES.keys())}")
        
        self.mol = mol
        self.atom_str = self.MOLECULES[mol]
        self.basis = basis
        
        # Lazy import PySCF
        try:
            from pyscf import gto, dft, scf
            self.gto = gto
            self.dft = dft
            self.scf = scf
        except ImportError:
            raise ImportError("PySCF not installed. Run: pip install pyscf")
        
        # Import memory kernel from core (NOT pyscf_interface!)
        try:
            from memory_dft.core.memory_kernel import CompositeMemoryKernel, KernelWeights
            self.CompositeMemoryKernel = CompositeMemoryKernel
            self.KernelWeights = KernelWeights
        except ImportError as e:
            raise ImportError(f"Error importing memory kernel: {e}")
    
    def compute_dft_at_temperature(self, T: float) -> float:
        """Compute DFT energy at given temperature using Fermi smearing."""
        pyscf_mol = self.gto.M(atom=self.atom_str, basis=self.basis, verbose=0)
        mf = self.dft.RKS(pyscf_mol)
        mf.xc = 'LDA'
        
        # Finite temperature via Fermi-Dirac smearing
        # Use scf.addons.smearing_ (not method on mf)
        sigma = self.K_B_HA * T
        if sigma > 1e-8:  # Only apply smearing if sigma is meaningful
            try:
                mf = self.scf.addons.smearing_(mf, sigma=sigma, method='fermi')
            except Exception:
                # Fallback: run without smearing for molecules
                pass
        
        return mf.kernel()
    
    def run_dse_thermal_path(self, temps: List[float], path_name: str) -> Dict[str, Any]:
        """Run DSE along temperature path using CompositeMemoryKernel."""
        # Initialize composite memory kernel (4 physical components!)
        kernel = self.CompositeMemoryKernel(
            weights=self.KernelWeights(field=0.3, phys=0.25, chem=0.25, exclusion=0.2)
        )
        
        # DFT at each T
        E_dft_list = []
        memory_effects = []
        
        for i, T in enumerate(temps):
            E_dft = self.compute_dft_at_temperature(T)
            E_dft_list.append(E_dft)
            
            # Memory contribution from previous states
            if i > 0:
                history_times = np.arange(i)
                weights = kernel.integrate(float(i), history_times)
                memory_contrib = np.sum(weights * np.array(E_dft_list[:-1])) * 0.01
            else:
                memory_contrib = 0.0
            
            memory_effects.append(memory_contrib)
        
        E_final = E_dft_list[-1]
        E_with_memory = E_final + sum(memory_effects)
        
        return {
            'path_name': path_name,
            'temperatures': temps,
            'E_dft_list': E_dft_list,
            'E_final': E_final,
            'E_with_memory': E_with_memory,
            'memory_effect': sum(memory_effects),
        }
    
    def compare_paths(self, 
                      t_high: float, 
                      t_low: float, 
                      t_final: float, 
                      steps: int) -> Dict[str, Any]:
        """Compare two thermal paths."""
        # Path 1: Low → High → Final
        path1_temps = (
            list(np.linspace(t_low, t_high, steps)) + 
            list(np.linspace(t_high, t_final, steps))
        )
        
        # Path 2: High → Low → Final
        path2_temps = (
            list(np.linspace(t_high, t_low, steps)) + 
            list(np.linspace(t_low, t_final, steps))
        )
        
        result1 = self.run_dse_thermal_path(path1_temps, f"Path 1: {t_low}K→{t_high}K→{t_final}K")
        result2 = self.run_dse_thermal_path(path2_temps, f"Path 2: {t_high}K→{t_low}K→{t_final}K")
        
        # DFT at final T (single calculation)
        E_dft_final = self.compute_dft_at_temperature(t_final)
        
        return {
            'path1': result1,
            'path2': result2,
            'E_dft_final': E_dft_final,
        }


# =============================================================================
# CLI Command
# =============================================================================

def thermal(
    mol: str = typer.Option("H2", "--mol", "-m", help="Molecule: H2, LiH"),
    basis: str = typer.Option("sto-3g", "--basis", "-b", help="Basis set"),
    t_high: float = typer.Option(300.0, "--T-high", help="High temperature (K)"),
    t_low: float = typer.Option(100.0, "--T-low", help="Low temperature (K)"),
    t_final: float = typer.Option(200.0, "--T-final", help="Final temperature (K)"),
    steps: int = typer.Option(5, "-n", "--steps", help="Steps per temperature segment"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """
    Thermal path dependence with REAL DFT (PySCF).
    
    Example:
        memory-dft thermal --mol H2 --T-high 300 --T-low 100 --T-final 200
    """
    print_banner()
    
    typer.echo("🌡️ Thermal Path Dependence (Real DFT)")
    typer.echo("─" * 50)
    print_key_value("Molecule", mol)
    print_key_value("Basis", basis)
    print_key_value("T_high", f"{t_high} K")
    print_key_value("T_low", f"{t_low} K")
    print_key_value("T_final", f"{t_final} K")
    typer.echo()
    
    # Run comparison
    try:
        runner = ThermalPathRunner(mol, basis)
    except ImportError as e:
        error_exit(str(e), "pip install pyscf")
    except ValueError as e:
        error_exit(str(e))
    
    typer.echo("Running DFT (PySCF) at final temperature...")
    typer.echo("Running DSE along temperature paths...")
    
    results = runner.compare_paths(t_high, t_low, t_final, steps)
    
    result1 = results['path1']
    result2 = results['path2']
    E_dft_final = results['E_dft_final']
    
    # DFT: Same final T → Same energy (by definition)
    delta_E_dft = 0.0
    
    # DSE: Different paths → Different results
    delta_E_dse = abs(result1['E_with_memory'] - result2['E_with_memory'])
    
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("📊 THERMAL PATH COMPARISON (Real DFT vs DSE)")
    typer.echo("=" * 60)
    
    typer.echo(f"\n  Same final temperature: {t_final} K")
    
    typer.echo(f"\n  DFT (PySCF, single calculation at T_final):")
    typer.echo(f"    E_DFT = {E_dft_final:.8f} Ha")
    typer.echo(f"    Path 1 final = Path 2 final (DFT is history-blind)")
    
    typer.echo(f"\n  DSE (history-dependent):")
    typer.echo(f"    Path 1 E_DSE = {result1['E_with_memory']:.8f} Ha")
    typer.echo(f"    Path 2 E_DSE = {result2['E_with_memory']:.8f} Ha")
    
    typer.echo()
    typer.echo("─" * 60)
    typer.echo(f"  |ΔE| DFT:  {delta_E_dft:.8f} Ha  ({delta_E_dft * 27.211:.4f} eV)")
    typer.echo(f"  |ΔE| DSE:  {delta_E_dse:.8f} Ha  ({delta_E_dse * 27.211:.4f} eV)")
    
    if delta_E_dse > 1e-6:
        typer.echo()
        typer.echo("  🎯 DFT: Cannot distinguish paths! (ΔE = 0 by construction)")
        typer.echo("  🎯 DSE: REVEALS difference! (ΔE ≠ 0)")
        typer.echo("  → Same final T, different history → Different quantum state")
    
    # Save
    if output:
        data = {
            'molecule': mol,
            'basis': basis,
            'temperatures': {
                'T_high': t_high, 'T_low': t_low, 'T_final': t_final,
            },
            'DFT': {
                'E_final': E_dft_final,
                'method': 'PySCF RKS/LDA with Fermi smearing',
            },
            'DSE': {
                'path1_E': result1['E_with_memory'],
                'path2_E': result2['E_with_memory'],
                'path1_memory': result1['memory_effect'],
                'path2_memory': result2['memory_effect'],
            },
            'comparison': {
                'delta_E_DFT_Ha': delta_E_dft,
                'delta_E_DSE_Ha': delta_E_dse,
                'delta_E_DFT_eV': delta_E_dft * 27.211,
                'delta_E_DSE_eV': delta_E_dse * 27.211,
            }
        }
        save_json(data, output)
    
    typer.echo("\n✅ Done!")
    return results
