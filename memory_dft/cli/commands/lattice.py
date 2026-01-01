"""
Lattice Command
===============

2D Lattice simulation with various Hamiltonians.

Supported models:
  - heisenberg: Heisenberg XXX model
  - xy: XY model
  - kitaev: Kitaev honeycomb (rectangular approx)
  - ising: Transverse-field Ising model
  - hubbard: Hubbard model (spin representation)

Usage:
    memory-dft lattice --model heisenberg --Lx 3 --Ly 3
    memory-dft lattice --model kitaev --Kx 1.0 --Ky 0.8 --path-compare
    memory-dft lattice --model ising -J 1.0 -h 0.5

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from ..utils import (
    print_banner, print_section, print_key_value, 
    save_json, error_exit
)


# =============================================================================
# Lattice Simulation Runner
# =============================================================================

class LatticeRunner:
    """
    Runs 2D lattice simulations with various Hamiltonians.
    """
    
    SUPPORTED_MODELS = ['heisenberg', 'xy', 'kitaev', 'ising', 'hubbard']
    
    def __init__(self, lx: int, ly: int, model: str = 'heisenberg'):
        self.lx = lx
        self.ly = ly
        self.n_sites = lx * ly
        self.model = model.lower()
        
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model}. Use: {self.SUPPORTED_MODELS}")
        
        # Import core modules
        try:
            from memory_dft.core.lattice import LatticeGeometry2D
            from memory_dft.core.operators import SpinOperators
            from memory_dft.core.hamiltonian import HamiltonianBuilder
            from scipy.sparse.linalg import eigsh
            
            self.LatticeGeometry2D = LatticeGeometry2D
            self.SpinOperators = SpinOperators
            self.HamiltonianBuilder = HamiltonianBuilder
            self.eigsh = eigsh
        except ImportError as e:
            raise ImportError(f"Could not import modules: {e}")
        
        # Initialize lattice (periodic boundary conditions)
        self.lattice = self.LatticeGeometry2D(lx, ly, periodic_x=True, periodic_y=True)
        self.ops = self.SpinOperators(self.n_sites)
        self.builder = self.HamiltonianBuilder(self.lattice, self.ops)  # geometry + ops
    
    def build_hamiltonian(self, 
                          j: float = 1.0,
                          kx: float = 1.0,
                          ky: float = 0.8,
                          kz: float = 0.3,
                          h_field: float = 0.5) -> Any:
        """Build Hamiltonian based on model type."""
        # HamiltonianBuilder uses geometry internally for bonds
        if self.model == 'heisenberg':
            return self.builder.heisenberg(J=j)
        elif self.model == 'xy':
            return self.builder.xy(J=j)
        elif self.model == 'kitaev':
            return self.builder.kitaev_rect(Kx=kx, Ky=ky, Kz_diag=kz)
        elif self.model == 'ising':
            return self.builder.ising(J=j, h=h_field)
        elif self.model == 'hubbard':
            return self.builder.hubbard_spin(t=j, U=2.0)
        else:
            raise ValueError(f"Unknown model: {self.model}")
    
    def compute_ground_state(self, H) -> Dict[str, Any]:
        """Compute ground state energy and wavefunction."""
        try:
            E0, psi0 = self.eigsh(H, k=1, which='SA')
            return {
                'E0': float(E0[0]),
                'psi0': psi0[:, 0],
                'dim': H.shape[0],
            }
        except Exception as e:
            return {
                'E0': None,
                'psi0': None,
                'error': str(e),
            }
    
    def run_single(self, 
                   j: float = 1.0,
                   kx: float = 1.0,
                   ky: float = 0.8,
                   kz: float = 0.3,
                   h_field: float = 0.5) -> Dict[str, Any]:
        """Run single lattice simulation."""
        H = self.build_hamiltonian(j, kx, ky, kz, h_field)
        result = self.compute_ground_state(H)
        
        # Compute observables if ground state found
        if result['psi0'] is not None:
            psi = result['psi0']
            # Total magnetization (use pre-built S_total_z)
            Sz_total = self.ops.S_total_z
            mz = float(np.real(psi.conj() @ (Sz_total @ psi)))
            result['magnetization'] = mz
            
            # Nearest-neighbor correlation
            bonds = self.lattice.bonds_nn  # 属性アクセス
            if len(bonds) > 0:
                i, j_site = bonds[0]
                SiSj = self.ops.Sz[i] @ self.ops.Sz[j_site]  # リストアクセス
                corr = float(np.real(psi.conj() @ (SiSj @ psi)))
                result['nn_correlation'] = corr
        
        return result
    
    def run_path_comparison(self,
                            j: float = 1.0,
                            kx: float = 1.0,
                            ky: float = 0.8,
                            kz: float = 0.3,
                            h_field: float = 0.5) -> Dict[str, Any]:
        """Compare two paths to same final Hamiltonian."""
        # Path 1: h=0 → h=h_field (adiabatic)
        # Path 2: h=2*h_field → h=h_field (quench)
        
        # For simplicity, just compute at different intermediate points
        steps = 5
        
        # Path 1: Increase h from 0
        path1_energies = []
        for step in range(steps + 1):
            h = h_field * step / steps
            H = self.build_hamiltonian(j, kx, ky, kz, h)
            result = self.compute_ground_state(H)
            if result['E0'] is not None:
                path1_energies.append(result['E0'])
        
        # Path 2: Decrease h from 2*h_field
        path2_energies = []
        for step in range(steps + 1):
            h = 2 * h_field - h_field * step / steps
            H = self.build_hamiltonian(j, kx, ky, kz, h)
            result = self.compute_ground_state(H)
            if result['E0'] is not None:
                path2_energies.append(result['E0'])
        
        # Final state (both paths end at same H)
        H_final = self.build_hamiltonian(j, kx, ky, kz, h_field)
        final_result = self.compute_ground_state(H_final)
        
        return {
            'path1': {
                'label': f'h: 0 → {h_field}',
                'energies': path1_energies,
                'final_E': path1_energies[-1] if path1_energies else None,
            },
            'path2': {
                'label': f'h: {2*h_field} → {h_field}',
                'energies': path2_energies,
                'final_E': path2_energies[-1] if path2_energies else None,
            },
            'final_state': final_result,
        }


# =============================================================================
# CLI Command
# =============================================================================

def lattice(
    model: str = typer.Option("heisenberg", "--model", "-m", 
                              help="Model: heisenberg, xy, kitaev, ising, hubbard"),
    lx: int = typer.Option(2, "--Lx", help="Lattice size X"),
    ly: int = typer.Option(2, "--Ly", help="Lattice size Y"),
    j: float = typer.Option(1.0, "-J", "--J", help="Exchange coupling J"),
    kx: float = typer.Option(1.0, "--Kx", help="Kitaev Kx coupling"),
    ky: float = typer.Option(0.8, "--Ky", help="Kitaev Ky coupling"),
    kz: float = typer.Option(0.3, "--Kz", help="Kitaev Kz coupling"),
    h_field: float = typer.Option(0.5, "-h", "--h", help="Transverse field (Ising)"),
    path_compare: bool = typer.Option(False, "--path-compare", "-p", 
                                       help="Compare two paths to same final H"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", 
                                          help="Output JSON file"),
):
    """
    2D Lattice simulation with various Hamiltonians.
    
    Supported models:
      - heisenberg: Heisenberg XXX model
      - xy: XY model
      - kitaev: Kitaev honeycomb (rectangular approx)
      - ising: Transverse-field Ising model
      - hubbard: Hubbard model (spin representation)
    
    Examples:
        memory-dft lattice --model heisenberg --Lx 3 --Ly 3
        memory-dft lattice --model kitaev --Kx 1.0 --Ky 0.8 --path-compare
        memory-dft lattice --model ising -J 1.0 -h 0.5
    """
    print_banner()
    
    typer.echo("🔲 2D Lattice Simulation")
    typer.echo("─" * 50)
    print_key_value("Model", model)
    print_key_value("Lattice", f"{lx}×{ly}")
    
    if model == 'kitaev':
        print_key_value("Kx, Ky, Kz", f"{kx}, {ky}, {kz}")
    elif model == 'ising':
        print_key_value("J, h", f"{j}, {h_field}")
    else:
        print_key_value("J", j)
    
    print_key_value("Path compare", "ON" if path_compare else "OFF")
    typer.echo()
    
    # Initialize runner
    try:
        runner = LatticeRunner(lx, ly, model)
    except (ImportError, ValueError) as e:
        error_exit(str(e), "pip install -e .")
    
    typer.echo(f"  Hilbert space dim: 2^{runner.n_sites} = {2**runner.n_sites}")
    typer.echo()
    
    if path_compare:
        # Path comparison mode
        typer.echo("Running path comparison...")
        results = runner.run_path_comparison(j, kx, ky, kz, h_field)
        
        print_section("Path Comparison Results", "📊")
        
        p1 = results['path1']
        p2 = results['path2']
        
        typer.echo(f"\n  Path 1 ({p1['label']}):")
        typer.echo(f"    Final E: {p1['final_E']:.6f}")
        
        typer.echo(f"\n  Path 2 ({p2['label']}):")
        typer.echo(f"    Final E: {p2['final_E']:.6f}")
        
        if p1['final_E'] and p2['final_E']:
            diff = abs(p1['final_E'] - p2['final_E'])
            typer.echo(f"\n  |ΔE|: {diff:.8f}")
            
            if diff < 1e-10:
                typer.echo("  → Ground state is path-independent (as expected)")
                typer.echo("  → Memory effects appear in excited states/dynamics!")
    else:
        # Single simulation mode
        typer.echo("Computing ground state...")
        result = runner.run_single(j, kx, ky, kz, h_field)
        
        print_section("Results", "📊")
        
        if result['E0'] is not None:
            print_key_value("Ground state energy", f"{result['E0']:.6f}")
            if 'magnetization' in result:
                print_key_value("Magnetization <Sz>", f"{result['magnetization']:.4f}")
            if 'nn_correlation' in result:
                print_key_value("NN correlation", f"{result['nn_correlation']:.4f}")
        else:
            typer.echo(f"  ❌ Error: {result.get('error', 'Unknown')}")
        
        results = result
    
    # Save output
    if output:
        data = {
            'config': {
                'model': model,
                'Lx': lx,
                'Ly': ly,
                'J': j,
                'Kx': kx,
                'Ky': ky,
                'Kz': kz,
                'h': h_field,
                'path_compare': path_compare,
            },
            'results': results if isinstance(results, dict) else {'E0': results},
        }
        # Remove numpy arrays for JSON
        if 'psi0' in data['results']:
            del data['results']['psi0']
        save_json(data, output)
    
    typer.echo("\n✅ Done!")
    return results
