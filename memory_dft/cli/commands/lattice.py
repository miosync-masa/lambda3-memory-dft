"""
Lattice Command - True DSE Implementation
==========================================

2D Lattice simulation with time evolution and path dependence.

Core insight:
  --path-compare: Field path dependence
    Path 1: h=0 → h=h_final (field increase)
    Path 2: h=2*h_final → h=h_final (field decrease)
    → Same final H, DIFFERENT λ!

  --thermal: Temperature path dependence
    Path 1: T_low → T_high → T_final
    Path 2: T_high → T_low → T_final
    → Same final T, DIFFERENT λ!

Static DFT: Always same result (history-blind)
DSE: Path-dependent (quantum memory)

Supported models:
  - heisenberg: Heisenberg XXX model
  - xy: XY model
  - kitaev: Kitaev honeycomb (rectangular approx)
  - ising: Transverse-field Ising model

Usage:
    memory-dft lattice --model heisenberg --Lx 3 --Ly 3
    memory-dft lattice --model ising --Lx 3 --Ly 3 --path-compare
    memory-dft lattice --model heisenberg --Lx 3 --Ly 3 --thermal

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
# Lattice DSE Runner (True Time Evolution)
# =============================================================================

class LatticeDSERunner:
    """
    2D Lattice DSE with time evolution.
    
    Key difference from static calculation:
    - Static: eigsh(H) at each step → no memory
    - DSE: exp(-iHdt)|ψ⟩ → history preserved!
    """
    
    SUPPORTED_MODELS = ['heisenberg', 'xy', 'kitaev', 'ising']
    
    def __init__(self, lx: int, ly: int, model: str = 'heisenberg',
                 j: float = 1.0, verbose: bool = True):
        self.lx = lx
        self.ly = ly
        self.n_sites = lx * ly
        self.dim = 2 ** self.n_sites
        self.model = model.lower()
        self.j = j
        self.verbose = verbose
        
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model}. Use: {self.SUPPORTED_MODELS}")
        
        # Check dimension (avoid memory explosion)
        if self.dim > 2**16:
            raise ValueError(f"System too large: dim={self.dim}. Max 2^16=65536")
        
        # Import modules
        try:
            from memory_dft.core.lattice import LatticeGeometry2D
            from memory_dft.core.operators import SpinOperators
            from memory_dft.core.hamiltonian import HamiltonianBuilder
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp
            
            self.LatticeGeometry2D = LatticeGeometry2D
            self.SpinOperators = SpinOperators
            self.HamiltonianBuilder = HamiltonianBuilder
            self.eigsh = eigsh
            self.sp = sp
        except ImportError as e:
            raise ImportError(f"Could not import modules: {e}")
        
        # Boltzmann constant
        self.K_B_EV = 8.617333262e-5
        
        # Build lattice and operators
        self.lattice = self.LatticeGeometry2D(lx, ly, periodic_x=True, periodic_y=True)
        self.ops = self.SpinOperators(self.n_sites)
        self.builder = self.HamiltonianBuilder(self.lattice, self.ops)
        
        if verbose:
            print(f"  Lattice: {lx}×{ly} = {self.n_sites} sites")
            print(f"  Hilbert space: 2^{self.n_sites} = {self.dim}")
            print(f"  Model: {model}")
    
    def build_hamiltonian(self, h_field: float = 0.0, 
                          kx: float = 1.0, ky: float = 0.8, kz: float = 0.3):
        """Build Hamiltonian with optional transverse field."""
        if self.model == 'heisenberg':
            H = self.builder.heisenberg(J=self.j)
        elif self.model == 'xy':
            H = self.builder.xy(J=self.j)
        elif self.model == 'kitaev':
            H = self.builder.kitaev_rect(Kx=kx, Ky=ky, Kz_diag=kz)
        elif self.model == 'ising':
            H = self.builder.ising(J=self.j, h=h_field)
            return H  # Ising already includes field
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        # Add transverse field for non-Ising models
        if abs(h_field) > 1e-10:
            H = H + h_field * self.ops.S_total_x
        
        return H
    
    def build_H_KV(self, h_field: float = 0.0):
        """Build H_K (kinetic) and H_V (potential) separately for λ calculation."""
        # For spin models: K = exchange, V = field
        if self.model == 'heisenberg':
            H_K = self.builder.heisenberg(J=self.j)
        elif self.model == 'xy':
            H_K = self.builder.xy(J=self.j)
        elif self.model == 'ising':
            # Ising: K = ZZ interaction, V = transverse field
            H_K = self.sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
            for (i, j_site) in self.lattice.bonds_nn:
                H_K += self.j * self.ops.Sz[i] @ self.ops.Sz[j_site]
            H_V = h_field * self.ops.S_total_x
            return H_K, H_V
        else:
            H_K = self.build_hamiltonian(h_field=0)
        
        H_V = h_field * self.ops.S_total_x if abs(h_field) > 1e-10 else \
              self.sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        
        return H_K, H_V
    
    def compute_ground_state(self, H):
        """Compute ground state."""
        try:
            E0, psi0 = self.eigsh(H, k=1, which='SA')
            return E0[0], psi0[:, 0]
        except Exception as e:
            if self.verbose:
                print(f"  Warning: eigsh failed ({e}), using dense")
            eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
            return eigenvalues[0], eigenvectors[:, 0]
    
    def compute_lambda(self, psi: np.ndarray, H_K, H_V) -> float:
        """Compute stability parameter λ = |K|/|V|."""
        K = float(np.real(np.vdot(psi, H_K @ psi)))
        V = float(np.real(np.vdot(psi, H_V @ psi)))
        return abs(K) / (abs(V) + 1e-10)
    
    def time_evolve(self, psi: np.ndarray, H, dt: float) -> np.ndarray:
        """Time evolution: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩."""
        try:
            from memory_dft.solvers import lanczos_expm_multiply
            return lanczos_expm_multiply(H, psi, dt, krylov_dim=20)
        except ImportError:
            from scipy.linalg import expm
            U = expm(-1j * H.toarray() * dt)
            return U @ psi
    
    # =========================================================================
    # Single Simulation (Static)
    # =========================================================================
    
    def run_single(self, h_field: float = 0.5) -> Dict[str, Any]:
        """Run single static simulation."""
        H = self.build_hamiltonian(h_field)
        E0, psi0 = self.compute_ground_state(H)
        
        # Observables
        mz = float(np.real(np.vdot(psi0, self.ops.S_total_z @ psi0)))
        
        # NN correlation
        i, j = self.lattice.bonds_nn[0]
        SiSj = self.ops.Sz[i] @ self.ops.Sz[j]
        corr = float(np.real(np.vdot(psi0, SiSj @ psi0)))
        
        return {
            'E0': float(E0),
            'magnetization': mz,
            'nn_correlation': corr,
            'h_field': h_field,
        }
    
    # =========================================================================
    # Field Path Comparison (DSE)
    # =========================================================================
    
    def evolve_field_path(self, h_values: List[float], 
                          dt: float = 0.1, steps_per_h: int = 10) -> Dict[str, Any]:
        """
        Evolve along field path with time evolution.
        
        This is TRUE DSE - state evolves, history matters!
        """
        # Start from ground state at initial field
        h0 = h_values[0]
        H0 = self.build_hamiltonian(h0)
        _, psi = self.compute_ground_state(H0)
        
        # Track evolution
        times = []
        lambdas = []
        energies = []
        t = 0.0
        
        for h in h_values:
            H = self.build_hamiltonian(h)
            H_K, H_V = self.build_H_KV(h)
            
            for step in range(steps_per_h):
                # Compute observables
                E = float(np.real(np.vdot(psi, H @ psi)))
                lam = self.compute_lambda(psi, H_K, H_V)
                
                times.append(t)
                lambdas.append(lam)
                energies.append(E)
                
                # Time evolve
                psi = self.time_evolve(psi, H, dt)
                psi = psi / np.linalg.norm(psi)  # Normalize
                t += dt
        
        return {
            'times': times,
            'lambdas': lambdas,
            'energies': energies,
            'lambda_final': lambdas[-1] if lambdas else 0.0,
            'E_final': energies[-1] if energies else 0.0,
            'h_values': h_values,
        }
    
    def compare_field_paths(self, h_final: float = 0.5, 
                            steps: int = 5, dt: float = 0.1,
                            steps_per_h: int = 10) -> Dict[str, Any]:
        """
        Compare two field paths to same final H.
        
        Path 1: h=0 → h=h_final (increase)
        Path 2: h=2*h_final → h=h_final (decrease)
        """
        # Build paths
        path1_h = list(np.linspace(0, h_final, steps))
        path2_h = list(np.linspace(2*h_final, h_final, steps))
        
        if self.verbose:
            print(f"\n  Path 1: h=0 → h={h_final}")
            print(f"  Path 2: h={2*h_final} → h={h_final}")
        
        # Evolve both paths
        result1 = self.evolve_field_path(path1_h, dt, steps_per_h)
        result2 = self.evolve_field_path(path2_h, dt, steps_per_h)
        
        # Static (DFT) reference
        H_final = self.build_hamiltonian(h_final)
        H_K, H_V = self.build_H_KV(h_final)
        E_static, psi_static = self.compute_ground_state(H_final)
        lambda_static = self.compute_lambda(psi_static, H_K, H_V)
        
        delta_dse = abs(result1['lambda_final'] - result2['lambda_final'])
        
        return {
            'path1': {
                'label': f'h: 0→{h_final}',
                'lambda_final': result1['lambda_final'],
                'E_final': result1['E_final'],
            },
            'path2': {
                'label': f'h: {2*h_final}→{h_final}',
                'lambda_final': result2['lambda_final'],
                'E_final': result2['E_final'],
            },
            'static': {
                'lambda': lambda_static,
                'E': float(E_static),
            },
            'dse': {
                'delta_lambda': delta_dse,
            },
            'h_final': h_final,
        }
    
    # =========================================================================
    # Thermal Path Comparison (DSE)
    # =========================================================================
    
    def T_to_beta(self, T_kelvin: float, energy_scale: float = 1.0) -> float:
        """Convert temperature to inverse temperature."""
        if T_kelvin <= 0:
            return float('inf')
        return energy_scale / (self.K_B_EV * T_kelvin)
    
    def boltzmann_weights(self, eigenvalues: np.ndarray, beta: float) -> np.ndarray:
        """Compute Boltzmann weights."""
        if beta == float('inf'):
            weights = np.zeros(len(eigenvalues))
            weights[0] = 1.0
            return weights
        
        E_min = np.min(eigenvalues)
        E_shifted = eigenvalues - E_min
        boltzmann = np.exp(-beta * E_shifted)
        return boltzmann / np.sum(boltzmann)
    
    def diagonalize(self, H, n_states: int = 14):
        """Diagonalize Hamiltonian."""
        n_states = min(n_states, self.dim - 2)
        eigenvalues, eigenvectors = self.eigsh(H, k=n_states, which='SA')
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def evolve_thermal_path(self, temps: List[float], H, H_K, H_V,
                            eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                            dt: float = 0.1, steps_per_T: int = 10) -> Dict[str, Any]:
        """Evolve along temperature path."""
        # Initialize with thermal state at T0
        T0 = temps[0]
        beta0 = self.T_to_beta(T0)
        weights = self.boltzmann_weights(eigenvalues, beta0)
        
        # Active states
        active_mask = weights > 1e-10
        active_indices = np.where(active_mask)[0]
        
        evolved_psis = [eigenvectors[:, i].copy() for i in active_indices]
        evolved_weights = [weights[i] for i in active_indices]
        
        times = []
        lambdas = []
        t = 0.0
        
        for T in temps:
            for step in range(steps_per_T):
                # Thermal average λ
                K_total = 0.0
                V_total = 0.0
                
                for i, psi in enumerate(evolved_psis):
                    w = evolved_weights[i]
                    K = float(np.real(np.vdot(psi, H_K @ psi)))
                    V = float(np.real(np.vdot(psi, H_V @ psi)))
                    K_total += w * K
                    V_total += w * V
                
                lam = abs(K_total) / (abs(V_total) + 1e-10)
                times.append(t)
                lambdas.append(lam)
                
                # Time evolve each state
                for i in range(len(evolved_psis)):
                    evolved_psis[i] = self.time_evolve(evolved_psis[i], H, dt)
                    evolved_psis[i] = evolved_psis[i] / np.linalg.norm(evolved_psis[i])
                
                t += dt
        
        return {
            'times': times,
            'lambdas': lambdas,
            'lambda_final': lambdas[-1] if lambdas else 0.0,
            'n_active': len(active_indices),
        }
    
    def compare_thermal_paths(self, h_field: float = 0.5,
                              T_high: float = 300.0, T_low: float = 50.0,
                              T_final: float = 150.0,
                              steps: int = 5, n_states: int = 14) -> Dict[str, Any]:
        """Compare two thermal paths."""
        # Build Hamiltonian and diagonalize
        H = self.build_hamiltonian(h_field)
        H_K, H_V = self.build_H_KV(h_field)
        eigenvalues, eigenvectors = self.diagonalize(H, n_states)
        
        # Temperature paths
        path1_temps = list(np.linspace(T_low, T_high, steps)) + \
                      list(np.linspace(T_high, T_final, steps))
        path2_temps = list(np.linspace(T_high, T_low, steps)) + \
                      list(np.linspace(T_low, T_final, steps))
        
        if self.verbose:
            print(f"\n  Path 1: {T_low}K → {T_high}K → {T_final}K")
            print(f"  Path 2: {T_high}K → {T_low}K → {T_final}K")
        
        result1 = self.evolve_thermal_path(path1_temps, H, H_K, H_V,
                                            eigenvalues, eigenvectors)
        result2 = self.evolve_thermal_path(path2_temps, H, H_K, H_V,
                                            eigenvalues, eigenvectors)
        
        # Equilibrium (static)
        beta_final = self.T_to_beta(T_final)
        weights = self.boltzmann_weights(eigenvalues, beta_final)
        K_eq = sum(weights[n] * float(np.real(np.vdot(eigenvectors[:,n], 
                   H_K @ eigenvectors[:,n]))) for n in range(len(eigenvalues)))
        V_eq = sum(weights[n] * float(np.real(np.vdot(eigenvectors[:,n],
                   H_V @ eigenvectors[:,n]))) for n in range(len(eigenvalues)))
        lambda_eq = abs(K_eq) / (abs(V_eq) + 1e-10)
        
        delta_dse = abs(result1['lambda_final'] - result2['lambda_final'])
        
        return {
            'path1': {
                'label': f'{T_low}K→{T_high}K→{T_final}K',
                'lambda_final': result1['lambda_final'],
                'n_active': result1['n_active'],
            },
            'path2': {
                'label': f'{T_high}K→{T_low}K→{T_final}K',
                'lambda_final': result2['lambda_final'],
                'n_active': result2['n_active'],
            },
            'equilibrium': {
                'lambda': lambda_eq,
            },
            'dse': {
                'delta_lambda': delta_dse,
            },
            'T_final': T_final,
        }


# =============================================================================
# CLI Command
# =============================================================================

def lattice(
    model: str = typer.Option("heisenberg", "--model", "-m", 
                              help="Model: heisenberg, xy, kitaev, ising"),
    lx: int = typer.Option(3, "--Lx", help="Lattice size X"),
    ly: int = typer.Option(3, "--Ly", help="Lattice size Y"),
    j: float = typer.Option(1.0, "-J", "--J", help="Exchange coupling J"),
    h_field: float = typer.Option(0.5, "-h", "--h", help="Transverse field"),
    kx: float = typer.Option(1.0, "--Kx", help="Kitaev Kx"),
    ky: float = typer.Option(0.8, "--Ky", help="Kitaev Ky"),
    kz: float = typer.Option(0.3, "--Kz", help="Kitaev Kz"),
    path_compare: bool = typer.Option(False, "--path-compare", "-p",
                                       help="Compare field paths (DSE)"),
    thermal: bool = typer.Option(False, "--thermal", "-T",
                                  help="Compare thermal paths (DSE)"),
    t_high: float = typer.Option(300.0, "--T-high", help="High temperature (K)"),
    t_low: float = typer.Option(50.0, "--T-low", help="Low temperature (K)"),
    t_final: float = typer.Option(150.0, "--T-final", help="Final temperature (K)"),
    steps: int = typer.Option(5, "--steps", help="Steps per path segment"),
    output: Optional[Path] = typer.Option(None, "-o", "--output",
                                          help="Output JSON file"),
):
    """
    2D Lattice DSE simulation.
    
    Three modes:
      1. Single (default): Static ground state calculation
      2. --path-compare: Field path dependence (TRUE DSE)
      3. --thermal: Temperature path dependence (Thermal-DSE)
    
    DSE shows path dependence that DFT cannot capture!
    """
    print_banner()
    
    print_section("2D Lattice DSE", "🔲")
    print_key_value("Model", model)
    print_key_value("Lattice", f"{lx}×{ly}")
    print_key_value("J", str(j))
    
    if path_compare:
        print_key_value("Mode", "Field path comparison (DSE)")
        print_key_value("h_final", str(h_field))
    elif thermal:
        print_key_value("Mode", "Thermal path comparison (DSE)")
        print_key_value("T_high/T_low/T_final", f"{t_high}/{t_low}/{t_final} K")
    else:
        print_key_value("Mode", "Single (static)")
        print_key_value("h", str(h_field))
    
    typer.echo(f"\n  Hilbert space: 2^{lx*ly} = {2**(lx*ly)}")
    typer.echo("")
    
    # Initialize runner
    try:
        runner = LatticeDSERunner(lx, ly, model, j, verbose=True)
    except (ImportError, ValueError) as e:
        error_exit(str(e), "Check parameters or install dependencies")
    
    # Run appropriate mode
    if path_compare:
        typer.echo("\nRunning field path comparison (DSE)...")
        results = runner.compare_field_paths(h_field, steps)
        
        print_section("Field Path Results", "📊")
        
        p1 = results['path1']
        p2 = results['path2']
        
        typer.echo(f"  Path 1 ({p1['label']}):")
        typer.echo(f"    λ_final: {p1['lambda_final']:.4f}")
        typer.echo(f"    E_final: {p1['E_final']:.6f}")
        typer.echo("")
        
        typer.echo(f"  Path 2 ({p2['label']}):")
        typer.echo(f"    λ_final: {p2['lambda_final']:.4f}")
        typer.echo(f"    E_final: {p2['E_final']:.6f}")
        typer.echo("")
        
        typer.echo(f"  Static (DFT-like):")
        typer.echo(f"    λ: {results['static']['lambda']:.4f}")
        typer.echo(f"    E: {results['static']['E']:.6f}")
        typer.echo("")
        
        delta = results['dse']['delta_lambda']
        typer.echo(f"  DSE Path Difference:")
        typer.echo(f"    |Δλ|: {delta:.4f}")
        
        if delta > 0.01:
            typer.echo("\n  ✨ Path dependence detected!")
            typer.echo("  → Same final H, DIFFERENT quantum states")
        
    elif thermal:
        typer.echo("\nRunning thermal path comparison (DSE)...")
        results = runner.compare_thermal_paths(h_field, t_high, t_low, t_final, steps)
        
        print_section("Thermal Path Results", "📊")
        
        p1 = results['path1']
        p2 = results['path2']
        
        typer.echo(f"  Path 1 ({p1['label']}):")
        typer.echo(f"    Active states: {p1['n_active']}")
        typer.echo(f"    λ_final: {p1['lambda_final']:.4f}")
        typer.echo("")
        
        typer.echo(f"  Path 2 ({p2['label']}):")
        typer.echo(f"    Active states: {p2['n_active']}")
        typer.echo(f"    λ_final: {p2['lambda_final']:.4f}")
        typer.echo("")
        
        typer.echo(f"  Equilibrium (DFT-like at {t_final}K):")
        typer.echo(f"    λ: {results['equilibrium']['lambda']:.4f}")
        typer.echo("")
        
        delta = results['dse']['delta_lambda']
        typer.echo(f"  DSE Path Difference:")
        typer.echo(f"    |Δλ|: {delta:.4f}")
        
        if delta > 0.01:
            typer.echo("\n  ✨ Thermal path dependence detected!")
            typer.echo("  → Same final T, DIFFERENT quantum states")
        
    else:
        typer.echo("\nComputing ground state (static)...")
        results = runner.run_single(h_field)
        
        print_section("Results", "📊")
        typer.echo(f"  Ground state energy: {results['E0']:.6f}")
        typer.echo(f"  Magnetization <Sz>: {results['magnetization']:.4f}")
        typer.echo(f"  NN correlation: {results['nn_correlation']:.4f}")
    
    # Save results
    if output:
        save_json(results, output)
        typer.echo(f"\n💾 Results saved to {output}")
    
    typer.echo("\n✅ Done!")
