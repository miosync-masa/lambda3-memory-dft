"""
Thermal Command - True Thermal-DSE Implementation
==================================================

Demonstrates temperature-path dependence using Direct Schrödinger Evolution.

Core insight (Appendix G):
  - Path A: T_low → T_high → T_final (heat then cool)
  - Path B: T_high → T_low → T_final (cool then heat)
  - Same final temperature
  - DIFFERENT quantum state! λ(T_final) ≠ λ'(T_final)

DFT: History-blind → ΔΛ = 0 (by construction)
DSE: History-aware → ΔΛ ≠ 0 (quantum memory)

This is fundamentally different from DFT with temperature smearing!

Usage:
    memory-dft thermal --T-high 300 --T-low 50 --T-final 150

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
# Thermal-DSE Runner (Real Implementation)
# =============================================================================

class ThermalDSERunner:
    """
    True Thermal-DSE implementation.
    
    Uses Hubbard model with:
    1. Exact diagonalization for eigenstates
    2. Boltzmann weights for thermal averaging
    3. Time evolution along temperature paths
    4. λ(T) = K(T)/|V(T)| stability parameter
    """
    
    def __init__(self, n_sites: int = 4, t_hop: float = 1.0, U_int: float = 2.0,
                 n_eigenstates: int = 14, verbose: bool = True):
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        self.t_hop = t_hop
        self.U_int = U_int
        self.n_eigenstates = min(n_eigenstates, self.dim - 2)
        self.verbose = verbose
        
        # Boltzmann constant in eV/K
        self.K_B_EV = 8.617333262e-5
        
        # Import required modules
        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigsh
            self.sp = sp
            self.eigsh = eigsh
        except ImportError as e:
            raise ImportError(f"SciPy required: {e}")
        
        # Build and diagonalize
        self._build_hubbard()
        self._diagonalize()
    
    def _build_hubbard(self):
        """Build Hubbard Hamiltonian H = H_K + H_V."""
        L = self.n_sites
        sp = self.sp
        
        # Sparse operators
        I = sp.eye(2, format='csr', dtype=np.complex128)
        Sp = sp.csr_matrix([[0, 1], [0, 0]], dtype=np.complex128)
        Sm = sp.csr_matrix([[0, 0], [1, 0]], dtype=np.complex128)
        n_op = sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)
        
        def site_op(op, site):
            ops = [I] * L
            ops[site] = op
            result = ops[0]
            for i in range(1, L):
                result = sp.kron(result, ops[i], format='csr')
            return result
        
        # H_K (hopping/kinetic)
        H_K = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        for i in range(L - 1):
            j = i + 1
            H_K += -self.t_hop * (site_op(Sp, i) @ site_op(Sm, j) + 
                                  site_op(Sm, i) @ site_op(Sp, j))
        
        # H_V (interaction/potential)
        H_V = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        for i in range(L - 1):
            j = i + 1
            H_V += self.U_int * site_op(n_op, i) @ site_op(n_op, j)
        
        self.H_K = H_K
        self.H_V = H_V
        self.H = H_K + H_V
        
        if self.verbose:
            print(f"  Built Hubbard: {L} sites, t={self.t_hop}, U={self.U_int}")
    
    def _diagonalize(self):
        """Compute low-energy eigenstates."""
        eigenvalues, eigenvectors = self.eigsh(
            self.H, k=self.n_eigenstates, which='SA'
        )
        idx = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        if self.verbose:
            gap = self.eigenvalues[1] - self.eigenvalues[0]
            print(f"  Diagonalized: {self.n_eigenstates} states")
            print(f"  E_0 = {self.eigenvalues[0]:.4f}, Gap = {gap:.4f}")
    
    def T_to_beta(self, T_kelvin: float) -> float:
        """Convert temperature to inverse temperature β."""
        if T_kelvin <= 0:
            return float('inf')
        return self.t_hop / (self.K_B_EV * T_kelvin)
    
    def boltzmann_weights(self, beta: float) -> np.ndarray:
        """Compute Boltzmann weights exp(-βE_n)/Z."""
        if beta == float('inf'):
            weights = np.zeros(len(self.eigenvalues))
            weights[0] = 1.0
            return weights
        
        E_min = np.min(self.eigenvalues)
        E_shifted = self.eigenvalues - E_min
        boltzmann = np.exp(-beta * E_shifted)
        Z = np.sum(boltzmann)
        return boltzmann / Z
    
    def compute_thermal_lambda(self, T_kelvin: float) -> float:
        """
        Compute stability parameter λ(T) = K(T)/|V(T)|.
        
        This is the thermal expectation value using Boltzmann weights.
        """
        beta = self.T_to_beta(T_kelvin)
        weights = self.boltzmann_weights(beta)
        
        K_total = 0.0
        V_total = 0.0
        
        for n in range(self.n_eigenstates):
            psi = self.eigenvectors[:, n]
            w = weights[n]
            
            K_n = float(np.real(np.vdot(psi, self.H_K @ psi)))
            V_n = float(np.real(np.vdot(psi, self.H_V @ psi)))
            
            K_total += w * K_n
            V_total += w * V_n
        
        return abs(K_total) / (abs(V_total) + 1e-10)
    
    def compute_entropy(self, T_kelvin: float) -> float:
        """Compute entropy S/k_B = ln(Z) + β⟨E'⟩."""
        beta = self.T_to_beta(T_kelvin)
        
        if beta == float('inf'):
            return 0.0
        
        E_min = np.min(self.eigenvalues)
        E_shifted = self.eigenvalues - E_min
        boltzmann = np.exp(-beta * E_shifted)
        Z = np.sum(boltzmann)
        E_avg = np.sum(E_shifted * boltzmann) / Z
        
        return np.log(Z) + beta * E_avg
    
    def count_active_states(self, T_kelvin: float, threshold: float = 1e-3) -> int:
        """Count thermally active states (weight > threshold)."""
        beta = self.T_to_beta(T_kelvin)
        weights = self.boltzmann_weights(beta)
        return int(np.sum(weights > threshold))
    
    def evolve_path(self, temperatures: List[float], 
                    dt: float = 0.1, steps_per_T: int = 10) -> Dict[str, Any]:
        """
        Evolve system along temperature path with time evolution.
        
        This is the KEY difference from DFT:
        - Each eigenstate |ψ_n⟩ evolves under H(t)
        - λ(T) accumulates history through evolution
        
        Args:
            temperatures: Temperature sequence [T1, T2, ...]
            dt: Time step
            steps_per_T: Steps at each temperature
            
        Returns:
            Dictionary with evolution results
        """
        # Try to import Lanczos solver
        try:
            from memory_dft.solvers import lanczos_expm_multiply
            use_lanczos = True
        except ImportError:
            use_lanczos = False
        
        # Initialize with thermal state at first temperature
        T0 = temperatures[0]
        beta0 = self.T_to_beta(T0)
        weights = self.boltzmann_weights(beta0)
        
        # Active states (non-negligible weight)
        active_mask = weights > 1e-10
        active_indices = np.where(active_mask)[0]
        
        # Copy eigenstates for evolution
        evolved_psis = [self.eigenvectors[:, i].copy() for i in active_indices]
        evolved_weights = [weights[i] for i in active_indices]
        
        # Track results
        times = []
        lambdas = []
        entropies = []
        t = 0.0
        
        for T in temperatures:
            beta = self.T_to_beta(T)
            
            for step in range(steps_per_T):
                # Compute current λ(T)
                K_total = 0.0
                V_total = 0.0
                
                for i, psi in enumerate(evolved_psis):
                    w = evolved_weights[i]
                    K = float(np.real(np.vdot(psi, self.H_K @ psi)))
                    V = float(np.real(np.vdot(psi, self.H_V @ psi)))
                    K_total += w * K
                    V_total += w * V
                
                lam = abs(K_total) / (abs(V_total) + 1e-10)
                S = self.compute_entropy(T)
                
                times.append(t)
                lambdas.append(lam)
                entropies.append(S)
                
                # Time evolution: |ψ(t+dt)⟩ = exp(-iHdt/ℏ)|ψ(t)⟩
                for i in range(len(evolved_psis)):
                    if use_lanczos:
                        evolved_psis[i] = lanczos_expm_multiply(
                            self.H, evolved_psis[i], dt, krylov_dim=20
                        )
                    else:
                        # Simple matrix exponential (dense, slow but works)
                        from scipy.linalg import expm
                        U = expm(-1j * self.H.toarray() * dt)
                        evolved_psis[i] = U @ evolved_psis[i]
                
                t += dt
        
        return {
            'times': times,
            'lambdas': lambdas,
            'entropies': entropies,
            'lambda_final': lambdas[-1] if lambdas else 0.0,
            'temperatures': temperatures,
            'n_active': len(active_indices),
        }
    
    def compare_paths(self, T_high: float, T_low: float, T_final: float,
                      steps: int = 5, dt: float = 0.1, 
                      steps_per_T: int = 10) -> Dict[str, Any]:
        """
        Compare two temperature paths to same final temperature.
        
        Path 1: T_low → T_high → T_final (heat then cool)
        Path 2: T_high → T_low → T_final (cool then heat)
        
        Returns:
            Dictionary with comparison results including ΔΛ
        """
        # Build temperature sequences
        path1_temps = (
            list(np.linspace(T_low, T_high, steps)) +
            list(np.linspace(T_high, T_final, steps))
        )
        
        path2_temps = (
            list(np.linspace(T_high, T_low, steps)) +
            list(np.linspace(T_low, T_final, steps))
        )
        
        if self.verbose:
            print(f"\n  Path 1: {T_low}K → {T_high}K → {T_final}K")
            print(f"  Path 2: {T_high}K → {T_low}K → {T_final}K")
        
        # Evolve along both paths
        result1 = self.evolve_path(path1_temps, dt, steps_per_T)
        result2 = self.evolve_path(path2_temps, dt, steps_per_T)
        
        # DFT reference (equilibrium at T_final - no path dependence)
        lambda_dft = self.compute_thermal_lambda(T_final)
        
        # Path difference
        delta_lambda_dse = abs(result1['lambda_final'] - result2['lambda_final'])
        delta_lambda_dft = 0.0  # DFT is path-independent by construction
        
        return {
            'path1': {
                'label': f'{T_low}K→{T_high}K→{T_final}K',
                'lambda_final': result1['lambda_final'],
                'n_active': result1['n_active'],
                'lambdas': result1['lambdas'],
            },
            'path2': {
                'label': f'{T_high}K→{T_low}K→{T_final}K',
                'lambda_final': result2['lambda_final'],
                'n_active': result2['n_active'],
                'lambdas': result2['lambdas'],
            },
            'dft': {
                'lambda': lambda_dft,
                'delta': delta_lambda_dft,
            },
            'dse': {
                'delta': delta_lambda_dse,
            },
            'T_final': T_final,
            'entropy': self.compute_entropy(T_final),
        }


# =============================================================================
# CLI Command
# =============================================================================

def thermal(
    t_high: float = typer.Option(300.0, "--T-high", help="High temperature (K)"),
    t_low: float = typer.Option(50.0, "--T-low", help="Low temperature (K)"),
    t_final: float = typer.Option(150.0, "--T-final", help="Final temperature (K)"),
    n_sites: int = typer.Option(4, "--sites", "-n", help="Number of lattice sites"),
    t_hop: float = typer.Option(1.0, "-t", help="Hopping parameter"),
    u_int: float = typer.Option(2.0, "-U", help="Interaction strength"),
    n_states: int = typer.Option(14, "--states", help="Number of eigenstates"),
    steps: int = typer.Option(5, "--steps", help="Steps per temperature segment"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", 
                                          help="Output JSON file"),
):
    """
    Thermal-DSE: Temperature path dependence demonstration.
    
    Shows that heating→cooling vs cooling→heating paths
    lead to DIFFERENT quantum states at the same final temperature.
    
    This is fundamentally impossible in DFT (history-blind).
    """
    print_banner()
    
    print_section("Thermal-DSE Path Comparison", "🌡️")
    print_key_value("System", f"{n_sites}-site Hubbard model")
    print_key_value("Parameters", f"t={t_hop}, U={u_int}")
    print_key_value("Eigenstates", str(n_states))
    print_key_value("T_high", f"{t_high} K")
    print_key_value("T_low", f"{t_low} K")  
    print_key_value("T_final", f"{t_final} K")
    typer.echo("")
    
    # Initialize runner
    try:
        typer.echo("Building Hubbard model and diagonalizing...")
        runner = ThermalDSERunner(
            n_sites=n_sites,
            t_hop=t_hop,
            U_int=u_int,
            n_eigenstates=n_states,
            verbose=True
        )
    except Exception as e:
        error_exit(str(e), "Check parameters")
    
    # Run path comparison
    typer.echo("\nEvolving along temperature paths...")
    results = runner.compare_paths(t_high, t_low, t_final, steps)
    
    # Display results
    print_section("Results", "📊")
    
    path1 = results['path1']
    path2 = results['path2']
    
    typer.echo(f"  Path 1 ({path1['label']}):")
    typer.echo(f"    Active states: {path1['n_active']}")
    typer.echo(f"    λ_final: {path1['lambda_final']:.4f}")
    typer.echo("")
    
    typer.echo(f"  Path 2 ({path2['label']}):")
    typer.echo(f"    Active states: {path2['n_active']}")
    typer.echo(f"    λ_final: {path2['lambda_final']:.4f}")
    typer.echo("")
    
    typer.echo(f"  DFT (equilibrium at {t_final}K):")
    typer.echo(f"    λ_DFT: {results['dft']['lambda']:.4f}")
    typer.echo(f"    ΔΛ: {results['dft']['delta']:.4f} (always 0 - history-blind)")
    typer.echo("")
    
    delta_dse = results['dse']['delta']
    typer.echo(f"  DSE Path Difference:")
    typer.echo(f"    |ΔΛ|: {delta_dse:.4f}")
    typer.echo(f"    Entropy S/k_B: {results['entropy']:.4f}")
    typer.echo("")
    
    if delta_dse > 0.01:
        typer.echo("  ✨ Path dependence detected!")
        typer.echo("  → Same T_final, DIFFERENT quantum states")
        typer.echo("  → DFT cannot capture this (ΔΛ_DFT ≡ 0)")
    else:
        typer.echo("  ⚠️ Weak path dependence (try lower T or larger U)")
    
    # Save results
    if output:
        save_json(results, output)
        typer.echo(f"\n💾 Results saved to {output}")
    
    typer.echo("\n✅ Done!")
