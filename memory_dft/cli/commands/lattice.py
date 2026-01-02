"""
Lattice Command - True DSE Implementation (Unified Engine)
==========================================================

2D Lattice simulation with time evolution and path dependence.
Uses unified SparseEngine for all operations.

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
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from ..utils import (
    print_banner, print_section, print_key_value, 
    save_json, error_exit
)

# Memory quantification
try:
    from memory_dft.solvers.memory_indicators import MemoryIndicator, MemoryMetrics
    HAS_MEMORY_INDICATORS = True
except ImportError:
    HAS_MEMORY_INDICATORS = False


# =============================================================================
# Lattice DSE Runner (Using Unified SparseEngine)
# =============================================================================

class LatticeDSERunner:
    """
    2D Lattice DSE with time evolution.
    
    Uses unified SparseEngine for:
      - GPU/CPU automatic backend
      - All model Hamiltonians
      - λ calculation
      - Ground state computation
    
    Key difference from static calculation:
    - Static: eigsh(H) at each step → no memory
    - DSE: exp(-iHdt)|ψ⟩ → history preserved!
    """
    
    SUPPORTED_MODELS = ['heisenberg', 'xy', 'kitaev', 'ising']
    
    def __init__(self, lx: int, ly: int, model: str = 'heisenberg',
                 j: float = 1.0, energy_scale: float = 0.1,
                 use_gpu: bool = True, verbose: bool = True):
        self.lx = lx
        self.ly = ly
        self.n_sites = lx * ly
        self.dim = 2 ** self.n_sites
        self.model = model.lower()
        self.j = j
        self.energy_scale = energy_scale
        self.verbose = verbose
        
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model}. Use: {self.SUPPORTED_MODELS}")
        
        # Check dimension (avoid memory explosion)
        if self.dim > 2**16:
            raise ValueError(f"System too large: dim={self.dim}. Max 2^16=65536")
        
        # Import unified engine
        try:
            from memory_dft.core.sparse_engine_unified import SparseEngine
            self.SparseEngine = SparseEngine
        except ImportError as e:
            raise ImportError(f"Could not import SparseEngine: {e}")
        
        # Boltzmann constant
        self.K_B_EV = 8.617333262e-5
        
        # Build engine with square geometry
        self.engine = self.SparseEngine(self.n_sites, use_gpu=use_gpu, verbose=False)
        self.geometry = self.engine.build_square(lx, ly, periodic_x=True, periodic_y=True)
        
        if verbose:
            print(f"  Lattice: {lx}×{ly} = {self.n_sites} sites")
            print(f"  Hilbert space: 2^{self.n_sites} = {self.dim}")
            print(f"  Model: {model}")
            print(f"  Backend: {'GPU' if self.engine.use_gpu else 'CPU'}")
    
    def build_hamiltonian(self, h_field: float = 0.0, 
                          kx: float = 1.0, ky: float = 0.8, kz: float = 0.3):
        """Build Hamiltonian with optional transverse field."""
        bonds = self.geometry.bonds
        
        # Safety check
        if not bonds and self.model != 'kitaev':
            raise ValueError(f"No bonds found! Lattice {self.lx}x{self.ly} requires at least 2 sites.")
        
        if self.model == 'heisenberg':
            H = self.engine.build_heisenberg(bonds, J=self.j, split_KV=False)
        elif self.model == 'xy':
            H = self.engine.build_xy(bonds, J=self.j)
        elif self.model == 'kitaev':
            H = self.engine.build_kitaev(self.lx, self.ly, Kx=kx, Ky=ky, Kz=kz)
        elif self.model == 'ising':
            H = self.engine.build_ising(bonds, J=self.j, h=h_field, split_KV=False)
            return H  # Ising already includes field
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        # Verify Hamiltonian
        if H is None:
            raise ValueError(f"Failed to build {self.model} Hamiltonian.")
        
        # Add transverse field for non-Ising models
        if abs(h_field) > 1e-10:
            H = H + h_field * self.engine.S_total_x
        
        return H
    
    def build_H_KV(self, h_field: float = 0.0) -> Tuple:
        """Build H_K (kinetic) and H_V (potential) separately for λ calculation."""
        bonds = self.geometry.bonds
        
        # Safety check
        if not bonds and self.model != 'kitaev':
            raise ValueError(f"No bonds found! Lattice {self.lx}x{self.ly} requires at least 2 sites.")
        
        if self.model == 'heisenberg':
            H_K, H_V = self.engine.build_heisenberg(bonds, J=self.j, split_KV=True)
            if H_K is None or H_V is None:
                raise ValueError("Failed to build Heisenberg H_K/H_V")
            # Add field to H_V
            if abs(h_field) > 1e-10:
                H_V = H_V + h_field * self.engine.S_total_x
            return H_K, H_V
            
        elif self.model == 'xy':
            H_K = self.engine.build_xy(bonds, J=self.j)
            if H_K is None:
                raise ValueError("Failed to build XY Hamiltonian")
            H_V = h_field * self.engine.S_total_x if abs(h_field) > 1e-10 else \
                  self.engine.sparse.csr_matrix((self.dim, self.dim), dtype=np.complex128)
            return H_K, H_V
            
        elif self.model == 'ising':
            H_K, H_V = self.engine.build_ising(bonds, J=self.j, h=h_field, split_KV=True)
            if H_K is None or H_V is None:
                raise ValueError("Failed to build Ising H_K/H_V")
            return H_K, H_V
            
        elif self.model == 'kitaev':
            H = self.engine.build_kitaev(self.lx, self.ly)
            if H is None:
                raise ValueError("Failed to build Kitaev Hamiltonian")
            H_V = h_field * self.engine.S_total_x if abs(h_field) > 1e-10 else \
                  self.engine.sparse.csr_matrix((self.dim, self.dim), dtype=np.complex128)
            return H, H_V
        
        else:
            H_K = self.build_hamiltonian(h_field=0)
            H_V = self.engine.sparse.csr_matrix((self.dim, self.dim), dtype=np.complex128)
            return H_K, H_V
    
    def compute_ground_state(self, H):
        """Compute ground state using engine."""
        return self.engine.compute_ground_state(H)
    
    def compute_lambda(self, psi: np.ndarray, H_K, H_V) -> float:
        """Compute stability parameter λ = |K|/|V| using engine."""
        return self.engine.compute_lambda(psi, H_K, H_V)
    
    def time_evolve(self, psi: np.ndarray, H, dt: float) -> np.ndarray:
        """Time evolution: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩."""
        xp = self.engine.xp
        
        try:
            from memory_dft.solvers import lanczos_expm_multiply
            # lanczos_expm_multiply now auto-detects backend
            result = lanczos_expm_multiply(H, psi, dt, krylov_dim=20)
            # Ensure output is on correct backend
            if self.engine.use_gpu and isinstance(result, np.ndarray):
                return xp.asarray(result)
            return result
        except ImportError:
            # Fallback to dense expm
            from scipy.linalg import expm
            if self.engine.use_gpu:
                H_dense = H.toarray().get()
                psi_np = psi.get() if hasattr(psi, 'get') else psi
            else:
                H_dense = H.toarray()
                psi_np = psi
            U = expm(-1j * H_dense * dt)
            result = U @ psi_np
            if self.engine.use_gpu:
                return xp.asarray(result)
            return result
    
    # =========================================================================
    # Single Simulation (Static)
    # =========================================================================
    
    def run_single(self, h_field: float = 0.5) -> Dict[str, Any]:
        """Run single static simulation."""
        H = self.build_hamiltonian(h_field)
        E0, psi0 = self.compute_ground_state(H)
        
        # Observables
        mz = self.engine.compute_magnetization(psi0) * self.n_sites
        
        # NN correlation
        if len(self.geometry.bonds) > 0:
            i, j = self.geometry.bonds[0]
            corr = self.engine.compute_correlation(psi0, i, j, 'Z')
        else:
            corr = 0.0
        
        return {
            'E0': float(E0),
            'magnetization': float(mz),
            'nn_correlation': float(corr),
            'h_field': h_field,
        }
    
    # =========================================================================
    # Field Path Comparison (DSE)
    # =========================================================================
    
    def evolve_field_path(self, h_values: List[float], dt: float = 0.1) -> Dict[str, Any]:
        """
        Evolve along a field path with TRUE time evolution.
        
        Key: State evolves continuously, preserving history!
        """
        xp = self.engine.xp
        
        # Initial state: ground state at first h
        H0 = self.build_hamiltonian(h_values[0])
        E0, psi = self.compute_ground_state(H0)
        
        # Time evolution along path
        for h in h_values[1:]:
            H = self.build_hamiltonian(h)
            psi = self.time_evolve(psi, H, dt)
            psi = psi / xp.linalg.norm(psi)  # Renormalize
        
        # Final λ calculation
        H_K, H_V = self.build_H_KV(h_values[-1])
        H_final = H_K + H_V
        E_final = float(xp.real(xp.vdot(psi, H_final @ psi)))
        lambda_final = self.compute_lambda(psi, H_K, H_V)
        
        return {
            'lambda_final': lambda_final,
            'E_final': E_final,
            'psi_final': psi,
        }
    
    def compare_field_paths(self, h_final: float = 0.5, 
                            steps: int = 10) -> Dict[str, Any]:
        """
        Compare two different field paths to the same final Hamiltonian.
        
        Path 1: h = 0 → h_final (field increase)
        Path 2: h = 2*h_final → h_final (field decrease)
        
        DSE shows different results. DFT cannot distinguish!
        """
        dt = 0.1
        
        # Path 1: 0 → h_final
        path1_h = list(np.linspace(0, h_final, steps))
        
        # Path 2: 2*h_final → h_final
        path2_h = list(np.linspace(2 * h_final, h_final, steps))
        
        if self.verbose:
            print(f"\n  Path 1: h = 0 → {h_final}")
            print(f"  Path 2: h = {2*h_final} → {h_final}")
        
        result1 = self.evolve_field_path(path1_h, dt)
        result2 = self.evolve_field_path(path2_h, dt)
        
        # Static calculation (DFT-like)
        H_K, H_V = self.build_H_KV(h_final)
        H = H_K + H_V
        E_static, psi_static = self.compute_ground_state(H)
        lambda_static = self.compute_lambda(psi_static, H_K, H_V)
        
        delta_lambda = abs(result1['lambda_final'] - result2['lambda_final'])
        
        # Compute memory metrics
        memory_metrics = None
        if HAS_MEMORY_INDICATORS:
            memory_metrics = MemoryIndicator.compute_all(
                O_forward=result1['lambda_final'],
                O_backward=result2['lambda_final']
            )
        
        return {
            'path1': {
                'label': f'0→{h_final}',
                'lambda_final': result1['lambda_final'],
                'E_final': result1['E_final'],
            },
            'path2': {
                'label': f'{2*h_final}→{h_final}',
                'lambda_final': result2['lambda_final'],
                'E_final': result2['E_final'],
            },
            'static': {
                'lambda': lambda_static,
                'E': float(E_static),
            },
            'dse': {
                'delta_lambda': delta_lambda,
            },
            'memory_metrics': memory_metrics,
            'h_final': h_final,
        }
    
    # =========================================================================
    # Thermal Path (DSE)
    # =========================================================================
    
    def T_to_beta(self, T: float) -> float:
        """Temperature to inverse temperature β = E_scale / (k_B T)."""
        return self.energy_scale / (self.K_B_EV * T) if T > 0 else float('inf')
    
    def boltzmann_weights(self, eigenvalues: np.ndarray, beta: float) -> np.ndarray:
        """Compute Boltzmann weights."""
        E_shifted = eigenvalues - eigenvalues.min()
        exp_vals = np.exp(-beta * E_shifted)
        return exp_vals / exp_vals.sum()
    
    def diagonalize(self, H, n_states: int = 14):
        """Full diagonalization for thermal calculations."""
        xp = self.engine.xp
        
        try:
            # Sparse eigensolve for lowest n_states
            eigenvalues, eigenvectors = self.engine.eigsh(H, k=n_states, which='SA')
            # eigenvalues → CPU (for Boltzmann weights with np.exp)
            # eigenvectors → GPU (for matrix operations)
            if self.engine.use_gpu:
                eigenvalues = eigenvalues.get()
                # eigenvectors stays on GPU!
            return eigenvalues, eigenvectors
        except Exception:
            # Fallback to dense
            if self.engine.use_gpu:
                H_dense = H.toarray().get()
            else:
                H_dense = H.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            eigenvalues = eigenvalues[:n_states]
            eigenvectors = eigenvectors[:, :n_states]
            # Convert eigenvectors to GPU if needed
            if self.engine.use_gpu:
                eigenvectors = xp.asarray(eigenvectors)
            return eigenvalues, eigenvectors
    
    def evolve_thermal_path(self, temperatures: List[float], H, H_K, H_V,
                            eigenvalues: np.ndarray, eigenvectors,
                            dt: float = 0.1) -> Dict[str, Any]:
        """
        Evolve along temperature path with thermal ensemble tracking.
        
        Each eigenstate evolves, weighted by Boltzmann factors.
        Note: eigenvalues is NumPy, eigenvectors is GPU (CuPy) or CPU (NumPy)
        """
        xp = self.engine.xp
        n_states = len(eigenvalues)
        
        # Initialize: each eigenstate at its energy (stays on same device)
        psi_list = [eigenvectors[:, n].copy() for n in range(n_states)]
        
        # Temperature evolution
        for T in temperatures:
            beta = self.T_to_beta(T)
            weights = self.boltzmann_weights(eigenvalues, beta)
            
            # Time evolve each state
            for n in range(n_states):
                psi_list[n] = self.time_evolve(psi_list[n], H, dt)
                psi_list[n] = psi_list[n] / xp.linalg.norm(psi_list[n])
        
        # Final thermal average
        beta_final = self.T_to_beta(temperatures[-1])
        weights = self.boltzmann_weights(eigenvalues, beta_final)
        
        K_total = 0.0
        V_total = 0.0
        active_indices = []
        
        for n in range(n_states):
            if weights[n] > 1e-6:
                active_indices.append(n)
                psi_n = psi_list[n]
                K_n = float(xp.real(xp.vdot(psi_n, H_K @ psi_n)))
                V_n = float(xp.real(xp.vdot(psi_n, H_V @ psi_n)))
                K_total += weights[n] * K_n
                V_total += weights[n] * V_n
        
        lambda_final = abs(K_total) / (abs(V_total) + 1e-10)
        
        return {
            'lambda_final': lambda_final,
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
        xp = self.engine.xp
        beta_final = self.T_to_beta(T_final)
        weights = self.boltzmann_weights(eigenvalues, beta_final)
        K_eq = sum(weights[n] * float(xp.real(xp.vdot(eigenvectors[:,n], 
                   H_K @ eigenvectors[:,n]))) for n in range(len(eigenvalues)))
        V_eq = sum(weights[n] * float(xp.real(xp.vdot(eigenvectors[:,n],
                   H_V @ eigenvectors[:,n]))) for n in range(len(eigenvalues)))
        lambda_eq = abs(K_eq) / (abs(V_eq) + 1e-10)
        
        delta_dse = abs(result1['lambda_final'] - result2['lambda_final'])
        
        # Compute memory metrics
        memory_metrics = None
        if HAS_MEMORY_INDICATORS:
            memory_metrics = MemoryIndicator.compute_all(
                O_forward=result1['lambda_final'],
                O_backward=result2['lambda_final']
            )
        
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
            'memory_metrics': memory_metrics,
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
    energy_scale: float = typer.Option(0.1, "--energy-scale", "-E",
                                        help="Energy scale for β (eV). 0.1=organic, 1.0=metal"),
    path_compare: bool = typer.Option(False, "--path-compare", "-p",
                                       help="Compare field paths (DSE)"),
    thermal: bool = typer.Option(False, "--thermal", "-T",
                                  help="Compare thermal paths (DSE)"),
    t_high: float = typer.Option(300.0, "--T-high", help="High temperature (K)"),
    t_low: float = typer.Option(50.0, "--T-low", help="Low temperature (K)"),
    t_final: float = typer.Option(150.0, "--T-final", help="Final temperature (K)"),
    steps: int = typer.Option(5, "--steps", help="Steps per path segment"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU if available"),
    output: Optional[Path] = typer.Option(None, "-o", "--output",
                                          help="Output JSON file"),
):
    """
    2D Lattice DSE simulation (Unified SparseEngine).
    
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
        print_key_value("Energy scale", f"{energy_scale} eV")
        print_key_value("T_high/T_low/T_final", f"{t_high}/{t_low}/{t_final} K")
    else:
        print_key_value("Mode", "Single (static)")
        print_key_value("h", str(h_field))
    
    typer.echo(f"\n  Hilbert space: 2^{lx*ly} = {2**(lx*ly)}")
    typer.echo("")
    
    # Initialize runner
    try:
        runner = LatticeDSERunner(lx, ly, model, j, energy_scale, 
                                   use_gpu=gpu, verbose=True)
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
        
        # Memory metrics display
        metrics = results.get('memory_metrics')
        if metrics is not None:
            print_section("Memory Indicators", "🧠")
            typer.echo(f"  ΔO (path non-commutativity): {metrics.delta_O:.6f}")
            typer.echo(f"  Non-Markovian? {metrics.is_non_markovian()}")
            typer.echo("")
        
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
        
        # Memory metrics display
        metrics = results.get('memory_metrics')
        if metrics is not None:
            print_section("Memory Indicators", "🧠")
            typer.echo(f"  ΔO (path non-commutativity): {metrics.delta_O:.6f}")
            typer.echo(f"  Non-Markovian? {metrics.is_non_markovian()}")
            typer.echo("")
        
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
