"""
2D Ladder DSE Demonstration
===========================

This example demonstrates path-dependent effects in 2D lattice
systems using various Hamiltonians.

Supported Models:
  - Heisenberg
  - XY
  - Kitaev (rectangular approximation)
  - Ising with transverse field
  - Hubbard (spin representation)

Key insight:
  Different Hamiltonian switching orders lead to
  different vorticity (flux) patterns.
  
  DFT: Same final H → Same V
  DSE: Different path → Different V
  
Usage:
    python -m memory_dft.examples.ladder_2d
    
    or
    
    from memory_dft.examples.ladder_2d import main
    main()

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm as scipy_expm

# Import from refactored modules
from memory_dft.core import (
    LatticeGeometry2D,
    SpinOperators,
    HamiltonianBuilder,
    build_hamiltonian,
)
from memory_dft.solvers import lanczos_expm_multiply


# =============================================================================
# 2D Ladder DSE Solver (Simplified for Examples)
# =============================================================================

class Ladder2DSolver:
    """
    2D Lattice DSE solver for demonstrations.
    
    Uses refactored Memory-DFT modules.
    """
    
    def __init__(self, Lx: int = 3, Ly: int = 3, 
                 periodic_x: bool = False, periodic_y: bool = False,
                 verbose: bool = True):
        """
        Initialize solver.
        
        Args:
            Lx, Ly: Lattice dimensions
            periodic_x, periodic_y: Boundary conditions
            verbose: Print progress
        """
        # Use refactored modules!
        self.geom = LatticeGeometry2D(Lx, Ly, periodic_x, periodic_y)
        self.ops = SpinOperators(self.geom.N_spins)
        self.builder = HamiltonianBuilder(self.geom, self.ops)
        
        self.verbose = verbose
        self.H = None
        self.H_type = None
        self.V_op = None
        self.eigenvalues = None
        self.eigenvectors = None
        
        if verbose:
            print(f"🔲 Ladder2DSolver: {Lx}×{Ly} lattice")
            print(f"   N_spins = {self.geom.N_spins}, Dim = {self.geom.Dim:,}")
            print(f"   Bonds: {len(self.geom.bonds_nn)}, Plaquettes: {len(self.geom.plaquettes)}")
    
    def build_hamiltonian(self, H_type: str = 'heisenberg', **params):
        """
        Build Hamiltonian of specified type.
        
        Args:
            H_type: 'heisenberg', 'xy', 'xx', 'kitaev', 'ising', 'hubbard'
            **params: Model-specific parameters
        """
        self.H_type = H_type
        
        # Use refactored builder
        if H_type == 'heisenberg':
            self.H = self.builder.heisenberg(**params)
        elif H_type == 'xy':
            self.H = self.builder.xy(**params)
        elif H_type == 'xx':
            self.H = self.builder.xx(**params)
        elif H_type == 'kitaev':
            self.H = self.builder.kitaev_rect(**params)
        elif H_type == 'ising':
            self.H = self.builder.ising(**params)
        elif H_type == 'hubbard':
            self.H = self.builder.hubbard_spin(**params)
        else:
            raise ValueError(f"Unknown model: {H_type}")
        
        # Build vorticity operator
        self.V_op = self.builder.build_vorticity_operator()
        
        if self.verbose:
            print(f"   Built {H_type}")
        
        return self
    
    def diagonalize(self, n_eigenstates: int = 50):
        """Compute low-energy eigenstates."""
        if self.H is None:
            raise ValueError("Build Hamiltonian first!")
        
        n_eigenstates = min(n_eigenstates, self.geom.Dim - 2)
        
        if self.verbose:
            print(f"   Diagonalizing ({n_eigenstates} states)...")
        
        t0 = time.time()
        eigenvalues, eigenvectors = eigsh(self.H, k=n_eigenstates, which='SA')
        
        idx = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        self.n_eigenstates = n_eigenstates
        
        if self.verbose:
            gap = self.eigenvalues[1] - self.eigenvalues[0]
            print(f"   Done in {time.time()-t0:.2f}s, E_0={self.eigenvalues[0]:.4f}, Gap={gap:.4f}")
        
        return self
    
    def compute_vorticity(self, psi: np.ndarray) -> float:
        """Compute vorticity ⟨V⟩ for a state."""
        return float(np.real(np.vdot(psi, self.V_op @ psi)))
    
    def evolve_hamiltonian_path(self, 
                                 H_sequence: List[Tuple[str, dict]],
                                 dt: float = 0.1,
                                 steps_per_H: int = 10) -> Dict:
        """
        Evolve system through a sequence of Hamiltonians.
        
        Args:
            H_sequence: List of (H_type, params) tuples
            dt: Time step
            steps_per_H: Steps per Hamiltonian
            
        Returns:
            Results dictionary
        """
        if self.verbose:
            print(f"\n   Evolving through {len(H_sequence)} Hamiltonians...")
        
        # Start from ground state of first Hamiltonian
        self.build_hamiltonian(*H_sequence[0][0] if isinstance(H_sequence[0], tuple) else H_sequence[0], 
                               **H_sequence[0][1] if len(H_sequence[0]) > 1 else {})
        self.diagonalize(n_eigenstates=20)
        
        psi = self.eigenvectors[:, 0].copy()
        
        times = []
        vorticities = []
        energies = []
        
        t = 0.0
        
        for H_spec in H_sequence:
            if isinstance(H_spec, tuple):
                H_type, params = H_spec
            else:
                H_type, params = H_spec, {}
            
            # Build new Hamiltonian
            self.build_hamiltonian(H_type, **params)
            
            for step in range(steps_per_H):
                # Compute observables
                V = self.compute_vorticity(psi)
                E = float(np.real(np.vdot(psi, self.H @ psi)))
                
                times.append(t)
                vorticities.append(V)
                energies.append(E)
                
                # Evolve
                psi = lanczos_expm_multiply(self.H, psi, dt, krylov_dim=20)
                t += dt
        
        return {
            'times': times,
            'vorticities': vorticities,
            'energies': energies,
            'V_final': vorticities[-1] if vorticities else 0.0,
        }


# =============================================================================
# Demonstrations
# =============================================================================

def run_model_comparison():
    """Compare different Hamiltonian models on same lattice."""
    print("\n" + "=" * 70)
    print("🔲 MODEL COMPARISON TEST")
    print("=" * 70)
    
    solver = Ladder2DSolver(Lx=2, Ly=2, verbose=True)
    
    models = [
        ('heisenberg', {'J': 1.0}),
        ('xy', {'J': 1.0}),
        ('ising', {'J': 1.0, 'h': 0.5}),
        ('kitaev', {'Kx': 1.0, 'Ky': 0.8, 'Kz_diag': 0.3}),
    ]
    
    results = {}
    
    print("\n" + "-" * 50)
    for model_name, params in models:
        solver.build_hamiltonian(model_name, **params)
        solver.diagonalize(n_eigenstates=10)
        
        psi_0 = solver.eigenvectors[:, 0]
        V_0 = solver.compute_vorticity(psi_0)
        E_0 = solver.eigenvalues[0]
        gap = solver.eigenvalues[1] - solver.eigenvalues[0]
        
        results[model_name] = {
            'E_0': E_0,
            'gap': gap,
            'V_0': V_0,
        }
        
        print(f"\n   {model_name:12s}: E_0={E_0:8.4f}, Gap={gap:.4f}, V={V_0:8.4f}")
    
    print("\n" + "-" * 50)
    return results


def run_kitaev_path_dependence():
    """
    Kitaev model path dependence test.
    
    Path A: Kx-dominated → Ky-dominated
    Path B: Ky-dominated → Kx-dominated
    """
    print("\n" + "=" * 70)
    print("🔲 KITAEV PATH DEPENDENCE TEST")
    print("=" * 70)
    print("\nCompare different paths to same final Hamiltonian")
    
    solver = Ladder2DSolver(Lx=2, Ly=2, verbose=True)
    
    # Path A: Start with Kx > Ky, end with Kx = Ky
    path_A = [
        ('kitaev', {'Kx': 1.5, 'Ky': 0.5, 'Kz_diag': 0.0}),
        ('kitaev', {'Kx': 1.3, 'Ky': 0.7, 'Kz_diag': 0.0}),
        ('kitaev', {'Kx': 1.0, 'Ky': 1.0, 'Kz_diag': 0.0}),
    ]
    
    # Path B: Start with Ky > Kx, end with Kx = Ky
    path_B = [
        ('kitaev', {'Kx': 0.5, 'Ky': 1.5, 'Kz_diag': 0.0}),
        ('kitaev', {'Kx': 0.7, 'Ky': 1.3, 'Kz_diag': 0.0}),
        ('kitaev', {'Kx': 1.0, 'Ky': 1.0, 'Kz_diag': 0.0}),
    ]
    
    print("\n📍 Path A: Kx-dominated → isotropic")
    result_A = solver.evolve_hamiltonian_path(path_A, dt=0.1, steps_per_H=10)
    
    print("\n📍 Path B: Ky-dominated → isotropic")
    result_B = solver.evolve_hamiltonian_path(path_B, dt=0.1, steps_per_H=10)
    
    delta_V = abs(result_A['V_final'] - result_B['V_final'])
    
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"\n   Path A final vorticity: {result_A['V_final']:.6f}")
    print(f"   Path B final vorticity: {result_B['V_final']:.6f}")
    print(f"\n   ΔV = {delta_V:.6f}")
    
    if delta_V > 1e-6:
        print("\n   ✅ PATH DEPENDENCE DETECTED!")
        print("   → Same final H, different history → Different vorticity")
        print("   → DFT sees ΔV ≡ 0 (blind to path)")
    
    return result_A, result_B


def run_ising_transition():
    """
    Ising model quantum phase transition.
    
    h < J: Ordered (Ising) phase
    h > J: Disordered (paramagnetic) phase
    """
    print("\n" + "=" * 70)
    print("🔲 ISING QUANTUM PHASE TRANSITION")
    print("=" * 70)
    
    solver = Ladder2DSolver(Lx=2, Ly=2, verbose=True)
    
    print("\nSweeping transverse field h...")
    print("-" * 50)
    
    h_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    for h in h_values:
        solver.build_hamiltonian('ising', J=1.0, h=h)
        solver.diagonalize(n_eigenstates=10)
        
        psi_0 = solver.eigenvectors[:, 0]
        E_0 = solver.eigenvalues[0]
        gap = solver.eigenvalues[1] - solver.eigenvalues[0]
        
        # Magnetization
        Mz = float(np.real(np.vdot(psi_0, solver.ops.S_total_z @ psi_0)))
        
        results.append({
            'h': h,
            'E_0': E_0,
            'gap': gap,
            'Mz': Mz,
        })
        
        print(f"   h={h:.1f}: E_0={E_0:7.4f}, Gap={gap:.4f}, Mz={Mz:6.4f}")
    
    print("\n" + "-" * 50)
    print("   Note: Gap minimum indicates quantum critical point")
    
    return results


def run_size_scaling():
    """Test scaling with system size."""
    print("\n" + "=" * 70)
    print("🔲 SIZE SCALING TEST")
    print("=" * 70)
    
    sizes = [(2, 2), (2, 3), (3, 3)]
    results = []
    
    print("\n" + "-" * 50)
    for Lx, Ly in sizes:
        print(f"\n   Testing {Lx}×{Ly} lattice...")
        
        t0 = time.time()
        solver = Ladder2DSolver(Lx=Lx, Ly=Ly, verbose=False)
        solver.build_hamiltonian('heisenberg', J=1.0)
        solver.diagonalize(n_eigenstates=min(20, solver.geom.Dim - 2))
        
        psi_0 = solver.eigenvectors[:, 0]
        V_0 = solver.compute_vorticity(psi_0)
        E_0 = solver.eigenvalues[0]
        
        elapsed = time.time() - t0
        
        results.append({
            'Lx': Lx,
            'Ly': Ly,
            'N': Lx * Ly,
            'Dim': solver.geom.Dim,
            'E_0': E_0,
            'V_0': V_0,
            'time': elapsed,
        })
        
        print(f"   {Lx}×{Ly}: N={Lx*Ly}, Dim={solver.geom.Dim:,}, "
              f"E_0={E_0:.4f}, V={V_0:.4f}, t={elapsed:.2f}s")
    
    print("\n" + "-" * 50)
    return results


def main():
    """Main entry point."""
    print("\n" + "🔲 " * 20)
    print("  MEMORY-DFT: 2D LADDER EXAMPLES")
    print("🔲 " * 20)
    
    # Run all tests
    run_model_comparison()
    print("\n")
    
    run_kitaev_path_dependence()
    print("\n")
    
    run_ising_transition()
    print("\n")
    
    run_size_scaling()
    
    print("\n" + "=" * 70)
    print("✅ All 2D ladder examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
