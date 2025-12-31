"""
Hubbard Model Engine for Memory-DFT
===================================

Hubbard model implementation for chemical memory tests.

Hamiltonian:
  H = -t sum(c^dag_i c_j + h.c.) + U sum(n_i n_j) + sum(V_i n_i)

Chemical-specific features:
  - Site-specific potentials (adsorption/catalyst modeling)
  - Bond-length dependent hopping (reaction coordinate)
  - External field coupling

This is a thin wrapper around the core modules:
  - core/operators.py: SpinOperators (operator construction)
  - physics/rdm.py: compute_2rdm (2-RDM calculation)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HubbardResult:
    """
    Container for Hubbard model calculation results.
    
    Attributes:
        energy: Ground state energy
        psi: Ground state wavefunction
        lambda_val: Stability parameter (K/|V|)
        rdm2: Two-particle reduced density matrix (optional)
    """
    energy: float
    psi: np.ndarray
    lambda_val: float
    rdm2: Optional[np.ndarray] = None


class HubbardEngine:
    """
    Hubbard model solver with chemistry-specific features.
    
    This engine is specialized for chemical reaction simulations:
      - Site potentials for adsorption sites
      - Bond-length dependent hopping for reaction coordinates
      - External field for bias/perturbation
    
    For general spin physics, use core/operators.py + core/hamiltonian.py
    
    Usage:
        engine = HubbardEngine(L=4)
        result = engine.compute_full(t=1.0, U=2.0)
        print(f"Stability: {result.lambda_val}")
    """
    
    def __init__(self, L: int, verbose: bool = False):
        """
        Args:
            L: Number of lattice sites
            verbose: Print debug information
        """
        self.L = L
        self.dim = 2 ** L
        self.verbose = verbose
        
        # Build operators (lightweight, no SpinOperators dependency for now)
        self._build_operators()
        
        # Cache for number operators
        self._n_ops_cache: Optional[List] = None
        
        if self.verbose:
            print(f"HubbardEngine: L={L}, dim={self.dim}")
    
    def _build_operators(self):
        """Build basic operators for Hubbard model."""
        self.I = sp.eye(2, format='csr', dtype=np.complex128)
        self.Sp = sp.csr_matrix([[0, 1], [0, 0]], dtype=np.complex128)
        self.Sm = sp.csr_matrix([[0, 0], [1, 0]], dtype=np.complex128)
        self.n = sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)
    
    def _site_op(self, op, site: int) -> sp.csr_matrix:
        """Build operator acting on specific site."""
        ops = [self.I] * self.L
        ops[site] = op
        result = ops[0]
        for i in range(1, self.L):
            result = sp.kron(result, ops[i], format='csr')
        return result
    
    def _get_number_operators(self) -> List[sp.csr_matrix]:
        """Get cached number operators for 2-RDM computation."""
        if self._n_ops_cache is None:
            self._n_ops_cache = [self._site_op(self.n, i) for i in range(self.L)]
        return self._n_ops_cache
    
    # =========================================================================
    # Hamiltonian Construction (Chemistry-specific)
    # =========================================================================
    
    def build_hamiltonian(self, 
                          t: float = 1.0, 
                          U: float = 2.0, 
                          h: float = 0.0,
                          site_potentials: Optional[List[float]] = None,
                          bond_lengths: Optional[List[float]] = None) -> sp.csr_matrix:
        """
        Build the Hubbard Hamiltonian with chemistry features.
        
        Args:
            t: Hopping amplitude
            U: On-site interaction strength
            h: Global external field (Zeeman-like)
            site_potentials: Site-specific potentials [V_0, V_1, ...]
                             for modeling adsorption sites
            bond_lengths: Bond lengths for position-dependent hopping
                          (t_eff = t / R for reaction coordinate)
        
        Returns:
            Sparse Hamiltonian matrix in CSR format
        """
        H = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        
        # Kinetic (hopping) terms: -t sum(c^dag_i c_j + h.c.)
        for i in range(self.L - 1):
            j = i + 1
            
            # Bond-length dependent hopping
            t_eff = t
            if bond_lengths is not None and i < len(bond_lengths):
                t_eff = t / bond_lengths[i]
            
            Sp_i = self._site_op(self.Sp, i)
            Sm_i = self._site_op(self.Sm, i)
            Sp_j = self._site_op(self.Sp, j)
            Sm_j = self._site_op(self.Sm, j)
            
            H = H - t_eff * (Sp_i @ Sm_j + Sm_i @ Sp_j)
        
        # Interaction terms: U sum(n_i n_j) nearest-neighbor
        for i in range(self.L - 1):
            j = i + 1
            n_i = self._site_op(self.n, i)
            n_j = self._site_op(self.n, j)
            H = H + U * (n_i @ n_j)
        
        # Global external field: h sum(n_i)
        if abs(h) > 1e-10:
            for i in range(self.L):
                n_i = self._site_op(self.n, i)
                H = H + h * n_i
        
        # Site-specific potentials: sum(V_i n_i)
        if site_potentials is not None:
            for i, V_i in enumerate(site_potentials):
                if i < self.L and abs(V_i) > 1e-10:
                    n_i = self._site_op(self.n, i)
                    H = H + V_i * n_i
        
        return H
    
    def build_kinetic(self, t: float = 1.0, 
                      bond_lengths: Optional[List[float]] = None) -> sp.csr_matrix:
        """Build kinetic (hopping) Hamiltonian only."""
        return self.build_hamiltonian(t=t, U=0, h=0, bond_lengths=bond_lengths)
    
    def build_potential(self, U: float = 2.0, h: float = 0.0,
                        site_potentials: Optional[List[float]] = None) -> sp.csr_matrix:
        """Build potential (interaction + field) Hamiltonian only."""
        return self.build_hamiltonian(t=0, U=U, h=h, site_potentials=site_potentials)
    
    # =========================================================================
    # Ground State & Observables
    # =========================================================================
    
    def compute_ground_state(self, H: sp.csr_matrix) -> Tuple[float, np.ndarray]:
        """
        Compute ground state via sparse diagonalization.
        
        Returns:
            (energy, wavefunction) tuple
        """
        E, psi = eigsh(H, k=1, which='SA')
        return float(E[0]), psi[:, 0]
    
    def compute_stability_parameter(self, psi: np.ndarray, 
                                    H_K: sp.csr_matrix, 
                                    H_V: sp.csr_matrix) -> float:
        """
        Compute the dimensionless stability parameter K/|V|.
        
        Physical interpretation:
          - < 1: Kinetic dominated by binding (stable)
          - ~ 1: Critical balance
          - > 1: Kinetic exceeds binding (unstable)
        """
        K = float(np.real(np.vdot(psi, H_K @ psi)))
        V = float(np.real(np.vdot(psi, H_V @ psi)))
        
        if abs(V) < 1e-10:
            return float('inf') if K > 0 else 0.0
        
        return abs(K) / abs(V)
    
    # Backward compatibility alias
    compute_lambda = compute_stability_parameter
    
    # =========================================================================
    # 2-RDM Computation (delegates to physics/rdm.py)
    # =========================================================================
    
    def compute_2rdm(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute two-particle reduced density matrix.
        
        Delegates to physics/rdm.py for the actual computation.
        The 2-RDM encodes all two-body correlations.
        
        Returns:
            rdm2: Array of shape (L, L, L, L)
        """
        # Import here to avoid circular imports
        from memory_dft.physics.rdm import compute_2rdm
        
        number_ops = self._get_number_operators()
        return compute_2rdm(psi, self.L, number_ops=number_ops, method='diagonal')
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def compute_full(self, t: float = 1.0, U: float = 2.0, 
                     h: float = 0.0,
                     site_potentials: Optional[List[float]] = None,
                     bond_lengths: Optional[List[float]] = None,
                     compute_rdm2: bool = False) -> HubbardResult:
        """
        Perform complete calculation: H -> ground state -> stability.
        
        This is the main entry point for single-shot calculations.
        
        Returns:
            HubbardResult containing energy, wavefunction, 
            stability parameter, and optionally 2-RDM.
        """
        H = self.build_hamiltonian(t=t, U=U, h=h, 
                                   site_potentials=site_potentials,
                                   bond_lengths=bond_lengths)
        H_K = self.build_kinetic(t=t, bond_lengths=bond_lengths)
        H_V = self.build_potential(U=U, h=h, site_potentials=site_potentials)
        
        E, psi = self.compute_ground_state(H)
        lambda_val = self.compute_stability_parameter(psi, H_K, H_V)
        
        rdm2 = None
        if compute_rdm2:
            rdm2 = self.compute_2rdm(psi)
        
        return HubbardResult(
            energy=E,
            psi=psi,
            lambda_val=lambda_val,
            rdm2=rdm2
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HubbardEngine Test")
    print("=" * 60)
    
    for L in [4, 6, 8]:
        engine = HubbardEngine(L, verbose=True)
        result = engine.compute_full(t=1.0, U=2.0, compute_rdm2=True)
        
        print(f"  E = {result.energy:.6f}")
        print(f"  stability = {result.lambda_val:.4f}")
        print(f"  rdm2 shape = {result.rdm2.shape}")
        print()
    
    print("HubbardEngine test passed!")
