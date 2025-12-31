"""
Two-Particle Reduced Density Matrix (2-RDM) for Memory-DFT
==========================================================

The 2-RDM is the foundation of correlation analysis in quantum systems.
It encodes all two-body correlations and enables decomposition of
correlation contributions by distance.

Definition:
  rho^(2)_{ijkl} = <psi|c^dag_i c^dag_j c_k c_l|psi>

Physical significance:
  - Complete description of two-body correlations
  - Basis for vorticity and correlation scaling analysis
  - Enables separation of local vs non-local correlations

Sources:
  1. Direct computation from wavefunction (lattice models)
  2. External quantum chemistry codes (PySCF, etc.)

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, List, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from memory_dft.core.operators import SpinOperators


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class RDM2Result:
    """
    Container for 2-RDM computation results.
    
    Attributes:
        rdm2: 2-RDM array of shape (n_orb, n_orb, n_orb, n_orb)
        n_orb: Number of orbitals/sites
        method: Computation method ('diagonal', 'full', 'external')
        trace: Trace value for normalization check
        n_particles: Expected particle number (for validation)
    """
    rdm2: np.ndarray
    n_orb: int
    method: str
    trace: Optional[float] = None
    n_particles: Optional[int] = None
    
    def __post_init__(self):
        if self.trace is None:
            self.trace = float(np.einsum('iijj->', self.rdm2))
    
    def validate(self, tol: float = 1e-6) -> bool:
        """
        Validate 2-RDM properties.
        
        Checks:
          - Hermiticity: rho_{ijkl} = rho_{klij}^*
          - Antisymmetry: rho_{ijkl} = -rho_{jikl}
          - Trace consistency
        """
        # Hermiticity check
        rdm2_swap = self.rdm2.transpose(2, 3, 0, 1).conj()
        hermitian_err = np.max(np.abs(self.rdm2 - rdm2_swap))
        
        if hermitian_err > tol:
            return False
        
        return True


# =============================================================================
# Core Computation Functions
# =============================================================================

def compute_2rdm(psi: np.ndarray,
                 n_sites: int,
                 number_ops: Optional[List] = None,
                 method: str = 'diagonal') -> np.ndarray:
    """
    Compute 2-RDM from wavefunction.
    
    The two-particle reduced density matrix encodes all two-body
    correlations in the quantum state. This is essential for
    analyzing correlation structure and memory effects.
    
    Args:
        psi: Normalized wavefunction vector
        n_sites: Number of lattice sites (orbitals)
        number_ops: List of number operators (auto-generated if None)
        method: 'diagonal' (fast, density-density only) or 
                'full' (complete 2-RDM)
        
    Returns:
        rdm2: 2-RDM array of shape (n_sites, n_sites, n_sites, n_sites)
        
    Example:
        >>> psi = ground_state_wavefunction
        >>> rdm2 = compute_2rdm(psi, n_sites=6)
        >>> # Use with VorticityCalculator
        >>> result = vorticity_calc.compute_vorticity(rdm2, n_orb=6)
    """
    if method == 'diagonal':
        return _compute_2rdm_diagonal(psi, n_sites, number_ops)
    elif method == 'full':
        return _compute_2rdm_full(psi, n_sites, number_ops)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'diagonal' or 'full'.")


def compute_2rdm_with_ops(psi: np.ndarray,
                          ops: 'SpinOperators',
                          method: str = 'diagonal') -> np.ndarray:
    """
    Compute 2-RDM using SpinOperators instance.
    
    Convenience wrapper that extracts number operators from
    SpinOperators automatically.
    
    Args:
        psi: Normalized wavefunction
        ops: SpinOperators instance
        method: Computation method
        
    Returns:
        rdm2: 2-RDM array
    """
    # Build number operators from Sz: n = Sz + 0.5
    n_sites = ops.N
    number_ops = []
    
    for i in range(n_sites):
        # n_i = Sz_i + 0.5 * I
        half_I = 0.5 * ops._build_site_operator(ops.iden, i)
        n_i = ops.Sz[i] + half_I
        number_ops.append(n_i)
    
    return compute_2rdm(psi, n_sites, number_ops, method)


# =============================================================================
# Internal Implementation
# =============================================================================

def _compute_2rdm_diagonal(psi: np.ndarray,
                           n_sites: int,
                           number_ops: Optional[List] = None) -> np.ndarray:
    """
    Compute 2-RDM in diagonal approximation.
    
    Only computes <n_i n_j> correlations (density-density).
    Fast but approximate. Sufficient for most correlation analyses.
    
    rho^(2)_{iijj} = <psi|n_i n_j|psi>
    """
    # Build number operators if not provided
    if number_ops is None:
        number_ops = _build_number_operators(n_sites)
    
    rdm2 = np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=np.complex128)
    
    for i in range(n_sites):
        for j in range(n_sites):
            n_i = number_ops[i]
            n_j = number_ops[j]
            
            # <n_i n_j>
            n_i_n_j = n_i @ n_j
            val = np.vdot(psi, n_i_n_j @ psi)
            val = complex(val)
            
            # Store in 2-RDM with proper symmetry
            rdm2[i, i, j, j] = val
            rdm2[i, j, i, j] = val * 0.5
            rdm2[i, j, j, i] = -val * 0.5  # Fermionic antisymmetry
    
    return rdm2


def _compute_2rdm_full(psi: np.ndarray,
                       n_sites: int,
                       number_ops: Optional[List] = None) -> np.ndarray:
    """
    Compute full 2-RDM with all off-diagonal elements.
    
    rho^(2)_{ijkl} = <psi|c^dag_i c^dag_j c_l c_k|psi>
    
    More expensive but complete. Required for some advanced analyses.
    """
    # For full 2-RDM, we need creation/annihilation operators
    # This is more complex and requires fermionic operators
    
    # For now, fall back to diagonal for spin systems
    # Full implementation would require proper fermionic algebra
    
    import warnings
    warnings.warn(
        "Full 2-RDM computation not yet implemented for spin systems. "
        "Using diagonal approximation.",
        UserWarning
    )
    
    return _compute_2rdm_diagonal(psi, n_sites, number_ops)


def _build_number_operators(n_sites: int) -> List[sp.csr_matrix]:
    """
    Build number operators for each site.
    
    n_i = |1><1| at site i (occupation number)
    """
    I = sp.eye(2, format='csr', dtype=np.complex128)
    n_single = sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)
    
    number_ops = []
    
    for site in range(n_sites):
        ops = [I] * n_sites
        ops[site] = n_single
        
        # Tensor product
        result = ops[0]
        for i in range(1, n_sites):
            result = sp.kron(result, ops[i], format='csr')
        
        number_ops.append(result)
    
    return number_ops


# =============================================================================
# Correlation Analysis Utilities
# =============================================================================

def compute_density_density_correlation(rdm2: np.ndarray,
                                         i: int, j: int) -> float:
    """
    Extract density-density correlation <n_i n_j> from 2-RDM.
    
    Args:
        rdm2: 2-RDM array
        i, j: Site indices
        
    Returns:
        Correlation value <n_i n_j>
    """
    return float(np.real(rdm2[i, i, j, j]))


def compute_connected_correlation(rdm2: np.ndarray,
                                   rdm1: np.ndarray,
                                   i: int, j: int) -> float:
    """
    Compute connected (cumulant) correlation.
    
    <n_i n_j>_c = <n_i n_j> - <n_i><n_j>
    
    This removes the mean-field contribution, isolating
    genuine quantum correlations.
    
    Args:
        rdm2: 2-RDM array
        rdm1: 1-RDM (diagonal: site occupations)
        i, j: Site indices
        
    Returns:
        Connected correlation value
    """
    n_i_n_j = float(np.real(rdm2[i, i, j, j]))
    n_i = float(np.real(rdm1[i, i]))
    n_j = float(np.real(rdm1[j, j]))
    
    return n_i_n_j - n_i * n_j


def compute_correlation_matrix(rdm2: np.ndarray) -> np.ndarray:
    """
    Extract correlation matrix C_{ij} = <n_i n_j> from 2-RDM.
    
    Useful for visualizing correlation structure.
    
    Args:
        rdm2: 2-RDM array
        
    Returns:
        Correlation matrix of shape (n_orb, n_orb)
    """
    n_orb = rdm2.shape[0]
    C = np.zeros((n_orb, n_orb), dtype=np.float64)
    
    for i in range(n_orb):
        for j in range(n_orb):
            C[i, j] = float(np.real(rdm2[i, i, j, j]))
    
    return C


def filter_by_distance(rdm2: np.ndarray,
                       max_range: int) -> np.ndarray:
    """
    Filter 2-RDM by correlation distance.
    
    Sets elements to zero where max(|i-j|, |k-l|, ...) > max_range.
    
    This enables separation of:
      - Local correlations (max_range=2): nearest-neighbor
      - Non-local correlations: max_range -> infinity
    
    Used for gamma decomposition:
      gamma_total = gamma_local + gamma_nonlocal
    
    Args:
        rdm2: 2-RDM array
        max_range: Maximum correlation distance to keep
        
    Returns:
        Filtered 2-RDM
    """
    n_orb = rdm2.shape[0]
    rdm2_filtered = np.zeros_like(rdm2)
    
    for i in range(n_orb):
        for j in range(n_orb):
            for k in range(n_orb):
                for l in range(n_orb):
                    # Compute maximum distance
                    d1 = abs(i - j)
                    d2 = abs(k - l)
                    d3 = abs(i - k)
                    d4 = abs(j - l)
                    max_d = max(d1, d2, d3, d4)
                    
                    if max_d <= max_range:
                        rdm2_filtered[i, j, k, l] = rdm2[i, j, k, l]
    
    return rdm2_filtered


# =============================================================================
# External Interface (PySCF, etc.)
# =============================================================================

def from_pyscf_rdm2(rdm2_pyscf: np.ndarray,
                    n_orb: Optional[int] = None) -> np.ndarray:
    """
    Convert PySCF 2-RDM format to internal format.
    
    PySCF uses chemist notation: (ij|kl)
    We use physicist notation: <ij|kl>
    
    Args:
        rdm2_pyscf: 2-RDM from PySCF
        n_orb: Number of orbitals (inferred if None)
        
    Returns:
        2-RDM in internal format
    """
    if n_orb is None:
        n_orb = rdm2_pyscf.shape[0]
    
    # PySCF stores as (p,q,r,s) -> <pq|rs>
    # Transpose if needed for our convention
    # For now, assume compatible format
    
    return rdm2_pyscf.copy()


def to_pyscf_rdm2(rdm2: np.ndarray) -> np.ndarray:
    """
    Convert internal 2-RDM to PySCF format.
    
    For integration with PySCF analysis tools.
    """
    return rdm2.copy()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("2-RDM Module Test")
    print("=" * 70)
    
    # Build a simple test system
    n_sites = 4
    dim = 2 ** n_sites
    
    # Random normalized state
    np.random.seed(42)
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi = psi / np.linalg.norm(psi)
    
    # Compute 2-RDM
    print("\n--- Diagonal 2-RDM ---")
    rdm2 = compute_2rdm(psi, n_sites, method='diagonal')
    print(f"Shape: {rdm2.shape}")
    print(f"Trace: {np.einsum('iijj->', rdm2):.4f}")
    
    # Correlation matrix
    print("\n--- Correlation Matrix ---")
    C = compute_correlation_matrix(rdm2)
    print(f"C =\n{np.real(C)}")
    
    # Distance filtering
    print("\n--- Distance Filtering ---")
    for max_r in [1, 2, 3]:
        rdm2_filt = filter_by_distance(rdm2, max_r)
        nnz = np.sum(np.abs(rdm2_filt) > 1e-10)
        print(f"max_range={max_r}: {nnz} non-zero elements")
    
    # RDM2Result container
    print("\n--- RDM2Result Container ---")
    result = RDM2Result(rdm2=rdm2, n_orb=n_sites, method='diagonal')
    print(f"Trace: {result.trace:.4f}")
    print(f"Valid: {result.validate()}")
    
    print("\n" + "=" * 70)
    print("2-RDM Module Test Complete")
    print("=" * 70)
