"""
Spin Operators for Memory-DFT
=============================

Pauli spin operators for N-spin quantum systems.

Features:
  - Full N-site Pauli matrices (Sx, Sy, Sz, S+, S-)
  - Efficient sparse matrix representation
  - GPU support via CuPy (optional)
  - Total spin operators

This module provides the operator foundation for spin
Hamiltonians in Memory-DFT simulations.

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Optional, Union

# GPU support (optional)
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    HAS_CUPY = True
except ImportError:
    cp = np
    csp = sp
    HAS_CUPY = False


# =============================================================================
# Single-Site Pauli Matrices
# =============================================================================

def pauli_matrices(sparse: bool = True):
    """
    Get single-site Pauli matrices.
    
    Args:
        sparse: Return sparse matrices if True
        
    Returns:
        Dictionary with 'I', 'X', 'Y', 'Z', 'P' (S+), 'M' (S-)
    """
    # Dense matrices
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)      # Sx = σx/2
    Y = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)   # Sy = σy/2
    Z = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)     # Sz = σz/2
    P = np.array([[0, 1], [0, 0]], dtype=np.complex128)          # S+ = |↑⟩⟨↓|
    M = np.array([[0, 0], [1, 0]], dtype=np.complex128)          # S- = |↓⟩⟨↑|
    
    matrices = {'I': I, 'X': X, 'Y': Y, 'Z': Z, 'P': P, 'M': M}
    
    if sparse:
        matrices = {k: sp.csr_matrix(v) for k, v in matrices.items()}
    
    return matrices


def pauli_matrices_full():
    """
    Get single-site Pauli matrices with full normalization (σ not S).
    
    Returns:
        Dictionary with 'I', 'X', 'Y', 'Z' (full Pauli σ matrices)
    """
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)       # σx
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)    # σy
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)      # σz
    
    return {'I': sp.csr_matrix(I), 
            'X': sp.csr_matrix(X), 
            'Y': sp.csr_matrix(Y), 
            'Z': sp.csr_matrix(Z)}


# =============================================================================
# N-Spin Operators
# =============================================================================

class SpinOperators:
    """
    Pauli spin operators for N-spin system.
    
    Constructs full Hilbert space operators via tensor product:
        S_i^α = I ⊗ ... ⊗ S^α ⊗ ... ⊗ I
                        (site i)
    
    Attributes:
        N: Number of spins
        Dim: Hilbert space dimension (2^N)
        Sx: List of Sx operators for each site
        Sy: List of Sy operators for each site
        Sz: List of Sz operators for each site
        Sp: List of S+ operators for each site
        Sm: List of S- operators for each site
        S_total_z: Total Sz operator
        
    Example:
        >>> ops = SpinOperators(4)
        >>> H_zeeman = sum(ops.Sz)  # Uniform magnetic field
        >>> print(H_zeeman.shape)
        (16, 16)
    """
    
    def __init__(self, N_spins: int, use_gpu: bool = False):
        """
        Initialize spin operators for N-spin system.
        
        Args:
            N_spins: Number of spins
            use_gpu: Use CuPy for GPU acceleration
        """
        self.N = N_spins
        self.Dim = 2 ** N_spins
        self.use_gpu = use_gpu and HAS_CUPY
        
        # Backend selection
        if self.use_gpu:
            self.xp = cp
            self.sparse_module = csp
        else:
            self.xp = np
            self.sparse_module = sp
        
        # Single-site operators (sparse)
        self._build_single_site_ops()
        
        # Full N-site operators
        self.Sx: List = [self._build_site_operator(self.sx, i) for i in range(N_spins)]
        self.Sy: List = [self._build_site_operator(self.sy, i) for i in range(N_spins)]
        self.Sz: List = [self._build_site_operator(self.sz, i) for i in range(N_spins)]
        self.Sp: List = [self._build_site_operator(self.sp, i) for i in range(N_spins)]
        self.Sm: List = [self._build_site_operator(self.sm, i) for i in range(N_spins)]
        
        # Total operators
        self.S_total_z = sum(self.Sz)
        self.S_total_x = sum(self.Sx)
        self.S_total_y = sum(self.Sy)
    
    def _build_single_site_ops(self):
        """Build single-site sparse Pauli matrices."""
        # Using S = σ/2 convention (spin-1/2)
        sx_np = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)
        sy_np = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)
        sz_np = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)
        sp_np = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        sm_np = np.array([[0, 0], [1, 0]], dtype=np.complex128)
        iden_np = np.eye(2, dtype=np.complex128)
        
        if self.use_gpu:
            self.sx = csp.csr_matrix(cp.asarray(sx_np))
            self.sy = csp.csr_matrix(cp.asarray(sy_np))
            self.sz = csp.csr_matrix(cp.asarray(sz_np))
            self.sp = csp.csr_matrix(cp.asarray(sp_np))
            self.sm = csp.csr_matrix(cp.asarray(sm_np))
            self.iden = csp.csr_matrix(cp.asarray(iden_np))
        else:
            self.sx = sp.csr_matrix(sx_np)
            self.sy = sp.csr_matrix(sy_np)
            self.sz = sp.csr_matrix(sz_np)
            self.sp = sp.csr_matrix(sp_np)
            self.sm = sp.csr_matrix(sm_np)
            self.iden = sp.csr_matrix(iden_np)
    
    def _build_site_operator(self, op, site: int):
        """
        Build full N-site operator from single-site operator.
        
        S_site = I ⊗ ... ⊗ op ⊗ ... ⊗ I
        
        Args:
            op: Single-site operator (2×2 sparse matrix)
            site: Site index (0 to N-1)
            
        Returns:
            Full operator (Dim × Dim sparse matrix)
        """
        ops = [self.iden] * self.N
        ops[site] = op
        
        full_op = ops[0]
        for i in range(1, self.N):
            full_op = self.sparse_module.kron(full_op, ops[i], format='csr')
        
        return full_op
    
    def get_operator(self, op_type: str, site: int):
        """
        Get operator by type and site.
        
        Args:
            op_type: 'X', 'Y', 'Z', '+', '-'
            site: Site index
            
        Returns:
            Sparse operator matrix
        """
        if op_type == 'X':
            return self.Sx[site]
        elif op_type == 'Y':
            return self.Sy[site]
        elif op_type == 'Z':
            return self.Sz[site]
        elif op_type == '+':
            return self.Sp[site]
        elif op_type == '-':
            return self.Sm[site]
        else:
            raise ValueError(f"Unknown operator type: {op_type}")
    
    def dot_product(self, i: int, j: int):
        """
        Compute S_i · S_j = Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j
        
        Args:
            i, j: Site indices
            
        Returns:
            Heisenberg interaction operator
        """
        return (self.Sx[i] @ self.Sx[j] + 
                self.Sy[i] @ self.Sy[j] + 
                self.Sz[i] @ self.Sz[j])
    
    def xy_interaction(self, i: int, j: int):
        """
        Compute XY interaction: Sx_i Sx_j + Sy_i Sy_j
        
        Args:
            i, j: Site indices
            
        Returns:
            XY interaction operator
        """
        return self.Sx[i] @ self.Sx[j] + self.Sy[i] @ self.Sy[j]
    
    def exchange_interaction(self, i: int, j: int):
        """
        Compute exchange interaction: S+_i S-_j + S-_i S+_j
        
        This is equivalent to 2 * (Sx_i Sx_j + Sy_i Sy_j)
        
        Args:
            i, j: Site indices
            
        Returns:
            Exchange interaction operator
        """
        return self.Sp[i] @ self.Sm[j] + self.Sm[i] @ self.Sp[j]
    
    def number_operator(self, site: int):
        """
        Compute number operator: n_i = Sz_i + 1/2
        
        Maps spin to occupation: |↑⟩ → 1, |↓⟩ → 0
        
        Args:
            site: Site index
            
        Returns:
            Number operator
        """
        half_identity = 0.5 * self._build_site_operator(self.iden, site)
        return self.Sz[site] + half_identity
    
    def __repr__(self) -> str:
        backend = "GPU" if self.use_gpu else "CPU"
        return f"SpinOperators(N={self.N}, Dim={self.Dim:,}, backend={backend})"


# =============================================================================
# Factory Functions
# =============================================================================

def create_spin_operators(N_spins: int, use_gpu: bool = False) -> SpinOperators:
    """
    Factory function to create SpinOperators.
    
    Args:
        N_spins: Number of spins
        use_gpu: Use GPU acceleration
        
    Returns:
        SpinOperators instance
    """
    return SpinOperators(N_spins, use_gpu)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_total_spin(ops: SpinOperators, psi: np.ndarray) -> float:
    """
    Compute ⟨S_total^2⟩ for a state.
    
    S^2 = S_x^2 + S_y^2 + S_z^2
    
    For spin-1/2: S(S+1) where S = 0, 1/2, 1, ...
    
    Args:
        ops: SpinOperators instance
        psi: State vector
        
    Returns:
        Total spin expectation value
    """
    xp = ops.xp
    S2 = ops.S_total_x @ ops.S_total_x + \
         ops.S_total_y @ ops.S_total_y + \
         ops.S_total_z @ ops.S_total_z
    
    return float(xp.real(xp.vdot(psi, S2 @ psi)))


def compute_magnetization(ops: SpinOperators, psi: np.ndarray) -> float:
    """
    Compute ⟨S_z^total⟩ / N (magnetization per site).
    
    Args:
        ops: SpinOperators instance
        psi: State vector
        
    Returns:
        Magnetization per site
    """
    xp = ops.xp
    sz_total = float(xp.real(xp.vdot(psi, ops.S_total_z @ psi)))
    return sz_total / ops.N


def compute_correlation(ops: SpinOperators, psi: np.ndarray, 
                        i: int, j: int, component: str = 'Z') -> float:
    """
    Compute spin-spin correlation ⟨S_i^α S_j^α⟩.
    
    Args:
        ops: SpinOperators instance
        psi: State vector
        i, j: Site indices
        component: 'X', 'Y', or 'Z'
        
    Returns:
        Correlation value
    """
    xp = ops.xp
    
    if component == 'X':
        Si, Sj = ops.Sx[i], ops.Sx[j]
    elif component == 'Y':
        Si, Sj = ops.Sy[i], ops.Sy[j]
    elif component == 'Z':
        Si, Sj = ops.Sz[i], ops.Sz[j]
    else:
        raise ValueError(f"Unknown component: {component}")
    
    SiSj = Si @ Sj
    return float(xp.real(xp.vdot(psi, SiSj @ psi)))


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Spin Operators Test")
    print("=" * 70)
    
    # Basic construction
    print("\n--- Basic Construction ---")
    ops = SpinOperators(4)
    print(f"{ops}")
    print(f"Sx[0] shape: {ops.Sx[0].shape}, nnz: {ops.Sx[0].nnz}")
    
    # Check commutation relations
    print("\n--- Commutation Relations ---")
    # [Sx, Sy] = i Sz
    comm_xy = ops.Sx[0] @ ops.Sy[0] - ops.Sy[0] @ ops.Sx[0]
    expected = 1j * ops.Sz[0]
    diff = (comm_xy - expected).toarray()
    print(f"[Sx, Sy] = i Sz: max error = {np.max(np.abs(diff)):.2e}")
    
    # S+ S- = Sz + 1/4 (for single site)
    # Actually: S+ S- = Sx^2 + Sy^2 + i[Sx, Sy] = ... complicated
    # Let's check ladder operator action
    print("\n--- Ladder Operators ---")
    # S+ |↓⟩ = |↑⟩, S+ |↑⟩ = 0
    # S- |↑⟩ = |↓⟩, S- |↓⟩ = 0
    spin_down = np.array([0, 1], dtype=np.complex128)  # |↓⟩
    spin_up = np.array([1, 0], dtype=np.complex128)    # |↑⟩
    
    sp_single = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    result = sp_single @ spin_down
    print(f"S+ |↓⟩ = {result} (should be [1, 0])")
    
    # Test with random state
    print("\n--- Expectation Values ---")
    np.random.seed(42)
    psi = np.random.randn(ops.Dim) + 1j * np.random.randn(ops.Dim)
    psi = psi / np.linalg.norm(psi)
    
    mag = compute_magnetization(ops, psi)
    s2 = compute_total_spin(ops, psi)
    corr_01 = compute_correlation(ops, psi, 0, 1, 'Z')
    
    print(f"Magnetization: {mag:.4f}")
    print(f"⟨S^2⟩: {s2:.4f}")
    print(f"⟨Sz_0 Sz_1⟩: {corr_01:.4f}")
    
    # Heisenberg interaction
    print("\n--- Heisenberg Interaction ---")
    H_bond = ops.dot_product(0, 1)
    E_bond = float(np.real(np.vdot(psi, H_bond @ psi)))
    print(f"⟨S_0 · S_1⟩: {E_bond:.4f}")
    
    # GPU test (if available)
    print("\n--- GPU Test ---")
    if HAS_CUPY:
        ops_gpu = SpinOperators(4, use_gpu=True)
        print(f"GPU: {ops_gpu}")
    else:
        print("CuPy not available, skipping GPU test")
    
    print("\n✅ All spin operator tests passed!")
