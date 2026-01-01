"""
Sparse Hamiltonian Engine (Unified) for Memory-DFT
===================================================

Central engine for all Hamiltonian construction and quantum operations.

This module unifies:
  - sparse_engine.py: GPU/CPU sparse matrix operations
  - operators.py: Spin operators
  - hamiltonian.py: Model Hamiltonians
  - hubbard_engine.py: Chemistry-specific Hubbard model

Features:
  - CuPy + SciPy automatic backend selection
  - GPU acceleration when available
  - All spin models: Heisenberg, Ising, XY, Kitaev, Hubbard
  - λ = K/|V| stability parameter
  - 2-RDM computation (delegates to physics/rdm.py)

Usage:
    engine = SparseEngine(n_sites=6, use_gpu=True)
    H_K, H_V = engine.build_heisenberg(J=1.0)
    E0, psi0 = engine.compute_ground_state(H_K + H_V)
    lambda_val = engine.compute_lambda(psi0, H_K, H_V)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

# GPU support (optional)
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import eigsh as eigsh_gpu
    HAS_CUPY = True
except ImportError:
    import scipy.sparse as sp_module
    cp = np
    csp = sp_module
    HAS_CUPY = False

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh as eigsh_cpu


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SystemGeometry:
    """
    System geometry definition.
    
    Attributes:
        n_sites: Number of lattice sites
        bonds: List of (i, j) nearest-neighbor pairs
        plaquettes: List of site tuples forming plaquettes (optional)
        positions: Real-space positions (n_sites, 3) (optional)
    """
    n_sites: int
    bonds: List[Tuple[int, int]]
    plaquettes: Optional[List[Tuple[int, ...]]] = None
    positions: Optional[np.ndarray] = None
    
    @property
    def dim(self) -> int:
        """Hilbert space dimension."""
        return 2 ** self.n_sites
    
    @property
    def N_spins(self) -> int:
        """Alias for n_sites (compatibility)."""
        return self.n_sites
    
    @property
    def Dim(self) -> int:
        """Alias for dim (compatibility)."""
        return self.dim
    
    @property
    def bonds_nn(self) -> List[Tuple[int, int]]:
        """Alias for bonds (compatibility with LatticeGeometry2D)."""
        return self.bonds


@dataclass
class ComputeResult:
    """
    Result container for full computation.
    
    Attributes:
        energy: Ground state energy
        psi: Ground state wavefunction
        lambda_val: Stability parameter K/|V|
        rdm2: Two-particle RDM (optional)
        observables: Additional observables (optional)
    """
    energy: float
    psi: np.ndarray
    lambda_val: float
    rdm2: Optional[np.ndarray] = None
    observables: Optional[Dict[str, float]] = None


# Backward compatibility alias
HubbardResult = ComputeResult


# =============================================================================
# Sparse Engine (Unified)
# =============================================================================

class SparseEngine:
    """
    Unified sparse matrix engine for Memory-DFT.
    
    Combines all operator and Hamiltonian construction with
    automatic GPU/CPU backend selection.
    
    Key features:
      - All spin models (Heisenberg, Ising, XY, Kitaev, Hubbard)
      - Total spin operators (S_total_x, S_total_y, S_total_z)
      - Geometry construction (chain, ladder, square)
      - Ground state computation
      - λ = K/|V| stability analysis
      - 2-RDM computation
    
    Example:
        >>> engine = SparseEngine(n_sites=4)
        >>> geom = engine.build_chain(periodic=True)
        >>> H_K, H_V = engine.build_heisenberg(geom.bonds, J=1.0)
        >>> result = engine.compute_full(H_K, H_V)
        >>> print(f"λ = {result.lambda_val:.4f}")
    """
    
    def __init__(self, n_sites: int, use_gpu: bool = True, verbose: bool = True):
        """
        Initialize sparse engine.
        
        Args:
            n_sites: Number of lattice sites
            use_gpu: Use GPU acceleration if available
            verbose: Print progress messages
        """
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.verbose = verbose
        
        # Check dimension limit
        if self.dim > 2**20:
            raise ValueError(f"System too large: dim={self.dim}. Max 2^20=1,048,576")
        
        # Backend selection
        if self.use_gpu:
            self.xp = cp
            self.sparse = csp
            self.eigsh = eigsh_gpu
        else:
            self.xp = np
            self.sparse = sp
            self.eigsh = eigsh_cpu
        
        if verbose:
            print(f"🚀 SparseEngine: N={n_sites}, Dim={self.dim:,}")
            print(f"   Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (SciPy)'}")
            mem_dense = self.dim * self.dim * 16 / 1e9
            print(f"   Dense would need: {mem_dense:.1f} GB")
        
        # Build Pauli matrices
        self._build_pauli_matrices()
        
        # Build site operators (cached)
        self._build_site_operators()
        
        # Build total operators
        self._build_total_operators()
        
        # Cache for number operators (2-RDM)
        self._n_ops_cache: Optional[List] = None
    
    # =========================================================================
    # Operator Construction
    # =========================================================================
    
    def _build_pauli_matrices(self):
        """Build single-site Pauli matrices (sparse)."""
        # NumPy arrays first
        I_np = np.eye(2, dtype=np.complex128)
        X_np = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)      # Sx = σx/2
        Y_np = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)   # Sy = σy/2
        Z_np = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)     # Sz = σz/2
        Sp_np = np.array([[0, 1], [0, 0]], dtype=np.complex128)         # S+
        Sm_np = np.array([[0, 0], [1, 0]], dtype=np.complex128)         # S-
        n_np = np.array([[0, 0], [0, 1]], dtype=np.complex128)          # Number operator
        
        # Convert to sparse
        if self.use_gpu:
            self.I_single = csp.csr_matrix(cp.asarray(I_np))
            self.X_single = csp.csr_matrix(cp.asarray(X_np))
            self.Y_single = csp.csr_matrix(cp.asarray(Y_np))
            self.Z_single = csp.csr_matrix(cp.asarray(Z_np))
            self.Sp_single = csp.csr_matrix(cp.asarray(Sp_np))
            self.Sm_single = csp.csr_matrix(cp.asarray(Sm_np))
            self.n_single = csp.csr_matrix(cp.asarray(n_np))
        else:
            self.I_single = sp.csr_matrix(I_np)
            self.X_single = sp.csr_matrix(X_np)
            self.Y_single = sp.csr_matrix(Y_np)
            self.Z_single = sp.csr_matrix(Z_np)
            self.Sp_single = sp.csr_matrix(Sp_np)
            self.Sm_single = sp.csr_matrix(Sm_np)
            self.n_single = sp.csr_matrix(n_np)
    
    def _build_site_operator(self, op, site: int):
        """Build operator acting on specific site via tensor product."""
        ops = [self.I_single] * self.n_sites
        ops[site] = op
        
        result = ops[0]
        for i in range(1, self.n_sites):
            result = self.sparse.kron(result, ops[i], format='csr')
        
        return result
    
    def _build_site_operators(self):
        """Build all site operators (cached lists)."""
        self.Sx: List = [self._build_site_operator(self.X_single, i) 
                         for i in range(self.n_sites)]
        self.Sy: List = [self._build_site_operator(self.Y_single, i) 
                         for i in range(self.n_sites)]
        self.Sz: List = [self._build_site_operator(self.Z_single, i) 
                         for i in range(self.n_sites)]
        self.Sp: List = [self._build_site_operator(self.Sp_single, i) 
                         for i in range(self.n_sites)]
        self.Sm: List = [self._build_site_operator(self.Sm_single, i) 
                         for i in range(self.n_sites)]
    
    def _build_total_operators(self):
        """Build total spin operators."""
        self.S_total_x = sum(self.Sx)
        self.S_total_y = sum(self.Sy)
        self.S_total_z = sum(self.Sz)
    
    def get_site_operator(self, op_type: str, site: int):
        """
        Get operator for specific site.
        
        Args:
            op_type: 'X', 'Y', 'Z', '+', '-', 'I', 'n'
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
        elif op_type == 'I':
            return self._build_site_operator(self.I_single, site)
        elif op_type == 'n':
            return self._build_site_operator(self.n_single, site)
        else:
            raise ValueError(f"Unknown operator type: {op_type}")
    
    def _get_number_operators(self) -> List:
        """Get cached number operators for 2-RDM."""
        if self._n_ops_cache is None:
            self._n_ops_cache = [self._build_site_operator(self.n_single, i) 
                                  for i in range(self.n_sites)]
        return self._n_ops_cache
    
    # =========================================================================
    # Geometry Construction
    # =========================================================================
    
    def build_chain(self, L: Optional[int] = None, periodic: bool = True) -> SystemGeometry:
        """Build 1D chain geometry."""
        L = L or self.n_sites
        bonds = [(i, (i + 1) % L) for i in range(L)]
        if not periodic:
            bonds = bonds[:-1]
        return SystemGeometry(n_sites=L, bonds=bonds)
    
    def build_ladder(self, L: Optional[int] = None, periodic: bool = True) -> SystemGeometry:
        """Build 2-leg ladder geometry."""
        L = L or (self.n_sites // 2)
        N = 2 * L
        
        # Leg bonds
        leg0 = [(i, (i + 1) % L) for i in range(L)]
        leg1 = [(L + i, L + (i + 1) % L) for i in range(L)]
        
        # Rung bonds
        rungs = [(i, L + i) for i in range(L)]
        
        if not periodic:
            leg0 = leg0[:-1]
            leg1 = leg1[:-1]
        
        bonds = leg0 + leg1 + rungs
        
        # Plaquettes
        plaquettes = []
        for i in range(L if periodic else L - 1):
            bl, br = i, (i + 1) % L
            tl, tr = L + i, L + (i + 1) % L
            plaquettes.append((bl, br, tr, tl))
        
        return SystemGeometry(n_sites=N, bonds=bonds, plaquettes=plaquettes)
    
    def build_square(self, Lx: int, Ly: int, 
                     periodic_x: bool = True, 
                     periodic_y: bool = True) -> SystemGeometry:
        """Build 2D square lattice geometry."""
        N = Lx * Ly
        bonds = []
        
        def idx(x, y):
            return y * Lx + x
        
        for y in range(Ly):
            for x in range(Lx):
                # Horizontal bond
                if x < Lx - 1:
                    bonds.append((idx(x, y), idx(x + 1, y)))
                elif periodic_x:
                    bonds.append((idx(x, y), idx(0, y)))
                
                # Vertical bond
                if y < Ly - 1:
                    bonds.append((idx(x, y), idx(x, y + 1)))
                elif periodic_y:
                    bonds.append((idx(x, y), idx(x, 0)))
        
        return SystemGeometry(n_sites=N, bonds=bonds)
    
    # Aliases for compatibility
    build_chain_geometry = build_chain
    build_ladder_geometry = build_ladder
    
    # =========================================================================
    # Hamiltonian Construction
    # =========================================================================
    
    def build_heisenberg(self, bonds: Optional[List[Tuple[int, int]]] = None,
                         J: float = 1.0, Jz: Optional[float] = None,
                         split_KV: bool = True):
        """
        Build Heisenberg Hamiltonian.
        
        H = J Σ (Sx_i Sx_j + Sy_i Sy_j + Jz/J Sz_i Sz_j)
        
        Args:
            bonds: List of (i, j) pairs. If None, uses all-to-all.
            J: Exchange coupling (XY part)
            Jz: Z coupling (defaults to J for isotropic)
            split_KV: Return (H_K, H_V) tuple for λ analysis
            
        Returns:
            If split_KV: (H_kinetic, H_potential)
            Else: H_total
        """
        if bonds is None:
            bonds = self.build_chain().bonds
        
        Jz = Jz if Jz is not None else J
        
        H_K = None  # XY part (kinetic)
        H_V = None  # ZZ part (potential)
        
        for (i, j) in bonds:
            # XY interaction (kinetic)
            term_xy = J * (self.Sx[i] @ self.Sx[j] + self.Sy[i] @ self.Sy[j])
            H_K = term_xy if H_K is None else H_K + term_xy
            
            # ZZ interaction (potential)
            term_zz = Jz * self.Sz[i] @ self.Sz[j]
            H_V = term_zz if H_V is None else H_V + term_zz
        
        if split_KV:
            return H_K, H_V
        else:
            return H_K + H_V
    
    def build_ising(self, bonds: Optional[List[Tuple[int, int]]] = None,
                    J: float = 1.0, h: float = 0.0,
                    split_KV: bool = True):
        """
        Build transverse-field Ising Hamiltonian.
        
        H = J Σ Sz_i Sz_j + h Σ Sx_i
        
        Args:
            bonds: List of (i, j) pairs
            J: Ising coupling
            h: Transverse field
            split_KV: Return (H_K, H_V) tuple
            
        Returns:
            If split_KV: (H_K, H_V) where H_K=field, H_V=interaction
            Else: H_total
        """
        if bonds is None:
            bonds = self.build_chain().bonds
        
        # ZZ interaction (potential)
        H_V = None
        for (i, j) in bonds:
            term = J * self.Sz[i] @ self.Sz[j]
            H_V = term if H_V is None else H_V + term
        
        # Transverse field (kinetic)
        H_K = h * self.S_total_x
        
        if split_KV:
            return H_K, H_V
        else:
            return H_K + H_V
    
    def build_xy(self, bonds: Optional[List[Tuple[int, int]]] = None,
                 J: float = 1.0):
        """
        Build XY Hamiltonian.
        
        H = J Σ (Sx_i Sx_j + Sy_i Sy_j)
        """
        if bonds is None:
            bonds = self.build_chain().bonds
        
        H = None
        for (i, j) in bonds:
            term = J * (self.Sx[i] @ self.Sx[j] + self.Sy[i] @ self.Sy[j])
            H = term if H is None else H + term
        
        return H
    
    def build_xx(self, bonds: Optional[List[Tuple[int, int]]] = None,
                 J: float = 1.0):
        """
        Build XX Hamiltonian.
        
        H = J Σ Sx_i Sx_j
        """
        if bonds is None:
            bonds = self.build_chain().bonds
        
        H = None
        for (i, j) in bonds:
            term = J * self.Sx[i] @ self.Sx[j]
            H = term if H is None else H + term
        
        return H
    
    def build_kitaev(self, Lx: int, Ly: int,
                     Kx: float = 1.0, Ky: float = 0.8, Kz: float = 0.5):
        """
        Build Kitaev honeycomb model (rectangular approximation).
        
        H = Kx Σ_x-bonds Sx Sx + Ky Σ_y-bonds Sy Sy + Kz Σ_z-bonds Sz Sz
        
        Args:
            Lx, Ly: Lattice dimensions
            Kx, Ky, Kz: Bond-dependent couplings
            
        Returns:
            H_total (no K-V split for Kitaev)
        """
        H = None
        
        for y in range(Ly):
            for x in range(Lx):
                i = y * Lx + x
                
                # X-bonds (horizontal)
                if x < Lx - 1:
                    j = y * Lx + (x + 1)
                    term = Kx * self.Sx[i] @ self.Sx[j]
                    H = term if H is None else H + term
                
                # Y-bonds (vertical)
                if y < Ly - 1:
                    j = (y + 1) * Lx + x
                    term = Ky * self.Sy[i] @ self.Sy[j]
                    H = term if H is None else H + term
                
                # Z-bonds (diagonal, if applicable)
                if x < Lx - 1 and y < Ly - 1:
                    j = (y + 1) * Lx + (x + 1)
                    term = Kz * self.Sz[i] @ self.Sz[j]
                    H = term if H is None else H + term
        
        return H
    
    # Alias for compatibility
    def build_kitaev_rect(self, Kx: float = 1.0, Ky: float = 0.8, Kz_diag: float = 0.5):
        """Compatibility wrapper for kitaev."""
        # Infer Lx, Ly from n_sites
        Lx = int(np.sqrt(self.n_sites))
        Ly = self.n_sites // Lx
        return self.build_kitaev(Lx, Ly, Kx, Ky, Kz_diag)
    
    def build_hubbard(self, bonds: Optional[List[Tuple[int, int]]] = None,
                      t: float = 1.0, U: float = 2.0, h: float = 0.0,
                      site_potentials: Optional[List[float]] = None,
                      bond_lengths: Optional[List[float]] = None,
                      split_KV: bool = True):
        """
        Build Hubbard Hamiltonian (spin representation).
        
        H = -t Σ (c†_i c_j + h.c.) + U Σ n_i n_j + Σ V_i n_i + h Σ Sz_i
        
        Chemistry-specific features:
          - site_potentials: Site-dependent potentials (adsorption)
          - bond_lengths: Bond-length dependent hopping
        
        Args:
            bonds: List of (i, j) pairs
            t: Hopping parameter
            U: Interaction strength
            h: Magnetic field
            site_potentials: Site potentials [V_0, V_1, ...]
            bond_lengths: Bond lengths (modifies t)
            split_KV: Return (H_K, H_V) tuple
            
        Returns:
            If split_KV: (H_kinetic, H_potential)
            Else: H_total
        """
        if bonds is None:
            bonds = self.build_chain().bonds
        
        H_K = None  # Hopping
        H_V = None  # Interaction + on-site
        
        for idx, (i, j) in enumerate(bonds):
            # Hopping (XY-type in spin language)
            t_ij = t
            if bond_lengths is not None and idx < len(bond_lengths):
                # Exponential decay with bond length
                t_ij = t * np.exp(-0.5 * (bond_lengths[idx] - 1.0))
            
            term_hop = -t_ij * (self.Sp[i] @ self.Sm[j] + self.Sm[i] @ self.Sp[j])
            H_K = term_hop if H_K is None else H_K + term_hop
            
            # Density-density interaction
            n_i = self.Sz[i] + 0.5 * self._build_site_operator(self.I_single, i)
            n_j = self.Sz[j] + 0.5 * self._build_site_operator(self.I_single, j)
            term_U = U * n_i @ n_j
            H_V = term_U if H_V is None else H_V + term_U
        
        # Site potentials
        if site_potentials is not None:
            for i, V_i in enumerate(site_potentials):
                if i < self.n_sites and abs(V_i) > 1e-10:
                    n_i = self.Sz[i] + 0.5 * self._build_site_operator(self.I_single, i)
                    H_V = H_V + V_i * n_i
        
        # Magnetic field
        if abs(h) > 1e-10:
            H_V = H_V + h * self.S_total_z
        
        if split_KV:
            return H_K, H_V
        else:
            return H_K + H_V
    
    # Aliases
    build_heisenberg_hamiltonian = build_heisenberg
    build_hubbard_hamiltonian = build_hubbard
    
    # =========================================================================
    # Observables
    # =========================================================================
    
    def build_current_operator(self, bonds: Optional[List[Tuple[int, int]]] = None):
        """
        Build spin current operator.
        
        J = Σ 2(Sx_i Sy_j - Sy_i Sx_j)
        
        Corresponds to Λ_F (progress vector) in Λ³ theory.
        """
        if bonds is None:
            bonds = self.build_chain().bonds
        
        J_op = None
        for (i, j) in bonds:
            term = 2.0 * (self.Sx[i] @ self.Sy[j] - self.Sy[i] @ self.Sx[j])
            J_op = term if J_op is None else J_op + term
        
        return J_op
    
    def compute_magnetization(self, psi) -> float:
        """Compute magnetization per site <Sz>/N."""
        xp = self.xp
        sz_total = float(xp.real(xp.vdot(psi, self.S_total_z @ psi)))
        return sz_total / self.n_sites
    
    def compute_total_spin(self, psi) -> float:
        """Compute total spin <S²>."""
        xp = self.xp
        S2 = (self.S_total_x @ self.S_total_x + 
              self.S_total_y @ self.S_total_y + 
              self.S_total_z @ self.S_total_z)
        return float(xp.real(xp.vdot(psi, S2 @ psi)))
    
    def compute_correlation(self, psi, i: int, j: int, 
                            component: str = 'Z') -> float:
        """Compute spin-spin correlation <S_i^α S_j^α>."""
        xp = self.xp
        
        if component == 'X':
            Si, Sj = self.Sx[i], self.Sx[j]
        elif component == 'Y':
            Si, Sj = self.Sy[i], self.Sy[j]
        elif component == 'Z':
            Si, Sj = self.Sz[i], self.Sz[j]
        else:
            raise ValueError(f"Unknown component: {component}")
        
        SiSj = Si @ Sj
        return float(xp.real(xp.vdot(psi, SiSj @ psi)))
    
    # =========================================================================
    # Ground State & Stability
    # =========================================================================
    
    def compute_ground_state(self, H, k: int = 1):
        """
        Compute ground state via sparse diagonalization.
        
        Args:
            H: Hamiltonian (sparse matrix)
            k: Number of eigenvalues to compute
            
        Returns:
            (energy, wavefunction) or (energies, wavefunctions) if k > 1
        """
        try:
            E, psi = self.eigsh(H, k=k, which='SA')
            if k == 1:
                return float(E[0]), psi[:, 0]
            else:
                return E, psi
        except Exception as e:
            if self.verbose:
                print(f"  Warning: eigsh failed ({e}), using dense")
            # Fallback to dense
            if self.use_gpu:
                H_dense = H.toarray().get()
            else:
                H_dense = H.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            if k == 1:
                return float(eigenvalues[0]), eigenvectors[:, 0]
            else:
                return eigenvalues[:k], eigenvectors[:, :k]
    
    def compute_lambda(self, psi, H_K, H_V, epsilon: float = 1e-10) -> float:
        """
        Compute stability parameter λ = K / |V|.
        
        H-CSP/Λ³ theory core!
        
        Args:
            psi: State vector
            H_K: Kinetic Hamiltonian
            H_V: Potential Hamiltonian
            epsilon: Regularization for division
            
        Returns:
            λ value:
              λ < 1: Stable (binding dominates)
              λ = 1: Critical
              λ > 1: Unstable (kinetic dominates)
        """
        xp = self.xp
        
        K = float(xp.real(xp.vdot(psi, H_K @ psi)))
        V = float(xp.real(xp.vdot(psi, H_V @ psi)))
        
        return abs(K) / (abs(V) + epsilon)
    
    # Alias for backward compatibility
    compute_stability_parameter = compute_lambda
    
    # =========================================================================
    # 2-RDM (Delegates to physics/rdm.py)
    # =========================================================================
    
    def compute_2rdm(self, psi, method: str = 'diagonal') -> np.ndarray:
        """
        Compute two-particle reduced density matrix.
        
        Delegates to physics/rdm.py for the actual computation.
        The 2-RDM encodes all two-body correlations.
        
        Args:
            psi: Wavefunction
            method: 'diagonal' (fast) or 'full'
            
        Returns:
            rdm2: Array of shape (n_sites, n_sites, n_sites, n_sites)
        """
        from memory_dft.physics.rdm import compute_2rdm
        
        number_ops = self._get_number_operators()
        
        # Convert to numpy if on GPU
        if self.use_gpu:
            psi_np = psi.get() if hasattr(psi, 'get') else psi
            number_ops_np = [op.get() if hasattr(op, 'get') else op 
                             for op in number_ops]
        else:
            psi_np = psi
            number_ops_np = number_ops
        
        return compute_2rdm(psi_np, self.n_sites, 
                           number_ops=number_ops_np, method=method)
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def compute_full(self, H_K, H_V, 
                     compute_rdm2: bool = False,
                     compute_observables: bool = True) -> ComputeResult:
        """
        Perform complete calculation: H -> ground state -> stability.
        
        This is the main entry point for single-shot calculations.
        
        Args:
            H_K: Kinetic Hamiltonian
            H_V: Potential Hamiltonian
            compute_rdm2: Compute 2-RDM (expensive)
            compute_observables: Compute magnetization, etc.
            
        Returns:
            ComputeResult containing energy, wavefunction, λ, etc.
        """
        H = H_K + H_V
        E, psi = self.compute_ground_state(H)
        lambda_val = self.compute_lambda(psi, H_K, H_V)
        
        rdm2 = None
        if compute_rdm2:
            rdm2 = self.compute_2rdm(psi)
        
        observables = None
        if compute_observables:
            observables = {
                'magnetization': self.compute_magnetization(psi),
                'total_spin': self.compute_total_spin(psi),
            }
            if len(self.build_chain().bonds) > 0:
                i, j = self.build_chain().bonds[0]
                observables['nn_correlation'] = self.compute_correlation(psi, i, j)
        
        return ComputeResult(
            energy=E,
            psi=psi,
            lambda_val=lambda_val,
            rdm2=rdm2,
            observables=observables
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return {
            'n_sites': self.n_sites,
            'dim': self.dim,
            'use_gpu': self.use_gpu,
            'backend': 'CuPy' if self.use_gpu else 'SciPy',
            'has_cupy': HAS_CUPY,
        }
    
    def __repr__(self) -> str:
        backend = "GPU" if self.use_gpu else "CPU"
        return f"SparseEngine(N={self.n_sites}, Dim={self.dim:,}, backend={backend})"


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Original name
SparseHamiltonianEngine = SparseEngine

# For imports expecting SpinOperators-like interface
class SpinOperatorsCompat:
    """Compatibility wrapper for SpinOperators interface."""
    
    def __init__(self, N_spins: int, use_gpu: bool = False):
        self._engine = SparseEngine(N_spins, use_gpu=use_gpu, verbose=False)
        self.N = N_spins
        self.Dim = self._engine.dim
        self.Sx = self._engine.Sx
        self.Sy = self._engine.Sy
        self.Sz = self._engine.Sz
        self.Sp = self._engine.Sp
        self.Sm = self._engine.Sm
        self.S_total_x = self._engine.S_total_x
        self.S_total_y = self._engine.S_total_y
        self.S_total_z = self._engine.S_total_z


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Unified Sparse Engine Test")
    print("=" * 70)
    
    # Basic test
    print("\n--- Basic Construction ---")
    engine = SparseEngine(n_sites=4, use_gpu=False)
    print(f"{engine}")
    
    # Geometry
    print("\n--- Geometry ---")
    chain = engine.build_chain(periodic=True)
    print(f"Chain: {chain.n_sites} sites, {len(chain.bonds)} bonds")
    print(f"Bonds: {chain.bonds}")
    
    ladder = engine.build_ladder(L=2, periodic=False)
    print(f"Ladder: {ladder.n_sites} sites, {len(ladder.bonds)} bonds")
    
    # Hamiltonians
    print("\n--- Hamiltonians ---")
    
    H_K, H_V = engine.build_heisenberg(chain.bonds, J=1.0)
    H = H_K + H_V
    print(f"Heisenberg: shape={H.shape}, nnz={H.nnz}")
    
    H_ising = engine.build_ising(chain.bonds, J=1.0, h=0.5, split_KV=False)
    print(f"Ising: shape={H_ising.shape}, nnz={H_ising.nnz}")
    
    H_xy = engine.build_xy(chain.bonds, J=1.0)
    print(f"XY: shape={H_xy.shape}, nnz={H_xy.nnz}")
    
    H_K_hub, H_V_hub = engine.build_hubbard(chain.bonds, t=1.0, U=2.0)
    print(f"Hubbard: H_K nnz={H_K_hub.nnz}, H_V nnz={H_V_hub.nnz}")
    
    # Ground state
    print("\n--- Ground State ---")
    E0, psi0 = engine.compute_ground_state(H)
    print(f"E0 = {E0:.6f}")
    
    # Stability
    lambda_val = engine.compute_lambda(psi0, H_K, H_V)
    print(f"λ = {lambda_val:.4f}")
    
    # Observables
    mag = engine.compute_magnetization(psi0)
    print(f"Magnetization: {mag:.4f}")
    
    # Full computation
    print("\n--- Full Computation ---")
    result = engine.compute_full(H_K, H_V, compute_rdm2=False)
    print(f"E = {result.energy:.6f}")
    print(f"λ = {result.lambda_val:.4f}")
    print(f"Observables: {result.observables}")
    
    # Larger system
    print("\n--- Larger System (6 sites) ---")
    engine6 = SparseEngine(n_sites=6, use_gpu=False, verbose=True)
    geom6 = engine6.build_chain()
    H_K6, H_V6 = engine6.build_heisenberg(geom6.bonds)
    result6 = engine6.compute_full(H_K6, H_V6)
    print(f"E = {result6.energy:.6f}, λ = {result6.lambda_val:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Unified Sparse Engine Test Complete!")
    print("=" * 70)
