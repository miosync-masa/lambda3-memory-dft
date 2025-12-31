"""
Hamiltonian Builders for Memory-DFT
===================================

Build various spin Hamiltonians on lattice geometries.

Supported Models:
  - Heisenberg:  H = J Σ S_i · S_j
  - XY:          H = J Σ (Sx_i Sx_j + Sy_i Sy_j)
  - XX:          H = J Σ Sx_i Sx_j
  - Ising:       H = J Σ Sz_i Sz_j + h Σ Sx_i
  - Kitaev:      H = Kx Σ_x-bonds Sx Sx + Ky Σ_y-bonds Sy Sy + ...
  - Hubbard:     H = -t Σ (c†_i c_j + h.c.) + U Σ n_i n_j

Key Features:
  - Separates H_kinetic and H_potential for Λ calculations
  - Vorticity/flux operator construction
  - Compatible with Memory-DFT solvers

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import scipy.sparse as sparse
from typing import List, Tuple, Optional, Dict, Union

from .lattice import LatticeGeometry2D, SystemGeometry
from .operators import SpinOperators


# =============================================================================
# Hamiltonian Builder for 2D Lattices
# =============================================================================

class HamiltonianBuilder:
    """
    Build various spin Hamiltonians on 2D lattice.
    
    Works with LatticeGeometry2D and SpinOperators to construct
    sparse Hamiltonian matrices for different models.
    
    Attributes:
        geom: Lattice geometry
        ops: Spin operators
        Dim: Hilbert space dimension
        
    Example:
        >>> geom = LatticeGeometry2D(3, 3)
        >>> ops = SpinOperators(geom.N_spins)
        >>> builder = HamiltonianBuilder(geom, ops)
        >>> H = builder.heisenberg(J=1.0)
        >>> print(H.shape)
        (512, 512)
    """
    
    def __init__(self, geometry: LatticeGeometry2D, spin_ops: SpinOperators):
        """
        Initialize Hamiltonian builder.
        
        Args:
            geometry: 2D lattice geometry
            spin_ops: Spin operators for the system
        """
        self.geom = geometry
        self.ops = spin_ops
        self.Dim = geometry.Dim
        
        # Validate compatibility
        if geometry.N_spins != spin_ops.N:
            raise ValueError(
                f"Geometry has {geometry.N_spins} sites but "
                f"SpinOperators has {spin_ops.N} spins"
            )
    
    # =========================================================================
    # Standard Models
    # =========================================================================
    
    def heisenberg(self, J: float = 1.0, Jz: Optional[float] = None):
        """
        Heisenberg Hamiltonian.
        
        H = J Σ_{⟨i,j⟩} (Sx_i Sx_j + Sy_i Sy_j + Jz/J · Sz_i Sz_j)
        
        Args:
            J: Exchange coupling (XY part)
            Jz: Ising coupling (default: same as J)
            
        Returns:
            Sparse Hamiltonian matrix
        """
        if Jz is None:
            Jz = J
            
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        for (i, j) in self.geom.bonds_nn:
            H += J * (self.ops.Sx[i] @ self.ops.Sx[j] +
                      self.ops.Sy[i] @ self.ops.Sy[j])
            H += Jz * self.ops.Sz[i] @ self.ops.Sz[j]
        
        return H
    
    def heisenberg_KV(self, J: float = 1.0, Jz: Optional[float] = None):
        """
        Heisenberg Hamiltonian split into K (kinetic) and V (potential).
        
        For Λ = K/|V| calculations in Memory-DFT.
        
        H_K: XY part (spin hopping / kinetic-like)
        H_V: ZZ part (Ising / potential-like)
        
        Args:
            J: Exchange coupling
            Jz: Ising coupling (default: same as J)
            
        Returns:
            (H_K, H_V) tuple of sparse matrices
        """
        if Jz is None:
            Jz = J
            
        H_K = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        H_V = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        for (i, j) in self.geom.bonds_nn:
            # Kinetic (XY hopping)
            H_K += J * (self.ops.Sx[i] @ self.ops.Sx[j] +
                        self.ops.Sy[i] @ self.ops.Sy[j])
            # Potential (Ising)
            H_V += Jz * self.ops.Sz[i] @ self.ops.Sz[j]
        
        return H_K, H_V
    
    def xy(self, J: float = 1.0):
        """
        XY Hamiltonian.
        
        H = J Σ_{⟨i,j⟩} (Sx_i Sx_j + Sy_i Sy_j)
        
        Args:
            J: Exchange coupling
            
        Returns:
            Sparse Hamiltonian matrix
        """
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        for (i, j) in self.geom.bonds_nn:
            H += J * (self.ops.Sx[i] @ self.ops.Sx[j] +
                      self.ops.Sy[i] @ self.ops.Sy[j])
        
        return H
    
    def xx(self, J: float = 1.0):
        """
        XX Hamiltonian (transverse field Ising without field).
        
        H = J Σ_{⟨i,j⟩} Sx_i Sx_j
        
        Args:
            J: Coupling strength
            
        Returns:
            Sparse Hamiltonian matrix
        """
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        for (i, j) in self.geom.bonds_nn:
            H += J * self.ops.Sx[i] @ self.ops.Sx[j]
        
        return H
    
    def ising(self, J: float = 1.0, h: float = 0.0):
        """
        Transverse-field Ising model.
        
        H = J Σ_{⟨i,j⟩} Sz_i Sz_j + h Σ_i Sx_i
        
        Args:
            J: Ising coupling
            h: Transverse field strength
            
        Returns:
            Sparse Hamiltonian matrix
        """
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        # Ising interaction
        for (i, j) in self.geom.bonds_nn:
            H += J * self.ops.Sz[i] @ self.ops.Sz[j]
        
        # Transverse field
        for i in range(self.geom.N_spins):
            H += h * self.ops.Sx[i]
        
        return H
    
    def ising_KV(self, J: float = 1.0, h: float = 0.0):
        """
        Transverse-field Ising split into K and V.
        
        H_K: Transverse field (drives dynamics)
        H_V: Ising interaction (binding)
        
        Args:
            J: Ising coupling
            h: Transverse field
            
        Returns:
            (H_K, H_V) tuple
        """
        H_K = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        H_V = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        # Kinetic (transverse field)
        for i in range(self.geom.N_spins):
            H_K += h * self.ops.Sx[i]
        
        # Potential (Ising)
        for (i, j) in self.geom.bonds_nn:
            H_V += J * self.ops.Sz[i] @ self.ops.Sz[j]
        
        return H_K, H_V
    
    # =========================================================================
    # Kitaev Model
    # =========================================================================
    
    def kitaev_rect(self, Kx: float = 1.0, Ky: float = 1.0, Kz_diag: float = 0.0):
        """
        Kitaev model on rectangular lattice.
        
        H = Kx Σ_{x-bonds} Sx_i Sx_j 
          + Ky Σ_{y-bonds} Sy_i Sy_j
          + Kz Σ_{diagonal} Sz_i Sz_j
        
        Note: True Kitaev model requires honeycomb lattice.
        This is a rectangular approximation.
        
        Args:
            Kx: Coupling for x-direction bonds
            Ky: Coupling for y-direction bonds
            Kz_diag: Coupling for diagonal (plaquette) bonds
            
        Returns:
            Sparse Hamiltonian matrix
        """
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        # X-bonds
        for (i, j) in self.geom.bonds_x:
            H += Kx * self.ops.Sx[i] @ self.ops.Sx[j]
        
        # Y-bonds
        for (i, j) in self.geom.bonds_y:
            H += Ky * self.ops.Sy[i] @ self.ops.Sy[j]
        
        # Diagonal Z-bonds (through plaquettes)
        if Kz_diag != 0.0:
            for (bl, br, tr, tl) in self.geom.plaquettes:
                H += Kz_diag * self.ops.Sz[bl] @ self.ops.Sz[tr]
                H += Kz_diag * self.ops.Sz[br] @ self.ops.Sz[tl]
        
        return H
    
    # =========================================================================
    # Hubbard-like Model (Spin Representation)
    # =========================================================================
    
    def hubbard_spin(self, t: float = 1.0, U: float = 2.0):
        """
        Hubbard-like model in spin representation.
        
        H = -t Σ (S+_i S-_j + S-_i S+_j) + U Σ Sz_i Sz_j
        
        This is a simplified spin-1/2 version. For full
        fermionic Hubbard, use HubbardEngine in core/.
        
        Args:
            t: Hopping strength
            U: On-site interaction (mapped to Sz-Sz)
            
        Returns:
            Sparse Hamiltonian matrix
        """
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        for (i, j) in self.geom.bonds_nn:
            # Hopping (spin exchange)
            H += -t * (self.ops.Sp[i] @ self.ops.Sm[j] +
                       self.ops.Sm[i] @ self.ops.Sp[j])
            # Interaction
            H += U * self.ops.Sz[i] @ self.ops.Sz[j]
        
        return H
    
    def hubbard_spin_KV(self, t: float = 1.0, U: float = 2.0):
        """
        Hubbard-like split into K and V.
        
        H_K: Hopping (kinetic)
        H_V: Interaction (potential)
        
        Args:
            t: Hopping
            U: Interaction
            
        Returns:
            (H_K, H_V) tuple
        """
        H_K = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        H_V = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        for (i, j) in self.geom.bonds_nn:
            H_K += -t * (self.ops.Sp[i] @ self.ops.Sm[j] +
                         self.ops.Sm[i] @ self.ops.Sp[j])
            H_V += U * self.ops.Sz[i] @ self.ops.Sz[j]
        
        return H_K, H_V
    
    # =========================================================================
    # Observables / Operators
    # =========================================================================
    
    def build_vorticity_operator(self):
        """
        Static vorticity operator on plaquettes.
        
        V = Σ_{plaq} Σ_{i→j ∈ plaq} 2(Sx_i Sy_j - Sy_i Sx_j)
        
        This measures the "spin current" circulating around
        each plaquette - key observable for Memory-DFT.
        
        Returns:
            Sparse vorticity operator
        """
        V = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        
        for (bl, br, tr, tl) in self.geom.plaquettes:
            # Edges of plaquette (counterclockwise)
            loop_edges = [(bl, br), (br, tr), (tr, tl), (tl, bl)]
            
            for (i, j) in loop_edges:
                # Spin current: J_ij ~ Sx_i Sy_j - Sy_i Sx_j
                V += 2.0 * (self.ops.Sx[i] @ self.ops.Sy[j] -
                           self.ops.Sy[i] @ self.ops.Sx[j])
        
        return V
    
    def build_current_operator(self, bond: Tuple[int, int]):
        """
        Spin current operator for a single bond.
        
        J_{ij} = 2i(S+_i S-_j - S-_i S+_j)
               = 2(Sx_i Sy_j - Sy_i Sx_j)
        
        Args:
            bond: (i, j) site pair
            
        Returns:
            Sparse current operator
        """
        i, j = bond
        return 2.0 * (self.ops.Sx[i] @ self.ops.Sy[j] -
                      self.ops.Sy[i] @ self.ops.Sx[j])
    
    def build_magnetization_operator(self, direction: str = 'Z'):
        """
        Total magnetization operator.
        
        M_α = Σ_i S_i^α
        
        Args:
            direction: 'X', 'Y', or 'Z'
            
        Returns:
            Sparse magnetization operator
        """
        if direction == 'X':
            return self.ops.S_total_x
        elif direction == 'Y':
            return self.ops.S_total_y
        elif direction == 'Z':
            return self.ops.S_total_z
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def __repr__(self) -> str:
        return f"HamiltonianBuilder(geom={self.geom.Lx}×{self.geom.Ly}, Dim={self.Dim:,})"


# =============================================================================
# Factory Functions
# =============================================================================

def build_hamiltonian(model: str, 
                      geometry: LatticeGeometry2D,
                      spin_ops: Optional[SpinOperators] = None,
                      split_KV: bool = False,
                      **params):
    """
    Factory function to build Hamiltonians.
    
    Args:
        model: 'heisenberg', 'xy', 'xx', 'ising', 'kitaev', 'hubbard'
        geometry: Lattice geometry
        spin_ops: Spin operators (created if not provided)
        split_KV: Return (H_K, H_V) tuple for Λ calculations
        **params: Model-specific parameters
        
    Returns:
        Hamiltonian matrix (or tuple if split_KV=True)
        
    Example:
        >>> geom = LatticeGeometry2D(3, 3)
        >>> H_K, H_V = build_hamiltonian('heisenberg', geom, split_KV=True, J=1.0)
    """
    if spin_ops is None:
        spin_ops = SpinOperators(geometry.N_spins)
    
    builder = HamiltonianBuilder(geometry, spin_ops)
    
    model_lower = model.lower()
    
    if model_lower == 'heisenberg':
        if split_KV:
            return builder.heisenberg_KV(**params)
        return builder.heisenberg(**params)
    
    elif model_lower == 'xy':
        return builder.xy(**params)
    
    elif model_lower == 'xx':
        return builder.xx(**params)
    
    elif model_lower == 'ising':
        if split_KV:
            return builder.ising_KV(**params)
        return builder.ising(**params)
    
    elif model_lower == 'kitaev':
        return builder.kitaev_rect(**params)
    
    elif model_lower == 'hubbard':
        if split_KV:
            return builder.hubbard_spin_KV(**params)
        return builder.hubbard_spin(**params)
    
    else:
        raise ValueError(f"Unknown model: {model}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Hamiltonian Builder Test")
    print("=" * 70)
    
    # Setup
    print("\n--- Setup ---")
    geom = LatticeGeometry2D(3, 3, periodic_x=False, periodic_y=False)
    ops = SpinOperators(geom.N_spins)
    builder = HamiltonianBuilder(geom, ops)
    print(f"Geometry: {geom}")
    print(f"Builder: {builder}")
    
    # Test all models
    print("\n--- Build All Models ---")
    
    H_heis = builder.heisenberg(J=1.0)
    print(f"Heisenberg: shape={H_heis.shape}, nnz={H_heis.nnz}")
    
    H_xy = builder.xy(J=1.0)
    print(f"XY: shape={H_xy.shape}, nnz={H_xy.nnz}")
    
    H_xx = builder.xx(J=1.0)
    print(f"XX: shape={H_xx.shape}, nnz={H_xx.nnz}")
    
    H_ising = builder.ising(J=1.0, h=0.5)
    print(f"Ising: shape={H_ising.shape}, nnz={H_ising.nnz}")
    
    H_kitaev = builder.kitaev_rect(Kx=1.0, Ky=0.8, Kz_diag=0.5)
    print(f"Kitaev: shape={H_kitaev.shape}, nnz={H_kitaev.nnz}")
    
    H_hubbard = builder.hubbard_spin(t=1.0, U=2.0)
    print(f"Hubbard: shape={H_hubbard.shape}, nnz={H_hubbard.nnz}")
    
    # K-V split
    print("\n--- K-V Split (for Λ calculations) ---")
    H_K, H_V = builder.heisenberg_KV(J=1.0)
    print(f"H_K: nnz={H_K.nnz}")
    print(f"H_V: nnz={H_V.nnz}")
    
    # Verify H = H_K + H_V
    H_total = H_K + H_V
    H_direct = builder.heisenberg(J=1.0)
    diff = (H_total - H_direct).toarray()
    print(f"H_K + H_V = H? max diff = {np.max(np.abs(diff)):.2e}")
    
    # Vorticity operator
    print("\n--- Vorticity Operator ---")
    V_op = builder.build_vorticity_operator()
    print(f"Vorticity: shape={V_op.shape}, nnz={V_op.nnz}")
    
    # Test expectation value
    np.random.seed(42)
    psi = np.random.randn(geom.Dim) + 1j * np.random.randn(geom.Dim)
    psi = psi / np.linalg.norm(psi)
    
    V_exp = float(np.real(np.vdot(psi, V_op @ psi)))
    print(f"⟨V⟩ (random state): {V_exp:.4f}")
    
    # Factory function
    print("\n--- Factory Function ---")
    H_K2, H_V2 = build_hamiltonian('heisenberg', geom, split_KV=True, J=1.5, Jz=0.5)
    print(f"Factory built: H_K nnz={H_K2.nnz}, H_V nnz={H_V2.nnz}")
    
    print("\n✅ All Hamiltonian builder tests passed!")
