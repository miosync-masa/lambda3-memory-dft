"""
Lattice Geometry for Memory-DFT
===============================

2D lattice structures with configurable boundary conditions.

Features:
  - Arbitrary Lx × Ly lattice
  - Periodic / Open boundary conditions
  - Bond enumeration (nearest-neighbor, x-direction, y-direction)
  - Plaquette enumeration for flux/vorticity calculations

This module provides the geometric foundation for 2D quantum
lattice simulations in Memory-DFT.

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field


# =============================================================================
# 1D System Geometry (from sparse_engine.py - kept for compatibility)
# =============================================================================

@dataclass
class SystemGeometry:
    """
    General system geometry for 1D/arbitrary connectivity.
    
    Used by SparseHamiltonianEngine for simple chain/ladder systems.
    For full 2D lattice support, use LatticeGeometry2D.
    
    Attributes:
        n_sites: Number of lattice sites
        bonds: List of (i, j) pairs for connected sites
        plaquettes: List of site tuples forming closed loops (optional)
        positions: Real-space coordinates (optional)
    """
    n_sites: int
    bonds: List[Tuple[int, int]]
    plaquettes: List[Tuple[int, ...]] = None
    positions: np.ndarray = None
    
    @property
    def dim(self) -> int:
        """Hilbert space dimension (2^N for spin-1/2)"""
        return 2 ** self.n_sites
    
    @property
    def n_bonds(self) -> int:
        """Number of bonds"""
        return len(self.bonds)
    
    def __repr__(self) -> str:
        plaq_str = f", {len(self.plaquettes)} plaquettes" if self.plaquettes else ""
        return f"SystemGeometry(N={self.n_sites}, {self.n_bonds} bonds{plaq_str})"


# =============================================================================
# 2D Lattice Geometry
# =============================================================================

class LatticeGeometry2D:
    """
    2D lattice geometry with configurable boundary conditions.
    
    Supports rectangular lattices with independent periodic boundary
    conditions in x and y directions.
    
    Attributes:
        Lx: Number of sites in x-direction
        Ly: Number of sites in y-direction
        periodic_x: Periodic boundary in x-direction
        periodic_y: Periodic boundary in y-direction
        N_spins: Total number of spins (Lx * Ly)
        Dim: Hilbert space dimension (2^N_spins)
        
    Bond classifications:
        bonds_nn: All nearest-neighbor bonds (unique pairs)
        bonds_x: Bonds in x-direction (including periodic)
        bonds_y: Bonds in y-direction (including periodic)
        
    Example:
        >>> geom = LatticeGeometry2D(3, 3, periodic_x=False, periodic_y=False)
        >>> print(f"Sites: {geom.N_spins}, Bonds: {len(geom.bonds_nn)}")
        Sites: 9, Bonds: 12
        >>> print(f"Plaquettes: {len(geom.plaquettes)}")
        Plaquettes: 4
    """
    
    def __init__(self, 
                 Lx: int, 
                 Ly: int, 
                 periodic_x: bool = False, 
                 periodic_y: bool = False):
        """
        Initialize 2D lattice geometry.
        
        Args:
            Lx: Number of sites in x-direction
            Ly: Number of sites in y-direction
            periodic_x: Enable periodic boundary in x
            periodic_y: Enable periodic boundary in y
        """
        self.Lx = Lx
        self.Ly = Ly
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.N_spins = Lx * Ly
        self.Dim = 2 ** self.N_spins
        
        # Build geometry
        self.coords = self._build_coords()
        self.bonds_nn, self.bonds_x, self.bonds_y = self._build_nn_bonds()
        self.plaquettes = self._build_plaquettes()
    
    def idx(self, x: int, y: int) -> int:
        """
        Convert (x, y) coordinates to linear site index.
        
        Site ordering: row-major (y * Lx + x)
        
        Args:
            x: x-coordinate (0 to Lx-1)
            y: y-coordinate (0 to Ly-1)
            
        Returns:
            Linear site index
        """
        return y * self.Lx + x
    
    def coords_from_idx(self, i: int) -> Tuple[int, int]:
        """
        Convert linear site index to (x, y) coordinates.
        
        Args:
            i: Linear site index
            
        Returns:
            (x, y) coordinate tuple
        """
        return (i % self.Lx, i // self.Lx)
    
    def _build_coords(self) -> Dict[int, Tuple[int, int]]:
        """Build site index to coordinate mapping."""
        return {self.idx(x, y): (x, y) 
                for y in range(self.Ly) 
                for x in range(self.Lx)}
    
    def _build_nn_bonds(self) -> Tuple[List[Tuple[int, int]], 
                                        List[Tuple[int, int]], 
                                        List[Tuple[int, int]]]:
        """
        Build nearest-neighbor bond lists.
        
        Returns:
            bonds_nn: Unique NN bonds as sorted (i, j) pairs with i < j
            bonds_x: All x-direction bonds (may include i > j for periodic)
            bonds_y: All y-direction bonds (may include i > j for periodic)
        """
        bonds: Set[Tuple[int, int]] = set()
        bonds_x: List[Tuple[int, int]] = []
        bonds_y: List[Tuple[int, int]] = []
        
        for y in range(self.Ly):
            for x in range(self.Lx):
                i = self.idx(x, y)
                
                # x-direction bond
                if x + 1 < self.Lx or self.periodic_x:
                    j = self.idx((x + 1) % self.Lx, y)
                    if i < j:
                        bonds.add((i, j))
                    else:
                        bonds.add((j, i))
                    bonds_x.append((i, j))
                
                # y-direction bond
                if y + 1 < self.Ly or self.periodic_y:
                    j = self.idx(x, (y + 1) % self.Ly)
                    if i < j:
                        bonds.add((i, j))
                    else:
                        bonds.add((j, i))
                    bonds_y.append((i, j))
        
        return sorted(list(bonds)), bonds_x, bonds_y
    
    def _build_plaquettes(self) -> List[Tuple[int, int, int, int]]:
        """
        Build elementary plaquettes (square loops).
        
        Each plaquette is a 4-tuple (bl, br, tr, tl) representing
        bottom-left, bottom-right, top-right, top-left corners.
        
        Used for:
          - Flux/vorticity calculations
          - Wilson loop observables
          - Kitaev model diagnostics
          
        Returns:
            List of plaquette tuples
        """
        plaquettes: List[Tuple[int, int, int, int]] = []
        
        # Determine iteration range based on boundary conditions
        x_range = self.Lx if self.periodic_x else self.Lx - 1
        y_range = self.Ly if self.periodic_y else self.Ly - 1
        
        for y in range(y_range):
            for x in range(x_range):
                bl = self.idx(x, y)
                br = self.idx((x + 1) % self.Lx, y)
                tr = self.idx((x + 1) % self.Lx, (y + 1) % self.Ly)
                tl = self.idx(x, (y + 1) % self.Ly)
                plaquettes.append((bl, br, tr, tl))
        
        return plaquettes
    
    def to_system_geometry(self) -> SystemGeometry:
        """
        Convert to SystemGeometry for compatibility with SparseHamiltonianEngine.
        
        Returns:
            SystemGeometry instance with bonds and plaquettes
        """
        return SystemGeometry(
            n_sites=self.N_spins,
            bonds=self.bonds_nn,
            plaquettes=self.plaquettes
        )
    
    def get_site_neighbors(self, i: int) -> List[int]:
        """
        Get all nearest neighbors of site i.
        
        Args:
            i: Site index
            
        Returns:
            List of neighbor site indices
        """
        neighbors = []
        for (a, b) in self.bonds_nn:
            if a == i:
                neighbors.append(b)
            elif b == i:
                neighbors.append(a)
        return neighbors
    
    def get_bond_direction(self, i: int, j: int) -> Optional[str]:
        """
        Determine the direction of bond (i, j).
        
        Args:
            i, j: Site indices
            
        Returns:
            'x', 'y', or None if not a valid bond
        """
        if (i, j) in self.bonds_x or (j, i) in self.bonds_x:
            return 'x'
        if (i, j) in self.bonds_y or (j, i) in self.bonds_y:
            return 'y'
        return None
    
    def __repr__(self) -> str:
        bc_x = "P" if self.periodic_x else "O"
        bc_y = "P" if self.periodic_y else "O"
        return (f"LatticeGeometry2D({self.Lx}×{self.Ly}, BC={bc_x}{bc_y}, "
                f"N={self.N_spins}, Dim={self.Dim:,}, "
                f"bonds={len(self.bonds_nn)}, plaq={len(self.plaquettes)})")


# =============================================================================
# Convenience Aliases
# =============================================================================

# For backward compatibility with existing code
LatticeGeometry = LatticeGeometry2D


# =============================================================================
# Factory Functions
# =============================================================================

def create_chain(L: int, periodic: bool = True) -> SystemGeometry:
    """
    Create 1D chain geometry.
    
    Args:
        L: Number of sites
        periodic: Enable periodic boundary conditions
        
    Returns:
        SystemGeometry for 1D chain
    """
    if periodic:
        bonds = [(i, (i + 1) % L) for i in range(L)]
    else:
        bonds = [(i, i + 1) for i in range(L - 1)]
    
    return SystemGeometry(n_sites=L, bonds=bonds)


def create_ladder(L: int, periodic: bool = True) -> SystemGeometry:
    """
    Create 2-leg ladder geometry.
    
    Args:
        L: Number of rungs
        periodic: Enable periodic boundary in leg direction
        
    Returns:
        SystemGeometry for ladder
    """
    N = 2 * L
    
    # Leg bonds (along x)
    if periodic:
        leg0 = [(i, (i + 1) % L) for i in range(L)]
        leg1 = [(L + i, L + (i + 1) % L) for i in range(L)]
    else:
        leg0 = [(i, i + 1) for i in range(L - 1)]
        leg1 = [(L + i, L + i + 1) for i in range(L - 1)]
    
    # Rung bonds (along y)
    rungs = [(i, L + i) for i in range(L)]
    
    bonds = leg0 + leg1 + rungs
    
    # Plaquettes
    plaquettes = []
    plaq_range = L if periodic else L - 1
    for i in range(plaq_range):
        bl, br = i, (i + 1) % L
        tl, tr = L + i, L + (i + 1) % L
        plaquettes.append((bl, br, tr, tl))
    
    return SystemGeometry(n_sites=N, bonds=bonds, plaquettes=plaquettes)


def create_square_lattice(Lx: int, Ly: int, 
                          periodic_x: bool = False,
                          periodic_y: bool = False) -> LatticeGeometry2D:
    """
    Create 2D square lattice geometry.
    
    Args:
        Lx, Ly: Lattice dimensions
        periodic_x, periodic_y: Boundary conditions
        
    Returns:
        LatticeGeometry2D instance
    """
    return LatticeGeometry2D(Lx, Ly, periodic_x, periodic_y)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lattice Geometry Test")
    print("=" * 70)
    
    # 1D Chain
    print("\n--- 1D Chain ---")
    chain = create_chain(6, periodic=True)
    print(f"Chain: {chain}")
    print(f"Bonds: {chain.bonds}")
    
    # Ladder
    print("\n--- 2-Leg Ladder ---")
    ladder = create_ladder(4, periodic=False)
    print(f"Ladder: {ladder}")
    print(f"Bonds: {ladder.bonds}")
    print(f"Plaquettes: {ladder.plaquettes}")
    
    # 2D Lattice (Open BC)
    print("\n--- 2D Square Lattice (Open) ---")
    lat_open = LatticeGeometry2D(3, 3, periodic_x=False, periodic_y=False)
    print(f"{lat_open}")
    print(f"NN bonds: {lat_open.bonds_nn}")
    print(f"Plaquettes: {lat_open.plaquettes}")
    
    # 2D Lattice (Periodic)
    print("\n--- 2D Square Lattice (Periodic) ---")
    lat_pbc = LatticeGeometry2D(3, 3, periodic_x=True, periodic_y=True)
    print(f"{lat_pbc}")
    print(f"NN bonds ({len(lat_pbc.bonds_nn)}): {lat_pbc.bonds_nn}")
    print(f"Plaquettes ({len(lat_pbc.plaquettes)})")
    
    # Coordinate conversion
    print("\n--- Coordinate Test ---")
    geom = LatticeGeometry2D(4, 3)
    for i in range(geom.N_spins):
        x, y = geom.coords_from_idx(i)
        i_back = geom.idx(x, y)
        assert i == i_back, f"Coordinate roundtrip failed: {i} != {i_back}"
    print("✅ Coordinate conversion OK!")
    
    # SystemGeometry conversion
    print("\n--- SystemGeometry Conversion ---")
    sys_geom = lat_open.to_system_geometry()
    print(f"Converted: {sys_geom}")
    
    print("\n✅ All lattice geometry tests passed!")
