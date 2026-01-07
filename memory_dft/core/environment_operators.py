"""
Environment Operators for Memory-DFT
====================================

H-CSP理論に基づく環境作用素 B_θ の実装

  H_total = H_0 + B_θ(H_0)
  
  B_θ = B_θ_field × B_θ_env_phys × B_θ_env_chem
  
    Θ_field:     g, E, B, D（場：普遍的）
    Θ_env_phys:  T, σ, p, h（物理環境：局所的）
    Θ_env_chem:  c_O₂, c_Cl, pH（化学環境：不可逆）

【設計思想】
  core/sparse_engine_unified.py が H_0 を構築
  core/environment_operators.py が ΔH(T,σ,...) を構築
  solvers/time_evolution.py が H_total で時間発展

【H-CSP公理との対応】
  公理5（環境作用）：B_θ: H → H(θ)

【EDR方程式】（理論メモより）
  K = K_th + K_mech + K_EM + K_rad
  |V|_eff = |V|_mat - ΔV_EM - ΔV_rad - ΔV_oxide - ΔV_corr + ρ·p + ΔV_cap

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field

# GPU support
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_sparse = None
    HAS_CUPY = False

import scipy.sparse as sp

# Type hints
if TYPE_CHECKING:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry


# =============================================================================
# Physical Constants
# =============================================================================

K_B_EV = 8.617333262e-5   # Boltzmann constant in eV/K
K_B_J = 1.380649e-23      # Boltzmann constant in J/K
H_EV = 4.135667696e-15    # Planck constant in eV·s
HBAR_EV = 6.582119569e-16 # Reduced Planck constant in eV·s


# =============================================================================
# Basic Thermodynamic Utilities
# =============================================================================

def T_to_beta(T_kelvin: float, energy_scale: float = 1.0) -> float:
    """Convert temperature (K) to inverse temperature β."""
    if T_kelvin <= 0:
        return float('inf')
    return energy_scale / (K_B_EV * T_kelvin)


def beta_to_T(beta: float, energy_scale: float = 1.0) -> float:
    """Convert inverse temperature β to temperature (K)."""
    if beta == float('inf') or beta <= 0:
        return 0.0
    return energy_scale / (K_B_EV * beta)


def thermal_energy(T_kelvin: float) -> float:
    """Thermal energy k_B T in eV."""
    return K_B_EV * T_kelvin


def boltzmann_weights(eigenvalues: np.ndarray, beta: float) -> np.ndarray:
    """Compute Boltzmann weights exp(-β E_n) / Z."""
    if beta == float('inf'):
        E_min = eigenvalues[0]
        weights = np.zeros_like(eigenvalues, dtype=float)
        ground_mask = np.abs(eigenvalues - E_min) < 1e-10
        n_ground = np.sum(ground_mask)
        weights[ground_mask] = 1.0 / n_ground
        return weights
    
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    boltzmann = np.exp(-beta * E_shifted)
    Z = np.sum(boltzmann)
    return boltzmann / Z


def partition_function(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute partition function Z = Σ exp(-β E_n)."""
    if beta == float('inf'):
        return 1.0
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    return np.sum(np.exp(-beta * E_shifted))


def compute_entropy(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute entropy S/k_B from partition function."""
    if beta == float('inf'):
        E_min = eigenvalues[0]
        degeneracy = np.sum(np.abs(eigenvalues - E_min) < 1e-10)
        return np.log(degeneracy)
    
    E_min = float(eigenvalues[0])
    E_shifted = np.array(eigenvalues) - E_min
    boltzmann = np.exp(-beta * E_shifted)
    Z = np.sum(boltzmann)
    E_avg = np.sum(E_shifted * boltzmann) / Z
    S = np.log(Z) + beta * E_avg
    return S


def compute_free_energy(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute Helmholtz free energy F = -k_B T ln(Z)."""
    if beta == float('inf'):
        return eigenvalues[0]
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    Z = np.sum(np.exp(-beta * E_shifted))
    return E_min - np.log(Z) / beta


# =============================================================================
# Dislocation Data Structure
# =============================================================================

@dataclass
class Dislocation:
    """Single dislocation representation."""
    site: int
    burgers: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    slip_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    pinned: bool = False
    history: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.site not in self.history:
            self.history.append(self.site)
    
    @property
    def burgers_magnitude(self) -> float:
        return float(np.sqrt(sum(b**2 for b in self.burgers)))
    
    def move_to(self, new_site: int):
        if not self.pinned:
            self.history.append(new_site)
            self.site = new_site


def compute_peach_koehler_force(dislocation: Dislocation, stress: float) -> float:
    """
    Compute Peach-Koehler force on dislocation.
    
    F = σ × b (simplified scalar form)
    """
    return stress * dislocation.burgers_magnitude


# =============================================================================
# Environment Operator Base
# =============================================================================

class EnvironmentOperator:
    """
    Base class for environment operators B_θ.
    
    H_total = H_0 + B_θ(H_0)
    
    Subclasses implement specific environmental effects:
      - TemperatureOperator: T → t(T), K_th
      - StressOperator: σ → H_stress
      - EMFieldOperator: B, E → ΔV_EM
      - ChemicalOperator: 酸化、腐食 → ΔV_chem
    """
    
    def __init__(self, engine: 'SparseEngine'):
        """
        Initialize environment operator.
        
        Args:
            engine: SparseEngine instance
        """
        self.engine = engine
        self.n_sites = engine.n_sites
        self.dim = engine.dim
        self.use_gpu = engine.use_gpu
        self.xp = engine.xp
        self.sparse = engine.sparse
    
    def apply(self, H_K, H_V, **params):
        """
        Apply environment effect to Hamiltonian.
        
        Args:
            H_K: Kinetic Hamiltonian
            H_V: Potential Hamiltonian
            **params: Environment parameters
            
        Returns:
            (H_K_new, H_V_new): Modified Hamiltonians
        """
        raise NotImplementedError("Subclass must implement apply()")


# =============================================================================
# Temperature Operator (Θ_env_phys: T)
# =============================================================================

class TemperatureOperator(EnvironmentOperator):
    """
    Temperature-dependent Hamiltonian modification.
    
    H-CSP: Θ_env_phys の T 成分
    
    Physical models:
      - Lattice expansion: t(T) = t₀(1 - α(T - T_ref)/T_ref)
      - K_th = (3/2) k_B T
      
    【重要】温度は環境関手ではなく K_th への直接寄与
    """
    
    def __init__(self, engine: 'SparseEngine',
                 t0: float = 1.0,
                 U0: float = 2.0,
                 alpha_t: float = 1e-4,
                 alpha_U: float = 0.0,
                 T_ref: float = 300.0,
                 T_melt: float = 1811.0):
        """
        Initialize temperature operator.
        
        Args:
            engine: SparseEngine instance
            t0: Base hopping at T_ref
            U0: Base interaction at T_ref
            alpha_t: Temperature coefficient for t
            alpha_U: Temperature coefficient for U
            T_ref: Reference temperature (K)
            T_melt: Melting temperature (K)
        """
        super().__init__(engine)
        self.t0 = t0
        self.U0 = U0
        self.alpha_t = alpha_t
        self.alpha_U = alpha_U
        self.T_ref = T_ref
        self.T_melt = T_melt
    
    def t_eff(self, T: float) -> float:
        """
        Effective hopping t(T).
        
        t(T) = t₀ × (1 - α_t × (T - T_ref) / T_ref)
        
        Physical: Lattice expansion weakens bonds at high T
        """
        t = self.t0 * (1 - self.alpha_t * (T - self.T_ref) / self.T_ref)
        return max(0.1 * self.t0, t)  # Floor at 10%
    
    def U_eff(self, T: float) -> float:
        """
        Effective interaction U(T).
        
        U(T) = U₀ × (1 - α_U × (T - T_ref) / T_ref)
        """
        U = self.U0 * (1 - self.alpha_U * (T - self.T_ref) / self.T_ref)
        return max(0.1 * self.U0, U)
    
    def lambda_critical(self, T: float) -> float:
        """
        Temperature-dependent critical λ.
        
        λ_c(T) = λ_c,0 × (1 - T/T_melt)
        """
        if T >= self.T_melt:
            return 0.0
        return 0.5 * (1.0 - T / self.T_melt)
    
    def apply(self, H_K, H_V, geometry: 'SystemGeometry',
              T: float = 300.0, **kwargs) -> Tuple:
        """
        Apply temperature effect.
        
        Rebuilds Hamiltonian with t(T) and U(T).
        
        Args:
            H_K, H_V: Base Hamiltonians (ignored, rebuilt)
            geometry: System geometry
            T: Temperature (K)
            
        Returns:
            (H_K_new, H_V_new)
        """
        t = self.t_eff(T)
        U = self.U_eff(T)
        
        # Rebuild with temperature-dependent parameters
        H_K_new, H_V_new = self.engine.build_hubbard_with_defects(
            geometry,
            t=t,
            U=U,
            t_weak=kwargs.get('t_weak', 0.3 * t),
            vacancy_potential=kwargs.get('vacancy_potential', 100.0),
            strain_coupling=kwargs.get('strain_coupling', 0.1),
        )
        
        return H_K_new, H_V_new


# =============================================================================
# Stress Operator (Θ_env_phys: σ)
# =============================================================================

class StressOperator(EnvironmentOperator):
    """
    Stress-dependent Hamiltonian modification.
    
    H-CSP: Θ_env_phys の σ 成分
    
    K_mech = σ²/(2E)
    
    Adds stress gradient to potential energy.
    """
    
    def __init__(self, engine: 'SparseEngine', Lx: int = None, Ly: int = None):
        """
        Initialize stress operator.
        
        Args:
            engine: SparseEngine instance
            Lx, Ly: Lattice dimensions (for stress gradient)
        """
        super().__init__(engine)
        self.Lx = Lx or int(np.sqrt(engine.n_sites))
        self.Ly = Ly or engine.n_sites // self.Lx
    
    def build_stress_hamiltonian(self, sigma: float) -> Any:
        """
        Build stress gradient Hamiltonian.
        
        H_stress = σ × Σ_i (x_i - L/2) × n_i
        
        Creates gradient in x-direction.
        """
        xp = self.xp
        dim = self.dim
        n = self.n_sites
        
        diag = xp.zeros(dim, dtype=xp.float64)
        
        # Only for tractable system sizes
        if n <= 16:
            for state in range(dim):
                for site in range(n):
                    if (state >> site) & 1:
                        x = site % self.Lx
                        diag[state] += sigma * (x - self.Lx / 2) / self.Lx
        
        if self.use_gpu:
            return cp_sparse.diags(diag, format='csr', dtype=cp.complex128)
        else:
            return sp.diags(diag.astype(np.float64), format='csr', dtype=np.complex128)
    
    def apply(self, H_K, H_V, sigma: float = 0.0, **kwargs) -> Tuple:
        """
        Apply stress effect.
        
        Args:
            H_K, H_V: Base Hamiltonians
            sigma: Applied stress
            
        Returns:
            (H_K, H_V + H_stress)
        """
        if abs(sigma) < 1e-10:
            return H_K, H_V
        
        H_stress = self.build_stress_hamiltonian(sigma)
        H_V_new = H_V + H_stress
        
        return H_K, H_V_new


# =============================================================================
# Combined Environment Builder
# =============================================================================

class EnvironmentBuilder:
    """
    Combined environment operator builder.
    
    Applies multiple environmental effects in sequence:
      H_total = H_0 + ΔH(T) + ΔH(σ) + ΔH(B,E) + ...
    
    Usage:
        builder = EnvironmentBuilder(engine)
        H_K, H_V = builder.build(geometry, T=500, sigma=2.0)
    """
    
    def __init__(self, engine: 'SparseEngine',
                 t0: float = 1.0,
                 U0: float = 2.0,
                 alpha_t: float = 1e-4,
                 T_ref: float = 300.0,
                 T_melt: float = 1811.0,
                 Lx: int = None,
                 Ly: int = None):
        """
        Initialize environment builder.
        
        Args:
            engine: SparseEngine instance
            t0, U0: Base parameters
            alpha_t: Temperature coefficient
            T_ref: Reference temperature
            T_melt: Melting temperature
            Lx, Ly: Lattice dimensions
        """
        self.engine = engine
        self.n_sites = engine.n_sites
        
        # Temperature operator
        self.temp_op = TemperatureOperator(
            engine, t0=t0, U0=U0, alpha_t=alpha_t,
            T_ref=T_ref, T_melt=T_melt
        )
        
        # Stress operator
        self.stress_op = StressOperator(engine, Lx=Lx, Ly=Ly)
        
        # Store parameters
        self.t0 = t0
        self.U0 = U0
        self.T_ref = T_ref
        self.T_melt = T_melt
        
        # Dislocation management
        self.dislocations: List[Dislocation] = []
    
    def build(self, geometry: 'SystemGeometry',
              T: float = 300.0,
              sigma: float = 0.0,
              include_dislocations: bool = True,
              **kwargs) -> Tuple:
        """
        Build complete environment-modified Hamiltonian.
        
        H(T, σ) = H_0(T) + H_stress(σ) + H_dislocation
        
        Args:
            geometry: System geometry
            T: Temperature (K)
            sigma: Stress
            include_dislocations: Include dislocation effects
            **kwargs: Additional parameters
            
        Returns:
            (H_K, H_V): Environment-modified Hamiltonians
        """
        # Apply temperature (rebuilds H with t(T), U(T))
        H_K, H_V = self.temp_op.apply(None, None, geometry, T=T, **kwargs)
        
        # Mark dislocation sites as weak bonds
        if include_dislocations and self.dislocations:
            weak_bonds = list(getattr(geometry, 'weak_bonds', []) or [])
            neighbors = self._build_neighbor_map(geometry)
            
            for disl in self.dislocations:
                site = disl.site
                for n in neighbors.get(site, []):
                    bond = (min(site, n), max(site, n))
                    if bond not in weak_bonds:
                        weak_bonds.append(bond)
            
            geometry.weak_bonds = weak_bonds
            
            # Rebuild with updated weak bonds
            H_K, H_V = self.temp_op.apply(None, None, geometry, T=T, **kwargs)
        
        # Apply stress
        H_K, H_V = self.stress_op.apply(H_K, H_V, sigma=sigma)
        
        return H_K, H_V
    
    def _build_neighbor_map(self, geometry: 'SystemGeometry') -> Dict[int, List[int]]:
        """Build neighbor map from geometry."""
        neighbors = {i: [] for i in range(geometry.n_sites)}
        for (i, j) in geometry.bonds:
            neighbors[i].append(j)
            neighbors[j].append(i)
        return neighbors
    
    # -------------------------------------------------------------------------
    # Dislocation Management
    # -------------------------------------------------------------------------
    
    def add_dislocation(self, site: int,
                        burgers: Tuple[float, float, float] = (1, 0, 0)) -> Dislocation:
        """Add dislocation at site."""
        disl = Dislocation(site=site, burgers=burgers)
        self.dislocations.append(disl)
        return disl
    
    def clear_dislocations(self):
        """Remove all dislocations."""
        self.dislocations = []
    
    def get_dislocation_sites(self) -> List[int]:
        """Get list of dislocation sites."""
        return [d.site for d in self.dislocations]
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    def t_eff(self, T: float) -> float:
        """Get effective hopping at temperature T."""
        return self.temp_op.t_eff(T)
    
    def U_eff(self, T: float) -> float:
        """Get effective interaction at temperature T."""
        return self.temp_op.U_eff(T)
    
    def lambda_critical(self, T: float) -> float:
        """Get critical λ at temperature T."""
        return self.temp_op.lambda_critical(T)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Environment Operators Test")
    print("=" * 70)
    
    # Test basic utilities
    print("\n--- Thermodynamic Utilities ---")
    print(f"T=300K → β = {T_to_beta(300):.2f}")
    print(f"k_B T(300K) = {thermal_energy(300):.4f} eV")
    
    # Test with SparseEngine
    try:
        from memory_dft.core.sparse_engine_unified import SparseEngine
        
        print("\n--- Environment Builder Test ---")
        engine = SparseEngine(n_sites=4, use_gpu=False, verbose=False)
        geometry = engine.build_square_with_defects(2, 2)
        
        builder = EnvironmentBuilder(engine, t0=1.0, U0=2.0)
        
        # Test at different temperatures
        for T in [300, 600, 1000]:
            H_K, H_V = builder.build(geometry, T=T, sigma=0.0)
            t = builder.t_eff(T)
            print(f"  T={T}K: t_eff={t:.4f}")
        
        # Test with stress
        H_K, H_V = builder.build(geometry, T=300, sigma=2.0)
        print(f"\n  With σ=2.0: H_V nnz = {H_V.nnz}")
        
        # Test with dislocation
        builder.add_dislocation(site=0)
        H_K, H_V = builder.build(geometry, T=300, sigma=1.0)
        print(f"  With dislocation: weak_bonds = {geometry.weak_bonds}")
        
        print("\n" + "=" * 70)
        print("✅ Environment Operators Test Complete!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n⚠️ SparseEngine not available: {e}")
        print("Basic utilities tested successfully.")
