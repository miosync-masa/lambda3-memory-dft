"""
Thermo-Mechanical Solver (Simplified)
=====================================

熱 + 応力 + 転位 のシミュレータ
DSE機能は base.py に統合済み！

【設計】
  _build_hamiltonian() だけ実装
  → DSE機能は自動的に継承される

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

# CuPy support
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_sparse = None
    HAS_CUPY = False

import scipy.sparse as sp

# Base class (DSE built-in!)
from .base import (
    EngineeringSolver,
    SolverResult,
    MaterialParams,
    ProcessConditions,
    create_material,
)

# Physics
try:
    from memory_dft.physics.dislocation_dynamics import Dislocation
except ImportError:
    @dataclass
    class Dislocation:
        site: int
        burgers: Tuple[float, float, float] = (1.0, 0.0, 0.0)
        pinned: bool = False
        history: List[int] = field(default_factory=list)
        
        @property
        def burgers_magnitude(self) -> float:
            return float(np.sqrt(sum(b**2 for b in self.burgers)))
        
        def move_to(self, new_site: int):
            if not self.pinned:
                self.history.append(new_site)
                self.site = new_site


# =============================================================================
# Heat Treatment Types
# =============================================================================

class HeatTreatmentType(Enum):
    HOT_WORKING = "hot_working"
    COLD_WORKING = "cold_working"
    QUENCHING = "quenching"
    TEMPERING = "tempering"
    ANNEALING = "annealing"
    NORMALIZING = "normalizing"
    CUSTOM = "custom"


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class ThermoMechanicalResult(SolverResult):
    """Result container for thermo-mechanical simulation."""
    process_type: Optional[HeatTreatmentType] = None
    sigma_y: float = 0.0
    rho_dislocation: float = 0.0
    T_history: Optional[np.ndarray] = None
    sigma_history: Optional[np.ndarray] = None
    n_dislocations_initial: int = 0
    n_dislocations_final: int = 0
    pileup_count: int = 0
    k_hall_petch: Optional[float] = None
    sigma_0: Optional[float] = None


@dataclass
class HallPetchResult:
    """Hall-Petch simulation results"""
    grain_sizes: np.ndarray
    inv_sqrt_d: np.ndarray
    sigma_y: np.ndarray
    k_HP: float = 0.0
    sigma_0: float = 0.0
    pileup_counts: Optional[np.ndarray] = None


# =============================================================================
# Thermo-Mechanical Solver
# =============================================================================

class ThermoMechanicalSolver(EngineeringSolver):
    """
    Thermo-mechanical solver with DSE.
    
    【固有機能】
      - 温度依存 hopping t(T)
      - 応力 Hamiltonian H_stress
      - 転位 dynamics
    
    【継承機能 (from base.py)】
      - HistoryManager
      - MemoryKernel
      - H_memory
      - γ_memory
    """
    
    def __init__(self,
                 material: Union[MaterialParams, str] = None,
                 n_sites: int = 16,
                 Lx: int = None,
                 Ly: int = None,
                 use_gpu: bool = False,
                 use_memory: bool = True,
                 verbose: bool = True):
        """Initialize thermo-mechanical solver."""
        super().__init__(
            material=material,
            n_sites=n_sites,
            Lx=Lx,
            Ly=Ly,
            use_gpu=use_gpu,
            use_memory=use_memory,
            verbose=verbose
        )
        
        # Dislocation list
        self.dislocations: List[Dislocation] = []
        
        # Temperature coefficient
        self.alpha_T = 1e-4
        
        # Vacancies and weak bonds
        self.vacancies: List[int] = []
        self.weak_bonds: List[Tuple[int, int]] = []
        self.grain_boundary_sites: List[int] = []
    
    # =========================================================================
    # _build_hamiltonian (固有部分！)
    # =========================================================================
    
    def _build_hamiltonian(self, T: float, sigma: float) -> Tuple:
        """
        Build temperature and stress dependent Hamiltonian.
        
        H(T, σ) = H_K(T) + H_V + H_stress(σ)
        
        Note: H_memory は base.py で自動追加される！
        """
        xp = self.xp
        
        if self.engine is None:
            raise RuntimeError("SparseEngine not initialized")
        
        # Temperature-dependent hopping
        t_eff = self.material.t_hop * (1.0 - self.alpha_T * T / self.material.T_melt)
        t_eff = max(0.1 * self.material.t_hop, t_eff)
        
        # U/t ratio
        U = min(self.material.U_int, 3.0 * t_eff)
        
        # Build geometry
        if self.geometry is None:
            self.geometry = self.engine.build_square_with_defects(
                self.Lx, self.Ly,
                vacancies=self.vacancies,
                weak_bonds=self.weak_bonds
            )
        
        # Mark dislocation sites as weak bonds
        weak_bonds = list(self.geometry.weak_bonds or [])
        for disl in self.dislocations:
            neighbors = self._get_neighbors(disl.site)
            for n in neighbors:
                bond = (min(disl.site, n), max(disl.site, n))
                if bond not in weak_bonds:
                    weak_bonds.append(bond)
        self.geometry.weak_bonds = weak_bonds
        
        # Build Hubbard
        H_K, H_V = self.engine.build_hubbard_with_defects(
            self.geometry,
            t=t_eff,
            U=U,
            t_weak=0.5 * t_eff
        )
        
        # Add stress gradient
        if abs(sigma) > 1e-10:
            H_stress = self._build_stress_hamiltonian(sigma)
            H_V = H_V + H_stress
        
        return H_K, H_V
    
    def _build_stress_hamiltonian(self, sigma: float):
        """Build stress gradient term"""
        xp = self.xp
        n = self.n_sites
        dim = self.engine.dim
        
        diag = xp.zeros(dim, dtype=xp.float64)
        
        if n <= 16:
            for state in range(dim):
                for site in range(n):
                    if (state >> site) & 1:
                        x = site % self.Lx
                        diag[state] += sigma * (x - self.Lx / 2) / self.Lx
        
        return self._to_sparse(diag)
    
    def _get_neighbors(self, site: int) -> List[int]:
        """Get neighboring sites"""
        neighbors = []
        x = site % self.Lx
        y = site // self.Lx
        
        if x > 0:
            neighbors.append(site - 1)
        if x < self.Lx - 1:
            neighbors.append(site + 1)
        if y > 0:
            neighbors.append(site - self.Lx)
        if y < self.Ly - 1:
            neighbors.append(site + self.Lx)
        
        return neighbors
    
    # =========================================================================
    # solve() オーバーライド（転位動力学追加）
    # =========================================================================
    
    def solve(self, conditions: ProcessConditions, **kwargs) -> ThermoMechanicalResult:
        """Solve with dislocation dynamics."""
        xp = self.xp
        
        dt = kwargs.get('dt', 0.1)
        n_sub = kwargs.get('n_steps_per_point', 5)
        
        self.clear_history()
        
        if self.verbose:
            print(f"\n[Thermo-Mechanical DSE Solve] {conditions.n_steps} steps")
        
        # Initialize
        T0 = conditions.get_T_at(0)
        sigma0 = conditions.get_sigma_at(0)
        
        self.H_K, self.H_V = self._build_hamiltonian(T0, sigma0)
        self._add_memory_term()
        
        E0, self.psi = self.compute_ground_state()
        
        n_disl_initial = len(self.dislocations)
        
        # Evolution
        for step in range(conditions.n_steps):
            T = conditions.get_T_at(step)
            sigma = conditions.get_sigma_at(step)
            
            self.H_K, self.H_V = self._build_hamiltonian(T, sigma)
            self._add_memory_term()
            
            for _ in range(n_sub):
                self.evolve_step(dt)
            
            lam = self.compute_lambda()
            H = self.H_K + self.H_V
            E = float(xp.real(xp.vdot(self.psi, H @ self.psi)))
            
            self.lambda_history.append(lam)
            self.energy_history.append(E)
            self.T_history.append(T)
            self.sigma_history.append(sigma)
            
            # Dislocation motion
            self._attempt_dislocation_motion(T, sigma)
            
            # Check failure
            failed, fail_site = self.check_failure(T)
            if failed:
                if self.verbose:
                    print(f"  → Failure at step {step}, site {fail_site}")
                
                return ThermoMechanicalResult(
                    success=True,
                    failed=True,
                    failure_step=step,
                    failure_site=fail_site,
                    lambda_final=lam,
                    energy_final=E,
                    lambda_history=np.array(self.lambda_history),
                    energy_history=np.array(self.energy_history),
                    T_history=np.array(self.T_history),
                    sigma_history=np.array(self.sigma_history),
                    n_dislocations_initial=n_disl_initial,
                    n_dislocations_final=len(self.dislocations),
                    memory_contribution=np.array(self.memory_history) if self.memory_history else None,
                    gamma_memory=self.compute_gamma_memory(),
                )
            
            if self.verbose and step % max(1, conditions.n_steps // 5) == 0:
                gamma = self.compute_gamma_memory()
                print(f"  Step {step}: T={T:.0f}K, σ={sigma:.2f}, λ={lam:.4f}, γ={gamma:.3f}")
        
        return ThermoMechanicalResult(
            success=True,
            failed=False,
            lambda_final=self.lambda_history[-1],
            energy_final=self.energy_history[-1],
            lambda_history=np.array(self.lambda_history),
            energy_history=np.array(self.energy_history),
            T_history=np.array(self.T_history),
            sigma_history=np.array(self.sigma_history),
            n_dislocations_initial=n_disl_initial,
            n_dislocations_final=len(self.dislocations),
            pileup_count=self._count_pileup(),
            memory_contribution=np.array(self.memory_history) if self.memory_history else None,
            gamma_memory=self.compute_gamma_memory(),
        )
    
    # =========================================================================
    # Dislocation Dynamics
    # =========================================================================
    
    def _attempt_dislocation_motion(self, T: float, sigma: float):
        """Attempt to move dislocations."""
        if not self.dislocations:
            return
        
        lambda_c = self.material.lambda_critical_T(T)
        lambda_local = self.compute_lambda_local()
        
        for disl in self.dislocations:
            if disl.pinned:
                continue
            
            F = sigma * disl.burgers_magnitude
            neighbors = self._get_neighbors(disl.site)
            best_site = None
            best_score = -float('inf')
            
            for n in neighbors:
                if n >= len(lambda_local):
                    continue
                lam_n = lambda_local[n]
                
                if lam_n > 10 or lam_n < 0:
                    continue
                
                score = F * lam_n
                
                if score > best_score and lam_n > lambda_c * 0.5:
                    best_score = score
                    best_site = n
            
            if best_site is not None and best_score > lambda_c:
                disl.move_to(best_site)
                
                if self._is_grain_boundary(best_site):
                    disl.pinned = True
    
    def _is_grain_boundary(self, site: int) -> bool:
        return site in self.grain_boundary_sites
    
    def _count_pileup(self) -> int:
        return sum(1 for d in self.dislocations if d.pinned)
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def simulate_hot_working(self,
                              T_start: float = 1200,
                              T_end: float = 900,
                              sigma: float = 2.0,
                              n_steps: int = 50,
                              n_dislocations: int = 3) -> ThermoMechanicalResult:
        """Simulate hot working process."""
        if self.verbose:
            print(f"\n🔥 Hot Working: {T_start}K → {T_end}K, σ={sigma}")
        
        self.dislocations = []
        for i in range(n_dislocations):
            site = int(np.random.randint(0, self.n_sites))
            self.dislocations.append(Dislocation(site=site))
        
        T_path = np.linspace(T_start, T_end, n_steps)
        sigma_path = np.ones(n_steps) * sigma
        
        conditions = ProcessConditions(T=T_path, sigma=sigma_path)
        result = self.solve(conditions)
        result.process_type = HeatTreatmentType.HOT_WORKING
        
        return result
    
    def simulate_cold_working(self,
                               T: float = 300,
                               sigma_max: float = 3.0,
                               n_steps: int = 50,
                               n_dislocations: int = 5) -> ThermoMechanicalResult:
        """Simulate cold working process."""
        if self.verbose:
            print(f"\n❄️ Cold Working: T={T}K, σ: 0 → {sigma_max}")
        
        self.dislocations = []
        for i in range(n_dislocations):
            site = int(np.random.randint(0, self.n_sites))
            self.dislocations.append(Dislocation(site=site))
        
        T_path = np.ones(n_steps) * T
        sigma_path = np.linspace(0, sigma_max, n_steps)
        
        conditions = ProcessConditions(T=T_path, sigma=sigma_path)
        result = self.solve(conditions)
        result.process_type = HeatTreatmentType.COLD_WORKING
        
        return result
    
    def simulate_quenching(self,
                           T_start: float = 1100,
                           T_end: float = 300,
                           cooling_rate: float = 100,
                           n_steps: int = 50) -> ThermoMechanicalResult:
        """Simulate quenching (rapid cooling)."""
        if self.verbose:
            print(f"\n💨 Quenching: {T_start}K → {T_end}K")
        
        tau = (T_start - T_end) / cooling_rate
        t = np.linspace(0, 5 * tau, n_steps)
        T_path = T_end + (T_start - T_end) * np.exp(-t / tau)
        
        dT_dt = -cooling_rate * np.exp(-t / tau)
        sigma_path = 0.1 * np.abs(dT_dt)
        
        conditions = ProcessConditions(T=T_path, sigma=sigma_path)
        result = self.solve(conditions)
        result.process_type = HeatTreatmentType.QUENCHING
        
        return result
    
    def simulate_tempering(self,
                           T_temper: float = 600,
                           T_start: float = 300,
                           hold_time: int = 20,
                           n_steps: int = 50) -> ThermoMechanicalResult:
        """Simulate tempering."""
        if self.verbose:
            print(f"\n🌡️ Tempering: {T_start}K → {T_temper}K → {T_start}K")
        
        n_heat = n_steps // 3
        n_hold = hold_time
        n_cool = n_steps - n_heat - n_hold
        
        T_path = np.concatenate([
            np.linspace(T_start, T_temper, n_heat),
            np.ones(n_hold) * T_temper,
            np.linspace(T_temper, T_start, n_cool)
        ])
        
        sigma_path = np.zeros(len(T_path))
        
        conditions = ProcessConditions(T=T_path, sigma=sigma_path)
        result = self.solve(conditions)
        result.process_type = HeatTreatmentType.TEMPERING
        
        return result
    
    # =========================================================================
    # Hall-Petch
    # =========================================================================
    
    def simulate_hall_petch(self,
                            grain_sizes: List[int],
                            T: float = 300,
                            sigma_max: float = 5.0,
                            n_dislocations: int = 3,
                            n_stress_points: int = 25) -> HallPetchResult:
        """Simulate Hall-Petch relation."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔬 HALL-PETCH SIMULATION")
            print("=" * 60)
        
        from memory_dft.core.sparse_engine_unified import SparseEngine
        
        results = {'d': [], 'inv_sqrt_d': [], 'sigma_y': [], 'pileup': []}
        
        stress_range = np.linspace(0, sigma_max, n_stress_points)
        
        for gs in grain_sizes:
            if self.verbose:
                print(f"\n--- Grain size: {gs} sites ---")
            
            self.Lx = gs * 2
            self.Ly = 2
            self.n_sites = self.Lx * self.Ly
            
            if self.n_sites > 16:
                if self.verbose:
                    print(f"  Skip: n_sites={self.n_sites} > 16")
                continue
            
            self.engine = SparseEngine(self.n_sites, use_gpu=self.use_gpu, verbose=False)
            self.geometry = None
            
            if self.history_manager is not None:
                self.history_manager.clear()
            self.memory_history = []
            
            self.grain_boundary_sites = [gs + y * self.Lx for y in range(self.Ly)]
            
            self.dislocations = []
            for i in range(n_dislocations):
                site = (i % self.Ly) * self.Lx + (gs // 2)
                if site < self.n_sites:
                    self.dislocations.append(Dislocation(site=site))
            
            sigma_y = sigma_max
            
            for sigma in stress_range:
                self.H_K, self.H_V = self._build_hamiltonian(T, sigma)
                self._add_memory_term()
                E0, self.psi = self.compute_ground_state()
                
                self._attempt_dislocation_motion(T, sigma)
                
                lambda_local = self.compute_lambda_local()
                gb_lambdas = [lambda_local[s] for s in self.grain_boundary_sites 
                             if s < len(lambda_local) and 0 < lambda_local[s] < 10]
                
                lambda_gb = np.mean(gb_lambdas) if gb_lambdas else 0.5
                
                if lambda_gb > self.material.lambda_critical_T(T):
                    sigma_y = sigma
                    if self.verbose:
                        print(f"  Yield at σ = {sigma:.3f}, λ_GB = {lambda_gb:.3f}")
                    break
            
            d_nm = gs * self.material.burgers_vector / 10
            results['d'].append(d_nm)
            results['inv_sqrt_d'].append(1.0 / np.sqrt(d_nm))
            results['sigma_y'].append(sigma_y)
            results['pileup'].append(self._count_pileup())
        
        if not results['d']:
            return HallPetchResult(
                grain_sizes=np.array(grain_sizes),
                inv_sqrt_d=np.array([]),
                sigma_y=np.array([]),
            )
        
        coeffs = np.polyfit(results['inv_sqrt_d'], results['sigma_y'], 1)
        k_HP = coeffs[0]
        sigma_0 = coeffs[1]
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("📊 HALL-PETCH FIT")
            print("=" * 60)
            print(f"  σ_y = σ_0 + k/√d")
            print(f"  σ_0 = {sigma_0:.4f}")
            print(f"  k   = {k_HP:.4f}")
            print(f"  γ_memory = {self.compute_gamma_memory():.3f}")
            print("=" * 60)
        
        return HallPetchResult(
            grain_sizes=np.array(grain_sizes),
            inv_sqrt_d=np.array(results['inv_sqrt_d']),
            sigma_y=np.array(results['sigma_y']),
            k_HP=k_HP,
            sigma_0=sigma_0,
            pileup_counts=np.array(results['pileup']),
        )


# =============================================================================
# Main (Test)
# =============================================================================

def main():
    """Test thermo-mechanical solver"""
    print("\n" + "🔥" * 25)
    print("  THERMO-MECHANICAL SOLVER TEST")
    print(f"  CuPy available: {HAS_CUPY}")
    print("🔥" * 25)
    
    solver = ThermoMechanicalSolver(
        material='Fe',
        n_sites=12,
        Lx=4,
        Ly=3,
        verbose=True,
        use_memory=True,
        use_gpu=HAS_CUPY
    )
    
    print("\n" + "-" * 40)
    print("Test: Cold Working")
    print("-" * 40)
    result = solver.simulate_cold_working(T=300, sigma_max=3.0, n_steps=20)
    print(f"  Final λ: {result.lambda_final:.4f}")
    print(f"  γ_memory: {result.gamma_memory:.3f}")
    print(f"  Failed: {result.failed}")
    
    print("\n" + "=" * 60)
    print("✅ Thermo-Mechanical Test Complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
