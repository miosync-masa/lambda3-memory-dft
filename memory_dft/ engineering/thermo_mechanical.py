"""
Thermo-Mechanical Solver
========================

熱 + 応力 + 転位 の統合シミュレータ

用途:
  - 熱間加工 (Hot Working): 高温 + 応力
  - 冷間加工 (Cold Working): 室温 + 応力 + 加工硬化
  - 焼入れ (Quenching): 急冷 + 残留応力
  - 焼戻し (Tempering): 再加熱 + 転位再配列
  - Hall-Petch 検証: 粒径 vs 降伏応力

物理モデル:
  - Hubbard ハミルトニアン: H = -t Σ c†c + U Σ n↑n↓
  - 温度依存: t(T), λ_critical(T)
  - 応力: H_stress = σ × (勾配項)
  - 転位: ピーチ・ケーラー力 + パイルアップ

Λ³ での統合:
  λ(T, σ) = K(T, σ) / |V(T, σ)|
  
  破壊条件: λ > λ_critical(T)
  
  高温: λ_critical ↓ → 動きやすい
  高応力: λ ↑ → 不安定化

使用例:
    from memory_dft.engineering import ThermoMechanicalSolver
    
    solver = ThermoMechanicalSolver(material='Fe')
    
    # 熱間加工シミュレーション
    result = solver.simulate_hot_working(
        T_start=1200,   # K
        T_end=900,      # K
        sigma=2.0,      # arb
        n_steps=50
    )
    
    # Hall-Petch 検証
    hp_result = solver.simulate_hall_petch(
        grain_sizes=[4, 6, 8, 10, 12],
        T=300,
        n_dislocations=3
    )

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.sparse.linalg import eigsh, expm_multiply

from .base import (
    EngineeringSolver,
    SolverResult,
    MaterialParams,
    ProcessConditions,
    create_material,
)

# Memory-DFT imports
try:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry
    from memory_dft.physics.dislocation_dynamics import DislocationDynamics, Dislocation
except ImportError:
    SparseEngine = Any
    SystemGeometry = Any
    DislocationDynamics = None


# =============================================================================
# Heat Treatment Types
# =============================================================================

class HeatTreatmentType(Enum):
    """Types of heat treatment processes"""
    HOT_WORKING = "hot_working"       # 熱間加工
    COLD_WORKING = "cold_working"     # 冷間加工
    QUENCHING = "quenching"           # 焼入れ
    TEMPERING = "tempering"           # 焼戻し
    ANNEALING = "annealing"           # 焼なまし
    NORMALIZING = "normalizing"       # 焼ならし
    CUSTOM = "custom"                 # カスタム経路


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class ThermoMechanicalResult(SolverResult):
    """
    Result container for thermo-mechanical simulation.
    """
    # Process info
    process_type: Optional[HeatTreatmentType] = None
    
    # Final state
    sigma_y: float = 0.0              # Yield stress
    rho_dislocation: float = 0.0      # Dislocation density
    
    # Path info
    T_history: Optional[np.ndarray] = None
    sigma_history: Optional[np.ndarray] = None
    
    # Dislocation info
    n_dislocations_initial: int = 0
    n_dislocations_final: int = 0
    pileup_count: int = 0
    
    # Hall-Petch (if applicable)
    k_hall_petch: Optional[float] = None
    sigma_0: Optional[float] = None


@dataclass
class HallPetchResult:
    """Hall-Petch simulation results"""
    grain_sizes: np.ndarray
    inv_sqrt_d: np.ndarray
    sigma_y: np.ndarray
    
    # Fit parameters
    k_HP: float = 0.0
    sigma_0: float = 0.0
    
    # Additional
    pileup_counts: Optional[np.ndarray] = None
    duality_indices: Optional[np.ndarray] = None


# =============================================================================
# Thermo-Mechanical Solver
# =============================================================================

class ThermoMechanicalSolver(EngineeringSolver):
    """
    Thermo-mechanical solver combining heat and stress effects.
    
    Integrates:
      - thermodynamics.py: Temperature-dependent H(T)
      - dislocation_dynamics.py: Stress-driven dislocation motion
      - Λ³ theory: Unified failure criterion
    
    Key features:
      1. Temperature-dependent hopping: t(T) = t₀(1 - αT)
      2. Temperature-dependent λ_critical: λ_c(T) = λ_c₀(1 - T/T_m)
      3. Dislocation motion: F = σ × b, move if λ_local > λ_c
      4. Pileup at grain boundaries → Hall-Petch
    """
    
    def __init__(self,
                 material: Union[MaterialParams, str] = None,
                 n_sites: int = 16,
                 Lx: int = None,
                 Ly: int = None,
                 use_gpu: bool = False,
                 verbose: bool = True):
        """
        Initialize thermo-mechanical solver.
        
        Args:
            material: Material parameters
            n_sites: Total sites (or Lx × Ly)
            Lx, Ly: 2D lattice dimensions (optional)
            use_gpu: GPU acceleration
            verbose: Print progress
        """
        super().__init__(material, n_sites, use_gpu, verbose)
        
        # 2D geometry
        if Lx is not None and Ly is not None:
            self.Lx = Lx
            self.Ly = Ly
            self.n_sites = Lx * Ly
        else:
            self.Lx = int(np.sqrt(n_sites))
            self.Ly = self.n_sites // self.Lx
        
        # Dislocation dynamics
        self.dislocations: List[Dislocation] = []
        self.dd_engine: Optional[DislocationDynamics] = None
        
        # Temperature coefficient for hopping
        self.alpha_T = 1e-4  # t(T) = t₀(1 - α×T)
        
        if verbose:
            print(f"  Lattice: {self.Lx} × {self.Ly}")
    
    # -------------------------------------------------------------------------
    # Hamiltonian Construction
    # -------------------------------------------------------------------------
    
    def _build_hamiltonian(self, T: float, sigma: float) -> Tuple:
        """
        Build temperature and stress dependent Hamiltonian.
        
        H(T, σ) = H_K(T) + H_V + H_stress(σ)
        
        Temperature effects:
          - t(T) = t₀ × (1 - α × T / T_melt)
          - Softer at high T
          
        Stress effects:
          - Gradient term proportional to σ
        """
        if self.engine is None:
            raise RuntimeError("SparseEngine not initialized")
        
        # Temperature-dependent hopping
        t_eff = self.material.t_hop * (1.0 - self.alpha_T * T / self.material.T_melt)
        t_eff = max(0.1 * self.material.t_hop, t_eff)  # Don't go negative
        
        U = self.material.U_int
        
        # Build geometry if needed
        if self.geometry is None:
            self.geometry = self.engine.build_square_with_defects(
                self.Lx, self.Ly,
                vacancies=getattr(self, 'vacancies', []),
                weak_bonds=getattr(self, 'weak_bonds', [])
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
        
        # Build Hubbard with defects
        self.H_K, self.H_V = self.engine.build_hubbard_with_defects(
            self.geometry,
            t=t_eff,
            U=U,
            t_weak=0.3 * t_eff
        )
        
        # Add stress gradient
        if sigma > 0:
            H_stress = self._build_stress_hamiltonian(sigma)
            self.H_V = self.H_V + H_stress
        
        return self.H_K, self.H_V
    
    def _build_stress_hamiltonian(self, sigma: float):
        """Build stress gradient term"""
        import scipy.sparse as sp
        
        n = self.geometry.n_sites
        dim = 2**n if n <= 10 else n
        
        diag = np.zeros(dim)
        
        if n <= 10:
            # Full Hilbert space diagonal
            for state in range(dim):
                for site in range(n):
                    if (state >> site) & 1:
                        x = site % self.Lx
                        diag[state] += sigma * (x - self.Lx / 2) / self.Lx
        
        return sp.diags(diag, format='csr')
    
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
    
    # -------------------------------------------------------------------------
    # Main Solve Methods
    # -------------------------------------------------------------------------
    
    def solve(self, conditions: ProcessConditions, **kwargs) -> ThermoMechanicalResult:
        """
        Main solver: evolve system along T(t), σ(t) path.
        
        Args:
            conditions: ProcessConditions with T and sigma paths
            **kwargs: Additional options (dt, n_steps_per_point, etc.)
            
        Returns:
            ThermoMechanicalResult
        """
        dt = kwargs.get('dt', 0.1)
        n_sub = kwargs.get('n_steps_per_point', 5)
        
        self.clear_history()
        
        if self.verbose:
            print(f"\n[Solve] {conditions.n_steps} steps")
        
        # Initialize
        T0 = conditions.get_T_at(0)
        sigma0 = conditions.get_sigma_at(0)
        
        self._build_hamiltonian(T0, sigma0)
        E0, self.psi = self.compute_ground_state()
        
        n_disl_initial = len(self.dislocations)
        
        # Evolution
        for step in range(conditions.n_steps):
            T = conditions.get_T_at(step)
            sigma = conditions.get_sigma_at(step)
            
            # Update Hamiltonian
            self._build_hamiltonian(T, sigma)
            
            # Sub-steps
            for _ in range(n_sub):
                self.evolve_step(dt)
            
            # Compute observables
            lam = self.compute_lambda()
            E = float(np.real(np.vdot(self.psi, (self.H_K + self.H_V) @ self.psi)))
            
            # Store history
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
                )
            
            if self.verbose and step % max(1, conditions.n_steps // 5) == 0:
                print(f"  Step {step}: T={T:.0f}K, σ={sigma:.2f}, λ={lam:.4f}")
        
        # Final result
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
        )
    
    def _attempt_dislocation_motion(self, T: float, sigma: float):
        """
        Attempt to move dislocations based on T and σ.
        
        Motion condition:
          λ_local > λ_critical(T) AND F > F_threshold
        """
        if not self.dislocations:
            return
        
        lambda_c = self.material.lambda_critical_T(T)
        lambda_local = self.compute_lambda_local()
        
        for disl in self.dislocations:
            if disl.pinned:
                continue
            
            # Peach-Koehler force
            F = sigma * disl.burgers_magnitude
            
            # Find best move
            neighbors = self._get_neighbors(disl.site)
            best_site = None
            best_score = -np.inf
            
            for n in neighbors:
                if n >= len(lambda_local):
                    continue
                lam_n = lambda_local[n]
                score = F * lam_n
                
                if score > best_score and lam_n > lambda_c * 0.5:
                    best_score = score
                    best_site = n
            
            # Move if conditions met
            if best_site is not None and best_score > lambda_c:
                disl.move_to(best_site)
                
                # Pin at grain boundary
                if self._is_grain_boundary(best_site):
                    disl.pinned = True
    
    def _is_grain_boundary(self, site: int) -> bool:
        """Check if site is at grain boundary"""
        gb_sites = getattr(self, 'grain_boundary_sites', [])
        return site in gb_sites
    
    def _count_pileup(self) -> int:
        """Count dislocations at grain boundaries"""
        return sum(1 for d in self.dislocations if d.pinned)
    
    # -------------------------------------------------------------------------
    # Convenience Methods for Common Processes
    # -------------------------------------------------------------------------
    
    def simulate_hot_working(self,
                              T_start: float = 1200,
                              T_end: float = 900,
                              sigma: float = 2.0,
                              n_steps: int = 50,
                              n_dislocations: int = 3) -> ThermoMechanicalResult:
        """
        Simulate hot working process.
        
        High temperature + constant stress.
        Dislocations move easily, dynamic recrystallization possible.
        """
        if self.verbose:
            print(f"\n🔥 Hot Working: {T_start}K → {T_end}K, σ={sigma}")
        
        # Add initial dislocations
        self.dislocations = []
        for i in range(n_dislocations):
            site = np.random.randint(0, self.n_sites)
            self.dislocations.append(Dislocation(site=site))
        
        # Create path
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
        """
        Simulate cold working process.
        
        Room temperature + increasing stress.
        Work hardening due to dislocation accumulation.
        """
        if self.verbose:
            print(f"\n❄️ Cold Working: T={T}K, σ: 0 → {sigma_max}")
        
        # Add initial dislocations
        self.dislocations = []
        for i in range(n_dislocations):
            site = np.random.randint(0, self.n_sites)
            self.dislocations.append(Dislocation(site=site))
        
        # Create path (constant T, increasing σ)
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
        """
        Simulate quenching (rapid cooling).
        
        Fast cooling → residual stress → martensite (in steel).
        """
        if self.verbose:
            print(f"\n💨 Quenching: {T_start}K → {T_end}K")
        
        # Exponential cooling
        tau = (T_start - T_end) / cooling_rate
        t = np.linspace(0, 5 * tau, n_steps)
        T_path = T_end + (T_start - T_end) * np.exp(-t / tau)
        
        # Thermal stress from rapid cooling
        dT_dt = -cooling_rate * np.exp(-t / tau)
        sigma_path = 0.1 * np.abs(dT_dt)  # Simplified thermal stress
        
        conditions = ProcessConditions(T=T_path, sigma=sigma_path)
        result = self.solve(conditions)
        result.process_type = HeatTreatmentType.QUENCHING
        
        return result
    
    def simulate_tempering(self,
                           T_temper: float = 600,
                           T_start: float = 300,
                           hold_time: int = 20,
                           n_steps: int = 50) -> ThermoMechanicalResult:
        """
        Simulate tempering (reheat after quench).
        
        Moderate temperature → stress relief, dislocation rearrangement.
        """
        if self.verbose:
            print(f"\n🌡️ Tempering: {T_start}K → {T_temper}K (hold) → {T_start}K")
        
        # Heat up, hold, cool down
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
    
    # -------------------------------------------------------------------------
    # Hall-Petch Simulation
    # -------------------------------------------------------------------------
    
    def simulate_hall_petch(self,
                            grain_sizes: List[int],
                            T: float = 300,
                            sigma_max: float = 5.0,
                            n_dislocations: int = 3,
                            n_stress_points: int = 25) -> HallPetchResult:
        """
        Simulate Hall-Petch relation: σ_y vs 1/√d
        
        For each grain size:
          1. Create grain structure with GB
          2. Add dislocations
          3. Increase stress until yield
          4. Record σ_y
          
        Args:
            grain_sizes: List of grain sizes (sites per grain)
            T: Temperature (K)
            sigma_max: Maximum stress
            n_dislocations: Dislocations per grain
            n_stress_points: Stress resolution
            
        Returns:
            HallPetchResult with σ_y vs 1/√d and fitted k_HP
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔬 HALL-PETCH SIMULATION")
            print("=" * 60)
            print(f"  Grain sizes: {grain_sizes}")
            print(f"  T = {T}K, σ_max = {sigma_max}")
        
        results = {
            'd': [],
            'inv_sqrt_d': [],
            'sigma_y': [],
            'pileup': [],
        }
        
        stress_range = np.linspace(0, sigma_max, n_stress_points)
        
        for gs in grain_sizes:
            if self.verbose:
                print(f"\n--- Grain size: {gs} sites ---")
            
            # Setup geometry with 2 grains
            self.Lx = gs * 2
            self.Ly = max(4, gs // 2)
            self.n_sites = self.Lx * self.Ly
            
            # Reset engine for new size
            self.engine = SparseEngine(self.n_sites, use_gpu=False, verbose=False)
            self.geometry = None
            
            # Grain boundary at center
            self.grain_boundary_sites = [
                y * self.Lx + gs for y in range(self.Ly)
            ]
            
            # Add dislocations in left grain
            self.dislocations = []
            for i in range(n_dislocations):
                site = (i % self.Ly) * self.Lx + (gs // 2)
                self.dislocations.append(Dislocation(site=site))
            
            # Find yield stress
            sigma_y = sigma_max  # Default if no yield
            
            for sigma in stress_range:
                self._build_hamiltonian(T, sigma)
                self.compute_ground_state()
                
                # Attempt dislocation motion
                self._attempt_dislocation_motion(T, sigma)
                
                # Check if GB is stressed (pileup)
                lambda_local = self.compute_lambda_local()
                lambda_gb = np.mean([
                    lambda_local[s] for s in self.grain_boundary_sites
                    if s < len(lambda_local)
                ])
                
                # Yield condition: GB λ exceeds critical
                if lambda_gb > self.material.lambda_critical_T(T):
                    sigma_y = sigma
                    if self.verbose:
                        print(f"  Yield at σ = {sigma:.3f}")
                    break
            
            # Record
            d_nm = gs * self.material.burgers_vector / 10  # nm
            results['d'].append(d_nm)
            results['inv_sqrt_d'].append(1.0 / np.sqrt(d_nm))
            results['sigma_y'].append(sigma_y)
            results['pileup'].append(self._count_pileup())
        
        # Fit Hall-Petch
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
# Plotting Utilities
# =============================================================================

def plot_thermo_mechanical_result(result: ThermoMechanicalResult, save: bool = True):
    """Plot thermo-mechanical simulation results"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (0,0) T and σ paths
    ax = axes[0, 0]
    ax2 = ax.twinx()
    
    steps = np.arange(len(result.T_history))
    ax.plot(steps, result.T_history, 'r-', label='T (K)', linewidth=2)
    ax2.plot(steps, result.sigma_history, 'b--', label='σ', linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature (K)', color='r')
    ax2.set_ylabel('Stress', color='b')
    ax.set_title('Process Path')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # (0,1) λ history
    ax = axes[0, 1]
    ax.plot(steps, result.lambda_history, 'g-', linewidth=2)
    ax.axhline(y=0.5, color='k', linestyle='--', label='λ_critical (300K)')
    ax.set_xlabel('Step')
    ax.set_ylabel('λ = K/|V|')
    ax.set_title('Stability Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (1,0) Energy history
    ax = axes[1, 0]
    ax.plot(steps, result.energy_history, 'm-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('System Energy')
    ax.grid(True, alpha=0.3)
    
    # (1,1) Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
    Process: {result.process_type.value if result.process_type else 'Custom'}
    
    Failed: {result.failed}
    λ_final: {result.lambda_final:.4f}
    E_final: {result.energy_final:.4f}
    
    Dislocations: {result.n_dislocations_initial} → {result.n_dislocations_final}
    Pileup: {result.pileup_count}
    """
    
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Summary')
    
    plt.tight_layout()
    
    if save:
        fname = f'thermo_mechanical_{result.process_type.value if result.process_type else "result"}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"\n[Saved] {fname}")
    
    plt.close()
    return fig


def plot_hall_petch_result(result: HallPetchResult, save: bool = True):
    """Plot Hall-Petch results"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (0) σ_y vs 1/√d
    ax = axes[0]
    ax.scatter(result.inv_sqrt_d, result.sigma_y, s=100, c='blue', label='Data')
    
    # Fit line
    x_fit = np.linspace(0, max(result.inv_sqrt_d) * 1.1, 100)
    y_fit = result.sigma_0 + result.k_HP * x_fit
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, 
            label=f'Fit: k={result.k_HP:.3f}, σ₀={result.sigma_0:.3f}')
    
    ax.set_xlabel('1/√d (nm⁻¹/²)', fontsize=12)
    ax.set_ylabel('σ_y', fontsize=12)
    ax.set_title('Hall-Petch Relation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (1) σ_y vs d
    ax = axes[1]
    d_nm = 1.0 / result.inv_sqrt_d**2
    ax.scatter(d_nm, result.sigma_y, s=100, c='green')
    ax.set_xlabel('Grain size d (nm)', fontsize=12)
    ax.set_ylabel('σ_y', fontsize=12)
    ax.set_title('Yield Stress vs Grain Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        fname = 'hall_petch_thermo_mechanical.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"\n[Saved] {fname}")
    
    plt.close()
    return fig


# =============================================================================
# Main (Test)
# =============================================================================

def main():
    """Test thermo-mechanical solver"""
    print("\n" + "🔥" * 25)
    print("  THERMO-MECHANICAL SOLVER TEST")
    print("🔥" * 25)
    
    # Create solver
    solver = ThermoMechanicalSolver(
        material='Fe',
        n_sites=16,
        Lx=4,
        Ly=4,
        verbose=True
    )
    
    # Test 1: Hot working
    print("\n" + "-" * 40)
    print("Test 1: Hot Working")
    print("-" * 40)
    result_hot = solver.simulate_hot_working(
        T_start=1200,
        T_end=900,
        sigma=2.0,
        n_steps=30
    )
    print(f"  Final λ: {result_hot.lambda_final:.4f}")
    print(f"  Failed: {result_hot.failed}")
    
    # Test 2: Cold working
    print("\n" + "-" * 40)
    print("Test 2: Cold Working")
    print("-" * 40)
    solver2 = ThermoMechanicalSolver(material='Fe', n_sites=16, verbose=True)
    result_cold = solver2.simulate_cold_working(
        T=300,
        sigma_max=3.0,
        n_steps=30
    )
    print(f"  Final λ: {result_cold.lambda_final:.4f}")
    print(f"  Failed: {result_cold.failed}")
    
    # Test 3: Hall-Petch
    print("\n" + "-" * 40)
    print("Test 3: Hall-Petch")
    print("-" * 40)
    solver3 = ThermoMechanicalSolver(material='Fe', verbose=True)
    hp_result = solver3.simulate_hall_petch(
        grain_sizes=[3, 4, 5, 6],
        T=300,
        n_dislocations=2
    )
    print(f"  k_HP = {hp_result.k_HP:.4f}")
    print(f"  σ_0 = {hp_result.sigma_0:.4f}")
    
    # Plot
    plot_hall_petch_result(hp_result, save=True)
    
    print("\n" + "=" * 60)
    print("✅ Thermo-Mechanical Solver Test Complete!")
    print("=" * 60)
    
    return hp_result


if __name__ == "__main__":
    main()
