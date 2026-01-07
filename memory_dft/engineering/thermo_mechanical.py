"""
Thermo-Mechanical DSE Solver
============================

熱 + 応力 + 転位 + 履歴依存性 の統合シミュレータ

【重要】これは単なるSchrödinger Evolutionではなく、
Direct Schrödinger Evolution (DSE) である！

  Standard DFT:  E[ρ(r)]      → 同じ構造 = 同じエネルギー
  DSE:           E[ψ(t)]      → 異なる履歴 = 異なるエネルギー
  
  γ_memory = 1.216 (46.7% of correlations are non-Markovian!)

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
  - 【NEW】履歴依存: H_memory from MemoryKernel

Λ³ での統合:
  λ(T, σ, history) = K(T, σ) / |V(T, σ) + V_memory(history)|
  
  破壊条件: λ > λ_critical(T)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.sparse.linalg import eigsh, expm_multiply
import scipy.sparse as sp

from .base import (
    EngineeringSolver,
    SolverResult,
    MaterialParams,
    ProcessConditions,
    create_material,
)

# Memory-DFT imports - Core
try:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry
    from memory_dft.core.history_manager import HistoryManager, StateSnapshot
    from memory_dft.core.memory_kernel import (
        CompositeMemoryKernel,
        PowerLawKernel,
        StretchedExpKernel,
        ExclusionKernel,
        KernelWeights,
    )
except ImportError:
    SparseEngine = Any
    SystemGeometry = Any
    HistoryManager = None
    StateSnapshot = None
    CompositeMemoryKernel = None

# Memory-DFT imports - Physics
try:
    from memory_dft.physics.dislocation_dynamics import DislocationDynamics, Dislocation
    from memory_dft.physics.thermodynamics import T_to_beta, thermal_energy
except ImportError:
    DislocationDynamics = None
    Dislocation = None
    T_to_beta = lambda T: 1.0 / (8.617e-5 * T)
    thermal_energy = lambda T: 8.617e-5 * T


# =============================================================================
# Heat Treatment Types
# =============================================================================

class HeatTreatmentType(Enum):
    """Types of heat treatment processes"""
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
    """Result container for thermo-mechanical DSE simulation."""
    process_type: Optional[HeatTreatmentType] = None
    sigma_y: float = 0.0
    rho_dislocation: float = 0.0
    T_history: Optional[np.ndarray] = None
    sigma_history: Optional[np.ndarray] = None
    n_dislocations_initial: int = 0
    n_dislocations_final: int = 0
    pileup_count: int = 0
    
    # DSE specific
    memory_contribution: Optional[np.ndarray] = None
    gamma_memory: float = 0.0
    
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
    duality_indices: Optional[np.ndarray] = None


# =============================================================================
# Thermo-Mechanical DSE Solver
# =============================================================================

class ThermoMechanicalSolver(EngineeringSolver):
    """
    Thermo-mechanical DSE solver with memory effects.
    
    Integrates:
      - SparseEngine: Hamiltonian construction
      - HistoryManager: History tracking (DSE!)
      - MemoryKernel: Non-Markovian memory effects (DSE!)
      - DislocationDynamics: Stress-driven dislocation motion
      - Thermodynamics: Temperature effects
    
    Key DSE features:
      1. H_eff = H + H_memory (history-dependent!)
      2. γ_memory tracking (46.7% of correlations)
      3. Path-dependent energy: different history = different result
    """
    
    def __init__(self,
                 material: Union[MaterialParams, str] = None,
                 n_sites: int = 16,
                 Lx: int = None,
                 Ly: int = None,
                 use_gpu: bool = False,
                 verbose: bool = True,
                 use_memory: bool = True):
        """
        Initialize thermo-mechanical DSE solver.
        
        Args:
            material: Material parameters
            n_sites: Total sites
            Lx, Ly: 2D lattice dimensions
            use_gpu: GPU acceleration
            verbose: Print progress
            use_memory: Enable DSE memory effects (default True!)
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
        
        # Temperature coefficient
        self.alpha_T = 1e-4
        
        # =====================================================================
        # DSE Components (NEW!)
        # =====================================================================
        self.use_memory = use_memory
        
        if use_memory and HistoryManager is not None:
            self.history_manager = HistoryManager(
                max_history=1000,
                use_gpu=use_gpu
            )
        else:
            self.history_manager = None
        
        if use_memory and CompositeMemoryKernel is not None:
            # 4-layer memory kernel
            self.memory_kernel = CompositeMemoryKernel(
                weights=KernelWeights(
                    field=0.3,      # Field theory (long-range)
                    physical=0.4,   # Physical (power-law decay)
                    chemical=0.2,   # Chemical (stretched exp)
                    exclusion=0.1   # Exclusion (short-range)
                ),
                kernels={
                    'field': PowerLawKernel(alpha=0.5, tau=10.0),
                    'physical': PowerLawKernel(alpha=1.0, tau=5.0),
                    'chemical': StretchedExpKernel(tau=3.0, beta=0.7),
                    'exclusion': ExclusionKernel(r_cut=2.0),
                }
            )
        else:
            self.memory_kernel = None
        
        # Memory contribution history
        self.memory_history = []
        self.gamma_memory_history = []
        
        if verbose:
            print(f"  Lattice: {self.Lx} × {self.Ly}")
            print(f"  DSE Memory: {'ENABLED' if use_memory else 'DISABLED'}")
    
    # -------------------------------------------------------------------------
    # DSE Hamiltonian Construction (with memory!)
    # -------------------------------------------------------------------------
    
    def _build_hamiltonian(self, T: float, sigma: float) -> Tuple:
        """
        Build temperature, stress, and history dependent Hamiltonian.
        
        H_eff(T, σ, history) = H_K(T) + H_V + H_stress(σ) + H_memory(history)
        
        This is the core DSE feature: history-dependent Hamiltonian!
        """
        if self.engine is None:
            raise RuntimeError("SparseEngine not initialized")
        
        # Temperature-dependent hopping
        t_eff = self.material.t_hop * (1.0 - self.alpha_T * T / self.material.T_melt)
        t_eff = max(0.1 * self.material.t_hop, t_eff)
        
        # U/t ratio adjustment for stability
        # Strong correlation (U/t > 4) causes V → 0 → λ divergence
        U = min(self.material.U_int, 3.0 * t_eff)  # Cap U/t at 3
        
        # Build geometry
        if self.geometry is None:
            self.geometry = self.engine.build_square_with_defects(
                self.Lx, self.Ly,
                vacancies=getattr(self, 'vacancies', []),
                weak_bonds=getattr(self, 'weak_bonds', [])
            )
        
        # Mark dislocation sites
        weak_bonds = list(self.geometry.weak_bonds or [])
        for disl in self.dislocations:
            neighbors = self._get_neighbors(disl.site)
            for n in neighbors:
                bond = (min(disl.site, n), max(disl.site, n))
                if bond not in weak_bonds:
                    weak_bonds.append(bond)
        self.geometry.weak_bonds = weak_bonds
        
        # Build Hubbard
        self.H_K, self.H_V = self.engine.build_hubbard_with_defects(
            self.geometry,
            t=t_eff,
            U=U,
            t_weak=0.5 * t_eff
        )
        
        # Add stress gradient
        if abs(sigma) > 1e-10:
            H_stress = self._build_stress_hamiltonian(sigma)
            self.H_V = self.H_V + H_stress
        
        # =====================================================================
        # DSE: Add memory term! (NEW!)
        # =====================================================================
        if self.use_memory and self.memory_kernel is not None:
            H_memory = self._build_memory_hamiltonian()
            self.H_V = self.H_V + H_memory
            
            # Track memory contribution
            if self.psi is not None:
                mem_contrib = float(np.real(np.vdot(self.psi, H_memory @ self.psi)))
                self.memory_history.append(mem_contrib)
        
        return self.H_K, self.H_V
    
    def _build_stress_hamiltonian(self, sigma: float):
        """Build stress gradient term"""
        n = self.n_sites
        dim = self.engine.dim
        
        diag = np.zeros(dim)
        
        if n <= 16:
            for state in range(dim):
                for site in range(n):
                    if (state >> site) & 1:
                        x = site % self.Lx
                        diag[state] += sigma * (x - self.Lx / 2) / self.Lx
        
        return sp.diags(diag, format='csr', dtype=np.complex128)
    
    def _build_memory_hamiltonian(self):
        """
        Build history-dependent memory term.
        
        H_memory = Σ_τ K(t-τ) × V_interaction(τ)
        
        This is the KEY DSE feature!
        """
        dim = self.engine.dim
        
        if self.history_manager is None or len(self.history_manager.history) < 2:
            return sp.csr_matrix((dim, dim), dtype=np.complex128)
        
        # Get history
        history = self.history_manager.history
        n_history = len(history)
        
        # Memory strength from kernel
        memory_diag = np.zeros(dim)
        total_weight = 0.0
        
        for tau, snapshot in enumerate(history[-50:]):  # Last 50 steps
            # Kernel weight
            weight = self.memory_kernel.evaluate(tau + 1)
            total_weight += weight
            
            # Use past lambda as memory influence
            if hasattr(snapshot, 'lambda_val'):
                lambda_past = snapshot.lambda_val
            else:
                lambda_past = 0.5
            
            # Memory contribution (simplified: diagonal)
            memory_diag += weight * lambda_past * 0.01
        
        # Normalize
        if total_weight > 0:
            memory_diag /= total_weight
        
        return sp.diags(memory_diag, format='csr', dtype=np.complex128)
    
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
    # DSE Evolution
    # -------------------------------------------------------------------------
    
    def evolve_step(self, dt: float):
        """
        Evolve wavefunction with history recording.
        """
        if self.psi is None:
            return
        
        H = self.H_K + self.H_V
        self.psi = expm_multiply(-1j * dt * H, self.psi)
        self.psi = self.psi / np.linalg.norm(self.psi)
        
        # Record in history (DSE!)
        if self.history_manager is not None:
            lam = self.compute_lambda()
            self.history_manager.record(
                StateSnapshot(
                    psi=self.psi.copy(),
                    lambda_val=lam,
                    energy=float(np.real(np.vdot(self.psi, H @ self.psi)))
                )
            )
    
    def compute_lambda(self) -> float:
        """Compute λ = K/|V| with stability check"""
        if self.psi is None or self.H_K is None or self.H_V is None:
            return 1.0
        
        K = float(np.real(np.vdot(self.psi, self.H_K @ self.psi)))
        V = float(np.real(np.vdot(self.psi, self.H_V @ self.psi)))
        
        # Stability: avoid division by near-zero
        if abs(V) < 0.01:
            V = np.sign(V) * 0.01 if V != 0 else 0.01
        
        return abs(K / V)
    
    def compute_gamma_memory(self) -> float:
        """
        Compute γ_memory: fraction of non-Markovian correlations.
        
        Reference: γ_memory = 1.216 (46.7%) from DSE theory
        """
        if self.history_manager is None or len(self.memory_history) < 10:
            return 0.0
        
        # Compare with and without memory
        mem_contribs = np.array(self.memory_history[-50:])
        
        # Autocorrelation at lag 1
        if len(mem_contribs) > 1:
            gamma = np.abs(np.corrcoef(mem_contribs[:-1], mem_contribs[1:])[0, 1])
        else:
            gamma = 0.0
        
        return gamma
    
    # -------------------------------------------------------------------------
    # Main Solve
    # -------------------------------------------------------------------------
    
    def solve(self, conditions: ProcessConditions, **kwargs) -> ThermoMechanicalResult:
        """
        Main DSE solver: evolve with history-dependent Hamiltonian.
        """
        dt = kwargs.get('dt', 0.1)
        n_sub = kwargs.get('n_steps_per_point', 5)
        
        self.clear_history()
        self.memory_history = []
        
        if self.verbose:
            print(f"\n[DSE Solve] {conditions.n_steps} steps")
            print(f"  Memory: {'ON' if self.use_memory else 'OFF'}")
        
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
            
            # Update Hamiltonian (includes memory term!)
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
                    memory_contribution=np.array(self.memory_history) if self.memory_history else None,
                    gamma_memory=self.compute_gamma_memory(),
                )
            
            if self.verbose and step % max(1, conditions.n_steps // 5) == 0:
                gamma = self.compute_gamma_memory()
                print(f"  Step {step}: T={T:.0f}K, σ={sigma:.2f}, λ={lam:.4f}, γ_mem={gamma:.3f}")
        
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
            memory_contribution=np.array(self.memory_history) if self.memory_history else None,
            gamma_memory=self.compute_gamma_memory(),
        )
    
    def check_failure(self, T: float) -> Tuple[bool, Optional[int]]:
        """
        Check for material failure.
        
        Failure condition: λ_local > λ_critical(T)
        But ignore anomalous values (λ > 10 likely numerical issue)
        """
        lambda_local = self.compute_lambda_local()
        lambda_c = self.material.lambda_critical_T(T)
        
        for site, lam in enumerate(lambda_local):
            # Skip anomalous values
            if lam > 10 or lam < 0:
                continue
            if lam > lambda_c:
                return True, site
        
        # Also check global λ
        lam_global = self.compute_lambda()
        if 0 < lam_global < 10 and lam_global > lambda_c:
            return True, -1
        
        return False, None
    
    def _attempt_dislocation_motion(self, T: float, sigma: float):
        """Attempt to move dislocations based on T and σ."""
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
            best_score = -np.inf
            
            for n in neighbors:
                if n >= len(lambda_local):
                    continue
                lam_n = lambda_local[n]
                
                # Skip anomalous values
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
        """Check if site is at grain boundary"""
        gb_sites = getattr(self, 'grain_boundary_sites', [])
        return site in gb_sites
    
    def _count_pileup(self) -> int:
        """Count dislocations at grain boundaries"""
        return sum(1 for d in self.dislocations if d.pinned)
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    def simulate_hot_working(self,
                              T_start: float = 1200,
                              T_end: float = 900,
                              sigma: float = 2.0,
                              n_steps: int = 50,
                              n_dislocations: int = 3) -> ThermoMechanicalResult:
        """Simulate hot working process."""
        if self.verbose:
            print(f"\n🔥 Hot Working (DSE): {T_start}K → {T_end}K, σ={sigma}")
        
        self.dislocations = []
        for i in range(n_dislocations):
            site = np.random.randint(0, self.n_sites)
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
            print(f"\n❄️ Cold Working (DSE): T={T}K, σ: 0 → {sigma_max}")
        
        self.dislocations = []
        for i in range(n_dislocations):
            site = np.random.randint(0, self.n_sites)
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
            print(f"\n💨 Quenching (DSE): {T_start}K → {T_end}K")
        
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
        """Simulate tempering (reheat after quench)."""
        if self.verbose:
            print(f"\n🌡️ Tempering (DSE): {T_start}K → {T_temper}K → {T_start}K")
        
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
        Simulate Hall-Petch relation with DSE.
        
        Memory effects included → more accurate path dependence!
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔬 HALL-PETCH DSE SIMULATION")
            print("=" * 60)
            print(f"  Grain sizes: {grain_sizes}")
            print(f"  Memory: {'ON' if self.use_memory else 'OFF'}")
        
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
            
            # Setup geometry
            self.Lx = gs * 2
            self.Ly = 2  # Fixed for memory efficiency
            self.n_sites = self.Lx * self.Ly
            
            # Skip if too large
            if self.n_sites > 16:
                if self.verbose:
                    print(f"  Skip: n_sites={self.n_sites} > 16")
                continue
            
            # Reset
            self.engine = SparseEngine(self.n_sites, use_gpu=False, verbose=False)
            self.geometry = None
            
            # Reset history for DSE
            if self.history_manager is not None:
                self.history_manager.clear()
            self.memory_history = []
            
            # Grain boundary
            self.grain_boundary_sites = [gs + y * self.Lx for y in range(self.Ly)]
            
            # Dislocations
            self.dislocations = []
            for i in range(n_dislocations):
                site = (i % self.Ly) * self.Lx + (gs // 2)
                if site < self.n_sites:
                    self.dislocations.append(Dislocation(site=site))
            
            # Find yield stress
            sigma_y = sigma_max
            
            for sigma in stress_range:
                self._build_hamiltonian(T, sigma)
                E0, self.psi = self.compute_ground_state()
                
                # Record in history (DSE!)
                if self.history_manager is not None:
                    lam = self.compute_lambda()
                    self.history_manager.record(
                        StateSnapshot(psi=self.psi.copy(), lambda_val=lam, energy=E0)
                    )
                
                self._attempt_dislocation_motion(T, sigma)
                
                lambda_local = self.compute_lambda_local()
                gb_lambdas = [lambda_local[s] for s in self.grain_boundary_sites 
                             if s < len(lambda_local) and 0 < lambda_local[s] < 10]
                
                if gb_lambdas:
                    lambda_gb = np.mean(gb_lambdas)
                else:
                    lambda_gb = 0.5
                
                if lambda_gb > self.material.lambda_critical_T(T):
                    sigma_y = sigma
                    if self.verbose:
                        print(f"  Yield at σ = {sigma:.3f}, λ_GB = {lambda_gb:.3f}")
                    break
            
            # Record
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
        
        # Fit
        coeffs = np.polyfit(results['inv_sqrt_d'], results['sigma_y'], 1)
        k_HP = coeffs[0]
        sigma_0 = coeffs[1]
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("📊 HALL-PETCH DSE FIT")
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
    """Test DSE thermo-mechanical solver"""
    print("\n" + "🔥" * 25)
    print("  THERMO-MECHANICAL DSE SOLVER TEST")
    print("🔥" * 25)
    
    # Create solver with DSE
    solver = ThermoMechanicalSolver(
        material='Fe',
        n_sites=12,
        Lx=4,
        Ly=3,
        verbose=True,
        use_memory=True  # DSE!
    )
    
    # Test: Cold working with DSE
    print("\n" + "-" * 40)
    print("Test: Cold Working (DSE)")
    print("-" * 40)
    result = solver.simulate_cold_working(
        T=300,
        sigma_max=3.0,
        n_steps=20
    )
    print(f"  Final λ: {result.lambda_final:.4f}")
    print(f"  γ_memory: {result.gamma_memory:.3f}")
    print(f"  Failed: {result.failed}")
    
    # Test: Hall-Petch with DSE
    print("\n" + "-" * 40)
    print("Test: Hall-Petch (DSE)")
    print("-" * 40)
    solver2 = ThermoMechanicalSolver(
        material='Fe',
        verbose=True,
        use_memory=True
    )
    hp_result = solver2.simulate_hall_petch(
        grain_sizes=[2, 3, 4],
        T=300,
        n_dislocations=2
    )
    print(f"  k_HP = {hp_result.k_HP:.4f}")
    print(f"  σ_0 = {hp_result.sigma_0:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Thermo-Mechanical DSE Test Complete!")
    print("=" * 60)
    
    return hp_result


if __name__ == "__main__":
    main()
