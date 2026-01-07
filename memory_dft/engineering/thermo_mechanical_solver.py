"""
Thermo-Mechanical Solver (Unified)
==================================

熱＋応力の時間発展ソルバー（DSE内蔵）

【設計思想】
  旧: base.py + thermo_mechanical.py の分離
  新: thermo_mechanical_solver.py に統合！
  
  EnvironmentBuilder を使って H(T,σ) を構築
  DSE（履歴依存性）で時間発展

【温度パス】
  T(t) = [T0, T1, T2, ...]
  各ステップで H(T(t)) が変化 → 真の経路依存性！

【H-CSP対応】
  公理5（環境作用）: B_θ(T,σ) で H を修正
  EDR方程式: λ = K/|V| で破壊判定

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field

# CuPy support
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_sparse = None
    cp_eigsh = None
    HAS_CUPY = False

# SciPy (CPU fallback)
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# Core imports
try:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry
    from memory_dft.core.environment_operators import (
        EnvironmentBuilder,
        T_to_beta,
        Dislocation,
        compute_peach_koehler_force,
    )
    from memory_dft.core.history_manager import HistoryManager
    from memory_dft.core.memory_kernel import SimpleMemoryKernel
except ImportError:
    # Development fallback - try local files
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try importing from local environment_operators.py
        from environment_operators import (
            EnvironmentBuilder,
            T_to_beta,
            Dislocation,
            compute_peach_koehler_force,
        )
        
        # For SparseEngine, we need a minimal mock or skip
        SparseEngine = None
        SystemGeometry = None
        HistoryManager = None
        SimpleMemoryKernel = None
        
        print("  (Using local environment_operators.py)")
    except ImportError:
        SparseEngine = None
        SystemGeometry = None
        EnvironmentBuilder = None
        HistoryManager = None
        SimpleMemoryKernel = None
        T_to_beta = lambda T, scale=1.0: scale / (8.617e-5 * T) if T > 0 else float('inf')


# =============================================================================
# Material Parameters
# =============================================================================

@dataclass
class MaterialParams:
    """Material parameters for engineering calculations."""
    name: str = "Fe"
    E_bond: float = 4.28          # eV
    Z_bulk: int = 8               # Coordination number
    Z_surface: int = 6
    lattice_constant: float = 2.87  # Å
    burgers_vector: float = 2.48    # Å
    T_melt: float = 1811.0        # K
    T_debye: float = 470.0        # K
    E_modulus: float = 211.0      # GPa
    nu_poisson: float = 0.29
    sigma_y0: float = 250.0       # MPa
    t_hop: float = 1.0            # Hopping parameter
    U_int: float = 5.0            # Interaction parameter
    lambda_critical: float = 0.5  # Critical λ
    xi_gb: float = 0.75           # Grain boundary factor
    delta_L: float = 0.1          # Lindemann parameter
    alpha_t: float = 1e-4         # Temperature coefficient for t
    
    @property
    def G_shear(self) -> float:
        """Shear modulus."""
        return self.E_modulus / (2 * (1 + self.nu_poisson))
    
    @property
    def U_over_t(self) -> float:
        """U/t ratio."""
        return self.U_int / self.t_hop
    
    def lambda_critical_T(self, T: float) -> float:
        """Temperature-dependent critical λ."""
        if T >= self.T_melt:
            return 0.0
        return self.lambda_critical * (1.0 - T / self.T_melt)


# =============================================================================
# Solver Result
# =============================================================================

@dataclass
class SolverResult:
    """Result container for solver."""
    success: bool = True
    message: str = ""
    
    # Final values
    energy_final: float = 0.0
    lambda_final: float = 0.0
    
    # History
    lambda_history: Optional[np.ndarray] = None
    energy_history: Optional[np.ndarray] = None
    T_history: Optional[np.ndarray] = None
    sigma_history: Optional[np.ndarray] = None
    
    # Failure info
    failed: bool = False
    failure_step: Optional[int] = None
    failure_site: Optional[int] = None
    failure_T: Optional[float] = None
    
    # DSE specific
    memory_contribution: Optional[np.ndarray] = None
    gamma_memory: float = 0.0
    
    # Extra data
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HallPetchResult:
    """Result container for Hall-Petch simulation."""
    grain_sizes: np.ndarray
    yield_stresses: np.ndarray
    lambda_values: np.ndarray
    
    # Fit parameters
    sigma_0: float = 0.0  # Friction stress
    k_hp: float = 0.0     # Hall-Petch coefficient
    
    # Theory comparison
    k_theory: Optional[float] = None
    
    def fit_hall_petch(self):
        """Fit σ_y = σ_0 + k/√d"""
        d_inv_sqrt = 1.0 / np.sqrt(self.grain_sizes)
        
        # Linear fit: σ_y vs 1/√d
        A = np.vstack([d_inv_sqrt, np.ones_like(d_inv_sqrt)]).T
        k, sigma_0 = np.linalg.lstsq(A, self.yield_stresses, rcond=None)[0]
        
        self.k_hp = k
        self.sigma_0 = sigma_0
        
        return self.sigma_0, self.k_hp


# =============================================================================
# Thermo-Mechanical Solver
# =============================================================================

class ThermoMechanicalSolver:
    """
    Unified thermo-mechanical solver with DSE.
    
    温度 T(t) と応力 σ(t) のパスに沿って時間発展。
    EnvironmentBuilder で H(T,σ) を構築。
    DSE で履歴依存性を追跡。
    
    Usage:
        solver = ThermoMechanicalSolver(material='Fe', Lx=4, Ly=4)
        
        # Temperature path (heat treatment)
        T_path = [300, 600, 900, 600, 300]  # Heat and cool
        sigma_path = [0, 0, 0, 0, 0]
        
        result = solver.solve(T_path, sigma_path)
        
        # Hall-Petch
        hp_result = solver.simulate_hall_petch([10, 20, 50, 100])
    """
    
    def __init__(self,
                 material: Union[MaterialParams, str] = None,
                 n_sites: int = 16,
                 Lx: int = None,
                 Ly: int = None,
                 use_memory: bool = True,
                 use_gpu: bool = True,
                 verbose: bool = True):
        """
        Initialize thermo-mechanical solver.
        
        Args:
            material: MaterialParams or material name ('Fe', 'Al', etc.)
            n_sites: Number of lattice sites
            Lx, Ly: 2D lattice dimensions
            use_memory: Enable DSE memory effects
            use_gpu: Use GPU acceleration (requires CuPy)
            verbose: Print progress
        """
        # =====================================================================
        # GPU Setup
        # =====================================================================
        self.use_gpu = use_gpu and HAS_CUPY
        if self.use_gpu:
            self.xp = cp
            self.sp_module = cp_sparse
        else:
            self.xp = np
            self.sp_module = sp
        
        # =====================================================================
        # Material
        # =====================================================================
        if material is None:
            self.material = MaterialParams()
        elif isinstance(material, str):
            self.material = create_material(material)
        else:
            self.material = material
        
        # =====================================================================
        # Geometry
        # =====================================================================
        if Lx is not None and Ly is not None:
            self.Lx = Lx
            self.Ly = Ly
            self.n_sites = Lx * Ly
        else:
            self.n_sites = n_sites
            self.Lx = int(np.sqrt(n_sites))
            self.Ly = self.n_sites // self.Lx
        
        self.verbose = verbose
        
        # =====================================================================
        # Engine
        # =====================================================================
        if SparseEngine is not None:
            self.engine = SparseEngine(
                self.n_sites, 
                use_gpu=self.use_gpu, 
                verbose=False
            )
        else:
            self.engine = None
            if verbose:
                print("⚠️ SparseEngine not available")
        
        # =====================================================================
        # Environment Builder
        # =====================================================================
        if self.engine is not None and EnvironmentBuilder is not None:
            self.env_builder = EnvironmentBuilder(
                self.engine,
                t0=self.material.t_hop,
                U0=self.material.U_int,
                alpha_t=self.material.alpha_t,
                T_ref=300.0,
                T_melt=self.material.T_melt,
                Lx=self.Lx,
                Ly=self.Ly,
            )
        else:
            self.env_builder = None
        
        # =====================================================================
        # Geometry (default: square with defects)
        # =====================================================================
        if self.engine is not None:
            self.geometry = self.engine.build_square_with_defects(
                self.Lx, self.Ly, 
                vacancies=[], 
                weak_bonds=[]
            )
        else:
            self.geometry = None
        
        # =====================================================================
        # State
        # =====================================================================
        self.H_K = None
        self.H_V = None
        self.psi = None
        self.current_time = 0.0
        
        # =====================================================================
        # DSE Components
        # =====================================================================
        self.use_memory = use_memory
        
        if use_memory and HistoryManager is not None:
            self.history_manager = HistoryManager(
                max_history=1000,
                compression_threshold=500,
                use_gpu=self.use_gpu
            )
        else:
            self.history_manager = None
        
        if use_memory and SimpleMemoryKernel is not None:
            self.memory_kernel = SimpleMemoryKernel(
                eta=0.2,
                tau=5.0,
                gamma=0.5
            )
        else:
            self.memory_kernel = None
        
        # =====================================================================
        # History Arrays
        # =====================================================================
        self.lambda_history: List[float] = []
        self.energy_history: List[float] = []
        self.T_history: List[float] = []
        self.sigma_history: List[float] = []
        self.memory_history: List[float] = []
        
        if verbose:
            print("=" * 60)
            print("ThermoMechanicalSolver")
            print("=" * 60)
            print(f"  Material: {self.material.name}")
            print(f"  Lattice: {self.Lx} × {self.Ly} = {self.n_sites} sites")
            print(f"  U/t: {self.material.U_over_t:.1f}")
            print(f"  DSE Memory: {'ENABLED' if use_memory else 'DISABLED'}")
            print(f"  Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
            print("=" * 60)
    
    # =========================================================================
    # Array Utilities
    # =========================================================================
    
    def _to_device(self, arr):
        """Convert to GPU if enabled."""
        if self.use_gpu and not isinstance(arr, cp.ndarray):
            return cp.asarray(arr)
        return arr
    
    def _to_host(self, arr):
        """Convert to CPU."""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    # =========================================================================
    # Hamiltonian Construction
    # =========================================================================
    
    def build_hamiltonian(self, T: float, sigma: float = 0.0) -> Tuple:
        """
        Build H(T, σ) using EnvironmentBuilder.
        
        Args:
            T: Temperature (K)
            sigma: Stress
            
        Returns:
            (H_K, H_V): Kinetic and potential Hamiltonians
        """
        if self.env_builder is None:
            raise RuntimeError("EnvironmentBuilder not available")
        
        H_K, H_V = self.env_builder.build(
            self.geometry,
            T=T,
            sigma=sigma,
            include_dislocations=True
        )
        
        self.H_K = H_K
        self.H_V = H_V
        
        return H_K, H_V
    
    def _add_memory_term(self):
        """Add DSE memory term to H_V."""
        if not self.use_memory or self.memory_kernel is None:
            return
        
        if self.psi is None or self.engine is None:
            return
        
        xp = self.xp
        dim = self.engine.dim
        
        # Get memory contribution
        psi_host = self._to_host(self.psi)
        delta_lambda = self.memory_kernel.compute_memory_contribution(
            self.current_time, psi_host
        )
        
        # Build diagonal memory Hamiltonian
        memory_diag = xp.ones(dim, dtype=xp.float64) * delta_lambda * 0.01
        
        if self.use_gpu:
            H_memory = cp_sparse.diags(memory_diag, format='csr', dtype=cp.complex128)
        else:
            H_memory = sp.diags(memory_diag.astype(np.float64), format='csr', dtype=np.complex128)
        
        self.H_V = self.H_V + H_memory
        
        # Record
        self.memory_history.append(delta_lambda)
    
    # =========================================================================
    # Ground State & Evolution
    # =========================================================================
    
    def compute_ground_state(self) -> Tuple[float, Any]:
        """Compute ground state of current Hamiltonian."""
        H = self.H_K + self.H_V
        
        if self.use_gpu:
            try:
                E0, psi0 = cp_eigsh(H, k=1, which='SA')
                self.psi = psi0[:, 0]
                self.psi = self.psi / cp.linalg.norm(self.psi)
                return float(E0[0]), self.psi
            except Exception:
                pass
        
        # CPU fallback
        if self.use_gpu:
            H_cpu = H.get() if hasattr(H, 'get') else H.toarray()
        else:
            H_cpu = H
        
        H_sp = sp.csr_matrix(H_cpu) if not sp.issparse(H_cpu) else H_cpu
        E0, psi0 = eigsh(H_sp, k=1, which='SA')
        psi = psi0[:, 0]
        psi = psi / np.linalg.norm(psi)
        
        if self.use_gpu:
            self.psi = cp.asarray(psi)
        else:
            self.psi = psi
        
        return float(E0[0]), self.psi
    
    def evolve_step(self, dt: float = 0.1):
        """Single time evolution step with DSE."""
        if self.psi is None:
            return
        
        xp = self.xp
        H = self.H_K + self.H_V
        
        # Euler evolution
        self.psi = self.psi - 1j * dt * (H @ self.psi)
        self.psi = self.psi / xp.linalg.norm(self.psi)
        
        self.current_time += dt
        
        # Record in DSE history
        lam = self.compute_lambda()
        psi_host = self._to_host(self.psi)
        energy = float(xp.real(xp.vdot(self.psi, H @ self.psi)))
        
        if self.history_manager is not None:
            self.history_manager.add(
                time=self.current_time,
                state=psi_host.copy(),
                energy=energy,
                lambda_density=lam
            )
        
        if self.memory_kernel is not None and hasattr(self.memory_kernel, 'add_state'):
            self.memory_kernel.add_state(self.current_time, lam, psi_host.copy())
        
        return self.psi
    
    # =========================================================================
    # Lambda & Failure
    # =========================================================================
    
    def compute_lambda(self, psi=None) -> float:
        """Compute global λ = K/|V|."""
        xp = self.xp
        
        if psi is None:
            psi = self.psi
        if psi is None or self.H_K is None or self.H_V is None:
            return 0.0
        
        K = float(xp.real(xp.vdot(psi, self.H_K @ psi)))
        V = float(xp.real(xp.vdot(psi, self.H_V @ psi)))
        
        if abs(V) < 0.01:
            V = 0.01 if V >= 0 else -0.01
        
        return abs(K / V)
    
    def compute_lambda_local(self, psi=None) -> np.ndarray:
        """Compute local λ at each site."""
        if self.engine is None or self.geometry is None:
            return np.ones(self.n_sites) * self.compute_lambda(psi)
        
        if psi is None:
            psi = self.psi
        
        result = self.engine.compute_local_lambda(
            psi, self.H_K, self.H_V, self.geometry
        )
        
        return self._to_host(result)
    
    def check_failure(self, T: float = 300.0) -> Tuple[bool, Optional[int]]:
        """Check if λ > λ_critical."""
        lambda_c = self.material.lambda_critical_T(T)
        
        try:
            lambda_local = self.compute_lambda_local()
            for site, lam in enumerate(lambda_local):
                if 0 < lam < 10 and lam > lambda_c:
                    return True, site
        except Exception:
            pass
        
        lam_global = self.compute_lambda()
        if 0 < lam_global < 10 and lam_global > lambda_c:
            return True, -1
        
        return False, None
    
    def compute_gamma_memory(self) -> float:
        """Compute γ_memory: non-Markovian fraction."""
        if len(self.memory_history) < 10:
            return 0.0
        
        mem = np.array(self.memory_history[-50:])
        if len(mem) > 1:
            gamma = np.abs(np.corrcoef(mem[:-1], mem[1:])[0, 1])
            return 0.0 if np.isnan(gamma) else gamma
        return 0.0
    
    # =========================================================================
    # Dislocation Management (via EnvironmentBuilder)
    # =========================================================================
    
    def add_dislocation(self, site: int, burgers: Tuple[float, float, float] = (1, 0, 0)):
        """Add dislocation at site."""
        if self.env_builder is not None:
            return self.env_builder.add_dislocation(site, burgers)
    
    def clear_dislocations(self):
        """Remove all dislocations."""
        if self.env_builder is not None:
            self.env_builder.clear_dislocations()
    
    # =========================================================================
    # Main Solver
    # =========================================================================
    
    def solve(self,
              T_path: Union[List[float], np.ndarray],
              sigma_path: Union[List[float], np.ndarray] = None,
              dt: float = 0.1,
              n_sub_steps: int = 5) -> SolverResult:
        """
        Solve along temperature/stress path.
        
        Args:
            T_path: Temperature path [T0, T1, T2, ...]
            sigma_path: Stress path (default: all zeros)
            dt: Time step for evolution
            n_sub_steps: Sub-steps per path point
            
        Returns:
            SolverResult with history and failure info
        """
        T_path = np.array(T_path)
        n_steps = len(T_path)
        
        if sigma_path is None:
            sigma_path = np.zeros(n_steps)
        else:
            sigma_path = np.array(sigma_path)
        
        # Clear history
        self.clear_history()
        
        if self.verbose:
            print(f"\n[Solve] {n_steps} steps")
            print(f"  T: {T_path[0]:.0f}K → {T_path[-1]:.0f}K")
            print(f"  DSE: {'ON' if self.use_memory else 'OFF'}")
        
        xp = self.xp
        
        # Initialize
        T0, sigma0 = T_path[0], sigma_path[0]
        self.build_hamiltonian(T0, sigma0)
        self._add_memory_term()
        E0, _ = self.compute_ground_state()
        
        # Evolution loop
        for step in range(n_steps):
            T = T_path[step]
            sigma = sigma_path[step]
            
            # Rebuild H(T, σ)
            self.build_hamiltonian(T, sigma)
            self._add_memory_term()
            
            # Sub-steps
            for _ in range(n_sub_steps):
                self.evolve_step(dt)
            
            # Record
            lam = self.compute_lambda()
            H = self.H_K + self.H_V
            E = float(xp.real(xp.vdot(self.psi, H @ self.psi)))
            
            self.lambda_history.append(lam)
            self.energy_history.append(E)
            self.T_history.append(T)
            self.sigma_history.append(sigma)
            
            # Check failure
            failed, fail_site = self.check_failure(T)
            if failed:
                if self.verbose:
                    print(f"  → Failure at step {step}, T={T:.0f}K, site={fail_site}")
                
                return SolverResult(
                    success=True,
                    failed=True,
                    failure_step=step,
                    failure_site=fail_site,
                    failure_T=T,
                    lambda_final=lam,
                    energy_final=E,
                    lambda_history=np.array(self.lambda_history),
                    energy_history=np.array(self.energy_history),
                    T_history=np.array(self.T_history),
                    sigma_history=np.array(self.sigma_history),
                    memory_contribution=np.array(self.memory_history) if self.memory_history else None,
                    gamma_memory=self.compute_gamma_memory(),
                )
            
            if self.verbose and step % max(1, n_steps // 5) == 0:
                gamma = self.compute_gamma_memory()
                print(f"  Step {step}: T={T:.0f}K, σ={sigma:.2f}, λ={lam:.4f}, γ={gamma:.3f}")
        
        if self.verbose:
            print(f"  ✅ Complete: λ_final={self.lambda_history[-1]:.4f}")
        
        return SolverResult(
            success=True,
            failed=False,
            lambda_final=self.lambda_history[-1],
            energy_final=self.energy_history[-1],
            lambda_history=np.array(self.lambda_history),
            energy_history=np.array(self.energy_history),
            T_history=np.array(self.T_history),
            sigma_history=np.array(self.sigma_history),
            memory_contribution=np.array(self.memory_history) if self.memory_history else None,
            gamma_memory=self.compute_gamma_memory(),
        )
    
    # =========================================================================
    # Hall-Petch Simulation
    # =========================================================================
    
    def simulate_hall_petch(self,
                            grain_sizes: List[float],
                            T: float = 300.0,
                            sigma_max: float = 10.0,
                            sigma_steps: int = 20) -> HallPetchResult:
        """
        Simulate Hall-Petch relation: σ_y = σ_0 + k/√d
        
        For each grain size d:
          1. Create geometry with grain boundary
          2. Increase stress until failure (λ > λ_c)
          3. Record yield stress σ_y
        
        Args:
            grain_sizes: List of grain sizes [d1, d2, ...]
            T: Temperature (K)
            sigma_max: Maximum stress to try
            sigma_steps: Number of stress increments
            
        Returns:
            HallPetchResult with fit parameters
        """
        grain_sizes = np.array(grain_sizes)
        yield_stresses = []
        lambda_values = []
        
        if self.verbose:
            print(f"\n[Hall-Petch] {len(grain_sizes)} grain sizes")
            print(f"  T = {T:.0f}K")
        
        sigma_range = np.linspace(0, sigma_max, sigma_steps)
        
        for d in grain_sizes:
            # Setup geometry with grain boundary spacing ~ d
            # For small quantum system, d maps to weak bond density
            n_weak = max(1, int(self.n_sites / d))
            weak_bonds = [(i, i+1) for i in range(0, min(n_weak, self.n_sites-1))]
            
            if self.geometry is not None:
                self.geometry.weak_bonds = weak_bonds
            
            # Find yield stress
            sigma_y = sigma_max
            lam_at_yield = 0.0
            
            for sigma in sigma_range:
                self.build_hamiltonian(T, sigma)
                _, _ = self.compute_ground_state()
                
                lam = self.compute_lambda()
                failed, _ = self.check_failure(T)
                
                if failed:
                    sigma_y = sigma
                    lam_at_yield = lam
                    break
            
            yield_stresses.append(sigma_y)
            lambda_values.append(lam_at_yield)
            
            if self.verbose:
                print(f"  d={d:.1f}: σ_y={sigma_y:.2f}, λ={lam_at_yield:.4f}")
        
        result = HallPetchResult(
            grain_sizes=grain_sizes,
            yield_stresses=np.array(yield_stresses),
            lambda_values=np.array(lambda_values),
        )
        
        # Fit Hall-Petch
        sigma_0, k_hp = result.fit_hall_petch()
        
        # Theoretical k (from H-CSP/Λ³ theory)
        G = self.material.G_shear
        xi = self.material.xi_gb
        E_bond = self.material.E_bond
        delta_L = self.material.delta_L
        
        # k = √(G × ξ × E_bond × (1 - δ/δ_L)²) / √(π × λ)
        # Simplified: k ∝ √(G × E_bond)
        lambda_avg = np.mean(lambda_values) if lambda_values else 0.5
        if lambda_avg > 0:
            k_theory = np.sqrt(G * xi * E_bond * (1 - 0.05/delta_L)**2) / np.sqrt(np.pi * lambda_avg)
            result.k_theory = k_theory
        
        if self.verbose:
            print(f"\n  Hall-Petch fit: σ_y = {sigma_0:.2f} + {k_hp:.2f}/√d")
            if result.k_theory:
                print(f"  Theory k = {result.k_theory:.2f}")
        
        return result
    
    # =========================================================================
    # History Management
    # =========================================================================
    
    def clear_history(self):
        """Clear all history."""
        self.lambda_history = []
        self.energy_history = []
        self.T_history = []
        self.sigma_history = []
        self.memory_history = []
        self.current_time = 0.0
        
        if self.history_manager is not None:
            self.history_manager.clear()
        
        if self.memory_kernel is not None and hasattr(self.memory_kernel, 'clear'):
            self.memory_kernel.clear()


# =============================================================================
# Material Factory
# =============================================================================

def create_material(name: str) -> MaterialParams:
    """Create MaterialParams for common materials."""
    materials = {
        'Fe': MaterialParams(
            name='Fe', E_bond=4.28, Z_bulk=8, T_melt=1811,
            E_modulus=211, t_hop=1.0, U_int=5.0, alpha_t=1.2e-5,
        ),
        'Al': MaterialParams(
            name='Al', E_bond=3.39, Z_bulk=12, T_melt=933,
            E_modulus=70, t_hop=1.2, U_int=3.0, alpha_t=2.3e-5,
        ),
        'Cu': MaterialParams(
            name='Cu', E_bond=3.49, Z_bulk=12, T_melt=1358,
            E_modulus=130, t_hop=1.1, U_int=4.0, alpha_t=1.7e-5,
        ),
        'Ti': MaterialParams(
            name='Ti', E_bond=4.85, Z_bulk=12, T_melt=1941,
            E_modulus=116, t_hop=0.9, U_int=5.5, alpha_t=0.9e-5,
        ),
    }
    
    return materials.get(name, materials['Fe'])


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ThermoMechanicalSolver Test")
    print("=" * 70)
    
    # Check dependencies
    print("\n--- Dependency Check ---")
    print(f"  CuPy: {'✅' if HAS_CUPY else '❌'}")
    print(f"  SparseEngine: {'✅' if SparseEngine is not None else '❌'}")
    print(f"  EnvironmentBuilder: {'✅' if EnvironmentBuilder is not None else '❌'}")
    
    if SparseEngine is None or EnvironmentBuilder is None:
        print("\n⚠️ Core dependencies not available.")
        print("   This file should be placed in memory_dft/engineering/")
        print("   or run with proper PYTHONPATH set.")
        print("\n--- Basic Import Test ---")
        print(f"  T_to_beta(300) = {T_to_beta(300):.2f}")
        print(f"  MaterialParams: {MaterialParams()}")
        print("\n✅ Basic structures OK!")
    else:
        try:
            # Create solver
            solver = ThermoMechanicalSolver(
                material='Fe',
                Lx=4, Ly=4,
                use_memory=True,
                use_gpu=HAS_CUPY,
                verbose=True
            )
            
            # Test 1: Simple temperature path
            print("\n--- Test 1: Temperature Path ---")
            T_path = [300, 400, 500, 600, 500, 400, 300]
            result = solver.solve(T_path, dt=0.1, n_sub_steps=3)
            print(f"  λ history: {result.lambda_history}")
            print(f"  γ_memory: {result.gamma_memory:.4f}")
            
            # Test 2: Hall-Petch
            print("\n--- Test 2: Hall-Petch ---")
            hp_result = solver.simulate_hall_petch(
                grain_sizes=[5, 10, 20, 50],
                T=300,
                sigma_max=5.0,
                sigma_steps=10
            )
            print(f"  σ_0 = {hp_result.sigma_0:.2f}")
            print(f"  k_HP = {hp_result.k_hp:.2f}")
            
            print("\n" + "=" * 70)
            print("✅ ThermoMechanicalSolver Test Complete!")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
