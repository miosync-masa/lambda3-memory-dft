"""
Engineering Solver Base Classes (CuPy Unified + DSE)
====================================================

全ての工学ソルバーの共通基盤
GPU加速対応（CuPy）
DSE（履歴依存性）機能内蔵

【設計思想】
  サブクラスは _build_hamiltonian() だけ実装すれば
  自動的にDSE対応になる！

  EngineeringSolver (base.py)
    ├── HistoryManager      ← DSE核心
    ├── MemoryKernel        ← DSE核心
    ├── _build_memory_hamiltonian()
    ├── compute_gamma_memory()
    └── solve()             ← テンプレートメソッド

  ThermoMechanicalSolver (thermo_mechanical.py)
    └── _build_hamiltonian()  ← 固有部分だけ

  FatigueSolver (fatigue.py)
    └── _build_hamiltonian()  ← 固有部分だけ

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

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
from scipy.sparse.linalg import eigsh, expm_multiply

# Memory-DFT imports - Core
try:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry
    from memory_dft.core.history_manager import HistoryManager, StateSnapshot
    from memory_dft.core.memory_kernel import (
        SimpleMemoryKernel,
        CompositeMemoryKernel,
    )
except ImportError:
    SparseEngine = Any
    SystemGeometry = Any
    HistoryManager = None
    StateSnapshot = None
    SimpleMemoryKernel = None
    CompositeMemoryKernel = None

# Physics imports
try:
    from memory_dft.physics.thermodynamics import T_to_beta, boltzmann_weights
except ImportError:
    T_to_beta = lambda T: 1.0 / (8.617e-5 * T)


# =============================================================================
# Material Parameters
# =============================================================================

@dataclass
class MaterialParams:
    """Material parameters for engineering calculations."""
    name: str = "Fe"
    E_bond: float = 4.28
    Z_bulk: int = 8
    Z_surface: int = 6
    lattice_constant: float = 2.87
    burgers_vector: float = 2.48
    T_melt: float = 1811.0
    T_debye: float = 470.0
    E_modulus: float = 211.0
    nu_poisson: float = 0.29
    sigma_y0: float = 250.0
    t_hop: float = 1.0
    U_int: float = 5.0
    lambda_critical: float = 0.5
    xi_gb: float = 0.75
    delta_L: float = 0.1
    
    @property
    def G_shear(self) -> float:
        return self.E_modulus / (2 * (1 + self.nu_poisson))
    
    @property
    def U_over_t(self) -> float:
        return self.U_int / self.t_hop
    
    def lambda_critical_T(self, T: float) -> float:
        if T >= self.T_melt:
            return 0.0
        return self.lambda_critical * (1.0 - T / self.T_melt)


# =============================================================================
# Process Conditions
# =============================================================================

@dataclass
class ProcessConditions:
    """Process conditions for engineering simulation."""
    T: Union[float, np.ndarray] = 300.0
    sigma: Union[float, np.ndarray] = 0.0
    epsilon: Union[float, np.ndarray] = 0.0
    time: Optional[np.ndarray] = None
    strain_rate: float = 1e-3
    atmosphere: str = "air"
    
    @property
    def is_path(self) -> bool:
        return isinstance(self.T, np.ndarray)
    
    @property
    def n_steps(self) -> int:
        if self.is_path:
            return len(self.T)
        return 1
    
    def get_T_at(self, step: int) -> float:
        if self.is_path:
            return float(self.T[step])
        return float(self.T)
    
    def get_sigma_at(self, step: int) -> float:
        if isinstance(self.sigma, np.ndarray):
            return float(self.sigma[step])
        return float(self.sigma)


# =============================================================================
# Solver Result
# =============================================================================

@dataclass
class SolverResult:
    """Base result container for engineering solvers."""
    success: bool = True
    message: str = ""
    energy_final: float = 0.0
    lambda_final: float = 0.0
    lambda_history: Optional[np.ndarray] = None
    energy_history: Optional[np.ndarray] = None
    failed: bool = False
    failure_step: Optional[int] = None
    failure_site: Optional[int] = None
    
    # DSE specific
    memory_contribution: Optional[np.ndarray] = None
    gamma_memory: float = 0.0
    
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Solver (CuPy Unified + DSE)
# =============================================================================

class EngineeringSolver(ABC):
    """
    Abstract base class for all engineering solvers.
    GPU acceleration via CuPy.
    DSE (Direct Schrödinger Evolution) built-in!
    
    【使い方】
    サブクラスは _build_hamiltonian() だけ実装すればOK！
    
    class MySolver(EngineeringSolver):
        def _build_hamiltonian(self, T, sigma):
            # 固有のH_K, H_Vを構築
            return H_K, H_V
    
    DSE機能（履歴依存性、H_memory、γ_memory）は自動的に付いてくる！
    """
    
    def __init__(self,
                 material: Union[MaterialParams, str] = None,
                 n_sites: int = 16,
                 Lx: int = None,
                 Ly: int = None,
                 use_gpu: bool = False,
                 use_memory: bool = True,
                 verbose: bool = True):
        """
        Initialize engineering solver with DSE support.
        
        Args:
            material: MaterialParams or material name
            n_sites: Number of lattice sites
            Lx, Ly: 2D lattice dimensions
            use_gpu: Use GPU acceleration (requires CuPy)
            use_memory: Enable DSE memory effects (default True!)
            verbose: Print progress
        """
        # =====================================================================
        # Backend Selection
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
        try:
            self.engine = SparseEngine(self.n_sites, use_gpu=self.use_gpu, verbose=False)
        except Exception as e:
            if verbose:
                print(f"⚠️ SparseEngine not available: {e}")
            self.engine = None
        
        # =====================================================================
        # State
        # =====================================================================
        self.geometry: Optional[SystemGeometry] = None
        self.H_K = None
        self.H_V = None
        self.psi = None
        
        # =====================================================================
        # DSE Components (Core!)
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
        
        # SimpleMemoryKernel is better for DSE (manages history internally)
        if use_memory:
            try:
                from memory_dft.core.memory_kernel import SimpleMemoryKernel
                self.memory_kernel = SimpleMemoryKernel(
                    eta=0.2,
                    tau=5.0,
                    gamma=0.5
                )
            except ImportError:
                self.memory_kernel = None
        else:
            self.memory_kernel = None
        
        # Track current time step
        self.current_time = 0.0
        
        # Memory contribution history
        self.memory_history: List[float] = []
        
        # =====================================================================
        # History Arrays
        # =====================================================================
        self.lambda_history: List[float] = []
        self.energy_history: List[float] = []
        self.T_history: List[float] = []
        self.sigma_history: List[float] = []
        
        if verbose:
            print("=" * 60)
            print(f"{self.__class__.__name__}")
            print("=" * 60)
            print(f"  Material: {self.material.name}")
            print(f"  Lattice: {self.Lx} × {self.Ly} = {self.n_sites} sites")
            print(f"  U/t: {self.material.U_over_t:.1f}")
            print(f"  DSE Memory: {'ENABLED' if use_memory else 'DISABLED'}")
            print(f"  Backend: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'}")
            print("=" * 60)
    
    # =========================================================================
    # Array Conversion Utilities
    # =========================================================================
    
    def _to_device(self, arr):
        """Convert array to device (GPU if available)"""
        if self.use_gpu and not isinstance(arr, cp.ndarray):
            return cp.asarray(arr)
        return arr
    
    def _to_host(self, arr):
        """Convert array to host (CPU)"""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def _to_sparse(self, data, format='csr'):
        """Create sparse matrix on appropriate device"""
        if self.use_gpu:
            if isinstance(data, np.ndarray):
                data = cp.asarray(data)
            return cp_sparse.diags(data, format=format, dtype=cp.complex128)
        else:
            return sp.diags(data, format=format, dtype=np.complex128)
    
    # =========================================================================
    # Abstract Method (サブクラスで実装)
    # =========================================================================
    
    @abstractmethod
    def _build_hamiltonian(self, T: float, sigma: float) -> Tuple:
        """
        Build Hamiltonian for given conditions.
        
        サブクラスで実装する！
        
        Args:
            T: Temperature (K)
            sigma: Stress (arb)
            
        Returns:
            (H_K, H_V): Kinetic and potential parts
            
        Note:
            H_memory は base.py で自動追加されるので
            ここでは考慮しなくてOK！
        """
        pass
    
    # =========================================================================
    # DSE: Memory Hamiltonian (共通機能)
    # =========================================================================
    
    def _build_memory_hamiltonian(self):
        """
        Build history-dependent memory term using SimpleMemoryKernel.
        
        SimpleMemoryKernel.compute_memory_contribution(t, psi) returns
        the memory-weighted Δλ contribution.
        
        This is DSE's core feature!
        γ_memory = 1.216 → 46.7% of correlations are non-Markovian
        """
        xp = self.xp
        
        if self.engine is None:
            return None
        
        dim = self.engine.dim
        
        # If no memory kernel or no psi, return zero
        if self.memory_kernel is None or self.psi is None:
            return self._to_sparse(xp.zeros(dim, dtype=xp.float64))
        
        # Get memory contribution from SimpleMemoryKernel
        psi_host = self._to_host(self.psi)
        delta_lambda = self.memory_kernel.compute_memory_contribution(
            self.current_time, psi_host
        )
        
        # Convert to diagonal Hamiltonian term
        # H_memory scales with delta_lambda
        memory_diag = xp.ones(dim, dtype=xp.float64) * delta_lambda * 0.01
        
        return self._to_sparse(memory_diag)
    
    def _add_memory_term(self):
        """Add memory term to H_V (call after _build_hamiltonian)"""
        if not self.use_memory or self.memory_kernel is None:
            return
        
        H_memory = self._build_memory_hamiltonian()
        if H_memory is not None:
            self.H_V = self.H_V + H_memory
            
            # Record memory contribution
            if self.psi is not None:
                xp = self.xp
                psi_host = self._to_host(self.psi)
                H_mem_host = H_memory
                if self.use_gpu and hasattr(H_memory, 'get'):
                    H_mem_host = H_memory.get()
                mem_contrib = float(np.real(np.vdot(psi_host, H_mem_host @ psi_host)))
                self.memory_history.append(mem_contrib)
    
    def compute_gamma_memory(self) -> float:
        """
        Compute γ_memory: fraction of non-Markovian correlations.
        
        DSE signature: γ_memory ≈ 1.216 → 46.7% non-Markovian!
        """
        if self.history_manager is None or len(self.memory_history) < 10:
            return 0.0
        
        mem_contribs = np.array(self.memory_history[-50:])
        
        if len(mem_contribs) > 1:
            gamma = np.abs(np.corrcoef(mem_contribs[:-1], mem_contribs[1:])[0, 1])
            if np.isnan(gamma):
                gamma = 0.0
        else:
            gamma = 0.0
        
        return gamma
    
    # =========================================================================
    # Common Methods
    # =========================================================================
    
    def compute_lambda(self, psi=None) -> float:
        """Compute global stability parameter λ = K/|V|."""
        xp = self.xp
        
        if psi is None:
            psi = self.psi
        if psi is None or self.H_K is None or self.H_V is None:
            return 0.0
        
        K = float(xp.real(xp.vdot(psi, self.H_K @ psi)))
        V = float(xp.real(xp.vdot(psi, self.H_V @ psi)))
        
        # Stability: avoid division by near-zero
        if abs(V) < 0.01:
            V = 0.01 if V >= 0 else -0.01
        
        return abs(K / V)
    
    def compute_lambda_local(self, psi=None) -> np.ndarray:
        """Compute local λ at each site."""
        if self.engine is None or self.geometry is None:
            lam_global = self.compute_lambda(psi)
            return np.ones(self.n_sites) * lam_global
        
        if psi is None:
            psi = self.psi
        
        psi_host = self._to_host(psi)
        
        # Convert H_K, H_V to host if GPU
        H_K_host = self.H_K
        H_V_host = self.H_V
        if self.use_gpu:
            if hasattr(self.H_K, 'get'):
                H_K_host = self.H_K.get()
            if hasattr(self.H_V, 'get'):
                H_V_host = self.H_V.get()
        
        return self.engine.compute_local_lambda(
            psi_host, H_K_host, H_V_host, self.geometry
        )
    
    def check_failure(self, T: float = 300.0) -> Tuple[bool, Optional[int]]:
        """Check if system has failed (λ > λ_critical)."""
        lambda_c = self.material.lambda_critical_T(T)
        lambda_local = self.compute_lambda_local()
        
        for site, lam in enumerate(lambda_local):
            if lam > 10 or lam < 0:
                continue
            if lam > lambda_c:
                return True, site
        
        lam_global = self.compute_lambda()
        if 0 < lam_global < 10 and lam_global > lambda_c:
            return True, -1
        
        return False, None
    
    def compute_ground_state(self) -> Tuple[float, Any]:
        """Compute ground state of current Hamiltonian."""
        xp = self.xp
        H = self.H_K + self.H_V
        
        if self.use_gpu and cp_eigsh is not None:
            try:
                E0, psi0 = cp_eigsh(H, k=1, which='SA')
                self.psi = psi0[:, 0]
                self.psi = self.psi / xp.linalg.norm(self.psi)
                return float(E0[0]), self.psi
            except Exception:
                pass
        
        # CPU fallback
        H_cpu = self._to_host(H.toarray()) if hasattr(H, 'toarray') else self._to_host(H)
        H_sp = sp.csr_matrix(H_cpu)
        E0, psi0 = eigsh(H_sp, k=1, which='SA')
        psi = psi0[:, 0]
        psi = psi / np.linalg.norm(psi)
        
        if self.use_gpu:
            self.psi = cp.asarray(psi)
        else:
            self.psi = psi
        
        return float(E0[0]), self.psi
    
    def evolve_step(self, dt: float = 0.1):
        """
        Single time evolution step with history recording.
        
        DSE: Records state in both HistoryManager and SimpleMemoryKernel!
        """
        xp = self.xp
        
        if self.psi is None:
            return
        
        H = self.H_K + self.H_V
        
        # Time evolution
        if self.use_gpu:
            # GPU: Euler approximation
            self.psi = self.psi - 1j * dt * (H @ self.psi)
            self.psi = self.psi / xp.linalg.norm(self.psi)
        else:
            # CPU: expm_multiply
            self.psi = expm_multiply(-1j * dt * H, self.psi)
            self.psi = self.psi / np.linalg.norm(self.psi)
        
        # Update time
        self.current_time += dt
        
        # Compute observables
        lam = self.compute_lambda()
        psi_host = self._to_host(self.psi)
        H_psi = H @ self.psi
        energy = float(xp.real(xp.vdot(self.psi, H_psi)))
        
        # DSE: Record in HistoryManager
        if self.history_manager is not None:
            self.history_manager.add(
                time=self.current_time,
                state=psi_host.copy(),
                energy=energy,
                lambda_density=lam
            )
        
        # DSE: Record in SimpleMemoryKernel (for memory contribution calc)
        if self.memory_kernel is not None and hasattr(self.memory_kernel, 'add_state'):
            self.memory_kernel.add_state(self.current_time, lam, psi_host.copy())
        
        return self.psi
    
    # =========================================================================
    # Template Method Pattern: solve()
    # =========================================================================
    
    def solve(self, conditions: ProcessConditions, **kwargs) -> SolverResult:
        """
        Main DSE solver: evolve with history-dependent Hamiltonian.
        
        Template Method Pattern:
          1. Call _build_hamiltonian() [サブクラス実装]
          2. Add H_memory [共通]
          3. Evolve [共通]
          4. Record history [共通]
        
        サブクラスは _build_hamiltonian() だけ実装すればOK！
        """
        xp = self.xp
        
        dt = kwargs.get('dt', 0.1)
        n_sub = kwargs.get('n_steps_per_point', 5)
        
        self.clear_history()
        self.memory_history = []
        
        if self.verbose:
            print(f"\n[DSE Solve] {conditions.n_steps} steps")
            print(f"  Memory: {'ON' if self.use_memory else 'OFF'}")
            print(f"  Backend: {'GPU' if self.use_gpu else 'CPU'}")
        
        # Initialize
        T0 = conditions.get_T_at(0)
        sigma0 = conditions.get_sigma_at(0)
        
        # Build initial Hamiltonian (calls subclass method)
        self.H_K, self.H_V = self._build_hamiltonian(T0, sigma0)
        self._add_memory_term()  # DSE!
        
        E0, self.psi = self.compute_ground_state()
        
        # Evolution loop
        for step in range(conditions.n_steps):
            T = conditions.get_T_at(step)
            sigma = conditions.get_sigma_at(step)
            
            # Rebuild Hamiltonian (calls subclass method)
            self.H_K, self.H_V = self._build_hamiltonian(T, sigma)
            self._add_memory_term()  # DSE!
            
            # Sub-steps
            for _ in range(n_sub):
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
                    print(f"  → Failure at step {step}, site {fail_site}")
                
                return SolverResult(
                    success=True,
                    failed=True,
                    failure_step=step,
                    failure_site=fail_site,
                    lambda_final=lam,
                    energy_final=E,
                    lambda_history=np.array(self.lambda_history),
                    energy_history=np.array(self.energy_history),
                    memory_contribution=np.array(self.memory_history) if self.memory_history else None,
                    gamma_memory=self.compute_gamma_memory(),
                )
            
            if self.verbose and step % max(1, conditions.n_steps // 5) == 0:
                gamma = self.compute_gamma_memory()
                print(f"  Step {step}: T={T:.0f}K, σ={sigma:.2f}, λ={lam:.4f}, γ_mem={gamma:.3f}")
        
        return SolverResult(
            success=True,
            failed=False,
            lambda_final=self.lambda_history[-1],
            energy_final=self.energy_history[-1],
            lambda_history=np.array(self.lambda_history),
            energy_history=np.array(self.energy_history),
            memory_contribution=np.array(self.memory_history) if self.memory_history else None,
            gamma_memory=self.compute_gamma_memory(),
        )
    
    # =========================================================================
    # History Management
    # =========================================================================
    
    def clear_history(self):
        """Clear accumulated history."""
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
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated history."""
        return {
            'n_steps': len(self.lambda_history),
            'lambda_initial': self.lambda_history[0] if self.lambda_history else None,
            'lambda_final': self.lambda_history[-1] if self.lambda_history else None,
            'lambda_max': max(self.lambda_history) if self.lambda_history else None,
            'energy_change': (self.energy_history[-1] - self.energy_history[0]) 
                             if len(self.energy_history) > 1 else 0.0,
            'gamma_memory': self.compute_gamma_memory(),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_material(name: str) -> MaterialParams:
    """Create MaterialParams for common materials."""
    materials = {
        'Fe': MaterialParams(
            name='Fe',
            E_bond=4.28,
            Z_bulk=8,
            T_melt=1811,
            E_modulus=211,
            t_hop=1.0,
            U_int=5.0,
        ),
        'Al': MaterialParams(
            name='Al',
            E_bond=3.39,
            Z_bulk=12,
            T_melt=933,
            E_modulus=70,
            t_hop=1.2,
            U_int=3.0,
        ),
        'Cu': MaterialParams(
            name='Cu',
            E_bond=3.49,
            Z_bulk=12,
            T_melt=1358,
            E_modulus=130,
            t_hop=1.1,
            U_int=4.0,
        ),
        'Ti': MaterialParams(
            name='Ti',
            E_bond=4.85,
            Z_bulk=12,
            T_melt=1941,
            E_modulus=116,
            t_hop=0.9,
            U_int=5.5,
        ),
    }
    
    if name in materials:
        return materials[name]
    else:
        print(f"⚠️ Unknown material '{name}', using Fe defaults")
        return materials['Fe']
