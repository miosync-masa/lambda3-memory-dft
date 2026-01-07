"""
Engineering Solver Base Classes (CuPy Unified)
==============================================

全ての工学ソルバーの共通基盤
GPU加速対応（CuPy）

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

# Memory-DFT imports
try:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry
    from memory_dft.physics.thermodynamics import T_to_beta, boltzmann_weights
except ImportError:
    SparseEngine = Any
    SystemGeometry = Any


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
    delta_L: float = 0.1  # Lindemann parameter
    
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
    memory_effect: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Solver (CuPy Unified)
# =============================================================================

class EngineeringSolver(ABC):
    """
    Abstract base class for all engineering solvers.
    GPU acceleration via CuPy.
    """
    
    def __init__(self,
                 material: Union[MaterialParams, str] = None,
                 n_sites: int = 16,
                 use_gpu: bool = False,
                 verbose: bool = True):
        """
        Initialize engineering solver.
        
        Args:
            material: MaterialParams or material name
            n_sites: Number of lattice sites
            use_gpu: Use GPU acceleration (requires CuPy)
            verbose: Print progress
        """
        # Backend selection
        self.use_gpu = use_gpu and HAS_CUPY
        if self.use_gpu:
            self.xp = cp
            self.sp_module = cp_sparse
        else:
            self.xp = np
            self.sp_module = sp
        
        # Material
        if material is None:
            self.material = MaterialParams()
        elif isinstance(material, str):
            self.material = create_material(material)
        else:
            self.material = material
        
        self.n_sites = n_sites
        self.verbose = verbose
        
        # Engine
        try:
            self.engine = SparseEngine(n_sites, use_gpu=self.use_gpu, verbose=False)
        except Exception as e:
            if verbose:
                print(f"⚠️ SparseEngine not available: {e}")
            self.engine = None
        
        # State
        self.geometry: Optional[SystemGeometry] = None
        self.H_K = None
        self.H_V = None
        self.psi = None
        
        # History
        self.lambda_history: List[float] = []
        self.energy_history: List[float] = []
        self.T_history: List[float] = []
        self.sigma_history: List[float] = []
        
        if verbose:
            print("=" * 60)
            print(f"{self.__class__.__name__}")
            print("=" * 60)
            print(f"  Material: {self.material.name}")
            print(f"  Sites: {n_sites}")
            print(f"  U/t: {self.material.U_over_t:.1f}")
            print(f"  Backend: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'}")
            print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Array Conversion Utilities
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def solve(self, conditions: ProcessConditions, **kwargs) -> SolverResult:
        """Main solver method."""
        pass
    
    @abstractmethod
    def _build_hamiltonian(self, T: float, sigma: float) -> Tuple:
        """Build Hamiltonian for given conditions."""
        pass
    
    # -------------------------------------------------------------------------
    # Common Methods
    # -------------------------------------------------------------------------
    
    def compute_lambda(self, psi=None) -> float:
        """Compute global stability parameter λ = K/|V|."""
        xp = self.xp
        
        if psi is None:
            psi = self.psi
        if psi is None or self.H_K is None or self.H_V is None:
            return 0.0
        
        K = float(xp.real(xp.vdot(psi, self.H_K @ psi)))
        V = float(xp.real(xp.vdot(psi, self.H_V @ psi)))
        
        return abs(K / V) if abs(V) > 1e-10 else 1.0
    
    def compute_lambda_local(self, psi=None) -> np.ndarray:
        """Compute local λ at each site."""
        if self.engine is None or self.geometry is None:
            return np.zeros(self.n_sites)
        
        if psi is None:
            psi = self.psi
        
        psi_host = self._to_host(psi)
        
        return self.engine.compute_local_lambda(
            psi_host, self.H_K, self.H_V, self.geometry
        )
    
    def check_failure(self, T: float = 300.0) -> Tuple[bool, Optional[int]]:
        """Check if system has failed (λ > λ_critical)."""
        lambda_c = self.material.lambda_critical_T(T)
        lambda_local = self.compute_lambda_local()
        
        max_site = int(np.argmax(lambda_local))
        max_lambda = lambda_local[max_site]
        
        if max_lambda > lambda_c:
            return True, max_site
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
        """Single time evolution step."""
        xp = self.xp
        
        if self.psi is None:
            return
        
        H = self.H_K + self.H_V
        
        if self.use_gpu:
            # GPU: Euler approximation
            self.psi = self.psi - 1j * dt * (H @ self.psi)
            self.psi = self.psi / xp.linalg.norm(self.psi)
        else:
            # CPU: expm_multiply
            self.psi = expm_multiply(-1j * dt * H, self.psi)
            self.psi = self.psi / np.linalg.norm(self.psi)
        
        return self.psi
    
    def clear_history(self):
        """Clear accumulated history."""
        self.lambda_history = []
        self.energy_history = []
        self.T_history = []
        self.sigma_history = []
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated history."""
        return {
            'n_steps': len(self.lambda_history),
            'lambda_initial': self.lambda_history[0] if self.lambda_history else None,
            'lambda_final': self.lambda_history[-1] if self.lambda_history else None,
            'lambda_max': max(self.lambda_history) if self.lambda_history else None,
            'energy_change': (self.energy_history[-1] - self.energy_history[0]) 
                             if len(self.energy_history) > 1 else 0.0,
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
