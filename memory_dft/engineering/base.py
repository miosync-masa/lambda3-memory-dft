"""
Engineering Solver Base Classes
===============================

全ての工学ソルバーの共通基盤

クラス階層:
  EngineeringSolver (基底)
    ├── ThermoMechanicalSolver
    ├── FatigueSolver
    ├── WearSolver
    ├── FormingSolver
    └── MachiningSolver

共通機能:
  - 材料パラメータ管理
  - Λ³場の計算
  - DSE履歴の追跡
  - 破壊判定

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Memory-DFT imports
try:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry
    from memory_dft.physics.thermodynamics import T_to_beta, boltzmann_weights
    from memory_dft.physics.dislocation_dynamics import DislocationDynamics, Dislocation
except ImportError:
    SparseEngine = Any
    SystemGeometry = Any


# =============================================================================
# Material Parameters
# =============================================================================

@dataclass
class MaterialParams:
    """
    Material parameters for engineering calculations.
    
    Contains both fundamental and derived properties.
    """
    # Identity
    name: str = "Fe"
    
    # Fundamental (from DFT/experiment)
    E_bond: float = 4.28          # Bond energy (eV)
    Z_bulk: int = 8               # Bulk coordination
    Z_surface: int = 6            # Surface coordination
    lattice_constant: float = 2.87  # Å (BCC Fe)
    burgers_vector: float = 2.48    # Å
    
    # Thermal
    T_melt: float = 1811.0        # Melting point (K)
    T_debye: float = 470.0        # Debye temperature (K)
    
    # Mechanical
    E_modulus: float = 211.0      # Young's modulus (GPa)
    nu_poisson: float = 0.29      # Poisson's ratio
    sigma_y0: float = 250.0       # Base yield stress (MPa)
    
    # Hubbard parameters (from Fe2 DSE)
    t_hop: float = 1.0            # Hopping (arb or eV)
    U_int: float = 5.0            # On-site interaction
    
    # Λ³ parameters
    lambda_critical: float = 0.5  # Critical stability
    xi_gb: float = 0.75           # GB weakness (Z_gb/Z_bulk)
    
    @property
    def G_shear(self) -> float:
        """Shear modulus from E and ν"""
        return self.E_modulus / (2 * (1 + self.nu_poisson))
    
    @property
    def U_over_t(self) -> float:
        """Correlation strength"""
        return self.U_int / self.t_hop
    
    def lambda_critical_T(self, T: float) -> float:
        """
        Temperature-dependent critical λ.
        
        λ_c(T) = λ_c0 × (1 - T/T_m)
        
        Higher T → lower threshold → easier failure.
        """
        if T >= self.T_melt:
            return 0.0
        return self.lambda_critical * (1.0 - T / self.T_melt)
    
    def __repr__(self) -> str:
        return f"MaterialParams({self.name}, E_bond={self.E_bond}eV, U/t={self.U_over_t:.1f})"


# =============================================================================
# Process Conditions
# =============================================================================

@dataclass
class ProcessConditions:
    """
    Process conditions for engineering simulation.
    
    Can represent:
      - Single point: T=300K, σ=100MPa
      - Path: T(t), σ(t) arrays
    """
    # Temperature (K)
    T: Union[float, np.ndarray] = 300.0
    
    # Stress (MPa or arb)
    sigma: Union[float, np.ndarray] = 0.0
    
    # Strain (optional)
    epsilon: Union[float, np.ndarray] = 0.0
    
    # Time (for paths)
    time: Optional[np.ndarray] = None
    
    # Strain rate (1/s)
    strain_rate: float = 1e-3
    
    # Additional
    atmosphere: str = "air"  # or "vacuum", "N2", etc.
    
    @property
    def is_path(self) -> bool:
        """Check if this is a path (vs single point)"""
        return isinstance(self.T, np.ndarray)
    
    @property
    def n_steps(self) -> int:
        """Number of steps in path"""
        if self.is_path:
            return len(self.T)
        return 1
    
    def get_T_at(self, step: int) -> float:
        """Get temperature at step"""
        if self.is_path:
            return float(self.T[step])
        return float(self.T)
    
    def get_sigma_at(self, step: int) -> float:
        """Get stress at step"""
        if isinstance(self.sigma, np.ndarray):
            return float(self.sigma[step])
        return float(self.sigma)


# =============================================================================
# Solver Result
# =============================================================================

@dataclass
class SolverResult:
    """
    Base result container for engineering solvers.
    
    Subclasses add specific fields.
    """
    # Success flag
    success: bool = True
    message: str = ""
    
    # Core results
    energy_final: float = 0.0
    lambda_final: float = 0.0
    lambda_history: Optional[np.ndarray] = None
    energy_history: Optional[np.ndarray] = None
    
    # Failure info
    failed: bool = False
    failure_step: Optional[int] = None
    failure_site: Optional[int] = None
    
    # DSE memory
    memory_effect: float = 0.0
    
    # Additional data
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "FAILED" if self.failed else "OK"
        return f"SolverResult({status}, λ={self.lambda_final:.4f})"


# =============================================================================
# Base Solver
# =============================================================================

class EngineeringSolver(ABC):
    """
    Abstract base class for all engineering solvers.
    
    Provides common infrastructure:
      - Material parameters
      - Sparse engine
      - DSE history tracking
      - λ field computation
      - Failure detection
    
    Subclasses implement:
      - solve(): Main calculation
      - _build_hamiltonian(): H(T, σ, ...)
    """
    
    def __init__(self,
                 material: Union[MaterialParams, str] = None,
                 n_sites: int = 16,
                 use_gpu: bool = False,
                 verbose: bool = True):
        """
        Initialize engineering solver.
        
        Args:
            material: MaterialParams or material name string
            n_sites: Number of lattice sites
            use_gpu: Use GPU acceleration
            verbose: Print progress
        """
        # Material
        if material is None:
            self.material = MaterialParams()
        elif isinstance(material, str):
            self.material = MaterialParams(name=material)
        else:
            self.material = material
        
        self.n_sites = n_sites
        self.verbose = verbose
        
        # Engine
        try:
            self.engine = SparseEngine(n_sites, use_gpu=use_gpu, verbose=False)
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
            print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def solve(self, conditions: ProcessConditions, **kwargs) -> SolverResult:
        """
        Main solver method.
        
        Args:
            conditions: Process conditions (T, σ, etc.)
            **kwargs: Solver-specific options
            
        Returns:
            SolverResult with calculation results
        """
        pass
    
    @abstractmethod
    def _build_hamiltonian(self, T: float, sigma: float) -> Tuple:
        """
        Build Hamiltonian for given conditions.
        
        Args:
            T: Temperature (K)
            sigma: Stress (arb)
            
        Returns:
            (H_K, H_V): Kinetic and potential parts
        """
        pass
    
    # -------------------------------------------------------------------------
    # Common Methods
    # -------------------------------------------------------------------------
    
    def compute_lambda(self, psi: np.ndarray = None) -> float:
        """
        Compute global stability parameter λ = K/|V|.
        
        Args:
            psi: Wavefunction (uses self.psi if None)
            
        Returns:
            λ value
        """
        if psi is None:
            psi = self.psi
        if psi is None or self.H_K is None or self.H_V is None:
            return 0.0
        
        K = float(np.real(np.vdot(psi, self.H_K @ psi)))
        V = float(np.real(np.vdot(psi, self.H_V @ psi)))
        
        return abs(K / V) if abs(V) > 1e-10 else 1.0
    
    def compute_lambda_local(self, psi: np.ndarray = None) -> np.ndarray:
        """
        Compute local λ at each site.
        
        Args:
            psi: Wavefunction
            
        Returns:
            Array of λ values per site
        """
        if self.engine is None or self.geometry is None:
            return np.zeros(self.n_sites)
        
        if psi is None:
            psi = self.psi
        
        return self.engine.compute_local_lambda(
            psi, self.H_K, self.H_V, self.geometry
        )
    
    def check_failure(self, T: float = 300.0) -> Tuple[bool, Optional[int]]:
        """
        Check if system has failed (λ > λ_critical).
        
        Args:
            T: Current temperature (affects λ_critical)
            
        Returns:
            (failed, failure_site): Failure status and location
        """
        lambda_c = self.material.lambda_critical_T(T)
        lambda_local = self.compute_lambda_local()
        
        # Find max λ
        max_site = int(np.argmax(lambda_local))
        max_lambda = lambda_local[max_site]
        
        if max_lambda > lambda_c:
            return True, max_site
        return False, None
    
    def compute_ground_state(self) -> Tuple[float, np.ndarray]:
        """Compute ground state of current Hamiltonian"""
        from scipy.sparse.linalg import eigsh
        
        H = self.H_K + self.H_V
        try:
            E0, psi0 = eigsh(H, k=1, which='SA')
            self.psi = psi0[:, 0]
            self.psi = self.psi / np.linalg.norm(self.psi)
            return E0[0], self.psi
        except Exception:
            dim = H.shape[0]
            self.psi = np.random.randn(dim) + 1j * np.random.randn(dim)
            self.psi = self.psi / np.linalg.norm(self.psi)
            return 0.0, self.psi
    
    def evolve_step(self, dt: float = 0.1) -> np.ndarray:
        """
        Single time evolution step.
        
        Args:
            dt: Time step
            
        Returns:
            Updated wavefunction
        """
        from scipy.sparse.linalg import expm_multiply
        
        H = self.H_K + self.H_V
        self.psi = expm_multiply(-1j * dt * H, self.psi)
        self.psi = self.psi / np.linalg.norm(self.psi)
        
        return self.psi
    
    def clear_history(self):
        """Clear accumulated history"""
        self.lambda_history = []
        self.energy_history = []
        self.T_history = []
        self.sigma_history = []
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated history"""
        return {
            'n_steps': len(self.lambda_history),
            'lambda_initial': self.lambda_history[0] if self.lambda_history else None,
            'lambda_final': self.lambda_history[-1] if self.lambda_history else None,
            'lambda_max': max(self.lambda_history) if self.lambda_history else None,
            'energy_change': (self.energy_history[-1] - self.energy_history[0]) 
                             if len(self.energy_history) > 1 else 0.0,
            'T_range': (min(self.T_history), max(self.T_history)) 
                       if self.T_history else (0, 0),
            'sigma_range': (min(self.sigma_history), max(self.sigma_history))
                           if self.sigma_history else (0, 0),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_material(name: str) -> MaterialParams:
    """
    Create MaterialParams for common materials.
    
    Supported: Fe, Al, Cu, Ti
    """
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
