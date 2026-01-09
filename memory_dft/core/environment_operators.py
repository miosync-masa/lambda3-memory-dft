"""
Environment Operators for Memory-DFT
====================================

H-CSP理論に基づく環境作用素 B_θ の実装

【設計思想 v2.0】
  
  ❌ 旧設計: H(T) を作る（経験則）
  ✅ 新設計: H は固定、温度は分布にだけ入る

  温度は「ハミルトニアンを変える」のではなく
  「固有状態の重み分布を変える」

  <O>(T) = Σ_n w_n(T) × <n|O|n>
  w_n(T) = exp(-βE_n) / Z

【H-CSP公理との対応】
  公理2（非可換）: 経路依存性は MemoryKernel で扱う
  公理5（環境作用）: 応力などの「場」はハミルトニアンに入れてよい
                    温度は分布に入る（環境関手ではない）

Author: Masamichi Iizumi, Tamaki Iizumi
Version: 2.0 - ThermalEnsemble導入、TemperatureOperator非推奨
"""

from __future__ import annotations

import numpy as np
import warnings
from typing import List, Tuple, Optional, Dict, Any, Union, TYPE_CHECKING, Callable
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
from scipy.sparse.linalg import eigsh

# Type hints
if TYPE_CHECKING:
    from memory_dft.core.sparse_engine_unified import (
        SparseEngine,
        SystemGeometry,
        HubbardAndersonGeometry,
        HubbardAndersonParams,
        LayerEnergies,
        LayerLambda,
    )

# =============================================================================
# Physical Constants
# =============================================================================

K_B_EV = 8.617333262e-5   # Boltzmann constant in eV/K
K_B_J = 1.380649e-23      # Boltzmann constant in J/K
H_EV = 4.135667696e-15    # Planck constant in eV·s
HBAR_EV = 6.582119569e-16 # Reduced Planck constant in eV·s


# =============================================================================
# Basic Thermodynamic Utilities (これらは正しい)
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
    """
    Compute Boltzmann weights exp(-β E_n) / Z.
    
    これが「温度」の正しい入れ方！
    ハミルトニアンではなく、固有状態の重みに入る。
    """
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


def compute_heat_capacity(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute heat capacity C_V = β² Var(E) in units of k_B."""
    if beta == float('inf'):
        return 0.0
    
    weights = boltzmann_weights(eigenvalues, beta)
    E_avg = np.sum(eigenvalues * weights)
    E2_avg = np.sum(eigenvalues**2 * weights)
    var_E = E2_avg - E_avg**2
    
    return beta**2 * var_E


def compute_free_energy(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute Helmholtz free energy F = -k_B T ln(Z)."""
    if beta == float('inf'):
        return eigenvalues[0]
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    Z = np.sum(np.exp(-beta * E_shifted))
    return E_min - np.log(Z) / beta


# =============================================================================
# ThermalEnsemble: 正しい有限温度計算 (v2.0 NEW!)
# =============================================================================

@dataclass
class ThermalObservable:
    """熱平均の結果を格納"""
    T: float
    beta: float
    value: float
    variance: float = 0.0
    n_states: int = 0
    
    
class ThermalEnsemble:
    """
    正しい有限温度計算クラス
    
    【設計思想】
    温度はハミルトニアンに入れない！
    温度は固有状態の重み分布にだけ入る！
    
    <O>(T) = Σ_n w_n(T) × <n|O|n>
    w_n(T) = exp(-βE_n) / Z
    
    Usage:
        # Step 1: ハミルトニアンは固定！
        H_K, H_V = engine.build_hubbard(bonds, t=1.0, U=2.0)
        H = H_K + H_V
        
        # Step 2: ThermalEnsemble を作る（固有状態を求める）
        ensemble = ThermalEnsemble(engine, H)
        
        # Step 3: 位相指標を登録
        ensemble.register_observable('Q', compute_winding)
        ensemble.register_observable('S_phase', compute_phase_entropy)
        
        # Step 4: 温度スキャン（ハミルトニアンは変わらない！）
        for T in T_values:
            Q_thermal = ensemble.thermal_average('Q', T)
            # これが「熱的振る舞い」の正しい計算
    """
    
    def __init__(self, 
                 engine: 'SparseEngine',
                 H: Any,
                 n_eigenstates: int = None,
                 compute_all: bool = False):
        """
        Initialize thermal ensemble.
        
        Args:
            engine: SparseEngine instance
            H: Hamiltonian (FIXED! 温度で変えない！)
            n_eigenstates: Number of eigenstates to compute (None = all)
            compute_all: If True, compute all eigenstates (exact)
        """
        self.engine = engine
        self.H = H
        self.dim = engine.dim
        self.use_gpu = engine.use_gpu
        self.xp = engine.xp
        
        # 固有状態を求める（一度だけ！）
        self._compute_eigenstates(n_eigenstates, compute_all)
        
        # Observable registry
        self._observables: Dict[str, np.ndarray] = {}
        self._observable_funcs: Dict[str, Callable] = {}
        
        # Cache for computed observables
        self._obs_cache: Dict[str, np.ndarray] = {}
    
    def _compute_eigenstates(self, n_eigenstates: int, compute_all: bool):
        """固有状態を計算（一度だけ！）"""
        
        if compute_all or self.dim <= 1024:
            # 厳密対角化
            if self.use_gpu:
                H_np = self.H.toarray().get() if hasattr(self.H, 'toarray') else self.H.get()
            else:
                H_np = self.H.toarray() if hasattr(self.H, 'toarray') else self.H
            
            E, V = np.linalg.eigh(H_np)
            self.eigenvalues = E
            self.eigenvectors = V
            self.n_eigenstates = len(E)
            
        else:
            # Lanczos法で低エネルギー状態のみ
            k = n_eigenstates or min(100, self.dim - 2)
            
            if self.use_gpu:
                H_cpu = self.H.get() if hasattr(self.H, 'get') else self.H
            else:
                H_cpu = self.H
            
            E, V = eigsh(H_cpu, k=k, which='SA')
            idx = np.argsort(E)
            self.eigenvalues = E[idx]
            self.eigenvectors = V[:, idx]
            self.n_eigenstates = k
        
        print(f"  ThermalEnsemble: {self.n_eigenstates} eigenstates computed")
        print(f"  E_0 = {self.eigenvalues[0]:.6f}")
        print(f"  E_1 = {self.eigenvalues[1]:.6f}")
        print(f"  Gap = {self.eigenvalues[1] - self.eigenvalues[0]:.6f}")
    
    def register_observable(self, name: str, func: Callable[[np.ndarray], float]):
        """
        Register an observable function.
        
        The function takes a wavefunction and returns a scalar.
        
        Args:
            name: Observable name (e.g., 'Q', 'S_phase', 'vorticity')
            func: Function psi -> float
        """
        self._observable_funcs[name] = func
        
        # Compute for all eigenstates
        values = np.zeros(self.n_eigenstates)
        for n in range(self.n_eigenstates):
            psi = self.eigenvectors[:, n]
            values[n] = func(psi)
        
        self._obs_cache[name] = values
        print(f"  Registered observable '{name}': computed for {self.n_eigenstates} states")
    
    def thermal_average(self, observable: str, T: float) -> ThermalObservable:
        """
        Compute thermal average of observable at temperature T.
        
        <O>(T) = Σ_n w_n(T) × O_n
        
        Args:
            observable: Name of registered observable
            T: Temperature (K)
            
        Returns:
            ThermalObservable with value and variance
        """
        if observable not in self._obs_cache:
            raise ValueError(f"Observable '{observable}' not registered. "
                           f"Available: {list(self._obs_cache.keys())}")
        
        beta = T_to_beta(T)
        weights = boltzmann_weights(self.eigenvalues, beta)
        
        O_values = self._obs_cache[observable]
        
        # Thermal average
        O_avg = np.sum(weights * O_values)
        
        # Variance
        O2_avg = np.sum(weights * O_values**2)
        variance = O2_avg - O_avg**2
        
        return ThermalObservable(
            T=T,
            beta=beta,
            value=O_avg,
            variance=variance,
            n_states=self.n_eigenstates
        )
    
    def thermal_average_operator(self, O: Any, T: float) -> float:
        """
        Compute thermal average of an operator.
        
        <O>(T) = Σ_n w_n(T) × <n|O|n>
        
        Args:
            O: Operator (sparse matrix)
            T: Temperature (K)
            
        Returns:
            float: Thermal average
        """
        beta = T_to_beta(T)
        weights = boltzmann_weights(self.eigenvalues, beta)
        
        result = 0.0
        for n in range(self.n_eigenstates):
            if weights[n] < 1e-15:
                continue
            psi = self.eigenvectors[:, n]
            O_nn = float(np.real(np.vdot(psi, O @ psi)))
            result += weights[n] * O_nn
        
        return result
    
    def temperature_scan(self, 
                         observable: str,
                         T_range: Tuple[float, float] = (10, 1000),
                         n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Scan temperature and compute thermal averages.
        
        Args:
            observable: Name of registered observable
            T_range: (T_min, T_max) in Kelvin
            n_points: Number of temperature points
            
        Returns:
            Dict with 'T', 'value', 'variance', 'd_value_dT'
        """
        T_values = np.linspace(T_range[0], T_range[1], n_points)
        values = np.zeros(n_points)
        variances = np.zeros(n_points)
        
        for i, T in enumerate(T_values):
            result = self.thermal_average(observable, T)
            values[i] = result.value
            variances[i] = result.variance
        
        # Compute derivative (for phase transition detection)
        d_value_dT = np.gradient(values, T_values)
        
        return {
            'T': T_values,
            'value': values,
            'variance': variances,
            'd_value_dT': d_value_dT
        }
    
    def detect_phase_transition(self,
                                 observable: str,
                                 T_range: Tuple[float, float] = (10, 1000),
                                 n_points: int = 100) -> Dict[str, Any]:
        """
        Detect phase transition from thermal average of observable.
        
        Phase transition = 急変点 = d<O>/dT の極値
        
        Args:
            observable: Name of registered observable
            T_range: Temperature range
            n_points: Number of points
            
        Returns:
            Dict with 'T_transition', 'type', 'scan_data'
        """
        scan = self.temperature_scan(observable, T_range, n_points)
        
        # Find maximum of |d<O>/dT|
        abs_deriv = np.abs(scan['d_value_dT'])
        idx_max = np.argmax(abs_deriv)
        T_transition = scan['T'][idx_max]
        
        # Determine transition type
        deriv_at_max = scan['d_value_dT'][idx_max]
        if deriv_at_max > 0:
            transition_type = 'increasing'
        else:
            transition_type = 'decreasing'
        
        return {
            'T_transition': T_transition,
            'type': transition_type,
            'max_derivative': abs_deriv[idx_max],
            'scan_data': scan
        }

    def get_thermal_state(self, T: float) -> np.ndarray:
        """
        温度 T での熱的混合状態を返す
        
        |ψ_thermal⟩ = Σ √w_n(T) |n⟩
        
        Args:
            T: 温度 (K)
            
        Returns:
            正規化された状態ベクトル
        """
        beta = T_to_beta(T)
        weights = boltzmann_weights(self.eigenvalues, beta)
        
        psi_thermal = np.zeros(self.dim, dtype=np.complex128)
        for n in range(self.n_eigenstates):
            if weights[n] > 1e-15:
                psi_thermal += np.sqrt(weights[n]) * self.eigenvectors[:, n]
        
        norm = np.linalg.norm(psi_thermal)
        if norm > 1e-10:
            psi_thermal /= norm
        
        return psi_thermal
    
    # =========================================================================
    # Thermodynamic quantities (computed correctly from ensemble)
    # =========================================================================
    
    def free_energy(self, T: float) -> float:
        """Helmholtz free energy F(T)."""
        return compute_free_energy(self.eigenvalues, T_to_beta(T))
    
    def entropy(self, T: float) -> float:
        """Entropy S(T)/k_B."""
        return compute_entropy(self.eigenvalues, T_to_beta(T))
    
    def heat_capacity(self, T: float) -> float:
        """Heat capacity C_V(T)/k_B."""
        return compute_heat_capacity(self.eigenvalues, T_to_beta(T))
    
    def internal_energy(self, T: float) -> float:
        """Internal energy <E>(T)."""
        beta = T_to_beta(T)
        weights = boltzmann_weights(self.eigenvalues, beta)
        return np.sum(weights * self.eigenvalues)


# =============================================================================
# Winding / Phase Observables (for ThermalEnsemble)
# =============================================================================

def compute_winding_number(psi: np.ndarray) -> float:
    """
    Compute winding number from wavefunction phase.
    
    Q = (1/2π) Σ Δθ_i
    """
    theta = np.angle(psi)
    n = len(theta)
    total_phase = 0.0
    
    for i in range(n - 1):
        dtheta = theta[i + 1] - theta[i]
        # Wrap to [-π, π]
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta <= -np.pi:
            dtheta += 2 * np.pi
        total_phase += dtheta
    
    return total_phase / (2 * np.pi)


def compute_phase_entropy(psi: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute entropy of phase distribution.
    
    S = -Σ p_i log(p_i)
    """
    theta = np.angle(psi)
    hist, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist / hist.sum()
    mask = hist > 0
    return -np.sum(hist[mask] * np.log(hist[mask]))


def compute_vorticity(psi: np.ndarray, H_K: Any, H_V: Any) -> float:
    """
    Compute vorticity V = |K| + |V|.
    
    This is the "knot complexity" measure.
    """
    K = float(np.real(np.vdot(psi, H_K @ psi)))
    V = float(np.real(np.vdot(psi, H_V @ psi)))
    return abs(K) + abs(V)


# =============================================================================
# Dislocation (これは場なのでOK)
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


# =============================================================================
# StressOperator (これは場なのでOK - ハミルトニアンに入れてよい)
# =============================================================================

class StressOperator:
    """
    Stress-dependent Hamiltonian modification.
    
    応力は「場」なのでハミルトニアンに入れてよい。
    （温度とは違う！）
    
    K_mech = σ²/(2E)
    """
    
    def __init__(self, engine: 'SparseEngine', Lx: int = None, Ly: int = None):
        self.engine = engine
        self.n_sites = engine.n_sites
        self.dim = engine.dim
        self.use_gpu = engine.use_gpu
        self.xp = engine.xp
        self.Lx = Lx or int(np.sqrt(engine.n_sites))
        self.Ly = Ly or engine.n_sites // self.Lx
    
    def build_stress_hamiltonian(self, sigma: float) -> Any:
        """Build stress gradient Hamiltonian."""
        xp = self.xp
        dim = self.dim
        n = self.n_sites
        
        diag = xp.zeros(dim, dtype=xp.float64)
        
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
    
    def apply(self, H_K, H_V, sigma: float = 0.0) -> Tuple:
        """Apply stress to Hamiltonian."""
        if abs(sigma) < 1e-10:
            return H_K, H_V
        
        H_stress = self.build_stress_hamiltonian(sigma)
        return H_K, H_V + H_stress


# =============================================================================
# DEPRECATED: TemperatureOperator (後方互換性のため残す)
# =============================================================================

class TemperatureOperator:
    """
    ⚠️ DEPRECATED: Do not use!
    
    This class modifies the Hamiltonian based on temperature,
    which is physically incorrect.
    
    Use ThermalEnsemble instead:
        ensemble = ThermalEnsemble(engine, H)
        <O>(T) = ensemble.thermal_average('O', T)
    
    温度はハミルトニアンを変えるものではない！
    温度は固有状態の重み分布を変えるもの！
    """
    
    def __init__(self, engine: 'SparseEngine', **kwargs):
        warnings.warn(
            "TemperatureOperator is DEPRECATED and physically incorrect!\n"
            "Temperature should NOT modify the Hamiltonian.\n"
            "Use ThermalEnsemble instead:\n"
            "  ensemble = ThermalEnsemble(engine, H)\n"
            "  result = ensemble.thermal_average('observable', T)",
            DeprecationWarning,
            stacklevel=2
        )
        self.engine = engine
        self.t0 = kwargs.get('t0', 1.0)
        self.U0 = kwargs.get('U0', 2.0)
        self.alpha_t = kwargs.get('alpha_t', 1e-4)
        self.T_ref = kwargs.get('T_ref', 300.0)
    
    def apply(self, *args, **kwargs):
        raise NotImplementedError(
            "TemperatureOperator.apply() is disabled.\n"
            "Use ThermalEnsemble for correct finite-temperature calculations."
        )


# =============================================================================
# EnvironmentBuilder (v2.0: 温度は分布で処理)
# =============================================================================

class EnvironmentBuilder:
    """
    Environment builder (v2.0).
    
    【v2.0 変更点】
    - 温度は ThermalEnsemble で処理
    - ハミルトニアンに入れるのは「場」のみ（応力、電磁場など）
    
    Usage:
        # Step 1: 基本ハミルトニアン
        H_K, H_V = engine.build_hubbard(bonds, t=1.0, U=2.0)
        
        # Step 2: 場の効果を追加（応力など）
        builder = EnvironmentBuilder(engine)
        H_K, H_V = builder.apply_stress(H_K, H_V, sigma=2.0)
        
        # Step 3: 有限温度は ThermalEnsemble で
        ensemble = ThermalEnsemble(engine, H_K + H_V)
        ensemble.register_observable('Q', compute_winding_number)
        result = ensemble.thermal_average('Q', T=500)
    """
    
    def __init__(self, engine: 'SparseEngine', Lx: int = None, Ly: int = None):
        self.engine = engine
        self.stress_op = StressOperator(engine, Lx=Lx, Ly=Ly)
        self.dislocations: List[Dislocation] = []
    
    def apply_stress(self, H_K, H_V, sigma: float) -> Tuple:
        """Apply stress field to Hamiltonian."""
        return self.stress_op.apply(H_K, H_V, sigma)
    
    def add_dislocation(self, site: int, 
                        burgers: Tuple[float, float, float] = (1, 0, 0)) -> Dislocation:
        """Add dislocation at site."""
        disl = Dislocation(site=site, burgers=burgers)
        self.dislocations.append(disl)
        return disl
    
    def clear_dislocations(self):
        """Remove all dislocations."""
        self.dislocations = []


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Environment Operators v2.0 Test")
    print("=" * 70)
    
    # Test thermodynamic utilities
    print("\n--- Thermodynamic Utilities ---")
    print(f"T=300K → β = {T_to_beta(300):.2f}")
    print(f"k_B T(300K) = {thermal_energy(300):.4f} eV")
    
    # Test ThermalEnsemble
    try:
        from memory_dft.core.sparse_engine_unified import SparseEngine
        
        print("\n--- ThermalEnsemble Test ---")
        engine = SparseEngine(n_sites=4, use_gpu=False, verbose=False)
        geometry = engine.build_chain(periodic=False)
        
        # Build Hamiltonian (FIXED! 温度で変えない！)
        H_K, H_V = engine.build_heisenberg(geometry.bonds, J=1.0)
        H = H_K + H_V
        
        # Create ensemble
        ensemble = ThermalEnsemble(engine, H, compute_all=True)
        
        # Register observables
        ensemble.register_observable('Q', compute_winding_number)
        ensemble.register_observable('S_phase', compute_phase_entropy)
        
        # Temperature scan
        print("\n--- Temperature Scan ---")
        for T in [10, 100, 300, 500, 1000]:
            Q = ensemble.thermal_average('Q', T)
            S = ensemble.thermal_average('S_phase', T)
            F = ensemble.free_energy(T)
            print(f"  T={T:4d}K: <Q>={Q.value:.4f}, <S_phase>={S.value:.4f}, F={F:.4f}")
        
        # Phase transition detection
        print("\n--- Phase Transition Detection ---")
        result = ensemble.detect_phase_transition('S_phase', T_range=(10, 1000))
        print(f"  T_transition = {result['T_transition']:.1f} K")
        print(f"  Type: {result['type']}")
        
        # Test deprecated warning
        print("\n--- Deprecated Warning Test ---")
        try:
            old_op = TemperatureOperator(engine)
        except Exception as e:
            print(f"  Caught: {type(e).__name__}")
        
        print("\n" + "=" * 70)
        print("✅ Environment Operators v2.0 Test Complete!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n⚠️ SparseEngine not available: {e}")
