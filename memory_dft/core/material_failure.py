#!/usr/bin/env python3
"""
Topology Module for Memory-DFT (Extended with Thermal & Stress Tests)
======================================================================

DSE (Direct Schrödinger Evolution) での位相トポロジー解析

【新機能】
  1. ThermalTopologyAnalyzer - 熱による位相変化（融解）
  2. StressTopologyAnalyzer - 応力による位相変化（破断）
  3. DSETopologyTest - 統合テストスイート

【物理的背景】
  熱（温度 T）:
    - 分布を変える（weights = exp(-βE)/Z）
    - Coherence↓ → 結び目が緩む → 融解
    - Z が低い場所（表面）から壊れる
    
  応力（σ）:
    - ハミルトニアンを変える（H → H + σ×位置項）
    - Δθ（位相差）が大きくなる → 結び目が切れる
    - 粒界（Δθ大）から壊れる

【破壊条件】
  λ(r) = K(r) / |V_eff(r)| → 1 で reconnection（破壊）
  V_eff(r) = V_0(r) + σ(r)
  
  熱: V_0(r) が下がる（Coherence低下）
  応力: σ(r) が V_eff を下げる

Author: Tamaki & Masamichi Iizumi
Date: 2025-01
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass, field
import warnings

# CuPy support (GPU)
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    HAS_CUPY = True
except ImportError:
    cp = np
    csp = None
    cp_eigsh = None
    HAS_CUPY = False
    warnings.warn("CuPy not available, falling back to NumPy")

# SciPy (CPU fallback)
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh as sp_eigsh

# Environment operators (ThermalEnsemble, etc.)
try:
    from .environment_operators import (
        ThermalEnsemble,
        ThermalObservable,
        boltzmann_weights,
        T_to_beta,
        K_B_EV,
        compute_winding_number,
        compute_phase_entropy,
    )
    HAS_ENV_OPS = True
except ImportError:
    HAS_ENV_OPS = False
    # Fallback constants
    K_B_EV = 8.617333262e-5  # eV/K


# =============================================================================
# Constants
# =============================================================================

k_B = K_B_EV  # Alias for compatibility


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TopologyResult:
    """Container for topological invariants."""
    Q_Lambda: float = 0.0           # Spin topological charge
    berry_phase: float = 0.0        # Berry phase (mod 2π)
    winding_number: int = 0         # Integer winding number
    zak_phase: float = 0.0          # Zak phase (0 or π)
    coherence: float = 1.0          # Phase coherence (0-1)
    phase_entropy: float = 0.0      # Phase distribution entropy
    
    # Per-site data
    site_phases: Optional[np.ndarray] = None
    local_lambda: Optional[np.ndarray] = None
    
    def is_topological(self, threshold: float = 0.1) -> bool:
        """Check if system has non-trivial topology."""
        return (abs(self.winding_number) >= 1 or 
                abs(self.Q_Lambda) > threshold)


@dataclass
class ThermalTopologyResult:
    """Result of thermal topology analysis."""
    T: float                          # Temperature
    coherence: float                  # Phase coherence at T
    Z_eff: float                      # Effective coordination number
    phase_entropy: float              # Phase distribution entropy
    lindemann_delta: float            # Lindemann parameter δ
    is_melted: bool                   # Whether system has melted
    
    # Detailed data
    eigenvalues: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None


@dataclass
class StressTopologyResult:
    """Result of stress topology analysis."""
    sigma: float                      # Applied stress
    max_lambda: float                 # Maximum local λ
    failure_site: int                 # Predicted failure site
    phase_gradient: Optional[np.ndarray] = None
    local_lambda: Optional[np.ndarray] = None
    is_failed: bool = False           # Whether λ > 1 anywhere


@dataclass
class FailurePrediction:
    """Prediction of material failure."""
    mode: str                         # 'thermal' or 'mechanical' or 'combined'
    failure_site: int                 # Where failure occurs
    critical_value: float             # T_m or σ_c
    lambda_at_failure: float          # λ value at failure
    mechanism: str                    # 'melting', 'fracture', 'creep'


# =============================================================================
# Utility Functions (CuPy compatible)
# =============================================================================

def get_xp(use_gpu: bool = True):
    """Get appropriate array module (CuPy or NumPy)."""
    if use_gpu and HAS_CUPY:
        return cp
    return np


def to_device(arr, use_gpu: bool = True):
    """Move array to GPU if available."""
    xp = get_xp(use_gpu)
    if use_gpu and HAS_CUPY:
        if not hasattr(arr, 'device'):
            return cp.asarray(arr)
    return arr


def to_host(arr):
    """Move array to CPU."""
    if hasattr(arr, 'get'):
        return arr.get()
    return arr


def eigsh_wrapper(H, k: int = 1, which: str = 'SA', use_gpu: bool = True):
    """Wrapper for eigenvalue solver (GPU/CPU)."""
    if use_gpu and HAS_CUPY and cp_eigsh is not None:
        try:
            if not hasattr(H, 'device'):
                H = csp.csr_matrix(H) if sp.issparse(H) else cp.asarray(H)
            E, V = cp_eigsh(H, k=k, which=which)
            return to_host(E), to_host(V)
        except Exception:
            pass
    
    # CPU fallback
    if hasattr(H, 'get'):
        H = H.get()
    if not sp.issparse(H):
        H = sp.csr_matrix(H)
    return sp_eigsh(H, k=k, which=which)


# =============================================================================
# Thermal Ensemble Helpers (for compatibility with environment_operators.py)
# =============================================================================

def _get_weights(eigenvalues: np.ndarray, T: float) -> np.ndarray:
    """Get Boltzmann weights at temperature T."""
    if HAS_ENV_OPS:
        return boltzmann_weights(eigenvalues, T_to_beta(T))
    else:
        # Fallback implementation
        if T <= 0:
            weights = np.zeros(len(eigenvalues))
            weights[0] = 1.0
            return weights
        beta = 1.0 / (k_B * T)
        E_shifted = eigenvalues - eigenvalues[0]
        weights = np.exp(-beta * E_shifted)
        return weights / weights.sum()


def _thermal_avg_value(result) -> float:
    """Extract float value from thermal average result."""
    if hasattr(result, 'value'):
        return result.value
    return float(result)


# =============================================================================
# Thermal Topology Analyzer
# =============================================================================

class ThermalTopologyAnalyzer:
    """
    熱による位相トポロジー変化の解析
    
    【物理】
      T↑ → 位相の揺らぎ δθ↑ → Coherence↓ → 結び目が緩む → 融解
      
    【融解条件】
      Lindemann: δ/a → δ_L ≈ 0.1 で融解
      位相的: Coherence → C_c で融解
    """
    
    def __init__(self, ensemble, use_gpu: bool = True):
        """
        Args:
            ensemble: ThermalEnsemble or LocalThermalEnsemble with pre-computed eigenstates
            use_gpu: Use GPU acceleration
        """
        self.ensemble = ensemble
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = get_xp(self.use_gpu)
        
        # Register topology observables
        self._register_topology_observables()
    
    def _register_topology_observables(self):
        """Register phase topology observables."""
        # Phase entropy
        def phase_entropy(psi):
            theta = np.angle(psi)
            hist, _ = np.histogram(theta, bins=20, range=(-np.pi, np.pi))
            p = hist / (hist.sum() + 1e-10)
            return -np.sum(p[p > 0] * np.log(p[p > 0]))
        
        # Winding number
        def winding_number(psi):
            theta = np.angle(psi)
            dtheta = np.diff(theta)
            # Wrap to [-π, π]
            dtheta = ((dtheta + np.pi) % (2 * np.pi)) - np.pi
            return np.sum(dtheta) / (2 * np.pi)
        
        # Phase variance (related to Lindemann)
        def phase_variance(psi):
            theta = np.angle(psi)
            return np.var(theta)
        
        self.ensemble.register_observable('phase_entropy', phase_entropy)
        self.ensemble.register_observable('winding', winding_number)
        self.ensemble.register_observable('phase_variance', phase_variance)
    
    def compute_coherence(self, T: float) -> float:
        """
        Compute phase coherence at temperature T.
        
        Coherence = |Σ exp(iθ_n) × w_n|
        
        High coherence: phases aligned → solid
        Low coherence: phases random → liquid
        """
        weights = _get_weights(self.ensemble.eigenvalues, T)
        
        # Phase factor from each eigenstate
        phase_sum = 0.0 + 0.0j
        for n in range(self.ensemble.n_eigenstates):
            psi = self.ensemble.eigenvectors[:, n]
            # Average phase of wavefunction
            avg_phase = np.angle(np.sum(psi))
            phase_sum += weights[n] * np.exp(1j * avg_phase)
        
        return float(abs(phase_sum))
    
    def compute_lindemann_parameter(self, T: float, a: float = 1.0) -> float:
        """
        Compute Lindemann parameter δ = sqrt(<u²>) / a.
        
        Uses phase variance as proxy for displacement.
        
        Args:
            T: Temperature
            a: Lattice constant (for normalization)
        """
        # Phase variance ∝ displacement variance
        phase_var = _thermal_avg_value(self.ensemble.thermal_average('phase_variance', T))
        
        # δ ≈ sqrt(phase_var) / π (normalized)
        delta = np.sqrt(phase_var) / np.pi
        
        return float(delta)
    
    def compute_effective_Z(self, T: float, Z_bulk: float = 12.0) -> float:
        """
        Compute effective coordination number at T.
        
        Z_eff = Z_bulk × Coherence
        
        As T↑, Coherence↓, Z_eff↓ (bonds effectively break)
        """
        coherence = self.compute_coherence(T)
        return Z_bulk * coherence
    
    def analyze_temperature(self, T: float, 
                            lindemann_critical: float = 0.1) -> ThermalTopologyResult:
        """
        Full thermal analysis at temperature T.
        """
        coherence = self.compute_coherence(T)
        Z_eff = self.compute_effective_Z(T)
        phase_entropy = _thermal_avg_value(self.ensemble.thermal_average('phase_entropy', T))
        delta = self.compute_lindemann_parameter(T)
        is_melted = delta > lindemann_critical
        
        return ThermalTopologyResult(
            T=T,
            coherence=coherence,
            Z_eff=Z_eff,
            phase_entropy=phase_entropy,
            lindemann_delta=delta,
            is_melted=is_melted,
            eigenvalues=self.ensemble.eigenvalues,
            weights=_get_weights(self.ensemble.eigenvalues, T)
        )
    
    def temperature_scan(self, T_range: np.ndarray,
                         lindemann_critical: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Scan temperature range and detect melting.
        
        Returns dict with T, coherence, Z_eff, delta, etc.
        """
        results = {
            'T': T_range,
            'coherence': np.zeros_like(T_range),
            'Z_eff': np.zeros_like(T_range),
            'phase_entropy': np.zeros_like(T_range),
            'lindemann_delta': np.zeros_like(T_range),
            'is_melted': np.zeros_like(T_range, dtype=bool)
        }
        
        for i, T in enumerate(T_range):
            res = self.analyze_temperature(T, lindemann_critical)
            results['coherence'][i] = res.coherence
            results['Z_eff'][i] = res.Z_eff
            results['phase_entropy'][i] = res.phase_entropy
            results['lindemann_delta'][i] = res.lindemann_delta
            results['is_melted'][i] = res.is_melted
        
        return results
    
    def detect_melting_point(self, T_range: np.ndarray,
                             lindemann_critical: float = 0.1) -> float:
        """
        Detect melting temperature T_m.
        
        T_m is where Lindemann parameter crosses critical value.
        """
        scan = self.temperature_scan(T_range, lindemann_critical)
        
        # Find first crossing
        for i, (T, is_melted) in enumerate(zip(T_range, scan['is_melted'])):
            if is_melted:
                if i > 0:
                    # Interpolate
                    delta_prev = scan['lindemann_delta'][i-1]
                    delta_curr = scan['lindemann_delta'][i]
                    T_prev = T_range[i-1]
                    
                    # Linear interpolation
                    frac = (lindemann_critical - delta_prev) / (delta_curr - delta_prev + 1e-10)
                    T_m = T_prev + frac * (T - T_prev)
                    return float(T_m)
                return float(T)
        
        # No melting detected
        return float('inf')


# =============================================================================
# Stress Topology Analyzer
# =============================================================================

class StressTopologyAnalyzer:
    """
    応力による位相トポロジー変化の解析
    
    【物理】
      応力 σ → H_V を変える → 位相勾配 ∇θ↑ → 粒界で λ→1 → 破断
      
    【破壊条件】
      λ(r) = K(r) / |V_eff(r)| → 1 で reconnection
      V_eff(r) = V_0(r) + σ(r)
    """
    
    def __init__(self, n_sites: int, use_gpu: bool = True):
        """
        Args:
            n_sites: Number of lattice sites
            use_gpu: Use GPU acceleration
        """
        self.n_sites = n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = get_xp(self.use_gpu)
    
    def compute_phase_gradient(self, psi: np.ndarray, 
                                bonds: List[Tuple[int, int]]) -> np.ndarray:
        """
        Compute phase gradient Δθ along bonds.
        
        Large Δθ = phase mismatch = grain boundary = weak point
        """
        xp = self.xp
        psi = to_device(psi, self.use_gpu)
        
        theta = xp.angle(psi[:self.n_sites] if len(psi) > self.n_sites else psi)
        theta = to_host(theta)
        
        gradients = np.zeros(len(bonds))
        for idx, (i, j) in enumerate(bonds):
            if i < len(theta) and j < len(theta):
                dtheta = theta[j] - theta[i]
                # Wrap to [-π, π]
                dtheta = ((dtheta + np.pi) % (2 * np.pi)) - np.pi
                gradients[idx] = abs(dtheta)
        
        return gradients
    
    def compute_local_K(self, psi: np.ndarray, H_K,
                        site: int) -> float:
        """
        Compute local kinetic energy at site.
        
        K(r) = contribution of site r to total K
        """
        xp = self.xp
        psi = to_device(psi, self.use_gpu)
        H_K = to_device(H_K, self.use_gpu)
        
        # Full K for now (local decomposition is complex)
        K_psi = H_K @ psi
        K_total = float(to_host(xp.real(xp.vdot(psi, K_psi))))
        
        # Approximate: equal distribution
        return K_total / self.n_sites
    
    def compute_local_V(self, psi: np.ndarray, H_V,
                        site: int) -> float:
        """
        Compute local potential energy at site.
        """
        xp = self.xp
        psi = to_device(psi, self.use_gpu)
        H_V = to_device(H_V, self.use_gpu)
        
        V_psi = H_V @ psi
        V_total = float(to_host(xp.real(xp.vdot(psi, V_psi))))
        
        return V_total / self.n_sites
    
    def compute_local_lambda(self, psi: np.ndarray,
                              H_K, H_V,
                              stress_field: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute local λ(r) = K(r) / |V_eff(r)|.
        
        Args:
            psi: Wavefunction
            H_K: Kinetic Hamiltonian
            H_V: Potential Hamiltonian
            stress_field: Applied stress at each site
            
        Returns:
            Array of λ values per site
        """
        xp = self.xp
        lambda_local = np.zeros(self.n_sites)
        
        for site in range(self.n_sites):
            K_local = self.compute_local_K(psi, H_K, site)
            V_local = self.compute_local_V(psi, H_V, site)
            
            # Add stress contribution
            if stress_field is not None and site < len(stress_field):
                V_eff = V_local + stress_field[site]
            else:
                V_eff = V_local
            
            # λ = |K| / |V_eff|
            lambda_local[site] = abs(K_local) / (abs(V_eff) + 1e-10)
        
        return lambda_local
    
    def apply_stress(self, H_V, sigma: float,
                     stress_sites: Optional[List[int]] = None,
                     stress_direction: np.ndarray = None) -> Any:
        """
        Apply stress to potential Hamiltonian.
        
        H_V_new = H_V + σ × Σ_i |i⟩⟨i| (for stressed sites)
        
        Args:
            H_V: Original potential Hamiltonian
            sigma: Stress magnitude
            stress_sites: Sites where stress is applied (None = all)
            stress_direction: Direction vector (for anisotropic stress)
        """
        xp = self.xp
        H_V = to_device(H_V, self.use_gpu)
        
        dim = H_V.shape[0]
        
        if stress_sites is None:
            # Uniform stress
            stress_matrix = sigma * xp.eye(dim, dtype=H_V.dtype)
        else:
            # Localized stress
            stress_matrix = xp.zeros((dim, dim), dtype=H_V.dtype)
            for site in stress_sites:
                if site < dim:
                    stress_matrix[site, site] = sigma
        
        return H_V + stress_matrix
    
    def analyze_stress(self, psi: np.ndarray,
                       H_K, H_V,
                       sigma: float,
                       stress_sites: Optional[List[int]] = None,
                       bonds: Optional[List[Tuple[int, int]]] = None) -> StressTopologyResult:
        """
        Full stress analysis.
        
        Returns prediction of failure site and critical λ.
        """
        # Apply stress
        H_V_stressed = self.apply_stress(H_V, sigma, stress_sites)
        
        # Compute stress field
        stress_field = np.zeros(self.n_sites)
        if stress_sites is not None:
            for site in stress_sites:
                if site < self.n_sites:
                    stress_field[site] = sigma
        else:
            stress_field[:] = sigma
        
        # Compute local λ
        lambda_local = self.compute_local_lambda(psi, H_K, H_V_stressed, stress_field)
        
        # Find maximum λ (failure point)
        failure_site = int(np.argmax(lambda_local))
        max_lambda = float(lambda_local[failure_site])
        
        # Phase gradient (if bonds provided)
        phase_gradient = None
        if bonds is not None:
            phase_gradient = self.compute_phase_gradient(psi, bonds)
        
        return StressTopologyResult(
            sigma=sigma,
            max_lambda=max_lambda,
            failure_site=failure_site,
            phase_gradient=phase_gradient,
            local_lambda=lambda_local,
            is_failed=(max_lambda >= 1.0)
        )
    
    def stress_scan(self, psi: np.ndarray,
                    H_K, H_V,
                    sigma_range: np.ndarray,
                    stress_sites: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Scan stress range and find critical stress.
        """
        results = {
            'sigma': sigma_range,
            'max_lambda': np.zeros_like(sigma_range),
            'failure_site': np.zeros_like(sigma_range, dtype=int),
            'is_failed': np.zeros_like(sigma_range, dtype=bool)
        }
        
        for i, sigma in enumerate(sigma_range):
            res = self.analyze_stress(psi, H_K, H_V, sigma, stress_sites)
            results['max_lambda'][i] = res.max_lambda
            results['failure_site'][i] = res.failure_site
            results['is_failed'][i] = res.is_failed
        
        return results
    
    def find_critical_stress(self, psi: np.ndarray,
                              H_K, H_V,
                              sigma_range: np.ndarray,
                              stress_sites: Optional[List[int]] = None) -> float:
        """
        Find critical stress σ_c where λ → 1.
        """
        scan = self.stress_scan(psi, H_K, H_V, sigma_range, stress_sites)
        
        # Find first failure
        for i, (sigma, is_failed) in enumerate(zip(sigma_range, scan['is_failed'])):
            if is_failed:
                if i > 0:
                    # Interpolate
                    lambda_prev = scan['max_lambda'][i-1]
                    lambda_curr = scan['max_lambda'][i]
                    sigma_prev = sigma_range[i-1]
                    
                    frac = (1.0 - lambda_prev) / (lambda_curr - lambda_prev + 1e-10)
                    sigma_c = sigma_prev + frac * (sigma - sigma_prev)
                    return float(sigma_c)
                return float(sigma)
        
        return float('inf')


# =============================================================================
# Combined Thermal-Stress Analyzer
# =============================================================================

class CombinedFailureAnalyzer:
    """
    熱と応力の協働効果を解析
    
    【物理】
      熱で結び目が緩む → 同じ応力でも切れやすい
      
      V_eff(r, T, σ) = V_0(r) × f(Coherence(T)) + σ(r)
      
      高温: V_0↓ → 小さい σ で破壊
      低温: V_0↑ → 大きい σ が必要
    """
    
    def __init__(self, thermal_analyzer: ThermalTopologyAnalyzer,
                 stress_analyzer: StressTopologyAnalyzer):
        self.thermal = thermal_analyzer
        self.stress = stress_analyzer
    
    def compute_effective_V0(self, T: float, V0_base: float = 1.0) -> float:
        """
        Compute temperature-dependent V_0.
        
        V_0(T) = V_0(0) × Coherence(T)
        """
        coherence = self.thermal.compute_coherence(T)
        return V0_base * coherence
    
    def predict_failure(self, psi: np.ndarray,
                        H_K, H_V,
                        T: float,
                        sigma: float,
                        stress_sites: Optional[List[int]] = None) -> FailurePrediction:
        """
        Predict failure under combined thermal-mechanical loading.
        """
        # Thermal analysis
        thermal_result = self.thermal.analyze_temperature(T)
        
        # Scale V_0 by coherence
        coherence = thermal_result.coherence
        H_V_scaled = H_V * coherence  # Effectively weaker bonds
        
        # Stress analysis with weakened bonds
        stress_result = self.stress.analyze_stress(
            psi, H_K, H_V_scaled, sigma, stress_sites
        )
        
        # Determine failure mode
        if thermal_result.is_melted:
            mode = 'thermal'
            mechanism = 'melting'
            critical_value = T
        elif stress_result.is_failed:
            mode = 'mechanical'
            mechanism = 'fracture'
            critical_value = sigma
        else:
            # Check for creep (high T + moderate σ)
            if T > 0.5 * 1000 and stress_result.max_lambda > 0.5:  # Rough threshold
                mode = 'combined'
                mechanism = 'creep'
                critical_value = sigma
            else:
                mode = 'none'
                mechanism = 'safe'
                critical_value = 0.0
        
        return FailurePrediction(
            mode=mode,
            failure_site=stress_result.failure_site,
            critical_value=critical_value,
            lambda_at_failure=stress_result.max_lambda,
            mechanism=mechanism
        )
    
    def compute_failure_surface(self, psi: np.ndarray,
                                 H_K, H_V,
                                 T_range: np.ndarray,
                                 sigma_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute failure surface in (T, σ) space.
        
        Returns 2D array of max_lambda values.
        """
        nT = len(T_range)
        nS = len(sigma_range)
        
        lambda_surface = np.zeros((nT, nS))
        failure_mode = np.empty((nT, nS), dtype=object)
        
        for i, T in enumerate(T_range):
            for j, sigma in enumerate(sigma_range):
                pred = self.predict_failure(psi, H_K, H_V, T, sigma)
                lambda_surface[i, j] = pred.lambda_at_failure
                failure_mode[i, j] = pred.mechanism
        
        return {
            'T': T_range,
            'sigma': sigma_range,
            'lambda_surface': lambda_surface,
            'failure_mode': failure_mode
        }


# =============================================================================
# Local Thermal Ensemble (for standalone testing)
# =============================================================================

class LocalThermalEnsemble:
    """
    軽量版 ThermalEnsemble（テスト用）
    
    environment_operators.py の ThermalEnsemble とは独立。
    DSETopologyTest で使用。
    """
    
    def __init__(self, H, n_eigenstates: int = 20, use_gpu: bool = False):
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = get_xp(self.use_gpu)
        
        self.H = H
        self.n_eigenstates = min(n_eigenstates, H.shape[0] - 2)
        
        # Compute eigenstates
        self.eigenvalues, self.eigenvectors = eigsh_wrapper(
            H, k=self.n_eigenstates, which='SA', use_gpu=self.use_gpu
        )
        
        # Sort by energy
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        
        # Observable cache
        self._obs_cache: Dict[str, np.ndarray] = {}
    
    def register_observable(self, name: str, func: Callable):
        """Register observable function."""
        values = np.zeros(self.n_eigenstates)
        for n in range(self.n_eigenstates):
            psi = self.eigenvectors[:, n]
            values[n] = func(psi)
        self._obs_cache[name] = values
    
    def thermal_average(self, observable: str, T: float) -> float:
        """Compute thermal average (returns float directly)."""
        if observable not in self._obs_cache:
            raise ValueError(f"Observable '{observable}' not registered")
        
        weights = _get_weights(self.eigenvalues, T)
        O_values = self._obs_cache[observable]
        
        return float(np.sum(weights * O_values))


# =============================================================================
# DSE Topology Test Suite
# =============================================================================

class DSETopologyTest:
    """
    DSE (Direct Schrödinger Evolution) での統合トポロジーテスト
    
    Tests:
      1. thermal_test: 温度スキャン → 融解検出
      2. stress_test: 応力スキャン → 破断検出
      3. combined_test: 熱+応力 → 破壊面
      4. coherence_test: Coherence-Entropy 関係
    """
    
    def __init__(self, n_sites: int = 4, use_gpu: bool = True):
        self.n_sites = n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = get_xp(self.use_gpu)
        
        # Will be set by build_test_system
        self.H_K = None
        self.H_V = None
        self.H = None
        self.ensemble = None
        self.thermal_analyzer = None
        self.stress_analyzer = None
    
    def build_hubbard_hamiltonian(self, t: float = 1.0, U: float = 2.0,
                                   bonds: Optional[List[Tuple[int, int]]] = None):
        """
        Build Hubbard model Hamiltonian.
        
        H = -t Σ c†_iσ c_jσ + U Σ n_i↑ n_i↓
        
        For simplicity, use spinless fermions on n_sites.
        """
        xp = self.xp
        n = self.n_sites
        dim = 2**n  # Fock space dimension
        
        if bonds is None:
            # Default: 1D chain with PBC
            bonds = [(i, (i+1) % n) for i in range(n)]
        
        self.bonds = bonds
        
        # Build H_K (hopping)
        H_K = xp.zeros((dim, dim), dtype=complex)
        
        for state in range(dim):
            for (i, j) in bonds:
                # Check if hopping is possible
                if (state >> i) & 1 and not ((state >> j) & 1):
                    # Hop from i to j
                    new_state = state ^ (1 << i) ^ (1 << j)
                    # Fermionic sign
                    sign = 1
                    for k in range(min(i, j) + 1, max(i, j)):
                        if (state >> k) & 1:
                            sign *= -1
                    H_K[new_state, state] += -t * sign
                    H_K[state, new_state] += -t * sign
        
        # Build H_V (interaction - for spinless, use nearest-neighbor)
        H_V = xp.zeros((dim, dim), dtype=complex)
        
        for state in range(dim):
            # Count occupied neighbors
            for (i, j) in bonds:
                ni = (state >> i) & 1
                nj = (state >> j) & 1
                H_V[state, state] += U * ni * nj
        
        self.H_K = H_K
        self.H_V = H_V
        self.H = H_K + H_V
        
        return H_K, H_V
    
    def build_test_system(self, t: float = 1.0, U: float = 2.0,
                          n_eigenstates: int = 10):
        """
        Build complete test system.
        """
        # Build Hamiltonian
        self.build_hubbard_hamiltonian(t, U)
        
        # Convert to host for eigensolver
        H_host = to_host(self.H)
        
        # Create thermal ensemble (using local version for standalone testing)
        self.ensemble = LocalThermalEnsemble(H_host, n_eigenstates, use_gpu=False)
        
        # Create analyzers
        self.thermal_analyzer = ThermalTopologyAnalyzer(self.ensemble, self.use_gpu)
        self.stress_analyzer = StressTopologyAnalyzer(self.n_sites, self.use_gpu)
        
        return self
    
    def thermal_test(self, T_range: np.ndarray = None,
                     verbose: bool = True) -> Dict:
        """
        Test thermal topology: temperature scan and melting detection.
        """
        if T_range is None:
            T_range = np.linspace(10, 2000, 50)
        
        if self.thermal_analyzer is None:
            self.build_test_system()
        
        results = self.thermal_analyzer.temperature_scan(T_range)
        T_m = self.thermal_analyzer.detect_melting_point(T_range)
        
        if verbose:
            print("=" * 60)
            print("THERMAL TOPOLOGY TEST")
            print("=" * 60)
            print(f"  Temperature range: {T_range[0]:.0f} - {T_range[-1]:.0f} K")
            print(f"  Melting point T_m: {T_m:.1f} K")
            print(f"  Coherence at T=100K: {results['coherence'][np.argmin(np.abs(T_range-100))]:.4f}")
            print(f"  Coherence at T=1000K: {results['coherence'][np.argmin(np.abs(T_range-1000))]:.4f}")
            print(f"  Max Lindemann δ: {results['lindemann_delta'].max():.4f}")
        
        results['T_m'] = T_m
        return results
    
    def stress_test(self, sigma_range: np.ndarray = None,
                    stress_sites: Optional[List[int]] = None,
                    verbose: bool = True) -> Dict:
        """
        Test stress topology: stress scan and fracture detection.
        """
        if sigma_range is None:
            sigma_range = np.linspace(0, 5, 50)
        
        if self.stress_analyzer is None:
            self.build_test_system()
        
        # Get ground state
        psi = self.ensemble.eigenvectors[:, 0]
        
        results = self.stress_analyzer.stress_scan(
            psi, 
            to_host(self.H_K), 
            to_host(self.H_V), 
            sigma_range, 
            stress_sites
        )
        sigma_c = self.stress_analyzer.find_critical_stress(
            psi, 
            to_host(self.H_K), 
            to_host(self.H_V), 
            sigma_range, 
            stress_sites
        )
        
        if verbose:
            print("=" * 60)
            print("STRESS TOPOLOGY TEST")
            print("=" * 60)
            print(f"  Stress range: {sigma_range[0]:.2f} - {sigma_range[-1]:.2f}")
            print(f"  Critical stress σ_c: {sigma_c:.3f}")
            print(f"  Failure site: {results['failure_site'][-1]}")
            print(f"  Max λ at σ=1: {results['max_lambda'][np.argmin(np.abs(sigma_range-1))]:.4f}")
        
        results['sigma_c'] = sigma_c
        return results
    
    def combined_test(self, T_range: np.ndarray = None,
                      sigma_range: np.ndarray = None,
                      verbose: bool = True) -> Dict:
        """
        Test combined thermal-mechanical loading.
        """
        if T_range is None:
            T_range = np.linspace(100, 1500, 20)
        if sigma_range is None:
            sigma_range = np.linspace(0, 3, 20)
        
        if self.thermal_analyzer is None or self.stress_analyzer is None:
            self.build_test_system()
        
        combined = CombinedFailureAnalyzer(self.thermal_analyzer, self.stress_analyzer)
        
        psi = self.ensemble.eigenvectors[:, 0]
        
        results = combined.compute_failure_surface(
            psi,
            to_host(self.H_K),
            to_host(self.H_V),
            T_range,
            sigma_range
        )
        
        if verbose:
            print("=" * 60)
            print("COMBINED THERMAL-STRESS TEST")
            print("=" * 60)
            print(f"  T range: {T_range[0]:.0f} - {T_range[-1]:.0f} K")
            print(f"  σ range: {sigma_range[0]:.2f} - {sigma_range[-1]:.2f}")
            
            # Find failure boundary (λ = 1)
            lambda_surf = results['lambda_surface']
            n_failure = np.sum(lambda_surf >= 1.0)
            print(f"  Failure points: {n_failure} / {lambda_surf.size}")
            
            # Report some specific points
            print(f"\n  Sample λ values:")
            print(f"    λ(T=500K, σ=1): {lambda_surf[np.argmin(np.abs(T_range-500)), np.argmin(np.abs(sigma_range-1))]:.3f}")
            print(f"    λ(T=1000K, σ=1): {lambda_surf[np.argmin(np.abs(T_range-1000)), np.argmin(np.abs(sigma_range-1))]:.3f}")
        
        return results
    
    def coherence_entropy_test(self, T_range: np.ndarray = None,
                                verbose: bool = True) -> Dict:
        """
        Test Coherence-Entropy relationship.
        
        As T↑: Coherence↓, Entropy↑
        """
        if T_range is None:
            T_range = np.linspace(10, 2000, 50)
        
        if self.thermal_analyzer is None:
            self.build_test_system()
        
        coherences = []
        entropies = []
        
        for T in T_range:
            coh = self.thermal_analyzer.compute_coherence(T)
            ent = self.ensemble.thermal_average('phase_entropy', T)
            coherences.append(coh)
            entropies.append(ent)
        
        coherences = np.array(coherences)
        entropies = np.array(entropies)
        
        # Correlation
        corr = np.corrcoef(coherences, entropies)[0, 1]
        
        if verbose:
            print("=" * 60)
            print("COHERENCE-ENTROPY TEST")
            print("=" * 60)
            print(f"  Correlation(Coherence, Entropy): {corr:.4f}")
            print(f"  Expected: negative (as Coherence↓, Entropy↑)")
            print(f"\n  At T=100K:  Coh={coherences[np.argmin(np.abs(T_range-100))]:.4f}, S={entropies[np.argmin(np.abs(T_range-100))]:.4f}")
            print(f"  At T=1000K: Coh={coherences[np.argmin(np.abs(T_range-1000))]:.4f}, S={entropies[np.argmin(np.abs(T_range-1000))]:.4f}")
        
        return {
            'T': T_range,
            'coherence': coherences,
            'entropy': entropies,
            'correlation': corr
        }
    
    def run_all_tests(self, verbose: bool = True) -> Dict:
        """
        Run all topology tests.
        """
        print("\n" + "🔬" * 30)
        print("DSE TOPOLOGY TEST SUITE")
        print("🔬" * 30 + "\n")
        
        results = {}
        
        # Build system
        self.build_test_system()
        print(f"✅ Built {self.n_sites}-site Hubbard system")
        print(f"   GPU: {self.use_gpu and HAS_CUPY}")
        print()
        
        # Run tests
        results['thermal'] = self.thermal_test(verbose=verbose)
        print()
        
        results['stress'] = self.stress_test(verbose=verbose)
        print()
        
        results['combined'] = self.combined_test(verbose=verbose)
        print()
        
        results['coherence_entropy'] = self.coherence_entropy_test(verbose=verbose)
        print()
        
        print("=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        
        return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run test suite
    test = DSETopologyTest(n_sites=4, use_gpu=HAS_CUPY)
    results = test.run_all_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Melting point T_m: {results['thermal']['T_m']:.1f} K")
    print(f"  Critical stress σ_c: {results['stress']['sigma_c']:.3f}")
    print(f"  Coherence-Entropy correlation: {results['coherence_entropy']['correlation']:.4f}")
