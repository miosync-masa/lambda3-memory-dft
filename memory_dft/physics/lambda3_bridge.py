"""
Stability and Memory Diagnostics for Memory-DFT
===============================================

This module provides diagnostic tools for analyzing
stability, criticality, and history-dependent behavior
in Memory-DFT simulations.

Core quantity:
  λ = K / |V_eff|
  (dimensionless stability indicator)

Physical interpretation:
  - λ < 1: bound / stable regime
  - λ ≈ 1: critical regime (onset of instability)
  - λ > 1: unbound / unstable regime

This formulation is equivalent to an energy-density ratio
used in mechanics and materials science, extended here
to quantum many-body dynamics with memory effects.

Key features:
  - Detection of critical transitions
  - Path-dependent stability analysis
  - Environmental renormalization of effective binding
  - Quantitative diagnostics for non-Markovian dynamics

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


class StabilityPhase(Enum):
    """Stability phases based on λ value."""
    STABLE = "stable"        # λ < 0.7
    CAUTION = "caution"      # 0.7 ≤ λ < 0.9
    CRITICAL = "critical"    # 0.9 ≤ λ < 1.0
    UNSTABLE = "unstable"    # λ ≥ 1.0


@dataclass
class LambdaState:
    """
    Complete description of the instantaneous stability state.

    Attributes:
        Lambda: Dimensionless stability parameter λ = K / |V_eff|
        K: Kinetic (or destabilizing) energy contribution
        V_eff: Effective binding energy magnitude
        Lambda_dot: Time derivative, used to detect dynamic instability
        K_components: Decomposition of kinetic contributions
        V_components: Decomposition of binding contributions
        phase: Qualitative stability regime inferred from λ
    """
    Lambda: float
    K: float
    V_eff: float
    Lambda_dot: float = 0.0
    K_components: Dict[str, float] = field(default_factory=dict)
    V_components: Dict[str, float] = field(default_factory=dict)
    phase: StabilityPhase = StabilityPhase.STABLE
    
    def __post_init__(self):
        self.phase = self._determine_phase()
    
    def _determine_phase(self) -> StabilityPhase:
        if self.Lambda < 0.7:
            return StabilityPhase.STABLE
        elif self.Lambda < 0.9:
            return StabilityPhase.CAUTION
        elif self.Lambda < 1.0:
            return StabilityPhase.CRITICAL
        else:
            return StabilityPhase.UNSTABLE


class Lambda3Calculator:
    """
    Stability calculator based on energy-density ratio.

    The stability parameter is defined as:
        λ = K / |V_eff|

    where:
    - K is the kinetic or destabilizing energy contribution
    - V_eff is the effective binding energy

    This quantity provides a dimensionless measure of
    proximity to instability and is particularly useful
    for detecting path-dependent and non-Markovian effects
    in time-dependent quantum simulations.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        self.lambda_history: List[Tuple[float, float]] = []
    
    def compute_lambda(self,
                       psi,
                       H_kinetic,
                       H_potential,
                       time: Optional[float] = None,
                       record: bool = True) -> LambdaState:
        """
        Compute stability parameter from wavefunction.
        
        Args:
            psi: Quantum state vector
            H_kinetic: Kinetic energy operator
            H_potential: Potential energy operator
            time: Current time (for derivative calculation)
            record: Whether to record in history
        
        Returns:
            LambdaState with full diagnostics
        """
        xp = self.xp
        
        K_psi = H_kinetic @ psi
        K = float(xp.real(xp.vdot(psi, K_psi)))
        
        V_psi = H_potential @ psi
        V = float(xp.real(xp.vdot(psi, V_psi)))
        V_eff = abs(V)
        
        Lambda = abs(K) / (V_eff + 1e-10)
        
        # Time derivative
        Lambda_dot = 0.0
        if time is not None and len(self.lambda_history) > 0:
            t_prev, L_prev = self.lambda_history[-1]
            dt = time - t_prev
            if dt > 0:
                Lambda_dot = (Lambda - L_prev) / dt
        
        if record and time is not None:
            self.lambda_history.append((time, Lambda))
        
        return LambdaState(
            Lambda=Lambda,
            K=K,
            V_eff=V_eff,
            Lambda_dot=Lambda_dot,
            K_components={'total': K},
            V_components={'total': V_eff}
        )
    
    def compute_edr(self,
                    psi,
                    H_kinetic,
                    H_potential,
                    environment: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compute environment-renormalized energy-density ratio.
    
        Extends the bare stability parameter by including
        external conditions such as temperature, fields, and
        chemical environment.
    
        The resulting ratio serves as a generalized stability
        indicator, analogous to criteria used in materials
        and mechanical engineering.
        """
        env = environment or {}
        state = self.compute_lambda(psi, H_kinetic, H_potential, record=False)
        
        K_total = state.K
        V_eff = state.V_eff
        
        # Temperature correction (thermal kinetic energy)
        if 'T' in env:
            k_B = 8.617e-5  # eV/K
            K_th = 1.5 * k_B * env['T']
            K_total += K_th
        
        # Magnetic field correction (binding reduction)
        if 'B' in env:
            beta_B = env.get('beta_B', 0.01)
            V_eff -= beta_B * env['B']**2
        
        # Oxidation correction (time-dependent degradation)
        if 'c_O2' in env and 't' in env:
            k_oxide = env.get('k_oxide', 0.001)
            V_eff -= k_oxide * env['c_O2'] * env['t']
        
        EDR = abs(K_total) / (abs(V_eff) + 1e-10)
        
        if EDR < 0.3:
            safety = "SAFE"
        elif EDR < 0.7:
            safety = "RECOMMENDED"
        elif EDR < 1.0:
            safety = "CAUTION"
        else:
            safety = "DANGER"
        
        return {
            'EDR': EDR,
            'K_total': K_total,
            'V_eff': V_eff,
            'safety': safety,
            'environment': env
        }
    
    def check_critical_transition(self,
                                  lambda_trajectory: np.ndarray,
                                  threshold: float = 1.0) -> Dict[str, Any]:
        """
        Detect crossings of the critical threshold λ = 1.
    
        Crossing λ = 1 indicates a transition between
        bound and unbound regimes, corresponding to
        instability, failure, or phase change.
        """
        crossings = []
        
        for i in range(1, len(lambda_trajectory)):
            L_prev = lambda_trajectory[i-1]
            L_curr = lambda_trajectory[i]
            
            if L_prev < threshold and L_curr >= threshold:
                crossings.append({
                    'index': i,
                    'type': 'destabilization',
                    'lambda_before': L_prev,
                    'lambda_after': L_curr
                })
            elif L_prev >= threshold and L_curr < threshold:
                crossings.append({
                    'index': i,
                    'type': 'stabilization',
                    'lambda_before': L_prev,
                    'lambda_after': L_curr
                })
        
        return {
            'n_crossings': len(crossings),
            'crossings': crossings,
            'max_lambda': float(np.max(lambda_trajectory)),
            'min_lambda': float(np.min(lambda_trajectory)),
            'ever_unstable': float(np.max(lambda_trajectory)) >= threshold
        }


class HCSPValidator:
    """
    Diagnostic checks for dynamical consistency
    in Memory-DFT simulations.

    These tests probe conservation, recursion,
    non-commutativity, and dynamical stability
    using only observable quantities.
    """
    
    @staticmethod
    def check_axiom1_hierarchy(lambda_series: List[LambdaState]) -> bool:
        """
        Check for smooth temporal evolution.
        
        Large jumps indicate violation of hierarchical
        constraint propagation.
        """
        if len(lambda_series) < 2:
            return True
        
        lambdas = [s.Lambda for s in lambda_series]
        max_jump = max(abs(lambdas[i+1] - lambdas[i]) for i in range(len(lambdas)-1))
        
        return max_jump < 0.5
    
    @staticmethod
    def check_axiom2_noncommutative(results_forward: Any, results_backward: Any) -> bool:
        """
        Check for path dependence.
        
        Forward and backward protocols leading to
        different final states indicate non-commutative,
        history-dependent dynamics.
        """
        L_forward = results_forward.lambdas[-1]
        L_backward = results_backward.lambdas[-1]
        
        return abs(L_forward - L_backward) > 1e-6
    
    @staticmethod
    def check_axiom3_conservation(lambda_series: List[float], 
                                  tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Check for global conservation.
        
        Tests whether the average λ remains stable over time,
        indicating conservation despite local fluctuations.
        """
        if len(lambda_series) < 10:
            return {'conserved': True, 'drift': 0}
        
        half = len(lambda_series) // 2
        mean_first = np.mean(lambda_series[:half])
        mean_second = np.mean(lambda_series[half:])
        
        drift = abs(mean_second - mean_first) / (mean_first + 1e-10)
        
        return {
            'conserved': drift < tolerance,
            'drift': drift,
            'mean_first': mean_first,
            'mean_second': mean_second
        }
    
    @staticmethod
    def check_axiom4_recursive(lambda_series: List[float]) -> Dict[str, Any]:
        """
        Check for temporal self-correlation.
        
        Significant autocorrelation indicates that
        the current state depends on its recent history,
        a hallmark of non-Markovian dynamics.
        """
        if len(lambda_series) < 20:
            return {'recursive': True, 'autocorr': 0}
        
        series = np.array(lambda_series)
        mean = np.mean(series)
        var = np.var(series)
        
        if var < 1e-10:
            return {'recursive': True, 'autocorr': 1.0}
        
        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
        
        return {
            'recursive': abs(autocorr) > 0.5,
            'autocorr': autocorr
        }
    
    @staticmethod
    def check_axiom5_pulsation(lambda_series: List[float],
                               window: int = 10) -> Dict[str, Any]:
        """
        Check for pulsative equilibrium.
        
        Tests for the combination of:
        - Local variation (system is dynamic)
        - Global stability (average is constant)
        
        This signature characterizes living/driven systems.
        """
        if len(lambda_series) < window * 2:
            return {'pulsation': False, 'reason': 'insufficient data'}
        
        series = np.array(lambda_series)
        
        local_var = np.mean(np.abs(np.diff(series[-window:])))
        global_std = np.std(series[-window:])
        global_mean = np.mean(series[-window:])
        relative_std = global_std / (global_mean + 1e-10)
        
        pulsation = (local_var > 1e-4) and (relative_std < 0.1)
        
        return {
            'pulsation': pulsation,
            'local_variation': local_var,
            'global_relative_std': relative_std,
            'interpretation': 'Living system signature!' if pulsation else 'Static or chaotic'
        }
    
    def validate_all(self, lambda_series: List[float]) -> Dict[str, Any]:
        """Run all diagnostic checks."""
        return {
            'conservation': self.check_axiom3_conservation(lambda_series),
            'recursion': self.check_axiom4_recursive(lambda_series),
            'pulsation': self.check_axiom5_pulsation(lambda_series)
        }

# =============================================================================
# Memory Kernel ↔ Environment Hierarchy Mapping
# =============================================================================

def map_kernel_to_environment():
    """
    Mapping between Memory Kernel components and physical environment classes

    This function provides a conceptual correspondence between different
    memory-kernel components used in Memory-DFT and their associated
    physical origins.

    | Memory kernel | Environment class | Physical interpretation        |
    |---------------|-------------------|--------------------------------|
    | K_field       | Field-like        | Non-local, scale-invariant     |
    | K_phys        | Structural        | Relaxation, dissipative        |
    | K_chem        | Chemical          | Irreversible, hysteretic       |
    """
    return {
        'field': {
            'kernel': 'PowerLaw',
            'environment': 'Field-like',
            'gamma': 1.0,
            'examples': ['gravity', 'electromagnetic fields', 'radiation'],
            'characteristic': 'non-local, scale-invariant'
        },
        'phys': {
            'kernel': 'StretchedExponential',
            'environment': 'Structural / Thermodynamic',
            'beta': 0.5,
            'examples': ['temperature', 'humidity', 'pressure'],
            'characteristic': 'relaxation, partially Markovian'
        },
        'chem': {
            'kernel': 'Step / Piecewise',
            'environment': 'Chemical',
            'examples': ['oxidation', 'corrosion', 'pH changes'],
            'characteristic': 'irreversible, hysteresis, path-dependent'
        }
    }

# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Stability Diagnostics Test")
    print("="*70)
    
    try:
        from memory_dft.core.sparse_engine import SparseHamiltonianEngine
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.sparse_engine import SparseHamiltonianEngine
    
    engine = SparseHamiltonianEngine(n_sites=4, use_gpu=False, verbose=False)
    geom = engine.build_chain_geometry(L=4)
    H_K, H_V = engine.build_heisenberg_hamiltonian(geom.bonds)
    
    psi = np.random.randn(engine.dim) + 1j * np.random.randn(engine.dim)
    psi = psi / np.linalg.norm(psi)
    
    calc = Lambda3Calculator(use_gpu=False)
    state = calc.compute_lambda(psi, H_K, H_V, time=0.0)
    
    print(f"\nStability State:")
    print(f"  λ      = {state.Lambda:.4f}")
    print(f"  K      = {state.K:.4f}")
    print(f"  |V|    = {state.V_eff:.4f}")
    print(f"  Phase: {state.phase.value}")
    
    # EDR with environment
    env = {'T': 300, 'B': 0.1, 'c_O2': 0.21, 't': 100}
    edr = calc.compute_edr(psi, H_K, H_V, environment=env)
    
    print(f"\nEDR (with environment):")
    print(f"  EDR    = {edr['EDR']:.4f}")
    print(f"  Safety: {edr['safety']}")
    
    # Diagnostic tests
    print("\n" + "="*70)
    print("Dynamical Diagnostics")
    print("="*70)
    
    t = np.linspace(0, 10, 100)
    lambda_series = 0.5 + 0.1 * np.sin(t) + 0.02 * np.random.randn(len(t))
    
    validator = HCSPValidator()
    results = validator.validate_all(list(lambda_series))
    
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    
    print("\n✅ Stability diagnostics OK!")
