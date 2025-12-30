"""
Memory Indicators for Memory-DFT
================================

Quantitative metrics for history-dependent quantum dynamics.

Three complementary measures:
1. Path non-commutativity (ΔO) - Different paths → Different outcomes
2. Temporal autocorrelation M(t) - How long does the system remember?
3. Gamma decomposition (γ_memory) - Non-Markovian fraction

These metrics answer the key question:
"How do you quantify memory?"

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MemoryMetrics:
    """Container for all memory indicators."""
    delta_O: float              # Path non-commutativity
    M_temporal: float           # Temporal memory integral
    gamma_memory: Optional[float] = None  # γ decomposition (if available)
    autocorr_time: float = 0.0  # Characteristic memory time
    
    def __repr__(self):
        s = f"MemoryMetrics:\n"
        s += f"  ΔO (path non-commutativity) = {self.delta_O:.6f}\n"
        s += f"  M (temporal memory)         = {self.M_temporal:.6f}\n"
        s += f"  τ_memory (autocorr time)    = {self.autocorr_time:.4f}\n"
        if self.gamma_memory is not None:
            s += f"  γ_memory (Non-Markovian)    = {self.gamma_memory:.4f}\n"
        return s
    
    def is_non_markovian(self, threshold: float = 0.01) -> bool:
        """Check if system shows significant memory effects."""
        return self.delta_O > threshold or self.M_temporal > threshold


class MemoryIndicator:
    """
    Calculator for memory indicators in quantum systems.
    
    Usage:
        indicator = MemoryIndicator()
        
        # From path comparison
        delta_O = indicator.path_noncommutativity(O_forward, O_backward)
        
        # From time series
        M, tau = indicator.temporal_memory(observable_series, dt)
        
        # From gamma decomposition
        gamma_mem = indicator.gamma_memory(gamma_total, gamma_local)
    """
    
    @staticmethod
    def path_noncommutativity(O_forward: float, O_backward: float) -> float:
        """
        Compute path non-commutativity indicator.
        
        ΔO = |O_{A→B} - O_{B→A}|
        
        This measures whether the order of operations matters.
        For Markovian systems: ΔO = 0 (path independent)
        For Non-Markovian: ΔO > 0 (history matters)
        
        Args:
            O_forward: Observable after path A→B
            O_backward: Observable after path B→A
            
        Returns:
            ΔO: Non-negative path difference
            
        Physical interpretation:
            - ΔO ~ 0.01: Weak memory effects
            - ΔO ~ 0.1:  Moderate memory effects
            - ΔO ~ 1.0:  Strong memory effects (e.g., hysteresis)
        """
        return abs(O_forward - O_backward)
    
    @staticmethod
    def path_noncommutativity_relative(O_forward: float, O_backward: float, 
                                        epsilon: float = 1e-10) -> float:
        """
        Relative path non-commutativity.
        
        ΔO_rel = |O_{A→B} - O_{B→A}| / (|O_{A→B}| + |O_{B→A}|) * 2
        
        Normalized to [0, 2] range.
        """
        denom = abs(O_forward) + abs(O_backward) + epsilon
        return 2 * abs(O_forward - O_backward) / denom
    
    @staticmethod
    def temporal_memory(series: np.ndarray, dt: float = 1.0) -> Tuple[float, float]:
        """
        Compute temporal memory from autocorrelation.
        
        M(t) = ∫₀ᵗ ⟨O(t)O(t')⟩_c dt'
        
        where ⟨...⟩_c is the connected (cumulant) correlator.
        
        Args:
            series: Time series of observable O(t)
            dt: Time step
            
        Returns:
            M: Integrated memory (area under autocorrelation)
            tau: Characteristic memory time (1/e decay)
        """
        if len(series) < 3:
            return 0.0, 0.0
            
        # Center the series (connected correlator)
        mean = np.mean(series)
        centered = series - mean
        
        # Autocorrelation
        n = len(centered)
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        
        # Normalize
        if autocorr[0] > 1e-10:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.0, 0.0
        
        # Integrated memory (trapezoidal)
        M = np.trapz(autocorr, dx=dt)
        
        # Characteristic time (1/e decay)
        tau = 0.0
        threshold = 1.0 / np.e
        for i, val in enumerate(autocorr):
            if val < threshold:
                tau = i * dt
                break
        else:
            tau = len(autocorr) * dt  # Never decayed
            
        return float(M), float(tau)
    
    @staticmethod
    def gamma_memory(gamma_total: float, gamma_local: float) -> float:
        """
        Non-Markovian gamma from distance decomposition.
        
        γ_memory = γ_total - γ_local
        
        Physical meaning:
            - γ_local: Correlations within short range (Markovian sector)
            - γ_total: All correlations including long-range
            - γ_memory: Non-local, history-dependent correlations
            
        Reference: Memory-DFT ED distance decomposition
        """
        return gamma_total - gamma_local
    
    @staticmethod
    def memory_fraction(gamma_total: float, gamma_local: float, 
                        epsilon: float = 1e-10) -> float:
        """
        Fraction of correlations that are non-Markovian.
        
        f_memory = γ_memory / γ_total
        
        Returns:
            Fraction in [0, 1]
        """
        gamma_mem = gamma_total - gamma_local
        return gamma_mem / (abs(gamma_total) + epsilon)
    
    @classmethod
    def compute_all(cls, 
                    O_forward: Optional[float] = None,
                    O_backward: Optional[float] = None,
                    series: Optional[np.ndarray] = None,
                    dt: float = 1.0,
                    gamma_total: Optional[float] = None,
                    gamma_local: Optional[float] = None) -> MemoryMetrics:
        """
        Compute all available memory indicators.
        
        Args:
            O_forward: Observable after forward path
            O_backward: Observable after backward path
            series: Time series of observable
            dt: Time step for series
            gamma_total: Total gamma (if available)
            gamma_local: Local gamma (if available)
            
        Returns:
            MemoryMetrics with all computed indicators
        """
        # Path non-commutativity
        delta_O = 0.0
        if O_forward is not None and O_backward is not None:
            delta_O = cls.path_noncommutativity(O_forward, O_backward)
        
        # Temporal memory
        M_temporal = 0.0
        tau = 0.0
        if series is not None and len(series) > 2:
            M_temporal, tau = cls.temporal_memory(series, dt)
        
        # Gamma decomposition
        gamma_mem = None
        if gamma_total is not None and gamma_local is not None:
            gamma_mem = cls.gamma_memory(gamma_total, gamma_local)
        
        return MemoryMetrics(
            delta_O=delta_O,
            M_temporal=M_temporal,
            gamma_memory=gamma_mem,
            autocorr_time=tau
        )


class HysteresisAnalyzer:
    """
    Analyze hysteresis loops for memory quantification.
    
    Hysteresis area is a direct measure of dissipated memory.
    """
    
    @staticmethod
    def compute_hysteresis_area(x_forward: np.ndarray, y_forward: np.ndarray,
                                 x_backward: np.ndarray, y_backward: np.ndarray) -> float:
        """
        Compute area enclosed by hysteresis loop.
        
        Uses shoelace formula for polygon area.
        
        Args:
            x_forward, y_forward: Forward sweep (x increasing)
            x_backward, y_backward: Backward sweep (x decreasing)
            
        Returns:
            Enclosed area (always positive)
        """
        # Combine into closed loop
        x = np.concatenate([x_forward, x_backward[::-1]])
        y = np.concatenate([y_forward, y_backward[::-1]])
        
        # Shoelace formula
        n = len(x)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]
        
        return abs(area) / 2.0
    
    @staticmethod
    def compute_hysteresis_metrics(x_forward: np.ndarray, y_forward: np.ndarray,
                                    x_backward: np.ndarray, y_backward: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive hysteresis analysis.
        
        Returns:
            Dictionary with:
            - area: Loop area
            - max_gap: Maximum y difference at same x
            - coercivity: x-intercept difference
        """
        area = HysteresisAnalyzer.compute_hysteresis_area(
            x_forward, y_forward, x_backward, y_backward
        )
        
        # Interpolate to common x grid
        x_common = np.linspace(
            max(x_forward.min(), x_backward.min()),
            min(x_forward.max(), x_backward.max()),
            100
        )
        
        y_fwd_interp = np.interp(x_common, x_forward, y_forward)
        y_bwd_interp = np.interp(x_common, x_backward[::-1], y_backward[::-1])
        
        max_gap = np.max(np.abs(y_fwd_interp - y_bwd_interp))
        
        return {
            'area': area,
            'max_gap': max_gap,
            'memory_strength': area / (np.ptp(x_forward) * np.ptp(y_forward) + 1e-10)
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Memory Indicators Test")
    print("="*60)
    
    indicator = MemoryIndicator()
    
    # Test 1: Path non-commutativity
    print("\n--- Test 1: Path Non-Commutativity ---")
    O_fwd = 0.5234
    O_bwd = 0.4821
    delta_O = indicator.path_noncommutativity(O_fwd, O_bwd)
    print(f"  O_forward  = {O_fwd:.4f}")
    print(f"  O_backward = {O_bwd:.4f}")
    print(f"  ΔO = {delta_O:.4f}")
    print(f"  → {'Non-Markovian!' if delta_O > 0.01 else 'Markovian'}")
    
    # Test 2: Temporal memory
    print("\n--- Test 2: Temporal Memory ---")
    t = np.linspace(0, 50, 500)
    # Decaying oscillation (memory effect)
    series = np.exp(-t/10) * np.cos(t) + 0.1 * np.random.randn(len(t))
    M, tau = indicator.temporal_memory(series, dt=0.1)
    print(f"  M (integrated memory) = {M:.4f}")
    print(f"  τ (memory time)       = {tau:.4f}")
    
    # Test 3: Gamma decomposition
    print("\n--- Test 3: Gamma Decomposition ---")
    gamma_total = 1.997
    gamma_local = 1.081
    gamma_mem = indicator.gamma_memory(gamma_total, gamma_local)
    frac = indicator.memory_fraction(gamma_total, gamma_local)
    print(f"  γ_total  = {gamma_total:.3f}")
    print(f"  γ_local  = {gamma_local:.3f}")
    print(f"  γ_memory = {gamma_mem:.3f}")
    print(f"  Memory fraction = {frac*100:.1f}%")
    
    # Test 4: Combined metrics
    print("\n--- Test 4: Combined Metrics ---")
    metrics = indicator.compute_all(
        O_forward=O_fwd,
        O_backward=O_bwd,
        series=series,
        dt=0.1,
        gamma_total=gamma_total,
        gamma_local=gamma_local
    )
    print(metrics)
    print(f"  Is Non-Markovian? {metrics.is_non_markovian()}")
    
    # Test 5: Hysteresis
    print("\n--- Test 5: Hysteresis Analysis ---")
    x_fwd = np.linspace(0, 1, 50)
    y_fwd = x_fwd**2
    x_bwd = np.linspace(1, 0, 50)
    y_bwd = x_bwd**0.5  # Different path!
    
    analyzer = HysteresisAnalyzer()
    hyst = analyzer.compute_hysteresis_metrics(x_fwd, y_fwd, x_bwd, y_bwd)
    print(f"  Hysteresis area     = {hyst['area']:.4f}")
    print(f"  Max gap             = {hyst['max_gap']:.4f}")
    print(f"  Memory strength     = {hyst['memory_strength']:.4f}")
    
    print("\n" + "="*60)
    print("✅ Memory Indicators Test Complete!")
    print("="*60)
