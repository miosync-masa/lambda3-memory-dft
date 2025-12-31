"""
Repulsive Memory Kernel - Compression History Tracking
======================================================

This module provides detailed compression history tracking
for the exclusion (repulsive) memory effect.

The core ExclusionKernel is now in memory_kernel.py.
This module provides:
  - CompressionEvent: Detailed compression event recording
  - RepulsiveMemoryKernel: Full compression history with hysteresis analysis
  - ExtendedCompositeKernel: Alias for CompositeMemoryKernel (backward compat)

Physical Motivation
-------------------
The same distance r = 0.8 A has DIFFERENT meaning:
  - Approaching (compression): preparing for Pauli repulsion
  - Departing (expansion): recovering from compression

This module tracks the detailed history of compression events
and computes the resulting enhancement of repulsive interactions.

Applications:
  - Diamond anvil cell (high-pressure)
  - AFM approach/retract curves
  - White layer formation in machining
  - Catalyst surface strain effects

Authors:
  Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Import unified kernel
try:
    from .memory_kernel import CompositeMemoryKernel, KernelWeights
except ImportError:
    # Direct execution
    from memory_kernel import CompositeMemoryKernel, KernelWeights


# =============================================================================
# Compression Event
# =============================================================================

@dataclass
class CompressionEvent:
    """
    Record of a compression event.
    
    Attributes:
        time: When compression occurred
        r_min: Minimum distance reached
        pressure: Local pressure estimate
        site: Which site was compressed (for multi-site systems)
    """
    time: float
    r_min: float
    pressure: float
    site: int = 0


# =============================================================================
# Repulsive Memory Kernel (Detailed Implementation)
# =============================================================================

class RepulsiveMemoryKernel:
    """
    History-dependent repulsive memory kernel with detailed tracking.
    
    This kernel tracks compression events and computes the
    enhancement of repulsive interactions due to prior compression.

    Effective interaction:
        V_rep_eff(r, t) = V_rep(r) * [1 + eta * integral K(t-t') * compression(t') dt']

    Parameters:
        eta_rep: Strength of repulsive memory
        tau_rep: Decay time of compression memory
        tau_recover: Recovery time scale
        r_critical: Distance threshold for compression detection
        n_power: Repulsion exponent (12 for Lennard-Jones)
    """
    
    def __init__(self,
                 eta_rep: float = 0.2,
                 tau_rep: float = 3.0,
                 tau_recover: float = 10.0,
                 r_critical: float = 0.8,
                 n_power: float = 12.0):
        self.eta_rep = eta_rep
        self.tau_rep = tau_rep
        self.tau_recover = tau_recover
        self.r_critical = r_critical
        self.n_power = n_power
        
        self.compression_history: List[CompressionEvent] = []
        self.state_history: List[Tuple[float, float, Optional[np.ndarray]]] = []
    
    def kernel_value(self, dt: float) -> float:
        """
        Temporal kernel for repulsive memory.

        K(dt) = exp(-dt / tau_rep) * [1 - exp(-dt / tau_recover)]

        Physical interpretation:
          - exp(-dt/tau_rep): forgetting of compression
          - 1 - exp(-dt/tau_rec): gradual elastic recovery
          - Product: maximal effect at intermediate times
        """
        if dt <= 0:
            return 0.0
        
        decay = np.exp(-dt / self.tau_rep)
        recovery = 1.0 - np.exp(-dt / self.tau_recover)
        
        return decay * recovery
    
    def add_compression(self, event: CompressionEvent):
        """Record a compression event."""
        self.compression_history.append(event)
        # Limit history size
        if len(self.compression_history) > 200:
            self.compression_history = self.compression_history[-200:]
    
    def add_state(self, t: float, r: float, psi: Optional[np.ndarray] = None):
        """
        Record state with distance.
        
        Automatically detects compression events when r < r_critical.
        """
        if r < self.r_critical:
            # Auto-detect compression
            pressure = (self.r_critical / r) ** self.n_power
            self.add_compression(CompressionEvent(
                time=t,
                r_min=r,
                pressure=pressure
            ))
        
        self.state_history.append((t, r, psi.copy() if psi is not None else None))
        if len(self.state_history) > 200:
            self.state_history = self.state_history[-200:]
    
    def compute_repulsion_enhancement(self, t: float, r_current: float) -> float:
        """
        Compute enhancement of repulsive interaction due to compression history.

        The enhancement depends on:
          - Severity of past compression
          - Time since compression
          - Current interparticle distance
        """
        if len(self.compression_history) == 0:
            return 0.0
        
        enhancement = 0.0
        
        for event in self.compression_history:
            dt = t - event.time
            if dt <= 0:
                continue
            
            # Kernel contribution
            K = self.kernel_value(dt)
            
            # Compression severity
            compression_factor = (self.r_critical / event.r_min) ** 2
            
            # Distance-dependent coupling
            distance_coupling = 1.0
            if r_current < self.r_critical:
                distance_coupling = self.r_critical / r_current
            
            enhancement += self.eta_rep * K * compression_factor * distance_coupling
        
        return enhancement
    
    def compute_effective_repulsion(self, r: float, t: float, A: float = 1.0) -> float:
        """
        Compute effective repulsive potential including memory.

        V_eff(r, t) = V_bare(r) * [1 + enhancement(t)]

        where V_bare(r) = A / r^n
        """
        V_bare = A / (r ** self.n_power)
        enhancement = self.compute_repulsion_enhancement(t, r)
        
        return V_bare * (1.0 + enhancement)
    
    def compute_lambda_contribution(self, t: float, psi: np.ndarray, 
                                    r_current: float) -> float:
        """
        Compute contribution to stability parameter from compression history.
        
        For integration with Memory-DFT solvers.
        """
        enhancement = self.compute_repulsion_enhancement(t, r_current)
        
        # State overlap with history (if available)
        overlap_factor = 1.0
        if len(self.state_history) > 0 and psi is not None:
            recent_states = [s for s in self.state_history[-10:] if s[2] is not None]
            if recent_states:
                overlaps = [abs(np.vdot(psi, s[2]))**2 for s in recent_states]
                overlap_factor = np.mean(overlaps)
        
        return enhancement * overlap_factor
    
    def get_hysteresis_curve(self, 
                             r_range: np.ndarray,
                             compression_history: List[float]) -> dict:
        """
        Compute hysteresis curve for compression-expansion cycle.
        
        A non-zero enclosed area signals non-Markovian memory effects.
        
        Args:
            r_range: Array of distances
            compression_history: List of distance values over time
        
        Returns:
            Dict with compression/expansion curves and hysteresis area
        """
        # Reset
        self.compression_history = []
        
        n_steps = len(compression_history)
        t_values = np.arange(n_steps) * 0.1
        
        # Compression phase
        V_compress = []
        for i, r in enumerate(compression_history[:n_steps//2]):
            t = t_values[i]
            self.add_state(t, r)
            V = self.compute_effective_repulsion(r, t)
            V_compress.append(V)
        
        # Expansion phase
        V_expand = []
        for i, r in enumerate(compression_history[n_steps//2:]):
            t = t_values[n_steps//2 + i]
            V = self.compute_effective_repulsion(r, t)
            V_expand.append(V)
        
        # Hysteresis area
        try:
            area = abs(np.trapezoid(V_compress) - np.trapezoid(V_expand))
        except AttributeError:
            # Older numpy
            area = abs(np.trapz(V_compress) - np.trapz(V_expand))
        
        return {
            'r_compress': compression_history[:n_steps//2],
            'r_expand': compression_history[n_steps//2:],
            'V_compress': V_compress,
            'V_expand': V_expand,
            'hysteresis_area': area
        }
    
    def clear(self):
        """Clear all history."""
        self.compression_history = []
        self.state_history = []
    
    def __repr__(self) -> str:
        return (
            f"RepulsiveMemoryKernel(\n"
            f"  eta_rep={self.eta_rep}, tau_rep={self.tau_rep}, "
            f"tau_recover={self.tau_recover}\n"
            f"  r_critical={self.r_critical}, n_power={self.n_power}\n"
            f"  compression_events={len(self.compression_history)}\n"
            f")"
        )


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

class ExtendedCompositeKernel(CompositeMemoryKernel):
    """
    Extended composite memory kernel (backward compatibility alias).
    
    This is now an alias for CompositeMemoryKernel which includes
    the exclusion kernel by default.
    
    Deprecated: Use CompositeMemoryKernel directly.
    """
    
    def __init__(self,
                 w_field: float = 0.30,
                 w_phys: float = 0.25,
                 w_chem: float = 0.25,
                 w_rep: float = 0.20):
        """
        Args:
            w_field: Weight for field kernel
            w_phys: Weight for physical kernel
            w_chem: Weight for chemical kernel
            w_rep: Weight for repulsive/exclusion kernel
        """
        weights = KernelWeights(
            field=w_field,
            phys=w_phys,
            chem=w_chem,
            exclusion=w_rep
        )
        
        super().__init__(
            weights=weights,
            include_exclusion=True
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Repulsive Memory Kernel Test")
    print("=" * 60)
    
    kernel = RepulsiveMemoryKernel(
        eta_rep=0.3,
        tau_rep=3.0,
        tau_recover=10.0,
        r_critical=0.8
    )
    
    print(f"\n{kernel}")
    
    # Simulate compression-expansion cycle
    print("\n--- Compression-Expansion Cycle ---")
    
    # Phase 1: Compression
    print("\nPhase 1: Compression")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        r = 1.0 - 0.15 * t  # Compressing
        psi = np.array([1.0, 0.0])
        kernel.add_state(t, r, psi)
        
        V_eff = kernel.compute_effective_repulsion(r, t)
        enhancement = kernel.compute_repulsion_enhancement(t, r)
        
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f}, enhancement={enhancement:.4f}")
    
    # Phase 2: Expansion
    print("\nPhase 2: Expansion (with memory!)")
    for t in [2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]:
        r = 0.7 + 0.05 * (t - 2.0)
        r = min(r, 1.0)
        
        V_eff = kernel.compute_effective_repulsion(r, t)
        enhancement = kernel.compute_repulsion_enhancement(t, r)
        
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f}, enhancement={enhancement:.4f}")
    
    # Hysteresis analysis
    print("\n--- Hysteresis Analysis ---")
    kernel.clear()
    
    r_compress = np.linspace(1.0, 0.6, 25)
    r_expand = np.linspace(0.6, 1.0, 25)
    r_path = np.concatenate([r_compress, r_expand])
    
    hysteresis = kernel.get_hysteresis_curve(
        r_range=np.linspace(0.6, 1.0, 50),
        compression_history=r_path.tolist()
    )
    
    print(f"  Hysteresis area: {hysteresis['hysteresis_area']:.4f}")
    print(f"  (Non-zero area = Memory effect!)")
    
    # ExtendedCompositeKernel (backward compat)
    print("\n--- ExtendedCompositeKernel (Backward Compat) ---")
    ext_kernel = ExtendedCompositeKernel()
    print(f"  Type: {type(ext_kernel).__name__}")
    print(f"  Weights: {ext_kernel.weights.to_dict()}")
    print(f"  Includes exclusion: {ext_kernel.include_exclusion}")
    
    print("\n" + "=" * 60)
    print("Repulsive Memory Kernel Test Complete!")
    print("=" * 60)
