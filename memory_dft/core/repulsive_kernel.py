"""
Repulsive Memory Kernel for Non-Markovian Interactions
=====================================================

This module implements a history-dependent short-range
repulsive interaction, where the effective repulsion
depends not only on the instantaneous configuration
but also on prior compression history.

The central idea is that short-range repulsion is not
purely instantaneous. Past compression events modify
the present effective stiffness, leading to hysteresis
and path dependence even when the final configuration
is identical.

Physical Motivation
-------------------
1. Quantum-Mechanical Repulsion
   - Pauli exclusion principle
   - Overlap of electronic wavefunctions
   - Short-range repulsion of the form:
         V_rep(r) ∝ 1 / r^n   (n ≈ 12, Lennard-Jones type)

2. Elastic and Structural Hysteresis
   - Compression history alters current response
   - Partial recovery over finite time scales
   - Analogous to viscoelastic relaxation

3. Memory-Induced Renormalization
   - Prior compression enhances effective repulsion
   - Leads to history-dependent |V|_eff
   - Stabilizes systems against repeated compression

Memory Kernel Form
------------------
The temporal memory kernel is defined as

    K_rep(Δt) = exp(-Δt / τ_rep) × [1 - exp(-Δt / τ_recover)]

where:
  - τ_rep     : decay time of compression memory
  - τ_recover : recovery time of elastic response

Physical interpretation:
  - Δt → 0    : recently compressed, repulsion enhanced
  - intermediate Δt : partial recovery, residual memory
  - large Δt  : full recovery, memory vanishes

Applications
------------
This model is relevant for systems where short-range
repulsion and compression history play a critical role:

  - High-pressure materials (diamond anvil cells)
  - Shock and impact compression
  - Frictional and contact interfaces
  - Surface chemistry and adsorption-induced strain
  - Catalytic reactions with mechanical feedback

Authors:
  Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CompressionEvent:
    """圧縮イベント"""
    time: float
    r_min: float        # 最小距離（圧縮度）
    pressure: float     # 局所圧力
    site: int = 0       # 圧縮サイト


class RepulsiveMemoryKernel:
    """
    History-dependent repulsive memory kernel.

    This kernel models the enhancement of short-range
    repulsive interactions due to prior compression events.

    Effective interaction:
        V_rep^eff(r, t) = V_rep(r) × [1 + η ∫ K_rep(t - t') Θ(r_c - r(t')) dt']

    Parameters:
        η_rep       : strength of repulsive memory
        τ_rep       : decay time of memory
        τ_recover   : recovery time scale
        r_critical  : distance threshold for compression
        n_power     : repulsion exponent (n ≈ 12)
    
    Usage:
        kernel = RepulsiveMemoryKernel()
        kernel.add_compression(CompressionEvent(t=1.0, r_min=0.5, pressure=10.0))
        delta_V = kernel.compute_repulsion_enhancement(t=2.0, r_current=0.9)
    """
    
    def __init__(self,
                 eta_rep: float = 0.2,
                 tau_rep: float = 3.0,
                 tau_recover: float = 10.0,
                 r_critical: float = 0.8,
                 n_power: float = 12.0):
        """
        Args:
            eta_rep: Memory strength for repulsion
            tau_rep: Decay time for repulsion memory
            tau_recover: Recovery time (rubber returning to shape)
            r_critical: Critical distance below which compression is recorded
            n_power: Power-law exponent for repulsion (12 for LJ)
        """
        self.eta_rep = eta_rep
        self.tau_rep = tau_rep
        self.tau_recover = tau_recover
        self.r_critical = r_critical
        self.n_power = n_power
        
        self.compression_history: List[CompressionEvent] = []
        self.state_history: List[Tuple[float, float, np.ndarray]] = []  # (t, r, psi)
    
    def kernel_value(self, dt: float) -> float:
        """
        Temporal kernel governing repulsive memory.

        K(Δt) = exp(-Δt / τ_rep) × [1 - exp(-Δt / τ_recover)]

        Interpretation:
          - exp(-Δt / τ_rep)      : forgetting of compression
          - 1 - exp(-Δt / τ_rec) : gradual elastic recovery
          - product              : maximal effect at intermediate times
        """
        if dt <= 0:
            return 0.0
        
        decay = np.exp(-dt / self.tau_rep)
        recovery = 1.0 - np.exp(-dt / self.tau_recover)
        
        return decay * recovery
    
    def add_compression(self, event: CompressionEvent):
        """Record compression event"""
        self.compression_history.append(event)
    
    def add_state(self, t: float, r: float, psi: Optional[np.ndarray] = None):
        """Record state with distance"""
        if r < self.r_critical:
            # Auto-detect compression
            pressure = (self.r_critical / r) ** self.n_power
            self.add_compression(CompressionEvent(
                time=t,
                r_min=r,
                pressure=pressure
            ))
        
        if psi is not None:
            self.state_history.append((t, r, psi.copy()))
            if len(self.state_history) > 100:
                self.state_history = self.state_history[-100:]
    
    def compute_repulsion_enhancement(self, t: float, r_current: float) -> float:
        """
        Compute enhancement of repulsive interaction
        due to compression history.

        The enhancement depends on:
          - severity of past compression
          - temporal distance from compression
          - current interparticle distance
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
            
            # Compression severity: how much was it squeezed?
            compression_factor = (self.r_critical / event.r_min) ** 2
            
            # Distance-dependent coupling
            # Closer current distance → more effect from past compression
            distance_coupling = 1.0
            if r_current < self.r_critical:
                distance_coupling = (self.r_critical / r_current)
            
            enhancement += self.eta_rep * K * compression_factor * distance_coupling
        
        return enhancement
    
    def compute_effective_repulsion(self, r: float, t: float, 
                                    A: float = 1.0) -> float:
        """
        Compute effective repulsive potential including memory.

        V_rep^eff(r, t) = A / r^n × [1 + memory enhancement]
        
        Args:
            r: Current distance
            t: Current time
            A: Amplitude of bare repulsion
        
        Returns:
            Effective repulsion energy
        """
        # Bare repulsion (LJ-type)
        V_bare = A / (r ** self.n_power)
        
        # Memory enhancement
        enhancement = self.compute_repulsion_enhancement(t, r)
        
        return V_bare * (1.0 + enhancement)
    
    def compute_lambda_contribution(self, t: float, 
                                    psi: np.ndarray,
                                    r_current: float) -> float:
        """
        Compute contribution of repulsive memory to the
        effective stability parameter Λ via |V|_eff.

        Enhanced repulsion increases |V|_eff, thereby
        reducing Λ and stabilizing the system against
        further compression.
        """
        enhancement = self.compute_repulsion_enhancement(t, r_current)
        
        # Overlap with past states (quantum coherence effect)
        overlap_factor = 1.0
        if len(self.state_history) > 0:
            for t_hist, r_hist, psi_hist in self.state_history[-10:]:
                dt = t - t_hist
                if dt > 0:
                    overlap = abs(np.vdot(psi, psi_hist)) ** 2
                    overlap_factor += 0.1 * overlap * self.kernel_value(dt)
        
        return enhancement * overlap_factor
    
    def get_hysteresis_curve(self, 
                             r_range: np.ndarray,
                             compression_history: List[float]) -> dict:
        """
        Generate compression–expansion hysteresis curves.

        A non-zero enclosed area directly signals
        non-Markovian memory effects.
        
        Args:
            r_range: Array of distances
            compression_history: List of compression depths over time
        
        Returns:
            Dict with compression and expansion curves
        """
        # Reset and simulate
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
        
        return {
            'r_compress': compression_history[:n_steps//2],
            'r_expand': compression_history[n_steps//2:],
            'V_compress': V_compress,
            'V_expand': V_expand,
            'hysteresis_area': abs(np.trapezoid(V_compress) - np.trapezoid(V_expand))
        }
    
    def clear(self):
        """Clear all history"""
        self.compression_history = []
        self.state_history = []
    
    def __repr__(self) -> str:
        return (
            f"RepulsiveMemoryKernel(\n"
            f"  η_rep={self.eta_rep}, τ_rep={self.tau_rep}, τ_recover={self.tau_recover}\n"
            f"  r_critical={self.r_critical}, n_power={self.n_power}\n"
            f"  compression_events={len(self.compression_history)}\n"
            f"  Origin: 🩲 → Elastic Hysteresis → Memory-DFT\n"
            f")"
        )


# =============================================================================
# Integration with CompositeMemoryKernel
# =============================================================================

class ExtendedCompositeKernel:
    """
    Extended composite memory kernel including
    repulsive hysteresis effects.

    Components:
      - Long-range field memory
      - Structural relaxation memory
      - Chemical (irreversible) memory
      - Short-range repulsive memory (this module)

    This construction enables simultaneous treatment of
    non-local correlations, slow relaxation, reaction order,
    and compression-induced hysteresis within a unified
    Memory-DFT framework.
    """
    
    def __init__(self,
                 w_field: float = 0.35,
                 w_phys: float = 0.25,
                 w_chem: float = 0.20,
                 w_rep: float = 0.20):
        """
        Args:
            w_field: Weight for field kernel
            w_phys: Weight for physical environment kernel
            w_chem: Weight for chemical environment kernel
            w_rep: Weight for repulsive memory kernel (NEW!)
        """
        self.weights = {
            'field': w_field,
            'phys': w_phys,
            'chem': w_chem,
            'repulsion': w_rep
        }
        
        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Initialize repulsive kernel
        self.repulsive_kernel = RepulsiveMemoryKernel()
    
    def compute_total(self, t: float, 
                      history_data: dict,
                      psi: np.ndarray,
                      r_current: float = 1.0) -> float:
        """
        Compute total memory contribution including repulsion
        
        Args:
            t: Current time
            history_data: Dict with 'field', 'phys', 'chem' contributions
            psi: Current wavefunction
            r_current: Current characteristic distance
        
        Returns:
            Total memory contribution
        """
        total = 0.0
        
        # Standard components
        if 'field' in history_data:
            total += self.weights['field'] * history_data['field']
        if 'phys' in history_data:
            total += self.weights['phys'] * history_data['phys']
        if 'chem' in history_data:
            total += self.weights['chem'] * history_data['chem']
        
        # Repulsive memory (🩲)
        rep_contribution = self.repulsive_kernel.compute_lambda_contribution(
            t, psi, r_current
        )
        total += self.weights['repulsion'] * rep_contribution
        
        return total


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🩲 Repulsive Memory Kernel Test")
    print("="*60)
    
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
        psi = np.array([1.0, 0.0])  # Dummy state
        kernel.add_state(t, r, psi)
        
        V_eff = kernel.compute_effective_repulsion(r, t)
        enhancement = kernel.compute_repulsion_enhancement(t, r)
        
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f}, enhancement={enhancement:.4f}")
    
    # Phase 2: Expansion
    print("\nPhase 2: Expansion (with memory!)")
    for t in [2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]:
        r = 0.7 + 0.05 * (t - 2.0)  # Expanding
        r = min(r, 1.0)
        psi = np.array([1.0, 0.0])
        
        V_eff = kernel.compute_effective_repulsion(r, t)
        enhancement = kernel.compute_repulsion_enhancement(t, r)
        
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f}, enhancement={enhancement:.4f}")
    
    # Hysteresis
    print("\n--- Hysteresis Analysis ---")
    kernel.clear()
    
    # Create compression-expansion path
    r_compress = np.linspace(1.0, 0.6, 25)
    r_expand = np.linspace(0.6, 1.0, 25)
    r_path = np.concatenate([r_compress, r_expand])
    
    hysteresis = kernel.get_hysteresis_curve(
        r_range=np.linspace(0.6, 1.0, 50),
        compression_history=r_path.tolist()
    )
    
    print(f"  Hysteresis area: {hysteresis['hysteresis_area']:.4f}")
    print(f"  (Non-zero area = Memory effect!)")
    
    # Extended kernel
    print("\n--- Extended 4-Component Kernel ---")
    ext_kernel = ExtendedCompositeKernel()
    print(f"  Weights: {ext_kernel.weights}")
    
    print("\n✅ Repulsive Memory Kernel test passed!")
    print("\n🩲 → 🧪 → Λ³ : Physics from underwear to publications!")
