"""
Memory Kernels for Non-Markovian Density Functional Theory
=========================================================

This module implements a physically motivated decomposition
of memory kernels for Memory-DFT simulations.

The total memory effect is represented as a weighted sum of
four kernel components with distinct physical origins:

  1. Long-range field memory (power-law kernel)
  2. Structural relaxation memory (stretched exponential)
  3. Irreversible chemical memory (step / hysteretic)
  4. Distance-direction memory (exclusion / repulsive)  [NEW]

Physical Insight (v0.4.0)
-------------------------
The same distance r = 0.8 A has DIFFERENT meaning depending on
whether the system is approaching or departing:

  - Approaching (r decreasing): preparing for compression
  - Departing (r increasing): recovering from compression

DFT sees only r = 0.8 A (same).
DSE sees the DIRECTION of change (different).

This is why we need the exclusion kernel - it tracks the
history of distance changes, not just the current distance.

Theoretical Background
----------------------
The total correlation exponent is decomposed as

    gamma_total = gamma_local + gamma_memory

Representative values (Hubbard model, U/t = 2):
  - gamma_total  (all distances)      = 2.604
  - gamma_local  (short-range only)   = 1.388
  - gamma_memory (difference)         = 1.216  (~47%)

Reference:
  S. H. Lie and J. Fullwood,
  Phys. Rev. Lett. 135, 230204 (2025)

Authors:
  Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass, field

# GPU support (optional)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


# =============================================================================
# Base Kernel Class
# =============================================================================

class MemoryKernelBase(ABC):
    """
    Abstract base class for temporal memory kernels.

    A memory kernel K(t, tau) assigns a weight to past states
    at time tau when evaluating the state at time t.
    """
    
    @abstractmethod
    def __call__(self, t: float, tau: float) -> float:
        """Compute K(t, tau)"""
        pass
    
    @abstractmethod
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        """Returns the weight vector for the entire history."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


# =============================================================================
# Component 1: Power-Law Kernel (Field Memory)
# =============================================================================

class PowerLawKernel(MemoryKernelBase):
    """
    Power-law memory kernel for field-induced correlations.

    K(t - tau) = A / (t - tau + epsilon)^gamma

    Physical characteristics:
    - Scale-invariant temporal correlations
    - Long-range memory
    - Strongly non-Markovian behavior

    Typical origins:
    - Electromagnetic fields
    - Collective electronic correlations
    - Radiative environments
    """
    
    def __init__(self, gamma: float = 1.2, amplitude: float = 1.0, epsilon: float = 1.0):
        self.gamma = gamma
        self.amplitude = amplitude
        self.epsilon = epsilon
        
    def __call__(self, t: float, tau: float) -> float:
        dt = t - tau + self.epsilon
        return self.amplitude / (dt ** self.gamma)
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        dt = t - history_times + self.epsilon
        weights = self.amplitude / (dt ** self.gamma)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights
    
    @property
    def name(self) -> str:
        return f"PowerLaw(gamma={self.gamma})"


# =============================================================================
# Component 2: Stretched Exponential Kernel (Structural Memory)
# =============================================================================

class StretchedExpKernel(MemoryKernelBase):
    """
    Stretched exponential kernel for structural relaxation.

    K(t - tau) = exp(-((t - tau) / tau0)^beta)

    Physical characteristics:
    - Non-exponential decay (glassy dynamics)
    - Multiple relaxation timescales
    - Partially non-Markovian

    Typical origins:
    - Structural relaxation
    - Viscoelastic response
    - Disorder-induced memory
    """
    
    def __init__(self, beta: float = 0.5, tau0: float = 10.0, amplitude: float = 1.0):
        self.beta = beta
        self.tau0 = tau0
        self.amplitude = amplitude
        
    def __call__(self, t: float, tau: float) -> float:
        dt = max(t - tau, 0)
        return self.amplitude * np.exp(-((dt / self.tau0) ** self.beta))
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        dt = np.maximum(t - history_times, 0)
        weights = self.amplitude * np.exp(-((dt / self.tau0) ** self.beta))
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights
    
    @property
    def name(self) -> str:
        return f"StretchedExp(beta={self.beta}, tau0={self.tau0})"


# =============================================================================
# Component 3: Step Kernel (Chemical Memory)
# =============================================================================

class StepKernel(MemoryKernelBase):
    """
    Step (sigmoid) kernel for irreversible chemical memory.

    K(t - tau) = A * sigmoid((t - tau - t_react) / width)

    Physical characteristics:
    - Irreversible memory
    - Strong path dependence
    - Non-commutative ordering effects

    Typical origins:
    - Chemical reactions
    - Surface modification
    - Oxidation, corrosion, bond formation
    """
    
    def __init__(self, reaction_time: float = 5.0, amplitude: float = 1.0, 
                 transition_width: float = 1.0):
        self.reaction_time = reaction_time
        self.amplitude = amplitude
        self.transition_width = transition_width
        
    def __call__(self, t: float, tau: float) -> float:
        dt = t - tau
        x = (dt - self.reaction_time) / self.transition_width
        return self.amplitude / (1 + np.exp(-x))
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        dt = t - history_times
        x = (dt - self.reaction_time) / self.transition_width
        weights = self.amplitude / (1 + np.exp(-x))
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights
    
    @property
    def name(self) -> str:
        return f"Step(t_react={self.reaction_time})"


# =============================================================================
# Component 4: Exclusion Kernel (Distance-Direction Memory)
# =============================================================================

class ExclusionKernel(MemoryKernelBase):
    """
    Exclusion (repulsive) kernel for distance-direction memory.

    K(dt) = exp(-dt / tau_rep) * [1 - exp(-dt / tau_recover)]

    Physical insight:
    The same distance r = 0.8 A means DIFFERENT things:
      - Approaching: system is being compressed
      - Departing: system is recovering from compression

    DFT cannot distinguish these. DSE can!

    Physical characteristics:
    - Compression history affects current repulsion
    - Elastic hysteresis
    - Direction-dependent response

    Typical origins:
    - Pauli exclusion principle
    - High-pressure compression
    - White layer formation in machining
    - AFM approach/retract curves
    """
    
    def __init__(self, 
                 tau_rep: float = 3.0, 
                 tau_recover: float = 10.0,
                 amplitude: float = 1.0):
        """
        Args:
            tau_rep: Decay time for compression memory
            tau_recover: Recovery time (elastic return)
            amplitude: Amplitude coefficient
        """
        self.tau_rep = tau_rep
        self.tau_recover = tau_recover
        self.amplitude = amplitude
        
    def __call__(self, t: float, tau: float) -> float:
        dt = t - tau
        if dt <= 0:
            return 0.0
        
        decay = np.exp(-dt / self.tau_rep)
        recovery = 1.0 - np.exp(-dt / self.tau_recover)
        
        return self.amplitude * decay * recovery
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        dt = t - history_times
        dt = np.maximum(dt, 1e-10)  # Avoid division issues
        
        decay = np.exp(-dt / self.tau_rep)
        recovery = 1.0 - np.exp(-dt / self.tau_recover)
        
        weights = self.amplitude * decay * recovery
        weights = np.maximum(weights, 0)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights
    
    @property
    def name(self) -> str:
        return f"Exclusion(tau_rep={self.tau_rep}, tau_recover={self.tau_recover})"


# =============================================================================
# Kernel Weights (4 Components)
# =============================================================================

@dataclass
class KernelWeights:
    """
    Relative weights of different physical memory channels.

    The weights reflect the relative importance of:
      - field: long-range field effects
      - phys: structural relaxation
      - chem: chemical irreversibility
      - exclusion: distance-direction (compression/expansion)
    """
    field: float = 0.30       # Power-law (field)
    phys: float = 0.25        # Stretched exp (structure)
    chem: float = 0.25        # Step (chemical)
    exclusion: float = 0.20   # Exclusion (distance direction) [NEW]
    
    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.field + self.phys + self.chem + self.exclusion
        if total > 0:
            self.field /= total
            self.phys /= total
            self.chem /= total
            self.exclusion /= total
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'field': self.field,
            'phys': self.phys,
            'chem': self.chem,
            'exclusion': self.exclusion
        }


# =============================================================================
# Composite Memory Kernel (4 Components - Unified)
# =============================================================================

class CompositeMemoryKernel:
    """
    Composite memory kernel with four physical components.

    The total kernel is constructed as:

        K(t - tau) = w_field * K_field
                   + w_phys  * K_phys
                   + w_chem  * K_chem
                   + w_excl  * K_exclusion   [NEW in v0.4.0]

    This additive structure allows independent control of
    distinct physical sources of non-Markovianity:

      1. Field: Long-range correlations (power-law)
      2. Phys: Structural relaxation (stretched exp)
      3. Chem: Irreversible reactions (step)
      4. Exclusion: Distance-direction memory (compression history)

    The exclusion component is key for understanding why
    the same distance can have different physical meaning
    depending on whether the system is approaching or departing.
    """
    
    def __init__(self, 
                 weights: Optional[KernelWeights] = None,
                 # Field kernel params
                 gamma_field: float = 1.216,
                 # Phys kernel params
                 beta_phys: float = 0.5,
                 tau0_phys: float = 10.0,
                 # Chem kernel params
                 t_react_chem: float = 5.0,
                 # Exclusion kernel params [NEW]
                 include_exclusion: bool = True,
                 tau_rep: float = 3.0,
                 tau_recover: float = 10.0):
        """
        Args:
            weights: Component weights (default: balanced 4-component)
            gamma_field: Power-law exponent (default: 1.216 from ED)
            beta_phys: Stretched exp exponent
            tau0_phys: Structural relaxation time
            t_react_chem: Chemical reaction time
            include_exclusion: Whether to include exclusion kernel
            tau_rep: Exclusion decay time
            tau_recover: Exclusion recovery time
        """
        self.weights = weights or KernelWeights()
        self.include_exclusion = include_exclusion
        
        # Adjust weights if exclusion is disabled
        if not include_exclusion:
            self.weights.exclusion = 0.0
        
        self.weights.normalize()
        
        # Build component kernels
        self.K_field = PowerLawKernel(gamma=gamma_field)
        self.K_phys = StretchedExpKernel(beta=beta_phys, tau0=tau0_phys)
        self.K_chem = StepKernel(reaction_time=t_react_chem)
        self.K_exclusion = ExclusionKernel(tau_rep=tau_rep, tau_recover=tau_recover)
        
    def __call__(self, t: float, tau: float) -> float:
        """Compute composite kernel value."""
        result = (
            self.weights.field * self.K_field(t, tau) +
            self.weights.phys * self.K_phys(t, tau) +
            self.weights.chem * self.K_chem(t, tau)
        )
        
        if self.include_exclusion:
            result += self.weights.exclusion * self.K_exclusion(t, tau)
        
        return result
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        """Compute integrated weights for history."""
        w_field = self.K_field.integrate(t, history_times)
        w_phys = self.K_phys.integrate(t, history_times)
        w_chem = self.K_chem.integrate(t, history_times)
        
        combined = (
            self.weights.field * w_field +
            self.weights.phys * w_phys +
            self.weights.chem * w_chem
        )
        
        if self.include_exclusion:
            w_excl = self.K_exclusion.integrate(t, history_times)
            combined += self.weights.exclusion * w_excl
        
        # Normalize
        if combined.sum() > 0:
            combined = combined / combined.sum()
            
        return combined
    
    def decompose(self, t: float, history_times: np.ndarray) -> dict:
        """Decompose kernel into components (for diagnostics)."""
        result = {
            'field': self.weights.field * self.K_field.integrate(t, history_times),
            'phys': self.weights.phys * self.K_phys.integrate(t, history_times),
            'chem': self.weights.chem * self.K_chem.integrate(t, history_times),
        }
        
        if self.include_exclusion:
            result['exclusion'] = self.weights.exclusion * self.K_exclusion.integrate(t, history_times)
        
        result['total'] = self.integrate(t, history_times)
        
        return result
    
    @property
    def name(self) -> str:
        return "CompositeMemoryKernel"
    
    def __repr__(self) -> str:
        components = [
            f"  weights: field={self.weights.field:.2f}, phys={self.weights.phys:.2f}, "
            f"chem={self.weights.chem:.2f}, exclusion={self.weights.exclusion:.2f}",
            f"  {self.K_field.name}",
            f"  {self.K_phys.name}",
            f"  {self.K_chem.name}",
        ]
        
        if self.include_exclusion:
            components.append(f"  {self.K_exclusion.name}")
        
        return "CompositeMemoryKernel(\n" + "\n".join(components) + "\n)"


# =============================================================================
# GPU-accelerated Composite Kernel
# =============================================================================

class CompositeMemoryKernelGPU:
    """
    GPU-accelerated implementation of the composite memory kernel.
    
    Designed for large-scale simulations where long history
    windows and fine time resolution are required.
    """
    
    def __init__(self, 
                 weights: Optional[KernelWeights] = None,
                 gamma_field: float = 1.0,
                 beta_phys: float = 0.5,
                 tau0_phys: float = 10.0,
                 t_react_chem: float = 5.0,
                 tau_rep: float = 3.0,
                 tau_recover: float = 10.0,
                 include_exclusion: bool = True,
                 epsilon: float = 1.0):
        
        self.weights = weights or KernelWeights()
        if not include_exclusion:
            self.weights.exclusion = 0.0
        self.weights.normalize()
        
        self.gamma_field = gamma_field
        self.beta_phys = beta_phys
        self.tau0_phys = tau0_phys
        self.t_react_chem = t_react_chem
        self.tau_rep = tau_rep
        self.tau_recover = tau_recover
        self.include_exclusion = include_exclusion
        self.epsilon = epsilon
        
    def integrate_gpu(self, t: float, history_times) -> 'cp.ndarray':
        """GPU-accelerated history weight computation."""
        dt = t - history_times
        
        # Power-law (field)
        w_field = 1.0 / ((dt + self.epsilon) ** self.gamma_field)
        
        # Stretched exp (phys)
        dt_positive = cp.maximum(dt, 0)
        w_phys = cp.exp(-((dt_positive / self.tau0_phys) ** self.beta_phys))
        
        # Step (chem)
        x = (dt - self.t_react_chem)
        w_chem = 1.0 / (1.0 + cp.exp(-x))
        
        # Normalize each
        w_field = w_field / (w_field.sum() + 1e-10)
        w_phys = w_phys / (w_phys.sum() + 1e-10)
        w_chem = w_chem / (w_chem.sum() + 1e-10)
        
        # Combine
        combined = (
            self.weights.field * w_field +
            self.weights.phys * w_phys +
            self.weights.chem * w_chem
        )
        
        # Exclusion kernel
        if self.include_exclusion:
            dt_safe = cp.maximum(dt, 1e-10)
            decay = cp.exp(-dt_safe / self.tau_rep)
            recovery = 1.0 - cp.exp(-dt_safe / self.tau_recover)
            w_excl = decay * recovery
            w_excl = cp.maximum(w_excl, 0)
            w_excl = w_excl / (w_excl.sum() + 1e-10)
            combined += self.weights.exclusion * w_excl
        
        return combined / (combined.sum() + 1e-10)


# =============================================================================
# Repulsive Memory Kernel (Distance-based)
# =============================================================================

class RepulsiveMemoryKernel:
    """
    Repulsive memory kernel for compression hysteresis.
    
    Uses ExclusionKernel internally for temporal memory effects.
    
    Physics: When atoms are compressed, they "remember" the compression
    and show enhanced repulsion during expansion (hysteresis).
    
    V_eff(r, t) = V_bare(r) × [1 + enhancement(history)]
    
    The enhancement depends on:
    - How close we got (minimum r in history)
    - How recently (ExclusionKernel decay)
    
    Key insight:
    Same r, different history → different V_eff!
    This is what DFT cannot capture but DSE can.
    """
    
    def __init__(self, 
                 eta_rep: float = 0.2, 
                 tau_rep: float = 3.0,
                 tau_recover: float = 10.0, 
                 r_critical: float = 0.8,
                 n_power: float = 12.0):
        """
        Args:
            eta_rep: Memory strength coefficient
            tau_rep: Decay time for compression memory
            tau_recover: Recovery time (elastic return)
            r_critical: Critical distance for compression activation
            n_power: Power for bare repulsion (r^-n)
        """
        self.eta_rep = eta_rep
        self.tau_rep = tau_rep
        self.tau_recover = tau_recover
        self.r_critical = r_critical
        self.n_power = n_power
        
        # Use ExclusionKernel for temporal weighting
        self._exclusion_kernel = ExclusionKernel(
            tau_rep=tau_rep,
            tau_recover=tau_recover,
            amplitude=1.0
        )
        
        self.compression_history: List[Tuple[float, float]] = []  # [(t, r), ...]
        self.state_history: List[Tuple[float, float, Optional[np.ndarray]]] = []
        self.r_min_history = float('inf')
    
    def add_state(self, t: float, r: float, psi: Optional[np.ndarray] = None):
        """Record state at time t with distance r."""
        self.state_history.append((t, r, psi))
        self.compression_history.append((t, r))
        if r < self.r_min_history:
            self.r_min_history = r
    
    def compute_repulsion_enhancement(self, t: float, r: float) -> float:
        """
        Compute memory-based enhancement using ExclusionKernel.
        
        Enhancement is higher when:
        1. We compressed below r_critical
        2. The compression was recent (ExclusionKernel decay)
        """
        if not self.compression_history:
            return 0.0
        
        enhancement = 0.0
        
        for t_past, r_past in self.compression_history:
            # Only count compressions below critical distance
            if r_past < self.r_critical:
                # Compression severity
                compression_depth = (self.r_critical - r_past) / self.r_critical
                
                # Use ExclusionKernel for temporal weighting
                kernel_weight = self._exclusion_kernel(t, t_past)
                
                enhancement += self.eta_rep * compression_depth * kernel_weight
        
        return enhancement
    
    def compute_effective_repulsion(self, r: float, t: float) -> float:
        """
        Compute effective repulsion with memory enhancement.
        
        V_eff = V_bare × (1 + enhancement)
        
        Same r, different history → different V_eff!
        """
        V_bare = 1.0 / (r ** self.n_power)
        enhancement = self.compute_repulsion_enhancement(t, r)
        return V_bare * (1.0 + enhancement)
    
    def clear(self):
        """Reset all history."""
        self.compression_history = []
        self.state_history = []
        self.r_min_history = float('inf')


# =============================================================================
# Catalyst Memory Kernel (Specialized)
# =============================================================================

@dataclass
class CatalystEvent:
    """Catalyst reaction event."""
    event_type: str  # 'adsorption', 'reaction', 'desorption'
    time: float
    site: int
    strength: float


class CatalystMemoryKernel:
    """
    Memory kernel specialized for catalytic systems.

    Encodes reaction history and order effects, including:
      - adsorption -> reaction: activation enhancement
      - reaction -> adsorption: deactivation or poisoning

    This kernel explicitly captures non-commutative
    reaction pathways.
    """
    
    def __init__(self, 
                 eta: float = 0.3,
                 tau_ads: float = 2.0,
                 tau_react: float = 5.0):
        self.eta = eta
        self.tau_ads = tau_ads
        self.tau_react = tau_react
        self.events: List[CatalystEvent] = []
        self.state_history: List[Tuple[float, float, np.ndarray]] = []
    
    def add_event(self, event: CatalystEvent):
        """Record a catalyst event."""
        self.events.append(event)
        if len(self.events) > 100:
            self.events = self.events[-100:]
    
    def add_state(self, t: float, lambda_val: float, psi: np.ndarray):
        """Record state for overlap calculations."""
        self.state_history.append((t, lambda_val, psi.copy()))
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def compute_catalyst_memory(self, t: float) -> float:
        """Compute catalyst memory contribution."""
        if len(self.events) == 0:
            return 0.0
        
        memory = 0.0
        
        for event in self.events:
            dt = t - event.time
            if dt <= 0:
                continue
            
            if event.event_type == 'adsorption':
                kernel = np.exp(-dt / self.tau_ads)
            elif event.event_type == 'reaction':
                kernel = np.exp(-dt / self.tau_react) * (1 + 0.5 * dt)
            else:
                kernel = np.exp(-dt / self.tau_ads) * 0.5
            
            memory += self.eta * kernel * event.strength
        
        return memory
    
    def compute_memory_contribution(self, t: float, psi: np.ndarray) -> float:
        """
        Compute memory contribution (compatible with SimpleMemoryKernel API).
        
        Combines catalyst event memory with state overlap effects.
        """
        # Base catalyst memory from events
        catalyst_mem = self.compute_catalyst_memory(t)
        
        # Add state overlap contribution if history available
        overlap_contribution = 0.0
        if len(self.state_history) > 0 and psi is not None:
            for t_hist, lambda_hist, psi_hist in self.state_history[-10:]:
                dt = t - t_hist
                if dt <= 0:
                    continue
                overlap = abs(np.vdot(psi, psi_hist))**2
                kernel = np.exp(-dt / self.tau_react)
                overlap_contribution += self.eta * 0.5 * kernel * lambda_hist * overlap
        
        return catalyst_mem + overlap_contribution
    
    def clear(self):
        """Clear event history."""
        self.events = []
        self.state_history = []


# =============================================================================
# Simple Memory Kernel (Lightweight)
# =============================================================================

class SimpleMemoryKernel:
    """
    Simple memory kernel for quick testing.
    
    Combines exponential decay with a weak polynomial correction.
    """
    
    def __init__(self, eta: float = 0.2, tau: float = 5.0, gamma: float = 0.5):
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.history: list = []
    
    def add_state(self, t: float, lambda_val: float, psi: np.ndarray):
        self.history.append((t, lambda_val, psi.copy()))
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def compute_memory_contribution(self, t: float, psi: np.ndarray) -> float:
        if len(self.history) == 0:
            return 0.0
        
        delta_lambda = 0.0
        
        for t_hist, lambda_hist, psi_hist in self.history:
            dt = t - t_hist
            if dt <= 0:
                continue
            
            kernel = np.exp(-dt / self.tau) * (1 + self.gamma * dt)
            overlap = abs(np.vdot(psi, psi_hist))**2
            delta_lambda += self.eta * kernel * lambda_hist * overlap
        
        return delta_lambda
    
    def clear(self):
        self.history = []


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Memory Kernel Test (v0.5.0 - 4 Components + RepulsiveMemory)")
    print("=" * 70)
    
    # Create composite kernel with all 4 components
    kernel = CompositeMemoryKernel(
        weights=KernelWeights(field=0.3, phys=0.25, chem=0.25, exclusion=0.2),
        gamma_field=1.216,
        beta_phys=0.5,
        tau0_phys=10.0,
        t_react_chem=5.0,
        tau_rep=3.0,
        tau_recover=10.0
    )
    
    print(kernel)
    print()
    
    # Test: weights at t=20
    history_times = np.arange(0, 20, 1.0)
    t_current = 20.0
    
    decomp = kernel.decompose(t_current, history_times)
    
    print(f"t = {t_current}, history length = {len(history_times)}")
    print()
    print("Component contributions:")
    for name, weights in decomp.items():
        if name != 'total':
            print(f"  {name}: max={weights.max():.4f} at tau={history_times[weights.argmax()]:.1f}")
    
    print()
    print("Integrated weights (last 5 steps):")
    for i in range(-5, 0):
        print(f"  tau={history_times[i]:.1f}: weight={decomp['total'][i]:.4f}")
    
    # Exclusion kernel specific test
    print()
    print("=" * 70)
    print("Exclusion Kernel Test (Distance Direction)")
    print("=" * 70)
    
    excl = ExclusionKernel(tau_rep=3.0, tau_recover=10.0)
    
    print("\nK(dt) for different time lags:")
    print("  dt      K(dt)    Interpretation")
    print("  " + "-" * 40)
    for dt in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        K_val = excl(dt, 0)
        if dt < 2:
            interp = "Recent compression, strong effect"
        elif dt < 10:
            interp = "Partial recovery"
        else:
            interp = "Mostly recovered"
        print(f"  {dt:5.1f}   {K_val:.4f}   {interp}")
    
    # RepulsiveMemoryKernel test
    print()
    print("=" * 70)
    print("RepulsiveMemoryKernel Test (Hysteresis)")
    print("=" * 70)
    
    rep_kernel = RepulsiveMemoryKernel(
        eta_rep=0.3,
        tau_rep=3.0,
        tau_recover=10.0,
        r_critical=0.8
    )
    
    # Simulate compression
    print("\nCompression phase (r: 1.2 → 0.6):")
    t = 0.0
    for r in np.linspace(1.2, 0.6, 5):
        rep_kernel.add_state(t, r)
        V_eff = rep_kernel.compute_effective_repulsion(r, t)
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f}")
        t += 1.0
    
    # Simulate expansion
    print("\nExpansion phase (r: 0.6 → 1.2):")
    for r in np.linspace(0.6, 1.2, 5):
        rep_kernel.add_state(t, r)
        V_eff = rep_kernel.compute_effective_repulsion(r, t)
        enhancement = rep_kernel.compute_repulsion_enhancement(t, r)
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f} (enhancement={enhancement:.4f})")
        t += 1.0
    
    # GPU test
    print()
    print("=" * 70)
    print("GPU Kernel Test")
    print("=" * 70)
    
    try:
        kernel_gpu = CompositeMemoryKernelGPU()
        history_gpu = cp.arange(0, 20, 1.0)
        weights_gpu = kernel_gpu.integrate_gpu(20.0, history_gpu)
        print(f"GPU weights shape: {weights_gpu.shape}")
        print(f"GPU weights sum: {float(weights_gpu.sum()):.6f}")
        print("GPU kernel works!")
    except Exception as e:
        print(f"GPU test skipped: {e}")
    
    print()
    print("=" * 70)
    print("Memory Kernel Test Complete!")
    print("=" * 70)
