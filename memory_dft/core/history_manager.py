"""
Memory Kernel for Memory-DFT
============================

Three-component Memory Kernel based on environment hierarchy:

| Component | Kernel Type      | Physics                    |
|-----------|------------------|----------------------------|
| Field     | PowerLawKernel   | Long-range, scale-invariant|
| Physical  | StretchedExpKernel| Structural relaxation     |
| Chemical  | StepKernel       | Irreversible reactions     |

Theoretical Background:
  The total correlation exponent decomposes as:
  
    γ_total = γ_local + γ_memory
  
  From exact diagonalization with distance filtering:
    γ_total (r=∞) = 2.604   ← Full correlations
    γ_local (r≤2) = 1.388   ← Markovian sector
    γ_memory      = 1.216   ← Non-Markovian extension (46.7%)
  
  This decomposition shows that nearly half of quantum correlations
  require history-dependent treatment beyond standard DFT.

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

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
    """Abstract base class for memory kernels."""
    
    @abstractmethod
    def __call__(self, t: float, tau: float) -> float:
        """Compute kernel value K(t, τ)."""
        pass
    
    @abstractmethod
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        """Return weight vector over entire history."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


# =============================================================================
# Field Component: Power-Law Kernel
# =============================================================================

class PowerLawKernel(MemoryKernelBase):
    """
    Power-law Memory Kernel for long-range correlations.
    
    K(t-τ) = A / (t - τ + ε)^γ
    
    Physical characteristics:
    - Scale-invariant decay
    - Long-range temporal correlations  
    - Non-Markovian dynamics
    - γ ≈ 1.216 (derived from ED distance decomposition)
    
    Corresponds to field-like environmental coupling
    (e.g., electromagnetic fields, gravitational effects).
    """
    
    def __init__(self, 
                 gamma: float = 1.216,
                 amplitude: float = 1.0,
                 epsilon: float = 0.1):
        """
        Args:
            gamma: Power-law exponent (default 1.216 from ED decomposition)
            amplitude: Overall strength
            epsilon: Short-time regularization
        """
        self.gamma = gamma
        self.amplitude = amplitude
        self.epsilon = epsilon
    
    def __call__(self, t: float, tau: float) -> float:
        dt = t - tau
        if dt <= 0:
            return 0.0
        return self.amplitude / (dt + self.epsilon) ** self.gamma
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        dt = t - history_times
        mask = dt > 0
        weights = np.zeros_like(history_times)
        weights[mask] = self.amplitude / (dt[mask] + self.epsilon) ** self.gamma
        return weights
    
    @property
    def name(self) -> str:
        return f"PowerLaw(γ={self.gamma:.3f})"


# =============================================================================
# Physical Environment: Stretched Exponential Kernel
# =============================================================================

class StretchedExpKernel(MemoryKernelBase):
    """
    Stretched exponential kernel for structural relaxation.
    
    K(t-τ) = A · exp(-(|t-τ|/τ₀)^β)
    
    Physical characteristics:
    - Multi-timescale relaxation
    - β < 1: sub-diffusive (glassy dynamics)
    - β = 1: simple exponential (Debye relaxation)
    - β > 1: compressed exponential (cooperative motion)
    
    Corresponds to physical environment effects
    (e.g., temperature-driven relaxation, structural rearrangement).
    """
    
    def __init__(self,
                 tau0: float = 10.0,
                 beta: float = 0.5,
                 amplitude: float = 1.0):
        """
        Args:
            tau0: Characteristic relaxation time
            beta: Stretching exponent (0.5 typical for glasses)
            amplitude: Overall strength
        """
        self.tau0 = tau0
        self.beta = beta
        self.amplitude = amplitude
    
    def __call__(self, t: float, tau: float) -> float:
        dt = abs(t - tau)
        if dt == 0:
            return self.amplitude
        return self.amplitude * np.exp(-(dt / self.tau0) ** self.beta)
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        dt = np.abs(t - history_times)
        return self.amplitude * np.exp(-(dt / self.tau0) ** self.beta)
    
    @property
    def name(self) -> str:
        return f"StretchedExp(β={self.beta:.2f}, τ₀={self.tau0:.1f})"


# =============================================================================
# Chemical Environment: Step Kernel
# =============================================================================

class StepKernel(MemoryKernelBase):
    """
    Step-function kernel for irreversible chemical processes.
    
    K(t-τ) = A · Θ(t - τ - t_react)
    
    Physical characteristics:
    - Sharp onset at reaction time
    - Irreversible state change
    - No decay (permanent memory)
    
    Corresponds to chemical environment effects
    (e.g., oxidation, corrosion, phase transitions).
    """
    
    def __init__(self,
                 t_react: float = 5.0,
                 amplitude: float = 1.0):
        """
        Args:
            t_react: Reaction/activation time
            amplitude: Post-reaction memory strength
        """
        self.t_react = t_react
        self.amplitude = amplitude
    
    def __call__(self, t: float, tau: float) -> float:
        if t - tau >= self.t_react:
            return self.amplitude
        return 0.0
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        mask = (t - history_times) >= self.t_react
        weights = np.zeros_like(history_times)
        weights[mask] = self.amplitude
        return weights
    
    @property
    def name(self) -> str:
        return f"Step(t_react={self.t_react:.1f})"


# =============================================================================
# Kernel Composition Weights
# =============================================================================

@dataclass
class KernelWeights:
    """
    Weights for combining memory kernel components.
    
    The composite kernel is:
      K_total = w_field·K_field + w_phys·K_phys + w_chem·K_chem
    
    Physical interpretation:
    - field: Long-range field effects (EM, gravitational)
    - phys: Local physical environment (T, P, humidity)
    - chem: Chemical environment (O₂, pH, corrosion)
    """
    field: float = 0.4
    phys: float = 0.4
    chem: float = 0.2
    
    def __post_init__(self):
        total = self.field + self.phys + self.chem
        if abs(total - 1.0) > 1e-6:
            self.field /= total
            self.phys /= total
            self.chem /= total
    
    def as_tuple(self) -> tuple:
        return (self.field, self.phys, self.chem)


# =============================================================================
# Composite Memory Kernel
# =============================================================================

class CompositeMemoryKernel:
    """
    Three-component composite memory kernel.
    
    Combines field, physical, and chemical memory effects
    with separate characteristic parameters for each.
    
    K_total(t,τ) = Σᵢ wᵢ · Kᵢ(t,τ)
    
    This decomposition reflects the hierarchical structure
    of environmental influences on quantum dynamics.
    """
    
    def __init__(self,
                 weights: Optional[KernelWeights] = None,
                 gamma_field: float = 1.216,
                 beta_phys: float = 0.5,
                 tau0_phys: float = 10.0,
                 t_react_chem: float = 5.0):
        self.weights = weights or KernelWeights()
        
        self.kernel_field = PowerLawKernel(gamma=gamma_field)
        self.kernel_phys = StretchedExpKernel(tau0=tau0_phys, beta=beta_phys)
        self.kernel_chem = StepKernel(t_react=t_react_chem)
    
    def __call__(self, t: float, tau: float) -> float:
        w = self.weights
        return (w.field * self.kernel_field(t, tau) +
                w.phys * self.kernel_phys(t, tau) +
                w.chem * self.kernel_chem(t, tau))
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        w = self.weights
        return (w.field * self.kernel_field.integrate(t, history_times) +
                w.phys * self.kernel_phys.integrate(t, history_times) +
                w.chem * self.kernel_chem.integrate(t, history_times))
    
    def decompose(self, t: float, history_times: np.ndarray) -> dict:
        return {
            'field': self.kernel_field.integrate(t, history_times),
            'phys': self.kernel_phys.integrate(t, history_times),
            'chem': self.kernel_chem.integrate(t, history_times),
            'total': self.integrate(t, history_times),
            'weights': self.weights.as_tuple()
        }
    
    def __repr__(self) -> str:
        return (
            f"CompositeMemoryKernel(\n"
            f"  weights: field={self.weights.field:.2f}, "
            f"phys={self.weights.phys:.2f}, chem={self.weights.chem:.2f}\n"
            f"  field:  {self.kernel_field.name}\n"
            f"  phys:   {self.kernel_phys.name}\n"
            f"  chem:   {self.kernel_chem.name}\n"
            f")"
        )


# =============================================================================
# GPU-Accelerated Composite Kernel
# =============================================================================

class CompositeMemoryKernelGPU:
    """GPU-accelerated composite memory kernel using CuPy."""
    
    def __init__(self,
                 weights: Optional[KernelWeights] = None,
                 gamma_field: float = 1.216,
                 beta_phys: float = 0.5,
                 tau0_phys: float = 10.0,
                 t_react_chem: float = 5.0):
        self.weights = weights or KernelWeights()
        self.gamma = gamma_field
        self.beta = beta_phys
        self.tau0 = tau0_phys
        self.t_react = t_react_chem
        self.epsilon = 0.1
        self.xp = cp if HAS_CUPY else np
    
    def integrate(self, t: float, history_times) -> np.ndarray:
        xp = self.xp
        history_times = xp.asarray(history_times)
        dt = t - history_times
        
        mask_field = dt > 0
        K_field = xp.zeros_like(dt)
        K_field[mask_field] = 1.0 / (dt[mask_field] + self.epsilon) ** self.gamma
        
        K_phys = xp.exp(-(xp.abs(dt) / self.tau0) ** self.beta)
        K_chem = xp.where(dt >= self.t_react, 1.0, 0.0)
        
        w = self.weights
        total = w.field * K_field + w.phys * K_phys + w.chem * K_chem
        
        if HAS_CUPY and isinstance(total, cp.ndarray):
            return cp.asnumpy(total)
        return total


# =============================================================================
# Simple Memory Kernel (Standalone Use)
# =============================================================================

class SimpleMemoryKernel:
    """
    Simplified memory kernel for standalone tests.
    
    K(t, t') = η · exp(-(t-t')/τ) · (1 + γ·(t-t'))
    
    Combines exponential decay with linear growth,
    providing a minimal model of history-dependent effects.
    """
    
    def __init__(self, eta: float = 0.3, tau: float = 5.0, gamma: float = 0.5):
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.state_history: List[Tuple[float, float, np.ndarray]] = []
    
    def add_state(self, t: float, lambda_val: float, psi: np.ndarray):
        self.state_history.append((t, lambda_val, psi.copy()))
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def compute_memory_contribution(self, t: float, psi: np.ndarray) -> float:
        if len(self.state_history) == 0:
            return 0.0
        
        contribution = 0.0
        for t_hist, lambda_hist, psi_hist in self.state_history:
            dt = t - t_hist
            if dt <= 0:
                continue
            K = self.eta * np.exp(-dt / self.tau) * (1 + self.gamma * dt)
            overlap = abs(np.vdot(psi, psi_hist)) ** 2
            contribution += K * lambda_hist * overlap
        
        return contribution
    
    def clear(self):
        self.state_history = []


# =============================================================================
# Catalyst Memory Kernel
# =============================================================================

@dataclass
class CatalystEvent:
    """Record of a catalytic event."""
    event_type: str   # 'adsorption' or 'reaction'
    time: float
    site: int
    strength: float


class CatalystMemoryKernel:
    """
    Memory kernel specialized for catalytic processes.
    
    Tracks the order of adsorption and reaction events,
    capturing the non-commutativity of reaction pathways:
    
      Adsorption → Reaction ≠ Reaction → Adsorption
    
    This path-dependence is crucial for heterogeneous catalysis.
    """
    
    def __init__(self, eta: float = 0.3, tau_ads: float = 3.0, tau_react: float = 5.0):
        self.eta = eta
        self.tau_ads = tau_ads
        self.tau_react = tau_react
        self.events: List[CatalystEvent] = []
        self.state_history: List[Tuple[float, float, np.ndarray]] = []
    
    def add_event(self, event: CatalystEvent):
        self.events.append(event)
    
    def add_state(self, t: float, lambda_val: float, psi: np.ndarray):
        self.state_history.append((t, lambda_val, psi.copy()))
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def compute_memory_contribution(self, t: float, psi: np.ndarray) -> float:
        if len(self.events) == 0 and len(self.state_history) == 0:
            return 0.0
        
        contribution = 0.0
        
        for event in self.events:
            dt = t - event.time
            if dt <= 0:
                continue
            tau = self.tau_ads if event.event_type == 'adsorption' else self.tau_react
            K = self.eta * np.exp(-dt / tau) * event.strength
            contribution += K
        
        order_factor = self._compute_order_factor()
        contribution *= order_factor
        
        for t_hist, lambda_hist, psi_hist in self.state_history[-10:]:
            dt = t - t_hist
            if dt > 0:
                overlap = abs(np.vdot(psi, psi_hist)) ** 2
                contribution += 0.1 * self.eta * np.exp(-dt / self.tau_react) * overlap * lambda_hist
        
        return contribution
    
    def _compute_order_factor(self) -> float:
        """Non-commutative order factor for reaction pathways."""
        if len(self.events) < 2:
            return 1.0
        
        factor = 1.0
        for i in range(1, len(self.events)):
            prev_type = self.events[i-1].event_type
            curr_type = self.events[i].event_type
            
            if prev_type == 'adsorption' and curr_type == 'reaction':
                factor *= 1.5  # Activation
            elif prev_type == 'reaction' and curr_type == 'adsorption':
                factor *= 0.7  # Passivation
        
        return factor
    
    def clear(self):
        self.events = []
        self.state_history = []

# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("History Manager Test")
    print("="*70)
    
    # CPU版テスト
    manager = HistoryManager(max_history=100)
    
    # ダミー履歴を追加
    for t in range(50):
        state = np.random.randn(16) + 1j * np.random.randn(16)
        state = state / np.linalg.norm(state)
        
        lambda_val = 0.5 + 0.5 * np.sin(t / 10)  # 時間変動するΛ
        
        manager.add(
            time=float(t),
            state=state,
            energy=-1.0 + 0.1 * t,
            lambda_density=lambda_val,
            observables={'J_x': np.random.randn()}
        )
    
    print(f"History size: {len(manager.history)}")
    print(f"Statistics: {manager.get_statistics()}")
    
    # Memory積分テスト
    from memory_kernel import CompositeMemoryKernel
    kernel = CompositeMemoryKernel()
    
    integral = manager.compute_memory_integral(kernel, t_current=50.0, observable_key='J_x')
    print(f"Memory integral (J_x): {integral:.6f}")
    
    memory_state = manager.compute_memory_state(kernel, t_current=50.0)
    print(f"Memory state norm: {np.linalg.norm(memory_state):.6f}")
    
    # GPU版テスト
    print()
    print("="*70)
    print("GPU History Manager Test")
    print("="*70)
    
    try:
        dim = 1024
        manager_gpu = HistoryManagerGPU(max_history=100, state_dim=dim)
        
        for t in range(50):
            state = cp.random.randn(dim) + 1j * cp.random.randn(dim)
            state = state / cp.linalg.norm(state)
            manager_gpu.add_fast(float(t), state, lambda_density=1.0)
            
        from memory_kernel import CompositeMemoryKernelGPU
        kernel_gpu = CompositeMemoryKernelGPU()
        
        mem_state = manager_gpu.compute_memory_state_fast(kernel_gpu, 50.0)
        print(f"GPU memory state norm: {float(cp.linalg.norm(mem_state)):.6f}")
        print("✅ GPU history manager works!")
        
    except Exception as e:
        print(f"⚠️ GPU test skipped: {e}")
