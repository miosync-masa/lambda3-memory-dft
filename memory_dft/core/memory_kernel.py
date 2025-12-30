"""
Memory Kernel for Memory-DFT
============================

H-CSP環境階層に基づく3成分Memory Kernel:
- Θ_field   → PowerLawKernel    (γ ~ 1.0, 場的・非局所)
- Θ_env_phys → StretchedExpKernel (β ~ 0.5, 構造緩和)
- Θ_env_chem → StepKernel         (不可逆・反応順序)

理論的背景:
  γ_total = γ_local + γ_memory
  
  ED距離分解による導出:
    γ_total (r=∞) = 2.604  ← 全相関
    γ_local (r≤2) = 1.388  ← Markovian (Lie & Fullwood PRL 2025)
    γ_memory      = 1.216  ← Non-Markovian extension (46.7%)
  
  → この差分が Memory kernel の寄与
  → DMRGは不要！ED + 距離フィルターで十分

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)
  "Quantum States Over Time are Uniquely Represented by a CPTP Map"

Author: Masamichi Iizumi, Tamaki Iizumi
Based on: Λ³/H-CSP Theory v2.0
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List
from dataclasses import dataclass

# GPU support (optional)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np  # Fallback to NumPy
    HAS_CUPY = False


# =============================================================================
# Base Kernel Class
# =============================================================================

class MemoryKernelBase(ABC):
    """Memory Kernel の抽象基底クラス"""
    
    @abstractmethod
    def __call__(self, t: float, tau: float) -> float:
        """K(t, τ) を計算"""
        pass
    
    @abstractmethod
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        """履歴全体に対する重みベクトルを返す"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


# =============================================================================
# Θ_field: Power-Law Kernel (場的記憶)
# =============================================================================

class PowerLawKernel(MemoryKernelBase):
    """
    Power-law Memory Kernel
    
    K(t-τ) = A / (t - τ + ε)^γ
    
    特徴:
    - スケール不変
    - 長距離相関
    - 非Markov
    - γ ≈ 1.2 (ED距離分解から導出: γ_memory = 1.216)
    
    H-CSP対応: Θ_field (重力、電磁場、放射線)
    """
    
    def __init__(self, gamma: float = 1.2, amplitude: float = 1.0, epsilon: float = 1.0):
        """
        Args:
            gamma: べき指数 (default: 1.0, 実験から導出)
            amplitude: 振幅係数
            epsilon: 正則化パラメータ (t=τ での発散回避)
        """
        self.gamma = gamma
        self.amplitude = amplitude
        self.epsilon = epsilon
        
    def __call__(self, t: float, tau: float) -> float:
        dt = t - tau + self.epsilon
        return self.amplitude / (dt ** self.gamma)
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        """履歴時刻に対する重みベクトル"""
        dt = t - history_times + self.epsilon
        weights = self.amplitude / (dt ** self.gamma)
        # 正規化
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights
    
    @property
    def name(self) -> str:
        return f"PowerLaw(γ={self.gamma})"


# =============================================================================
# Θ_env_phys: Stretched Exponential Kernel (構造的記憶)
# =============================================================================

class StretchedExpKernel(MemoryKernelBase):
    """
    Stretched Exponential Memory Kernel
    
    K(t-τ) = A * exp(-(（t-τ)/τ₀)^β)
    
    特徴:
    - 緩和現象
    - 時定数分布
    - 温度・歪み依存
    - 半Markov的
    
    H-CSP対応: Θ_env_phys (温度、湿度、外圧)
    """
    
    def __init__(self, beta: float = 0.5, tau0: float = 10.0, amplitude: float = 1.0):
        """
        Args:
            beta: ストレッチ指数 (0 < β < 1 で sub-exponential)
            tau0: 特性時間
            amplitude: 振幅係数
        """
        self.beta = beta
        self.tau0 = tau0
        self.amplitude = amplitude
        
    def __call__(self, t: float, tau: float) -> float:
        dt = t - tau
        if dt < 0:
            return 0.0
        return self.amplitude * np.exp(-((dt / self.tau0) ** self.beta))
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        dt = t - history_times
        dt = np.maximum(dt, 0)
        weights = self.amplitude * np.exp(-((dt / self.tau0) ** self.beta))
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights
    
    @property
    def name(self) -> str:
        return f"StretchedExp(β={self.beta}, τ₀={self.tau0})"


# =============================================================================
# Θ_env_chem: Step/Piecewise Kernel (化学的記憶)
# =============================================================================

class StepKernel(MemoryKernelBase):
    """
    Step/Piecewise Memory Kernel
    
    K(t-τ) = A * Θ(t - τ - t_react)  (ヘヴィサイド関数)
    
    特徴:
    - 不可逆
    - 反応順序依存
    - ヒステリシス
    - 完全非可換
    
    H-CSP対応: Θ_env_chem (酸化、腐食、pH変化)
    """
    
    def __init__(self, reaction_time: float = 5.0, amplitude: float = 1.0, 
                 transition_width: float = 1.0):
        """
        Args:
            reaction_time: 反応が「記憶される」までの時間
            amplitude: 振幅係数
            transition_width: ステップの滑らかさ (σ of sigmoid)
        """
        self.reaction_time = reaction_time
        self.amplitude = amplitude
        self.transition_width = transition_width
        
    def __call__(self, t: float, tau: float) -> float:
        dt = t - tau
        # Smooth step function (sigmoid)
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
# Composite Memory Kernel (統合カーネル)
# =============================================================================

@dataclass
class KernelWeights:
    """H-CSP環境階層に対応する重み"""
    field: float = 0.4    # Θ_field (電子的・場的)
    phys: float = 0.3     # Θ_env_phys (構造的)
    chem: float = 0.3     # Θ_env_chem (化学的)
    
    def normalize(self):
        total = self.field + self.phys + self.chem
        if total > 0:
            self.field /= total
            self.phys /= total
            self.chem /= total


class CompositeMemoryKernel:
    """
    統合 Memory Kernel
    
    K(t-τ) = w_field * K_field + w_phys * K_phys + w_chem * K_chem
    
    H-CSP環境階層:
    - Θ = Θ_field × Θ_env_phys × Θ_env_chem
    
    これが Memory-DFT の核心！
    """
    
    def __init__(self, 
                 weights: Optional[KernelWeights] = None,
                 gamma_field: float = 1.216,
                 beta_phys: float = 0.5,
                 tau0_phys: float = 10.0,
                 t_react_chem: float = 5.0):
        """
        Args:
            weights: 各カーネルの重み (系依存、学習 or 推定)
            gamma_field: Power-law 指数 (default: 1.216, ED距離分解から導出)
            beta_phys: Stretched exp 指数
            tau0_phys: 構造緩和時間
            t_react_chem: 化学反応時間
        """
        self.weights = weights or KernelWeights()
        self.weights.normalize()
        
        # 3つのカーネル (H-CSP環境階層)
        self.K_field = PowerLawKernel(gamma=gamma_field)
        self.K_phys = StretchedExpKernel(beta=beta_phys, tau0=tau0_phys)
        self.K_chem = StepKernel(reaction_time=t_react_chem)
        
    def __call__(self, t: float, tau: float) -> float:
        """統合カーネル値を計算"""
        return (
            self.weights.field * self.K_field(t, tau) +
            self.weights.phys * self.K_phys(t, tau) +
            self.weights.chem * self.K_chem(t, tau)
        )
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        """各成分の重みを統合した履歴重みベクトル"""
        w_field = self.K_field.integrate(t, history_times)
        w_phys = self.K_phys.integrate(t, history_times)
        w_chem = self.K_chem.integrate(t, history_times)
        
        combined = (
            self.weights.field * w_field +
            self.weights.phys * w_phys +
            self.weights.chem * w_chem
        )
        
        # 正規化
        if combined.sum() > 0:
            combined = combined / combined.sum()
            
        return combined
    
    def decompose(self, t: float, history_times: np.ndarray) -> dict:
        """各成分の寄与を分解して返す（診断用）"""
        return {
            'field': self.weights.field * self.K_field.integrate(t, history_times),
            'phys': self.weights.phys * self.K_phys.integrate(t, history_times),
            'chem': self.weights.chem * self.K_chem.integrate(t, history_times),
            'total': self.integrate(t, history_times)
        }
    
    def __repr__(self) -> str:
        return (
            f"CompositeMemoryKernel(\n"
            f"  weights: field={self.weights.field:.2f}, "
            f"phys={self.weights.phys:.2f}, chem={self.weights.chem:.2f}\n"
            f"  {self.K_field.name}\n"
            f"  {self.K_phys.name}\n"
            f"  {self.K_chem.name}\n"
            f")"
        )


# =============================================================================
# GPU-accelerated Kernel (CuPy版)
# =============================================================================

class CompositeMemoryKernelGPU:
    """
    GPU加速版 Memory Kernel
    
    大規模系での高速計算用
    """
    
    def __init__(self, 
                 weights: Optional[KernelWeights] = None,
                 gamma_field: float = 1.0,
                 beta_phys: float = 0.5,
                 tau0_phys: float = 10.0,
                 t_react_chem: float = 5.0,
                 epsilon: float = 1.0):
        
        self.weights = weights or KernelWeights()
        self.weights.normalize()
        
        # パラメータ
        self.gamma_field = gamma_field
        self.beta_phys = beta_phys
        self.tau0_phys = tau0_phys
        self.t_react_chem = t_react_chem
        self.epsilon = epsilon
        
    def integrate_gpu(self, t: float, history_times: cp.ndarray) -> cp.ndarray:
        """GPU上で履歴重みを計算"""
        dt = t - history_times
        
        # Power-law (field)
        w_field = 1.0 / ((dt + self.epsilon) ** self.gamma_field)
        
        # Stretched exp (phys)
        dt_positive = cp.maximum(dt, 0)
        w_phys = cp.exp(-((dt_positive / self.tau0_phys) ** self.beta_phys))
        
        # Step (chem)
        x = (dt - self.t_react_chem)
        w_chem = 1.0 / (1.0 + cp.exp(-x))
        
        # 正規化
        w_field = w_field / (w_field.sum() + 1e-10)
        w_phys = w_phys / (w_phys.sum() + 1e-10)
        w_chem = w_chem / (w_chem.sum() + 1e-10)
        
        # 統合
        combined = (
            self.weights.field * w_field +
            self.weights.phys * w_phys +
            self.weights.chem * w_chem
        )
        
        return combined / (combined.sum() + 1e-10)


# =============================================================================
# Catalyst Memory Kernel (触媒専用)
# =============================================================================

@dataclass
class CatalystEvent:
    """触媒反応イベント"""
    event_type: str  # 'adsorption', 'reaction', 'desorption'
    time: float
    site: int
    strength: float


class CatalystMemoryKernel:
    """
    触媒履歴専用 Memory Kernel
    
    反応順序を記憶:
    - adsorption → reaction: 活性化 (factor > 1)
    - reaction → adsorption: 不活性化 (factor < 1)
    
    H-CSP対応: Θ_env_chem の強化版
    
    Usage:
        kernel = CatalystMemoryKernel(eta=0.3)
        kernel.add_event(CatalystEvent('adsorption', t=1.0, site=0, strength=0.5))
        kernel.add_event(CatalystEvent('reaction', t=2.0, site=1, strength=0.3))
        delta = kernel.compute_memory_contribution(t=3.0, psi)
    """
    
    def __init__(self, eta: float = 0.3, tau_ads: float = 3.0, tau_react: float = 5.0):
        """
        Args:
            eta: Memory strength coefficient
            tau_ads: Adsorption memory timescale
            tau_react: Reaction memory timescale
        """
        self.eta = eta
        self.tau_ads = tau_ads
        self.tau_react = tau_react
        self.events: list = []
        self.history: list = []  # (t, Λ, psi) tuples
    
    def add_event(self, event: CatalystEvent):
        """Record catalyst event"""
        self.events.append(event)
    
    def add_state(self, t: float, lambda_val: float, psi: np.ndarray):
        """Record state in history"""
        self.history.append((t, lambda_val, psi.copy()))
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def compute_memory_contribution(self, t: float, psi: np.ndarray) -> float:
        """
        Compute memory contribution considering catalyst history
        
        Order effects:
        - adsorption before reaction → activation bonus (factor > 1)
        - reaction before adsorption → penalty (factor < 1)
        """
        if len(self.history) == 0:
            return 0.0
        
        delta_lambda = 0.0
        
        # Basic memory contribution
        for t_hist, lambda_hist, psi_hist in self.history:
            dt = t - t_hist
            if dt <= 0:
                continue
            
            kernel = np.exp(-dt / self.tau_react)
            overlap = abs(np.vdot(psi, psi_hist))**2
            delta_lambda += self.eta * kernel * lambda_hist * overlap
        
        # Order effect multiplier
        order_factor = self._compute_order_factor(t)
        delta_lambda *= order_factor
        
        return delta_lambda
    
    def _compute_order_factor(self, t: float) -> float:
        """
        Compute order-dependent factor
        
        adsorption → reaction: factor > 1 (activation)
        reaction → adsorption: factor < 1 (deactivation)
        """
        if len(self.events) < 2:
            return 1.0
        
        recent = [e for e in self.events if e.time <= t]
        if len(recent) < 2:
            return 1.0
        
        last_two = recent[-2:]
        
        if last_two[0].event_type == 'adsorption' and last_two[1].event_type == 'reaction':
            return 1.5  # Activation path
        elif last_two[0].event_type == 'reaction' and last_two[1].event_type == 'adsorption':
            return 0.7  # Deactivation path
        else:
            return 1.0
    
    def get_history_summary(self) -> dict:
        """Get summary of catalyst history"""
        return {
            'n_events': len(self.events),
            'n_states': len(self.history),
            'event_types': [e.event_type for e in self.events],
            'event_times': [e.time for e in self.events]
        }
    
    def clear(self):
        """Clear all history"""
        self.events = []
        self.history = []
    
    def __repr__(self) -> str:
        return (
            f"CatalystMemoryKernel(\n"
            f"  eta={self.eta}, tau_ads={self.tau_ads}, tau_react={self.tau_react}\n"
            f"  events={len(self.events)}, history={len(self.history)}\n"
            f")"
        )


# =============================================================================
# Simple Memory Kernel (for standalone tests)
# =============================================================================

class SimpleMemoryKernel:
    """
    Simplified Memory Kernel for basic tests
    
    K(t, t') = η * exp(-(t-t')/τ) * (1 + γ*(t-t'))
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
    print("="*70)
    print("Memory Kernel Test")
    print("="*70)
    
    # 統合カーネル生成
    kernel = CompositeMemoryKernel(
        weights=KernelWeights(field=0.5, phys=0.3, chem=0.2),
        gamma_field=1.216,  # ED距離分解から導出: γ_memory = 1.216
        beta_phys=0.5,
        tau0_phys=10.0,
        t_react_chem=5.0
    )
    
    print(kernel)
    print()
    
    # テスト: t=20 での履歴重み
    history_times = np.arange(0, 20, 1.0)
    t_current = 20.0
    
    decomp = kernel.decompose(t_current, history_times)
    
    print(f"t = {t_current}, history length = {len(history_times)}")
    print()
    print("各成分の寄与:")
    for name, weights in decomp.items():
        if name != 'total':
            print(f"  {name}: max={weights.max():.4f} at τ={history_times[weights.argmax()]:.1f}")
    
    print()
    print("統合重み (最近の5ステップ):")
    for i in range(-5, 0):
        print(f"  τ={history_times[i]:.1f}: weight={decomp['total'][i]:.4f}")
    
    # GPU版テスト
    print()
    print("="*70)
    print("GPU Kernel Test")
    print("="*70)
    
    try:
        kernel_gpu = CompositeMemoryKernelGPU()
        history_gpu = cp.arange(0, 20, 1.0)
        weights_gpu = kernel_gpu.integrate_gpu(20.0, history_gpu)
        print(f"GPU weights shape: {weights_gpu.shape}")
        print(f"GPU weights sum: {float(weights_gpu.sum()):.6f}")
        print("✅ GPU kernel works!")
    except Exception as e:
        print(f"⚠️ GPU test skipped: {e}")
