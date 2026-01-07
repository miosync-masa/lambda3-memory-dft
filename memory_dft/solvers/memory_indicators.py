"""
Memory Indicators for DSE
=========================

DSE 時間発展の結果を分析するメトリクス

【3つの指標】
  1. ΔO (Path non-commutativity) - 経路非可換性
  2. M(t) (Temporal memory) - 時間的記憶
  3. γ_memory (Gamma decomposition) - 非マルコフ成分

【使用例】
  from memory_dft.physics.memory_indicators import MemoryIndicator, MemoryMetrics
  from memory_dft.solvers.dse_solver import DSESolver
  
  # DSE 実行
  solver = DSESolver(H_K, H_V)
  result1 = solver.run(psi0, t_end=10)
  result2 = solver.run(psi0, t_end=10)  # 別の経路
  
  # メモリ指標を計算
  indicator = MemoryIndicator()
  metrics = indicator.from_dse_results(result1, result2)
  
  print(metrics)
  print(f"Non-Markovian? {metrics.is_non_markovian()}")

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from memory_dft.solvers.dse_solver import DSEResult


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MemoryMetrics:
    """メモリ指標のコンテナ"""
    delta_O: float                          # 経路非可換性
    M_temporal: float                       # 時間的記憶
    autocorr_time: float = 0.0              # 特性記憶時間
    gamma_memory: Optional[float] = None    # γ分解（あれば）
    
    # 追加診断
    delta_energy: float = 0.0               # エネルギー差
    delta_lambda: float = 0.0               # λ差
    
    def __repr__(self):
        s = f"MemoryMetrics:\n"
        s += f"  ΔO (path non-commutativity) = {self.delta_O:.6f}\n"
        s += f"  ΔE (energy difference)      = {self.delta_energy:.6f}\n"
        s += f"  Δλ (lambda difference)      = {self.delta_lambda:.6f}\n"
        s += f"  M (temporal memory)         = {self.M_temporal:.6f}\n"
        s += f"  τ_memory (autocorr time)    = {self.autocorr_time:.4f}\n"
        if self.gamma_memory is not None:
            s += f"  γ_memory (Non-Markovian)    = {self.gamma_memory:.4f}\n"
        return s
    
    def is_non_markovian(self, threshold: float = 0.01) -> bool:
        """有意な記憶効果があるか"""
        return (self.delta_O > threshold or 
                self.M_temporal > threshold or
                self.delta_lambda > threshold)
    
    def summary(self) -> str:
        """サマリー文字列"""
        status = "Non-Markovian ✅" if self.is_non_markovian() else "Markovian"
        return f"Memory: ΔO={self.delta_O:.4f}, Δλ={self.delta_lambda:.4f}, M={self.M_temporal:.4f} [{status}]"


# =============================================================================
# Memory Indicator Calculator
# =============================================================================

class MemoryIndicator:
    """
    DSE 結果からメモリ指標を計算
    
    Usage:
        indicator = MemoryIndicator()
        
        # DSE 結果から
        metrics = indicator.from_dse_results(result1, result2)
        
        # 手動で
        delta = indicator.path_noncommutativity(O1, O2)
        M, tau = indicator.temporal_memory(series, dt)
    """
    
    # =========================================================================
    # DSE Integration
    # =========================================================================
    
    @classmethod
    def from_dse_results(cls, 
                         result1: 'DSEResult',
                         result2: 'DSEResult',
                         observable_key: Optional[str] = None) -> MemoryMetrics:
        """
        2つの DSE 結果からメモリ指標を計算
        
        Args:
            result1, result2: DSEResult（異なる経路）
            observable_key: 比較する物理量（None なら λ を使用）
            
        Returns:
            MemoryMetrics
        """
        indicator = cls()
        
        # λ の差
        delta_lambda = indicator.path_noncommutativity(
            result1.final_lambda, result2.final_lambda
        )
        
        # エネルギーの差
        delta_energy = indicator.path_noncommutativity(
            result1.final_energy, result2.final_energy
        )
        
        # 物理量の差（指定されていれば）
        if observable_key and observable_key in result1.observables:
            O1 = result1.observables[observable_key][-1]
            O2 = result2.observables[observable_key][-1]
            delta_O = indicator.path_noncommutativity(O1, O2)
        else:
            delta_O = delta_lambda
        
        # 時間的記憶（result1 の λ 系列から）
        dt = result1.times[1] - result1.times[0] if len(result1.times) > 1 else 0.1
        M_temporal, tau = indicator.temporal_memory(
            np.array(result1.lambdas), dt
        )
        
        return MemoryMetrics(
            delta_O=delta_O,
            delta_energy=delta_energy,
            delta_lambda=delta_lambda,
            M_temporal=M_temporal,
            autocorr_time=tau
        )
    
    @classmethod
    def from_dse_result(cls, result: 'DSEResult') -> MemoryMetrics:
        """
        単一の DSE 結果から時間的メモリを計算
        """
        indicator = cls()
        
        dt = result.times[1] - result.times[0] if len(result.times) > 1 else 0.1
        M_temporal, tau = indicator.temporal_memory(
            np.array(result.lambdas), dt
        )
        
        return MemoryMetrics(
            delta_O=0.0,
            delta_energy=0.0,
            delta_lambda=0.0,
            M_temporal=M_temporal,
            autocorr_time=tau
        )
    
    # =========================================================================
    # Core Indicators
    # =========================================================================
    
    @staticmethod
    def path_noncommutativity(O_forward: float, O_backward: float) -> float:
        """
        経路非可換性 ΔO = |O_{A→B} - O_{B→A}|
        
        Markovian:     ΔO = 0（経路非依存）
        Non-Markovian: ΔO > 0（履歴依存）
        """
        return abs(O_forward - O_backward)
    
    @staticmethod
    def path_noncommutativity_relative(O_forward: float, O_backward: float,
                                        epsilon: float = 1e-10) -> float:
        """相対的経路非可換性（[0, 2] に正規化）"""
        denom = abs(O_forward) + abs(O_backward) + epsilon
        return 2 * abs(O_forward - O_backward) / denom
    
    @staticmethod
    def temporal_memory(series: np.ndarray, dt: float = 1.0) -> Tuple[float, float]:
        """
        自己相関からの時間的記憶
        
        M(t) = ∫₀ᵗ ⟨O(t)O(t')⟩_c dt'
        
        Returns:
            M: 積分記憶
            tau: 特性記憶時間（1/e 減衰）
        """
        if len(series) < 3:
            return 0.0, 0.0
        
        # 中心化（連結相関子）
        mean = np.mean(series)
        centered = series - mean
        
        # 自己相関
        n = len(centered)
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[n-1:]  # 正のラグのみ
        
        # 正規化
        if autocorr[0] > 1e-10:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.0, 0.0
        
        # 積分（台形則）
        M = np.trapz(autocorr, dx=dt)
        
        # 特性時間（1/e 減衰）
        tau = 0.0
        threshold = 1.0 / np.e
        for i, val in enumerate(autocorr):
            if val < threshold:
                tau = i * dt
                break
        else:
            tau = len(autocorr) * dt
        
        return float(M), float(tau)
    
    @staticmethod
    def gamma_memory(gamma_total: float, gamma_local: float) -> float:
        """
        γ 分解からの非マルコフ成分
        
        γ_memory = γ_total - γ_local
        """
        return gamma_total - gamma_local
    
    @staticmethod
    def memory_fraction(gamma_total: float, gamma_local: float,
                        epsilon: float = 1e-10) -> float:
        """非マルコフ成分の割合 [0, 1]"""
        gamma_mem = gamma_total - gamma_local
        return gamma_mem / (abs(gamma_total) + epsilon)
    
    # =========================================================================
    # Combined Computation
    # =========================================================================
    
    @classmethod
    def compute_all(cls,
                    O_forward: Optional[float] = None,
                    O_backward: Optional[float] = None,
                    series: Optional[np.ndarray] = None,
                    dt: float = 1.0,
                    gamma_total: Optional[float] = None,
                    gamma_local: Optional[float] = None) -> MemoryMetrics:
        """全ての利用可能な指標を計算"""
        
        delta_O = 0.0
        if O_forward is not None and O_backward is not None:
            delta_O = cls.path_noncommutativity(O_forward, O_backward)
        
        M_temporal = 0.0
        tau = 0.0
        if series is not None and len(series) > 2:
            M_temporal, tau = cls.temporal_memory(series, dt)
        
        gamma_mem = None
        if gamma_total is not None and gamma_local is not None:
            gamma_mem = cls.gamma_memory(gamma_total, gamma_local)
        
        return MemoryMetrics(
            delta_O=delta_O,
            M_temporal=M_temporal,
            gamma_memory=gamma_mem,
            autocorr_time=tau
        )


# =============================================================================
# Hysteresis Analyzer
# =============================================================================

class HysteresisAnalyzer:
    """
    ヒステリシスループの解析
    
    ループ面積 = 散逸された記憶
    """
    
    @staticmethod
    def compute_area(x_forward: np.ndarray, y_forward: np.ndarray,
                     x_backward: np.ndarray, y_backward: np.ndarray) -> float:
        """
        ヒステリシスループの面積（靴ひも公式）
        """
        x = np.concatenate([x_forward, x_backward[::-1]])
        y = np.concatenate([y_forward, y_backward[::-1]])
        
        n = len(x)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]
        
        return abs(area) / 2.0
    
    @staticmethod
    def compute_metrics(x_forward: np.ndarray, y_forward: np.ndarray,
                        x_backward: np.ndarray, y_backward: np.ndarray) -> Dict[str, float]:
        """
        包括的なヒステリシス解析
        
        Returns:
            area: ループ面積
            max_gap: 最大 y 差
            memory_strength: 正規化された記憶強度
        """
        area = HysteresisAnalyzer.compute_area(
            x_forward, y_forward, x_backward, y_backward
        )
        
        # 共通 x グリッドに補間
        x_min = max(x_forward.min(), x_backward.min())
        x_max = min(x_forward.max(), x_backward.max())
        x_common = np.linspace(x_min, x_max, 100)
        
        y_fwd_interp = np.interp(x_common, x_forward, y_forward)
        y_bwd_interp = np.interp(x_common, x_backward[::-1], y_backward[::-1])
        
        max_gap = float(np.max(np.abs(y_fwd_interp - y_bwd_interp)))
        
        # 正規化
        range_x = np.ptp(x_forward)
        range_y = np.ptp(y_forward)
        memory_strength = area / (range_x * range_y + 1e-10)
        
        return {
            'area': area,
            'max_gap': max_gap,
            'memory_strength': memory_strength
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Memory Indicators Test")
    print("=" * 60)
    
    indicator = MemoryIndicator()
    
    # Test 1: 経路非可換性
    print("\n--- Test 1: Path Non-Commutativity ---")
    O_fwd = 0.5234
    O_bwd = 0.4821
    delta_O = indicator.path_noncommutativity(O_fwd, O_bwd)
    print(f"  O_forward  = {O_fwd:.4f}")
    print(f"  O_backward = {O_bwd:.4f}")
    print(f"  ΔO = {delta_O:.4f}")
    print(f"  → {'Non-Markovian!' if delta_O > 0.01 else 'Markovian'}")
    
    # Test 2: 時間的記憶
    print("\n--- Test 2: Temporal Memory ---")
    t = np.linspace(0, 50, 500)
    series = np.exp(-t/10) * np.cos(t) + 0.1 * np.random.randn(len(t))
    M, tau = indicator.temporal_memory(series, dt=0.1)
    print(f"  M (integrated memory) = {M:.4f}")
    print(f"  τ (memory time)       = {tau:.4f}")
    
    # Test 3: γ 分解
    print("\n--- Test 3: Gamma Decomposition ---")
    gamma_total = 1.997
    gamma_local = 1.081
    gamma_mem = indicator.gamma_memory(gamma_total, gamma_local)
    frac = indicator.memory_fraction(gamma_total, gamma_local)
    print(f"  γ_total  = {gamma_total:.3f}")
    print(f"  γ_local  = {gamma_local:.3f}")
    print(f"  γ_memory = {gamma_mem:.3f}")
    print(f"  Memory fraction = {frac*100:.1f}%")
    
    # Test 4: 統合
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
    print(metrics.summary())
    
    # Test 5: ヒステリシス
    print("\n--- Test 5: Hysteresis ---")
    x_fwd = np.linspace(0, 1, 50)
    y_fwd = x_fwd ** 2
    x_bwd = np.linspace(1, 0, 50)
    y_bwd = x_bwd ** 0.5
    
    hyst = HysteresisAnalyzer.compute_metrics(x_fwd, y_fwd, x_bwd, y_bwd)
    print(f"  Area     = {hyst['area']:.4f}")
    print(f"  Max gap  = {hyst['max_gap']:.4f}")
    print(f"  Strength = {hyst['memory_strength']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Memory Indicators Test Complete!")
    print("=" * 60)
