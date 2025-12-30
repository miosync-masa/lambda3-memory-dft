"""
History Manager for Memory-DFT
==============================

履歴（過去の状態）を保持し、Λ重み付けを行う

H-CSP公理との対応:
- 公理3（全体保存）: 履歴の保存と流束計算
- 公理4（再帰生成）: Λ(t+Δt) = F(Λ(t), Λ̇(t))
- 公理5（拍動的平衡）: 履歴を通じた非平衡維持

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import cupy as cp
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class StateSnapshot:
    """状態のスナップショット"""
    time: float
    state: Union[np.ndarray, cp.ndarray]  # |ψ⟩ or ρ
    energy: Optional[float] = None
    lambda_density: Optional[float] = None  # Λ意味密度
    observables: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HistoryManager:
    """
    履歴管理クラス
    
    機能:
    1. 状態履歴の保持（メモリ効率的）
    2. Λ意味密度による重み付け
    3. Memory kernel との統合
    4. 履歴の圧縮・間引き
    """
    
    def __init__(self, 
                 max_history: int = 1000,
                 compression_threshold: int = 500,
                 use_gpu: bool = True):
        """
        Args:
            max_history: 保持する最大履歴数
            compression_threshold: 圧縮を開始する閾値
            use_gpu: GPU使用フラグ
        """
        self.max_history = max_history
        self.compression_threshold = compression_threshold
        self.use_gpu = use_gpu
        
        self.history: deque = deque(maxlen=max_history)
        self.compressed_history: List[StateSnapshot] = []
        
        # 統計情報
        self.total_snapshots = 0
        self.compression_count = 0
        
    def add(self, 
            time: float, 
            state: Union[np.ndarray, cp.ndarray],
            energy: Optional[float] = None,
            lambda_density: Optional[float] = None,
            observables: Optional[Dict[str, float]] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """状態を履歴に追加"""
        
        snapshot = StateSnapshot(
            time=time,
            state=state.copy() if hasattr(state, 'copy') else state,
            energy=energy,
            lambda_density=lambda_density,
            observables=observables or {},
            metadata=metadata or {}
        )
        
        self.history.append(snapshot)
        self.total_snapshots += 1
        
        # 必要に応じて圧縮
        if len(self.history) >= self.compression_threshold:
            self._compress_old_history()
            
    def _compress_old_history(self):
        """古い履歴を圧縮（間引き）"""
        # 古い半分を間引き
        n_to_compress = len(self.history) // 2
        old_snapshots = [self.history.popleft() for _ in range(n_to_compress)]
        
        # 重要なものだけ残す（Λ密度が高いもの）
        old_snapshots.sort(key=lambda s: s.lambda_density or 0, reverse=True)
        n_keep = n_to_compress // 4  # 1/4だけ残す
        
        self.compressed_history.extend(old_snapshots[:n_keep])
        self.compression_count += 1
        
    def get_history_states(self, 
                           n_recent: Optional[int] = None,
                           include_compressed: bool = False) -> List[StateSnapshot]:
        """履歴状態を取得"""
        result = list(self.history)
        
        if include_compressed:
            result = self.compressed_history + result
            
        if n_recent is not None:
            result = result[-n_recent:]
            
        return result
    
    def get_history_times(self) -> np.ndarray:
        """履歴の時刻配列を取得"""
        return np.array([s.time for s in self.history])
    
    def get_lambda_weights(self) -> np.ndarray:
        """Λ意味密度による重みを取得"""
        lambdas = np.array([
            s.lambda_density if s.lambda_density is not None else 1.0 
            for s in self.history
        ])
        # 正規化
        if lambdas.sum() > 0:
            lambdas = lambdas / lambdas.sum()
        return lambdas
    
    def compute_memory_integral(self, 
                                 kernel,
                                 t_current: float,
                                 observable_key: Optional[str] = None) -> float:
        """
        Memory 積分を計算
        
        ∫ K(t-τ) * Λ(τ) * O(τ) dτ
        
        Args:
            kernel: Memory kernel オブジェクト
            t_current: 現在時刻
            observable_key: 積分する物理量のキー（Noneなら状態ノルム）
        """
        if len(self.history) == 0:
            return 0.0
            
        times = self.get_history_times()
        lambda_weights = self.get_lambda_weights()
        
        # Kernel重みを取得
        kernel_weights = kernel.integrate(t_current, times)
        
        # 統合重み
        combined_weights = kernel_weights * lambda_weights
        combined_weights = combined_weights / (combined_weights.sum() + 1e-10)
        
        # 物理量を取得
        if observable_key is not None:
            values = np.array([
                s.observables.get(observable_key, 0.0) 
                for s in self.history
            ])
        else:
            values = np.array([
                float(np.linalg.norm(s.state) if isinstance(s.state, np.ndarray) 
                      else float(cp.linalg.norm(s.state)))
                for s in self.history
            ])
            
        # 積分
        return float(np.dot(combined_weights, values))
    
    def compute_memory_state(self,
                              kernel,
                              t_current: float) -> Union[np.ndarray, cp.ndarray]:
        """
        Memory項を含む状態の重ね合わせを計算
        
        |ψ_memory⟩ = Σ K(t-τ) * Λ(τ) * |ψ(τ)⟩
        
        これが Memory-DFT の核心操作！
        """
        if len(self.history) == 0:
            return None
            
        times = self.get_history_times()
        lambda_weights = self.get_lambda_weights()
        kernel_weights = kernel.integrate(t_current, times)
        
        combined_weights = kernel_weights * lambda_weights
        combined_weights = combined_weights / (combined_weights.sum() + 1e-10)
        
        # 状態の重ね合わせ
        states = [s.state for s in self.history]
        
        if self.use_gpu and isinstance(states[0], cp.ndarray):
            result = cp.zeros_like(states[0])
            for w, s in zip(combined_weights, states):
                result += w * s
        else:
            result = np.zeros_like(states[0])
            for w, s in zip(combined_weights, states):
                result += w * s
                
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            'current_history_size': len(self.history),
            'compressed_history_size': len(self.compressed_history),
            'total_snapshots': self.total_snapshots,
            'compression_count': self.compression_count,
            'memory_usage_mb': self._estimate_memory_usage() / 1e6
        }
    
    def _estimate_memory_usage(self) -> float:
        """メモリ使用量を推定（バイト）"""
        if len(self.history) == 0:
            return 0.0
        
        sample_state = self.history[0].state
        state_size = sample_state.nbytes if hasattr(sample_state, 'nbytes') else 0
        
        return state_size * (len(self.history) + len(self.compressed_history))
    
    def clear(self):
        """履歴をクリア"""
        self.history.clear()
        self.compressed_history.clear()
        self.total_snapshots = 0
        self.compression_count = 0


class LambdaDensityCalculator:
    """
    Λ意味密度の計算
    
    Λ(τ) = K(τ) / |V|_eff(τ)
    
    H-CSP理論における意味密度を量子状態から計算
    """
    
    @staticmethod
    def from_energy(kinetic: float, potential: float, epsilon: float = 1e-10) -> float:
        """
        エネルギーからΛを計算
        
        Λ = K / |V|_eff
        
        臨界条件:
        - Λ < 1: 安定
        - Λ = 1: 臨界（相転移）
        - Λ > 1: カタストロフィ
        """
        return kinetic / (abs(potential) + epsilon)
    
    @staticmethod
    def from_variance(state: Union[np.ndarray, cp.ndarray],
                      H_kinetic,
                      H_potential) -> float:
        """
        状態のエネルギー分散からΛを計算
        
        より精密な計算。揺らぎを考慮。
        """
        if isinstance(state, cp.ndarray):
            xp = cp
        else:
            xp = np
            
        # ⟨K⟩
        K_psi = H_kinetic @ state
        K_mean = float(xp.real(xp.vdot(state, K_psi)))
        
        # ⟨V⟩
        V_psi = H_potential @ state
        V_mean = float(xp.real(xp.vdot(state, V_psi)))
        
        return K_mean / (abs(V_mean) + 1e-10)
    
    @staticmethod
    def from_vorticity(vorticity: float, E_xc: float, N: int, epsilon: float = 1e-10) -> float:
        """
        Vorticity から γ を抽出し、Λ相当値を計算
        
        α = E_xc / V ∝ N^(-γ)
        
        PySCF vorticity との連携用
        """
        alpha = abs(E_xc) / (vorticity + epsilon)
        # γ相当の指標を返す
        return alpha


# =============================================================================
# GPU-optimized History Manager
# =============================================================================

class HistoryManagerGPU(HistoryManager):
    """
    GPU最適化版 History Manager
    
    大規模系での高速計算用
    """
    
    def __init__(self, max_history: int = 1000, state_dim: int = None):
        super().__init__(max_history=max_history, use_gpu=True)
        
        self.state_dim = state_dim
        if state_dim is not None:
            # 事前にGPUメモリを確保
            self._state_buffer = cp.zeros((max_history, state_dim), dtype=cp.complex128)
            self._time_buffer = cp.zeros(max_history, dtype=cp.float64)
            self._lambda_buffer = cp.zeros(max_history, dtype=cp.float64)
            self._current_idx = 0
            
    def add_fast(self, time: float, state: cp.ndarray, lambda_density: float = 1.0):
        """高速追加（事前確保バッファ使用）"""
        if self.state_dim is None:
            raise ValueError("state_dim must be set for fast mode")
            
        idx = self._current_idx % self.max_history
        self._state_buffer[idx] = state
        self._time_buffer[idx] = time
        self._lambda_buffer[idx] = lambda_density
        self._current_idx += 1
        
    def compute_memory_state_fast(self, kernel_gpu, t_current: float) -> cp.ndarray:
        """GPU上で高速にメモリ状態を計算"""
        n = min(self._current_idx, self.max_history)
        if n == 0:
            return cp.zeros(self.state_dim, dtype=cp.complex128)
            
        times = self._time_buffer[:n]
        lambdas = self._lambda_buffer[:n]
        states = self._state_buffer[:n]
        
        # Kernel重み（GPU）
        kernel_weights = kernel_gpu.integrate_gpu(t_current, times)
        
        # Λ重み
        lambda_weights = lambdas / (lambdas.sum() + 1e-10)
        
        # 統合重み
        combined = kernel_weights * lambda_weights
        combined = combined / (combined.sum() + 1e-10)
        
        # 状態の重ね合わせ（行列演算）
        # combined: (n,), states: (n, dim) → result: (dim,)
        result = cp.einsum('i,ij->j', combined, states)
        
        return result


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
