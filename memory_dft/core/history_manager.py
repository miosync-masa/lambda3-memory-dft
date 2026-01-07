"""
History Manager for Memory-DFT
==============================

大規模計算用の高度な履歴管理

【役割分担】
  MemoryKernel.history: 小〜中規模用（シンプル）
  HistoryManager: 大規模用（圧縮、統計、GPU最適化）

【特徴】
  1. メモリ効率的な履歴保存（圧縮・間引き）
  2. λ 重み付き平均
  3. MemoryKernel との統合
  4. GPU 最適化版

【H-CSP との対応】
  - 保存則: 履歴保存とフラックス計算
  - 再帰生成: λ(t+Δt) = F(λ(t), λ̇(t))
  - 拍動的平衡: 履歴による非平衡維持

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from collections import deque

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# Memory Kernel（新しい統一版）
try:
    from memory_kernel import MemoryKernel, HistoryEntry
    HAS_MEMORY_KERNEL = True
except ImportError:
    MemoryKernel = None
    HistoryEntry = None
    HAS_MEMORY_KERNEL = False


# =============================================================================
# State Snapshot
# =============================================================================

@dataclass
class StateSnapshot:
    """
    状態のスナップショット
    
    HistoryEntry より詳細な情報を保持
    """
    time: float
    state: np.ndarray
    position: float = 0.0                    # 位置（結合長など）
    velocity: float = 0.0                    # 速度 dr/dt
    energy: Optional[float] = None
    lambda_density: Optional[float] = None
    entropy: Optional[float] = None
    observables: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_history_entry(self) -> 'HistoryEntry':
        """MemoryKernel 用の HistoryEntry に変換"""
        if HistoryEntry is None:
            raise ImportError("HistoryEntry not available")
        
        return HistoryEntry(
            time=self.time,
            position=self.position,
            velocity=self.velocity,
            state=self.state,
            energy=self.energy,
            entropy=self.entropy,
            metadata=self.metadata
        )


# =============================================================================
# History Manager
# =============================================================================

class HistoryManager:
    """
    大規模計算用の履歴管理
    
    Features:
    1. メモリ効率的な履歴保存
    2. 自動圧縮（古い履歴の間引き）
    3. λ 重み付き平均
    4. MemoryKernel との統合
    
    Usage:
        manager = HistoryManager(max_history=1000)
        
        # 状態を追加
        manager.add(t=0.0, state=psi, r=1.0, lambda_density=0.8)
        
        # MemoryKernel と連携
        kernel = manager.create_memory_kernel(E_xc=-0.5)
        delta_E = kernel.compute_memory_contribution(t, current_state)
    """
    
    def __init__(self,
                 max_history: int = 1000,
                 compression_threshold: int = 500,
                 use_gpu: bool = True):
        """
        Args:
            max_history: 最大履歴数
            compression_threshold: 圧縮を開始する閾値
            use_gpu: GPU 使用フラグ
        """
        self.max_history = max_history
        self.compression_threshold = compression_threshold
        self.use_gpu = use_gpu and HAS_CUPY
        
        self.history: deque = deque(maxlen=max_history)
        self.compressed_history: List[StateSnapshot] = []
        
        # 統計
        self.total_snapshots = 0
        self.compression_count = 0
        
        # MemoryKernel キャッシュ
        self._kernel_cache: Optional[MemoryKernel] = None
        self._kernel_dirty = True
    
    def add(self,
            time: float,
            state: np.ndarray,
            r: float = 0.0,
            energy: Optional[float] = None,
            lambda_density: Optional[float] = None,
            observables: Optional[Dict[str, float]] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """
        状態を履歴に追加
        
        Args:
            time: 時刻
            state: 状態ベクトル
            r: 位置（結合長など）
            energy: エネルギー
            lambda_density: λ 密度
            observables: 観測量の辞書
            metadata: メタデータ
        """
        # 速度計算
        if len(self.history) > 0:
            prev = self.history[-1]
            dt = time - prev.time
            if dt > 1e-10:
                velocity = (r - prev.position) / dt
            else:
                velocity = 0.0
        else:
            velocity = 0.0
        
        snapshot = StateSnapshot(
            time=time,
            state=state.copy() if hasattr(state, 'copy') else state,
            position=r,
            velocity=velocity,
            energy=energy,
            lambda_density=lambda_density,
            observables=observables or {},
            metadata=metadata or {}
        )
        
        self.history.append(snapshot)
        self.total_snapshots += 1
        self._kernel_dirty = True
        
        # 圧縮チェック
        if len(self.history) >= self.compression_threshold:
            self._compress_old_history()
    
    def _compress_old_history(self):
        """古い履歴を圧縮（間引き）"""
        n_to_compress = len(self.history) // 2
        old_snapshots = [self.history.popleft() for _ in range(n_to_compress)]
        
        # λ 密度が高いものを優先的に保持
        old_snapshots.sort(
            key=lambda s: s.lambda_density if s.lambda_density is not None else 0,
            reverse=True
        )
        n_keep = max(1, n_to_compress // 4)
        
        self.compressed_history.extend(old_snapshots[:n_keep])
        self.compression_count += 1
    
    def get_history_states(self,
                            n_recent: Optional[int] = None,
                            include_compressed: bool = False) -> List[StateSnapshot]:
        """履歴を取得"""
        result = list(self.history)
        
        if include_compressed:
            result = self.compressed_history + result
        
        if n_recent is not None:
            result = result[-n_recent:]
        
        return result
    
    def get_history_times(self) -> np.ndarray:
        """時刻配列を取得"""
        return np.array([s.time for s in self.history])
    
    def get_lambda_weights(self) -> np.ndarray:
        """λ 重みを取得"""
        lambdas = np.array([
            s.lambda_density if s.lambda_density is not None else 1.0
            for s in self.history
        ])
        total = lambdas.sum()
        if total > 0:
            lambdas = lambdas / total
        return lambdas
    
    # =========================================================================
    # MemoryKernel Integration
    # =========================================================================
    
    def create_memory_kernel(self,
                              E_xc: float,
                              gamma_memory: Optional[float] = None,
                              **kwargs) -> 'MemoryKernel':
        """
        履歴から MemoryKernel を作成
        
        Args:
            E_xc: 相関エネルギー
            gamma_memory: γ_memory（指定しない場合はデフォルト）
            **kwargs: MemoryKernel への追加引数
            
        Returns:
            履歴が設定された MemoryKernel
        """
        if not HAS_MEMORY_KERNEL:
            raise ImportError("memory_kernel module not available")
        
        # キャッシュチェック
        if not self._kernel_dirty and self._kernel_cache is not None:
            return self._kernel_cache
        
        # Kernel 作成
        if gamma_memory is not None:
            kernel = MemoryKernel(gamma_memory=gamma_memory, 
                                   use_gpu=self.use_gpu, **kwargs)
        else:
            kernel = MemoryKernel(use_gpu=self.use_gpu, **kwargs)
        
        # 履歴を転送
        for snapshot in self.history:
            kernel.add_state(
                t=snapshot.time,
                r=snapshot.position,
                state=snapshot.state,
                energy=snapshot.energy
            )
        
        self._kernel_cache = kernel
        self._kernel_dirty = False
        
        return kernel
    
    def compute_memory_contribution(self,
                                     t: float,
                                     current_state: np.ndarray,
                                     E_xc: float = -1.0,
                                     **kwargs) -> float:
        """
        メモリ寄与を計算（便利メソッド）
        """
        kernel = self.create_memory_kernel(E_xc, **kwargs)
        return kernel.compute_memory_contribution(t, current_state)
    
    def compute_memory_state(self,
                              t: float,
                              E_xc: float = -1.0,
                              **kwargs) -> np.ndarray:
        """
        メモリ重み付け状態を計算
        
        |ψ_memory⟩ = Σ K(t-τ) × I(τ) × |ψ(τ)⟩
        """
        kernel = self.create_memory_kernel(E_xc, **kwargs)
        
        if len(kernel.history) == 0:
            return None
        
        xp = cp if self.use_gpu and HAS_CUPY else np
        
        # 重みを計算
        weights = kernel.integrate(t, np.array([e.time for e in kernel.history]))
        
        # 状態の重ね合わせ
        states = [e.state for e in kernel.history if e.state is not None]
        if len(states) == 0:
            return None
        
        result = xp.zeros_like(states[0])
        for w, s in zip(weights, states):
            if s is not None:
                result = result + w * s
        
        return result
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        stats = {
            'current_history_size': len(self.history),
            'compressed_history_size': len(self.compressed_history),
            'total_snapshots': self.total_snapshots,
            'compression_count': self.compression_count,
            'use_gpu': self.use_gpu,
        }
        
        if len(self.history) > 0:
            lambdas = [s.lambda_density for s in self.history 
                       if s.lambda_density is not None]
            if lambdas:
                stats['lambda_mean'] = float(np.mean(lambdas))
                stats['lambda_std'] = float(np.std(lambdas))
            
            velocities = [s.velocity for s in self.history]
            stats['velocity_mean'] = float(np.mean(velocities))
            stats['n_stretch'] = sum(1 for v in velocities if v > 0)
            stats['n_compress'] = sum(1 for v in velocities if v < 0)
            
            stats['memory_usage_mb'] = self._estimate_memory_usage() / 1e6
        
        return stats
    
    def _estimate_memory_usage(self) -> float:
        """メモリ使用量を推定（バイト）"""
        if len(self.history) == 0:
            return 0.0
        
        sample = self.history[0].state
        state_size = sample.nbytes if hasattr(sample, 'nbytes') else 0
        
        return state_size * (len(self.history) + len(self.compressed_history))
    
    def clear(self):
        """履歴をクリア"""
        self.history.clear()
        self.compressed_history.clear()
        self.total_snapshots = 0
        self.compression_count = 0
        self._kernel_cache = None
        self._kernel_dirty = True


# =============================================================================
# GPU-optimized History Manager
# =============================================================================

class HistoryManagerGPU(HistoryManager):
    """
    GPU 最適化版 HistoryManager
    
    大規模計算用にメモリを事前確保
    """
    
    def __init__(self,
                 max_history: int = 1000,
                 state_dim: int = None):
        super().__init__(max_history=max_history, use_gpu=True)
        
        if not HAS_CUPY:
            raise ImportError("CuPy required for GPU HistoryManager")
        
        self.state_dim = state_dim
        
        if state_dim is not None:
            # GPU メモリを事前確保
            self._state_buffer = cp.zeros((max_history, state_dim), dtype=cp.complex128)
            self._time_buffer = cp.zeros(max_history, dtype=cp.float64)
            self._position_buffer = cp.zeros(max_history, dtype=cp.float64)
            self._lambda_buffer = cp.zeros(max_history, dtype=cp.float64)
            self._current_idx = 0
    
    def add_fast(self,
                  time: float,
                  state: 'cp.ndarray',
                  r: float = 0.0,
                  lambda_density: float = 1.0):
        """
        高速追加（事前確保バッファ使用）
        """
        if self.state_dim is None:
            raise ValueError("state_dim must be set")
        
        idx = self._current_idx % self.max_history
        self._state_buffer[idx] = state
        self._time_buffer[idx] = time
        self._position_buffer[idx] = r
        self._lambda_buffer[idx] = lambda_density
        self._current_idx += 1
        self._kernel_dirty = True
    
    def compute_memory_state_fast(self,
                                   kernel: 'MemoryKernel',
                                   t_current: float) -> 'cp.ndarray':
        """
        GPU 上で高速にメモリ状態を計算
        """
        n = min(self._current_idx, self.max_history)
        if n == 0:
            return cp.zeros(self.state_dim, dtype=cp.complex128)
        
        times = self._time_buffer[:n]
        lambdas = self._lambda_buffer[:n]
        states = self._state_buffer[:n]
        
        # カーネル重み
        kernel_weights = kernel.integrate(t_current, cp.asnumpy(times))
        kernel_weights = cp.asarray(kernel_weights)
        
        # λ 重み
        lambda_weights = lambdas / (lambdas.sum() + 1e-10)
        
        # 合成重み
        combined = kernel_weights * lambda_weights
        combined = combined / (combined.sum() + 1e-10)
        
        # 状態の重ね合わせ（行列演算）
        result = cp.einsum('i,ij->j', combined, states)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計（GPU バッファ情報付き）"""
        stats = super().get_statistics()
        
        if self.state_dim is not None:
            stats['buffer_size'] = self._current_idx
            stats['buffer_capacity'] = self.max_history
            stats['gpu_memory_mb'] = (
                self._state_buffer.nbytes + 
                self._time_buffer.nbytes +
                self._position_buffer.nbytes +
                self._lambda_buffer.nbytes
            ) / 1e6
        
        return stats


# =============================================================================
# Lambda Density Calculator
# =============================================================================

class LambdaDensityCalculator:
    """
    λ（安定性パラメータ）の計算
    
    λ = K / |V|_eff
    
    臨界条件:
    - λ < 1: 安定（束縛）
    - λ = 1: 臨界（相転移）
    - λ > 1: 不安定（崩壊）
    """
    
    @staticmethod
    def from_energy(kinetic: float, potential: float, epsilon: float = 1e-10) -> float:
        """エネルギーから λ を計算"""
        return kinetic / (abs(potential) + epsilon)
    
    @staticmethod
    def from_state(state: np.ndarray,
                   H_kinetic,
                   H_potential) -> float:
        """状態から λ を計算"""
        xp = cp if HAS_CUPY and isinstance(state, cp.ndarray) else np
        
        # ⟨K⟩
        K_psi = H_kinetic @ state
        K_mean = float(xp.real(xp.vdot(state, K_psi)))
        
        # ⟨V⟩
        V_psi = H_potential @ state
        V_mean = float(xp.real(xp.vdot(state, V_psi)))
        
        return K_mean / (abs(V_mean) + 1e-10)
    
    @staticmethod
    def from_delta(delta: float, delta_L: float = 0.18) -> float:
        """
        δ から λ を推定
        
        λ ≈ (δ / δ_L)²  for δ < δ_L
        """
        return (delta / delta_L) ** 2


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("History Manager Test (Unified Version)")
    print("=" * 70)
    
    print(f"\nMemoryKernel available: {HAS_MEMORY_KERNEL}")
    print(f"CuPy available: {HAS_CUPY}")
    
    # 基本テスト
    print("\n[1] Basic HistoryManager Test")
    manager = HistoryManager(max_history=100, use_gpu=False)
    
    np.random.seed(42)
    dim = 16
    
    for t in range(50):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state = state / np.linalg.norm(state)
        
        r = 1.0 + 0.1 * np.sin(t / 5)
        lambda_val = 0.5 + 0.3 * np.cos(t / 10)
        
        manager.add(
            time=float(t),
            state=state,
            r=r,
            energy=-1.0 + 0.05 * t,
            lambda_density=lambda_val,
            observables={'J_x': np.random.randn()}
        )
    
    stats = manager.get_statistics()
    print(f"  History size: {stats['current_history_size']}")
    print(f"  Lambda mean: {stats.get('lambda_mean', 0):.4f}")
    print(f"  Velocity mean: {stats.get('velocity_mean', 0):.4f}")
    print(f"  N stretch: {stats.get('n_stretch', 0)}, N compress: {stats.get('n_compress', 0)}")
    
    # MemoryKernel 統合テスト
    if HAS_MEMORY_KERNEL:
        print("\n[2] MemoryKernel Integration Test")
        
        kernel = manager.create_memory_kernel(E_xc=-0.5, gamma_memory=1.2)
        print(f"  Created: {kernel}")
        print(f"  Kernel history: {len(kernel.history)} entries")
        
        current = np.random.randn(dim) + 1j * np.random.randn(dim)
        current = current / np.linalg.norm(current)
        
        delta_E = manager.compute_memory_contribution(t=50.0, current_state=current)
        print(f"  ΔE_mem = {delta_E:.6f}")
        
        mem_state = manager.compute_memory_state(t=50.0)
        if mem_state is not None:
            print(f"  Memory state norm: {np.linalg.norm(mem_state):.6f}")
    
    # GPU テスト
    if HAS_CUPY:
        print("\n[3] GPU HistoryManager Test")
        
        try:
            manager_gpu = HistoryManagerGPU(max_history=100, state_dim=256)
            
            for t in range(50):
                state = cp.random.randn(256) + 1j * cp.random.randn(256)
                state = state / cp.linalg.norm(state)
                manager_gpu.add_fast(float(t), state, r=1.0 + 0.1*t, lambda_density=1.0)
            
            stats_gpu = manager_gpu.get_statistics()
            print(f"  Buffer size: {stats_gpu.get('buffer_size', 0)}")
            print(f"  GPU memory: {stats_gpu.get('gpu_memory_mb', 0):.2f} MB")
            print("  ✅ GPU HistoryManager works!")
            
        except Exception as e:
            print(f"  ⚠️ GPU test failed: {e}")
    
    # LambdaDensityCalculator テスト
    print("\n[4] LambdaDensityCalculator Test")
    
    lambda_E = LambdaDensityCalculator.from_energy(kinetic=0.5, potential=-1.0)
    print(f"  from_energy(K=0.5, V=-1.0): λ = {lambda_E:.4f}")
    
    lambda_delta = LambdaDensityCalculator.from_delta(delta=0.09, delta_L=0.18)
    print(f"  from_delta(δ=0.09, δ_L=0.18): λ = {lambda_delta:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ History Manager Test Complete!")
    print("=" * 70)
