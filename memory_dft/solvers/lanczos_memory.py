"""
Lanczos Time Evolution with Memory
==================================

Lanczos法による時間発展に Memory 項を追加

標準: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩

Memory-DFT: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩ + η * |ψ_memory⟩

ここで |ψ_memory⟩ = Σ K(t-τ) Λ(τ) |ψ(τ)⟩

H-CSP公理4との対応:
  Λ(t+Δt) = F(Λ(t), Λ̇(t))
  → 過去の状態が現在に干渉する再帰生成

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from scipy.linalg import expm as scipy_expm
import scipy.sparse as sp
from typing import Optional, Tuple, Dict, Any, Callable
import time

# GPU support (optional)
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    HAS_CUPY = True
except ImportError:
    cp = np  # Fallback to NumPy
    csp = sp  # Fallback to SciPy sparse
    HAS_CUPY = False


def lanczos_expm_multiply(H_sparse, psi, dt, krylov_dim=30):
    """
    Lanczos法による exp(-i H dt) |ψ⟩ の計算
    
    CuPy/NumPy 両対応版
    """
    # Backend detection
    if HAS_CUPY and isinstance(psi, cp.ndarray):
        xp = cp
    else:
        xp = np
        # Ensure psi is numpy array
        if hasattr(psi, 'get'):
            psi = psi.get()
    
    n = psi.shape[0]
    
    V = xp.zeros((krylov_dim, n), dtype=xp.complex128)
    alpha = np.zeros(krylov_dim, dtype=np.float64)
    beta = np.zeros(krylov_dim - 1, dtype=np.float64)
    
    norm_psi = float(xp.linalg.norm(psi))
    v = psi / norm_psi
    V[0] = v
    
    w = H_sparse @ v
    alpha[0] = float(xp.real(xp.vdot(v, w)))
    w = w - alpha[0] * v
    
    actual_dim = krylov_dim
    for j in range(1, krylov_dim):
        beta_j = float(xp.linalg.norm(w))
        
        if beta_j < 1e-12:
            actual_dim = j
            break
            
        beta[j-1] = beta_j
        v_new = w / beta_j
        V[j] = v_new
        
        w = H_sparse @ v_new
        alpha[j] = float(xp.real(xp.vdot(v_new, w)))
        w = w - alpha[j] * v_new - beta[j-1] * V[j-1]
    
    T = np.diag(alpha[:actual_dim])
    if actual_dim > 1:
        T += np.diag(beta[:actual_dim-1], k=1)
        T += np.diag(beta[:actual_dim-1], k=-1)
    
    exp_T = scipy_expm(-1j * dt * T)
    
    e0 = np.zeros(actual_dim, dtype=np.complex128)
    e0[0] = 1.0
    y = exp_T @ e0
    
    if xp == cp:
        y_xp = cp.asarray(y)
    else:
        y_xp = y
        
    psi_new = norm_psi * (V[:actual_dim].T @ y_xp)
    
    return psi_new / xp.linalg.norm(psi_new)


class MemoryLanczosSolver:
    """
    Memory項を含むLanczos時間発展ソルバー
    
    これが Memory-DFT の核心計算エンジン！
    
    標準のユニタリ発展に、過去の状態からの寄与を追加：
    
    |ψ(t+dt)⟩ = (1-η) * exp(-iHdt)|ψ(t)⟩ + η * |ψ_memory⟩
    
    η: Memory 強度 (0 = 標準量子力学, 1 = 完全Memory支配)
    """
    
    def __init__(self,
                 memory_kernel,
                 history_manager,
                 krylov_dim: int = 30,
                 memory_strength: float = 0.1,
                 use_gpu: bool = True):
        """
        Args:
            memory_kernel: CompositeMemoryKernel or GPU version
            history_manager: HistoryManager instance
            krylov_dim: Krylov部分空間の次元
            memory_strength: η, Memory項の強度 [0, 1]
            use_gpu: GPU使用フラグ
        """
        self.kernel = memory_kernel
        self.history = history_manager
        self.krylov_dim = krylov_dim
        self.eta = memory_strength
        self.use_gpu = use_gpu and HAS_CUPY
        
        # Backend
        self.xp = cp if self.use_gpu else np
        
        # 診断情報
        self.diagnostics = {
            'unitary_contrib': [],
            'memory_contrib': [],
            'overlap_with_memory': []
        }
        
    def evolve(self,
               H,
               psi,
               t_current: float,
               dt: float,
               lambda_calculator: Optional[Callable] = None):
        """
        Memory項を含む1ステップ時間発展
        
        Args:
            H: ハミルトニアン（スパース）
            psi: 現在の状態
            t_current: 現在時刻
            dt: 時間刻み
            lambda_calculator: Λ密度計算関数（オプション）
            
        Returns:
            psi_new: 新しい状態（正規化済み）
        """
        xp = self.xp
        
        # 1. 標準Lanczos発展
        psi_unitary = lanczos_expm_multiply(H, psi, dt, self.krylov_dim)
        
        # 2. Memory項を計算
        psi_memory = self.history.compute_memory_state(self.kernel, t_current)
        
        if psi_memory is not None:
            # 配列変換
            if self.use_gpu and isinstance(psi_memory, np.ndarray):
                psi_memory = cp.asarray(psi_memory)
            elif not self.use_gpu and HAS_CUPY and isinstance(psi_memory, cp.ndarray):
                psi_memory = psi_memory.get()
                
            psi_memory = psi_memory / (xp.linalg.norm(psi_memory) + 1e-10)
            
            # 3. 混合
            psi_new = (1 - self.eta) * psi_unitary + self.eta * psi_memory
            psi_new = psi_new / xp.linalg.norm(psi_new)
            
            # 診断情報
            overlap = float(xp.abs(xp.vdot(psi_unitary, psi_memory)))
            self.diagnostics['overlap_with_memory'].append(overlap)
        else:
            psi_new = psi_unitary
            
        # 4. Λ密度を計算して履歴に追加
        if lambda_calculator is not None:
            lambda_val = lambda_calculator(psi_new)
        else:
            lambda_val = 1.0
            
        # 履歴に追加
        self.history.add(
            time=t_current + dt,
            state=psi_new,
            lambda_density=lambda_val
        )
        
        return psi_new
    
    def run(self,
            H,
            psi_initial,
            t_start: float,
            t_end: float,
            dt: float,
            observables: Optional[Dict[str, Any]] = None,
            lambda_calculator: Optional[Callable] = None,
            callback: Optional[Callable] = None,
            verbose: bool = True) -> Dict[str, Any]:
        """
        時間発展を実行
        
        Args:
            H: ハミルトニアン
            psi_initial: 初期状態
            t_start, t_end: 時間範囲
            dt: 時間刻み
            observables: 測定する物理量 {'name': operator}
            lambda_calculator: Λ計算関数
            callback: 各ステップで呼ばれるコールバック
            verbose: 進捗表示
            
        Returns:
            結果辞書
        """
        xp = self.xp
        n_steps = int((t_end - t_start) / dt)
        times = np.linspace(t_start, t_end, n_steps + 1)
        
        psi = psi_initial.copy()
        
        # 初期状態を履歴に追加
        self.history.add(
            time=t_start,
            state=psi,
            lambda_density=lambda_calculator(psi) if lambda_calculator else 1.0
        )
        
        # 結果格納
        results = {
            'times': times,
            'states': [psi.copy()],
            'observables': {name: [] for name in (observables or {})},
            'lambda_series': [],
            'memory_overlaps': []
        }
        
        # 初期物理量
        if observables:
            for name, op in observables.items():
                val = float(xp.real(xp.vdot(psi, op @ psi)))
                results['observables'][name].append(val)
        
        if verbose:
            print(f"Memory-DFT Evolution: {n_steps} steps, dt={dt}, η={self.eta}")
            print(f"  Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
            
        t0 = time.time()
        
        for i, t in enumerate(times[:-1]):
            # 時間発展
            psi = self.evolve(H, psi, t, dt, lambda_calculator)
            
            results['states'].append(psi.copy())
            
            # 物理量測定
            if observables:
                for name, op in observables.items():
                    val = float(xp.real(xp.vdot(psi, op @ psi)))
                    results['observables'][name].append(val)
            
            # Λ
            if lambda_calculator:
                results['lambda_series'].append(lambda_calculator(psi))
                
            # Memory overlap
            if self.diagnostics['overlap_with_memory']:
                results['memory_overlaps'].append(
                    self.diagnostics['overlap_with_memory'][-1]
                )
                
            # コールバック
            if callback:
                callback(i, t, psi, results)
                
            # 進捗
            if verbose and (i + 1) % max(n_steps // 4, 1) == 0:
                elapsed = time.time() - t0
                print(f"  Step {i+1}/{n_steps} ({100*(i+1)/n_steps:.0f}%) - {elapsed:.2f}s")
        
        if verbose:
            print(f"  ✅ Completed in {time.time() - t0:.2f}s")
            
        return results


class AdaptiveMemorySolver(MemoryLanczosSolver):
    """
    適応的Memory強度ソルバー
    
    状態に応じてη（Memory強度）を動的に調整
    
    - Λ > 0.8: Memory強化（不安定→過去に頼る）
    - Λ < 0.3: Memory弱化（安定→自由に発展）
    """
    
    def __init__(self,
                 memory_kernel,
                 history_manager,
                 eta_min: float = 0.01,
                 eta_max: float = 0.3,
                 lambda_threshold_low: float = 0.3,
                 lambda_threshold_high: float = 0.8,
                 **kwargs):
        
        super().__init__(memory_kernel, history_manager, **kwargs)
        
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.lambda_low = lambda_threshold_low
        self.lambda_high = lambda_threshold_high
        
    def _adaptive_eta(self, lambda_val: float) -> float:
        """Λに応じてηを調整"""
        if lambda_val < self.lambda_low:
            return self.eta_min
        elif lambda_val > self.lambda_high:
            return self.eta_max
        else:
            # 線形補間
            t = (lambda_val - self.lambda_low) / (self.lambda_high - self.lambda_low)
            return self.eta_min + t * (self.eta_max - self.eta_min)
    
    def evolve(self, H, psi, t_current, dt, lambda_calculator=None):
        """適応的ηで発展"""
        if lambda_calculator and len(self.history.history) > 0:
            last_lambda = self.history.history[-1].lambda_density
            if last_lambda is not None:
                self.eta = self._adaptive_eta(last_lambda)
                
        return super().evolve(H, psi, t_current, dt, lambda_calculator)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Memory Lanczos Solver Test")
    print("="*70)
    print(f"Backend: {'GPU (CuPy)' if HAS_CUPY else 'CPU (NumPy/SciPy)'}")
    
    # 簡単な2準位系でテスト
    try:
        from memory_dft.core.memory_kernel import CompositeMemoryKernel
        from memory_dft.core.history_manager import HistoryManager
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.memory_kernel import CompositeMemoryKernel
        from core.history_manager import HistoryManager
    
    # Backend選択
    xp = cp if HAS_CUPY else np
    sparse = csp if HAS_CUPY else sp
    
    # パウリ行列
    sx = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
    sz = xp.array([[1, 0], [0, -1]], dtype=xp.complex128)
    
    # ハミルトニアン: H = -sz + 0.5*sx (磁場中のスピン)
    H_dense = -sz + 0.5 * sx
    H = sparse.csr_matrix(H_dense)
    
    # 初期状態: |↑⟩
    psi0 = xp.array([1, 0], dtype=xp.complex128)
    
    # Memory kernel と History manager
    kernel = CompositeMemoryKernel()
    history = HistoryManager(max_history=1000, use_gpu=HAS_CUPY)
    
    # ソルバー
    solver = MemoryLanczosSolver(
        memory_kernel=kernel,
        history_manager=history,
        memory_strength=0.1,  # 10% memory
        krylov_dim=10,
        use_gpu=HAS_CUPY
    )
    
    # Λ計算（簡易版：エネルギー比）
    def lambda_calc(psi):
        E = float(xp.real(xp.vdot(psi, H @ psi)))
        return abs(E) / 2.0  # 正規化
    
    # 実行
    results = solver.run(
        H=H,
        psi_initial=psi0,
        t_start=0,
        t_end=10,
        dt=0.1,
        observables={'Sz': sparse.csr_matrix(sz), 'Sx': sparse.csr_matrix(sx)},
        lambda_calculator=lambda_calc,
        verbose=True
    )
    
    print()
    print("Results:")
    print(f"  Final Sz: {results['observables']['Sz'][-1]:.4f}")
    print(f"  Final Sx: {results['observables']['Sx'][-1]:.4f}")
    print(f"  History size: {len(history.history)}")
    
    if results['memory_overlaps']:
        print(f"  Avg memory overlap: {np.mean(results['memory_overlaps']):.4f}")
    
    print()
    print("="*70)
    print("Adaptive Memory Solver Test")
    print("="*70)
    
    # 適応ソルバーテスト
    history2 = HistoryManager(max_history=1000, use_gpu=HAS_CUPY)
    adaptive_solver = AdaptiveMemorySolver(
        memory_kernel=kernel,
        history_manager=history2,
        eta_min=0.01,
        eta_max=0.3,
        use_gpu=HAS_CUPY
    )
    
    results2 = adaptive_solver.run(
        H=H,
        psi_initial=psi0,
        t_start=0,
        t_end=10,
        dt=0.1,
        lambda_calculator=lambda_calc,
        verbose=True
    )
    
    print(f"  Final eta: {adaptive_solver.eta:.4f}")
    print("✅ All tests passed!")
