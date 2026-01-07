"""
DSE Solver - Direct Schrödinger Evolution
=========================================

Schrödinger 方程式を履歴付きで解く統一ソルバー

【DSE の本質】
  標準量子力学:
    iℏ ∂ψ/∂t = H ψ
    → 履歴なし、Markovian
    
  DSE (Direct Schrödinger Evolution):
    iℏ ∂ψ/∂t = H ψ + ∫ K(t-τ) F[ψ(τ)] dτ
    → 履歴あり、Non-Markovian
    → 材料の「記憶」を表現

【実装】
  |ψ(t+dt)⟩ = (1-η) exp(-iHdt)|ψ(t)⟩ + η |ψ_memory⟩
  
  |ψ_memory⟩ = Σ K(t-τ) × I(τ) × |ψ(τ)⟩

【Hamiltonian Sources】
  - sparse_engine + environment_operators（格子模型）
  - PySCF（分子、H を借りる）
  - カスタム（直接 H を渡す）

【使用例】
  # 直接 H を渡す
  solver = DSESolver(H_K, H_V, gamma_memory=1.2)
  result = solver.run(psi0, t_end=10.0, dt=0.1)
  
  # SparseEngine から
  solver = DSESolver.from_sparse_engine(engine, geometry, T=500, sigma=1.0)
  
  # PySCF から（H を借りる）
  solver = DSESolver.from_pyscf(mf)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm as scipy_expm
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
import time as time_module

# GPU support
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_sparse = None
    HAS_CUPY = False

# Memory Kernel
try:
    from memory_kernel import MemoryKernel
    HAS_MEMORY_KERNEL = True
except ImportError:
    MemoryKernel = None
    HAS_MEMORY_KERNEL = False

# History Manager
try:
    from history_manager import HistoryManager
    HAS_HISTORY_MANAGER = True
except ImportError:
    HistoryManager = None
    HAS_HISTORY_MANAGER = False


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class DSEResult:
    """DSE 時間発展の結果"""
    times: np.ndarray
    states: List[np.ndarray]
    energies: List[float]
    lambdas: List[float]
    memory_contributions: List[float]
    
    # オプション
    observables: Dict[str, List[float]] = field(default_factory=dict)
    
    # メタデータ
    n_steps: int = 0
    wall_time: float = 0.0
    use_memory: bool = True
    gamma_memory: float = 1.0
    eta: float = 0.1
    
    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]
    
    @property
    def final_energy(self) -> float:
        return self.energies[-1]
    
    @property
    def final_lambda(self) -> float:
        return self.lambdas[-1]
    
    @property
    def total_memory_effect(self) -> float:
        return sum(self.memory_contributions)
    
    @property
    def energy_drift(self) -> float:
        return abs(self.energies[-1] - self.energies[0])
    
    def check_pulsation(self, window: int = 10) -> Dict[str, Any]:
        """
        拍動的平衡（H-CSP 公理5）のチェック
        
        Λ̇ ≠ 0 かつ ⟨Λ(t+Δt)⟩ ≈ Λ(t)
        """
        if len(self.lambdas) < window * 2:
            return {'pulsation': False}
        
        lambdas = np.array(self.lambdas[-window*2:])
        
        # 局所変動
        local_var = np.mean(np.abs(np.diff(lambdas)))
        
        # 大域平均
        lambda_mean = np.mean(lambdas)
        lambda_std = np.std(lambdas)
        
        # 拍動判定: 変動ありかつ平均安定
        pulsation = local_var > 1e-4 and lambda_std / (lambda_mean + 1e-10) < 0.1
        
        return {
            'pulsation': pulsation,
            'local_variation': local_var,
            'lambda_mean': lambda_mean,
            'lambda_std': lambda_std
        }
    
    def summary(self) -> str:
        """結果サマリー"""
        puls = self.check_pulsation()
        return f"""
DSE Result Summary
==================
Steps: {self.n_steps}
Wall time: {self.wall_time:.2f}s
Memory: {'ON' if self.use_memory else 'OFF'} (γ={self.gamma_memory:.2f}, η={self.eta:.2f})

Energy:
  Initial: {self.energies[0]:.6f}
  Final: {self.energies[-1]:.6f}
  Drift: {self.energy_drift:.6f}

Lambda (Stability):
  Initial: {self.lambdas[0]:.4f}
  Final: {self.lambdas[-1]:.4f}
  Range: [{min(self.lambdas):.4f}, {max(self.lambdas):.4f}]

Memory Effect:
  Total: {self.total_memory_effect:.6f}
  Max: {max(self.memory_contributions):.6f}

Pulsation: {'Yes 🫀' if puls['pulsation'] else 'No'}
"""


# =============================================================================
# Lanczos Time Evolution
# =============================================================================

def lanczos_expm_multiply(H, psi, dt: float, krylov_dim: int = 30):
    """
    Lanczos 法による exp(-i H dt) |ψ⟩ の計算
    
    Args:
        H: Hamiltonian（スパース行列）
        psi: 状態ベクトル
        dt: 時間刻み
        krylov_dim: Krylov 部分空間の次元
        
    Returns:
        時間発展した状態
    """
    # Backend detection
    if HAS_CUPY and isinstance(psi, cp.ndarray):
        xp = cp
        is_gpu = True
    else:
        xp = np
        is_gpu = False
    
    n = psi.shape[0]
    
    # Krylov vectors
    V = xp.zeros((krylov_dim, n), dtype=xp.complex128)
    alpha = np.zeros(krylov_dim, dtype=np.float64)  # CPU for scipy_expm
    beta = np.zeros(krylov_dim - 1, dtype=np.float64)
    
    # Normalize
    norm_psi = float(xp.linalg.norm(psi))
    if norm_psi < 1e-15:
        return psi
    
    v = psi / norm_psi
    V[0] = v
    
    # First step
    w = H @ v
    alpha[0] = float(xp.real(xp.vdot(v, w)))
    w = w - alpha[0] * v
    
    # Build tridiagonal
    actual_dim = krylov_dim
    for j in range(1, krylov_dim):
        beta_j = float(xp.linalg.norm(w))
        
        if beta_j < 1e-12:
            actual_dim = j
            break
        
        beta[j-1] = beta_j
        v_new = w / beta_j
        V[j] = v_new
        
        w = H @ v_new
        alpha[j] = float(xp.real(xp.vdot(v_new, w)))
        w = w - alpha[j] * v_new - beta[j-1] * V[j-1]
    
    # Tridiagonal matrix (CPU)
    T = np.diag(alpha[:actual_dim])
    if actual_dim > 1:
        T += np.diag(beta[:actual_dim-1], k=1)
        T += np.diag(beta[:actual_dim-1], k=-1)
    
    # exp(-i dt T)
    exp_T = scipy_expm(-1j * dt * T)
    
    # Apply
    e0 = np.zeros(actual_dim, dtype=np.complex128)
    e0[0] = 1.0
    y = exp_T @ e0
    
    if is_gpu:
        y = cp.asarray(y)
    
    # Reconstruct
    psi_new = norm_psi * (V[:actual_dim].T @ y)
    psi_new = psi_new / xp.linalg.norm(psi_new)
    
    return psi_new


# =============================================================================
# DSE Solver
# =============================================================================

class DSESolver:
    """
    DSE (Direct Schrödinger Evolution) ソルバー
    
    Schrödinger 方程式を履歴付きで解く
    
    Features:
    - 標準量子力学モード（memory off）
    - DSE モード（memory on）
    - 適応的メモリ強度
    - 各種 Hamiltonian source 対応
    """
    
    def __init__(self,
                 H_kinetic,
                 H_potential,
                 gamma_memory: float = 1.0,
                 eta: float = 0.1,
                 krylov_dim: int = 30,
                 use_memory: bool = True,
                 use_gpu: bool = False,
                 max_history: int = 1000):
        """
        Args:
            H_kinetic: 運動エネルギー Hamiltonian
            H_potential: ポテンシャル Hamiltonian
            gamma_memory: メモリ指数
            eta: メモリ強度 [0, 1]
            krylov_dim: Krylov 次元
            use_memory: メモリ効果を使うか
            use_gpu: GPU 使用
            max_history: 最大履歴数
        """
        self.H_K = H_kinetic
        self.H_V = H_potential
        self.H = H_kinetic + H_potential
        
        self.gamma_memory = gamma_memory
        self.eta = eta
        self.krylov_dim = krylov_dim
        self.use_memory = use_memory
        self.use_gpu = use_gpu and HAS_CUPY
        
        self.xp = cp if self.use_gpu else np
        
        # Memory Kernel
        if use_memory and HAS_MEMORY_KERNEL:
            self.kernel = MemoryKernel(gamma_memory=gamma_memory, use_gpu=use_gpu)
        else:
            self.kernel = None
        
        # History Manager（オプション、大規模用）
        if HAS_HISTORY_MANAGER:
            self.history_manager = HistoryManager(max_history=max_history, use_gpu=use_gpu)
        else:
            self.history_manager = None
        
        # 内部状態
        self._history_states: List[np.ndarray] = []
        self._history_times: List[float] = []
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_sparse_engine(cls,
                           engine,
                           geometry,
                           T: float = 300.0,
                           sigma: float = 0.0,
                           gamma_memory: float = 1.0,
                           eta: float = 0.1,
                           **kwargs) -> 'DSESolver':
        """
        SparseEngine + EnvironmentBuilder から作成
        
        Args:
            engine: SparseEngine instance
            geometry: SystemGeometry
            T: 温度 (K)
            sigma: 応力
            gamma_memory: メモリ指数
            eta: メモリ強度
            **kwargs: EnvironmentBuilder への追加引数
        """
        try:
            from environment_operators import EnvironmentBuilder
        except ImportError:
            raise ImportError("environment_operators module required")
        
        builder = EnvironmentBuilder(engine, **kwargs)
        H_K, H_V = builder.build(geometry, T=T, sigma=sigma)
        
        return cls(H_K, H_V, gamma_memory=gamma_memory, eta=eta,
                   use_gpu=engine.use_gpu)
    
    @classmethod
    def from_pyscf(cls,
                   mf,
                   gamma_memory: float = 1.0,
                   eta: float = 0.1,
                   **kwargs) -> 'DSESolver':
        """
        PySCF から H を借りて作成
        
        Args:
            mf: 収束した SCF オブジェクト
            gamma_memory: メモリ指数
            eta: メモリ強度
        """
        try:
            from pyscf import gto
        except ImportError:
            raise ImportError("PySCF required")
        
        mol = mf.mol
        
        # 1電子積分から H を構築
        h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        
        # スパース行列として
        H_K = sp.csr_matrix(mol.intor('int1e_kin'), dtype=np.complex128)
        H_V = sp.csr_matrix(mol.intor('int1e_nuc'), dtype=np.complex128)
        
        # 注: これは1電子近似。多体効果は別途必要。
        
        return cls(H_K, H_V, gamma_memory=gamma_memory, eta=eta, **kwargs)
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def compute_lambda(self, psi) -> float:
        """
        Λ = K / |V| を計算
        
        Args:
            psi: 状態ベクトル
            
        Returns:
            安定性パラメータ Λ
        """
        xp = self.xp
        
        K = float(xp.real(xp.vdot(psi, self.H_K @ psi)))
        V = float(xp.real(xp.vdot(psi, self.H_V @ psi)))
        
        return abs(K) / (abs(V) + 1e-10)
    
    def compute_energy(self, psi) -> float:
        """エネルギーを計算"""
        xp = self.xp
        return float(xp.real(xp.vdot(psi, self.H @ psi)))
    
    def _compute_memory_state(self, t: float) -> Optional[np.ndarray]:
        """
        メモリ重み付け状態を計算
        
        |ψ_memory⟩ = Σ K(t-τ) × |ψ(τ)⟩
        """
        if len(self._history_states) == 0:
            return None
        
        xp = self.xp
        
        # カーネルからの重み
        if self.kernel is not None:
            times = np.array(self._history_times)
            weights = self.kernel.integrate(t, times)
        else:
            # フォールバック: 単純な指数減衰
            weights = np.array([
                np.exp(-(t - tau) / 10.0) for tau in self._history_times
            ])
        
        weights = weights / (weights.sum() + 1e-10)
        
        # 状態の重ね合わせ
        psi_mem = xp.zeros_like(self._history_states[0])
        for w, psi_past in zip(weights, self._history_states):
            psi_mem = psi_mem + w * psi_past
        
        norm = xp.linalg.norm(psi_mem)
        if norm > 1e-10:
            psi_mem = psi_mem / norm
        
        return psi_mem
    
    def evolve_step(self, psi, t: float, dt: float) -> Tuple[np.ndarray, float]:
        """
        1ステップの時間発展
        
        |ψ(t+dt)⟩ = (1-η) U(dt)|ψ(t)⟩ + η |ψ_memory⟩
        
        Args:
            psi: 現在の状態
            t: 現在時刻
            dt: 時間刻み
            
        Returns:
            (新しい状態, メモリ寄与)
        """
        xp = self.xp
        
        # 1. 標準 Lanczos 発展
        psi_unitary = lanczos_expm_multiply(self.H, psi, dt, self.krylov_dim)
        
        # 2. メモリ項
        memory_contrib = 0.0
        
        if self.use_memory and len(self._history_states) > 0:
            psi_memory = self._compute_memory_state(t)
            
            if psi_memory is not None:
                # 混合
                psi_new = (1 - self.eta) * psi_unitary + self.eta * psi_memory
                psi_new = psi_new / xp.linalg.norm(psi_new)
                
                # メモリ寄与を計算
                memory_contrib = float(xp.abs(xp.vdot(psi_unitary, psi_memory)))
            else:
                psi_new = psi_unitary
        else:
            psi_new = psi_unitary
        
        # 3. 履歴に追加
        self._history_states.append(psi_new.copy())
        self._history_times.append(t + dt)
        
        # 履歴の上限
        if len(self._history_states) > 1000:
            self._history_states = self._history_states[-1000:]
            self._history_times = self._history_times[-1000:]
        
        # 4. MemoryKernel にも追加
        if self.kernel is not None:
            r = self.compute_lambda(psi_new)  # Λ を位置として使う
            self.kernel.add_state(t + dt, r, psi_new)
        
        return psi_new, memory_contrib
    
    def run(self,
            psi_initial,
            t_end: float = 10.0,
            dt: float = 0.1,
            t_start: float = 0.0,
            observables: Optional[Dict[str, Any]] = None,
            callback: Optional[Callable] = None,
            verbose: bool = True) -> DSEResult:
        """
        時間発展を実行
        
        Args:
            psi_initial: 初期状態
            t_end: 終了時刻
            dt: 時間刻み
            t_start: 開始時刻
            observables: 測定する物理量 {'name': operator}
            callback: 各ステップで呼ばれる関数
            verbose: 進捗表示
            
        Returns:
            DSEResult
        """
        xp = self.xp
        
        # リセット
        self._history_states.clear()
        self._history_times.clear()
        if self.kernel is not None:
            self.kernel = MemoryKernel(gamma_memory=self.gamma_memory, 
                                        use_gpu=self.use_gpu)
        
        # 時間グリッド
        n_steps = int((t_end - t_start) / dt)
        times = np.linspace(t_start, t_end, n_steps + 1)
        
        # 初期状態
        psi = psi_initial.copy()
        if self.use_gpu and not isinstance(psi, cp.ndarray):
            psi = cp.asarray(psi)
        
        # 結果格納
        states = [psi.copy()]
        energies = [self.compute_energy(psi)]
        lambdas = [self.compute_lambda(psi)]
        memory_contribs = [0.0]
        obs_results = {name: [float(xp.real(xp.vdot(psi, op @ psi)))] 
                       for name, op in (observables or {}).items()}
        
        # 初期状態を履歴に
        self._history_states.append(psi.copy())
        self._history_times.append(t_start)
        
        if verbose:
            print(f"DSE Solver: {n_steps} steps, dt={dt}")
            print(f"  Memory: {'ON' if self.use_memory else 'OFF'} "
                  f"(γ={self.gamma_memory:.2f}, η={self.eta:.2f})")
            print(f"  Backend: {'GPU' if self.use_gpu else 'CPU'}")
        
        t0_wall = time_module.time()
        
        # 時間発展ループ
        for i, t in enumerate(times[:-1]):
            psi, mem_contrib = self.evolve_step(psi, t, dt)
            
            states.append(psi.copy())
            energies.append(self.compute_energy(psi))
            lambdas.append(self.compute_lambda(psi))
            memory_contribs.append(mem_contrib)
            
            # 物理量測定
            for name, op in (observables or {}).items():
                obs_results[name].append(float(xp.real(xp.vdot(psi, op @ psi))))
            
            # コールバック
            if callback:
                callback(i, t, psi)
            
            # 進捗
            if verbose and (i + 1) % max(n_steps // 4, 1) == 0:
                elapsed = time_module.time() - t0_wall
                print(f"  Step {i+1}/{n_steps}: Λ={lambdas[-1]:.4f}, "
                      f"E={energies[-1]:.4f}, t={elapsed:.2f}s")
        
        wall_time = time_module.time() - t0_wall
        
        if verbose:
            print(f"  ✅ Done in {wall_time:.2f}s")
        
        return DSEResult(
            times=times,
            states=states,
            energies=energies,
            lambdas=lambdas,
            memory_contributions=memory_contribs,
            observables=obs_results,
            n_steps=n_steps,
            wall_time=wall_time,
            use_memory=self.use_memory,
            gamma_memory=self.gamma_memory,
            eta=self.eta
        )
    
    def compare_with_standard(self,
                               psi_initial,
                               t_end: float = 10.0,
                               dt: float = 0.1,
                               verbose: bool = True) -> Tuple[DSEResult, DSEResult]:
        """
        DSE と標準量子力学を比較
        
        Returns:
            (dse_result, standard_result)
        """
        # DSE
        result_dse = self.run(psi_initial, t_end, dt, verbose=verbose)
        
        # 標準（メモリなし）
        solver_std = DSESolver(
            self.H_K, self.H_V,
            use_memory=False,
            use_gpu=self.use_gpu
        )
        result_std = solver_std.run(psi_initial, t_end, dt, verbose=verbose)
        
        if verbose:
            print("\n--- Comparison ---")
            print(f"DSE final Λ: {result_dse.final_lambda:.4f}")
            print(f"Std final Λ: {result_std.final_lambda:.4f}")
            print(f"Λ difference: {abs(result_dse.final_lambda - result_std.final_lambda):.4f}")
        
        return result_dse, result_std


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_dse(H, psi0, t_end: float = 10.0, dt: float = 0.1,
              gamma: float = 1.0, eta: float = 0.1,
              verbose: bool = True) -> DSEResult:
    """
    簡易 DSE 実行
    
    H = H_K + H_V の分離がない場合用
    """
    # H を全て「運動エネルギー」として扱う
    if sp.issparse(H):
        H_V = sp.csr_matrix(H.shape, dtype=H.dtype)
    else:
        H_V = np.zeros_like(H)
    
    solver = DSESolver(H, H_V, gamma_memory=gamma, eta=eta)
    return solver.run(psi0, t_end, dt, verbose=verbose)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DSE Solver Test")
    print("=" * 70)
    
    print(f"\nMemoryKernel available: {HAS_MEMORY_KERNEL}")
    print(f"HistoryManager available: {HAS_HISTORY_MANAGER}")
    print(f"CuPy available: {HAS_CUPY}")
    
    # 簡単な2準位系
    print("\n--- 2-Level System Test ---")
    
    # Pauli 行列
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    # H = -sz + 0.5*sx
    H_K = sp.csr_matrix(-sz, dtype=np.complex128)
    H_V = sp.csr_matrix(0.5 * sx, dtype=np.complex128)
    
    # 初期状態 |↑⟩
    psi0 = np.array([1, 0], dtype=np.complex128)
    
    # DSE Solver
    solver = DSESolver(H_K, H_V, gamma_memory=1.0, eta=0.1)
    result = solver.run(psi0, t_end=5.0, dt=0.1, verbose=True)
    
    print(result.summary())
    
    # 比較
    print("\n--- DSE vs Standard Comparison ---")
    result_dse, result_std = solver.compare_with_standard(psi0, t_end=5.0, dt=0.1)
    
    print("\n" + "=" * 70)
    print("✅ DSE Solver Test Complete!")
    print("=" * 70)
