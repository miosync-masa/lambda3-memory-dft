"""
Time Evolution Engine for Memory-DFT
====================================

高レベルの時間発展インターフェース

Features:
- Memory-DFT / 標準量子力学の切り替え
- 適応時間刻み
- 物理量のモニタリング
- Λ軌跡の追跡

H-CSP公理との対応:
- 公理4（再帰生成）: Λ(t+Δt) = F(Λ(t), Λ̇(t))
- 公理5（拍動的平衡）: Λ̇≠0 かつ ⟨Λ⟩≈const

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
import time

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from .memory_kernel import CompositeMemoryKernel
from .history_manager import HistoryManager
from ..solvers.lanczos_memory import MemoryLanczosSolver, AdaptiveMemorySolver, lanczos_expm_multiply


@dataclass
class EvolutionConfig:
    """時間発展の設定"""
    t_start: float = 0.0
    t_end: float = 10.0
    dt: float = 0.1
    
    # Memory-DFT設定
    use_memory: bool = True
    memory_strength: float = 0.1
    adaptive_memory: bool = False
    
    # カーネル設定
    gamma_field: float = 1.0
    beta_phys: float = 0.5
    tau0_phys: float = 10.0
    t_react_chem: float = 5.0
    
    # 適応時間刻み
    adaptive_dt: bool = False
    dt_min: float = 0.001
    dt_max: float = 1.0
    error_tol: float = 1e-6
    
    # その他
    krylov_dim: int = 30
    max_history: int = 1000
    verbose: bool = True


@dataclass
class EvolutionResult:
    """時間発展の結果"""
    times: np.ndarray
    states: List
    
    # 物理量
    energies: List[float] = field(default_factory=list)
    lambdas: List[float] = field(default_factory=list)
    observables: Dict[str, List[float]] = field(default_factory=dict)
    
    # Memory診断
    memory_overlaps: List[float] = field(default_factory=list)
    eta_history: List[float] = field(default_factory=list)
    
    # メタデータ
    config: EvolutionConfig = None
    wall_time: float = 0.0
    
    def get_final_state(self):
        """最終状態"""
        return self.states[-1]
    
    def get_lambda_trajectory(self) -> np.ndarray:
        """Λ軌跡"""
        return np.array(self.lambdas)
    
    def check_pulsation(self, window: int = 10) -> Dict[str, float]:
        """
        拍動的平衡（公理5）のチェック
        
        Λ̇ ≠ 0 かつ ⟨Λ(t+Δt)⟩ ≈ Λ(t)
        """
        if len(self.lambdas) < window * 2:
            return {'pulsation': False, 'lambda_var': 0, 'lambda_mean': 0}
        
        lambdas = np.array(self.lambdas)
        
        # 局所変動
        lambda_diff = np.abs(np.diff(lambdas))
        local_var = np.mean(lambda_diff[-window:])
        
        # 大域平均
        lambda_mean = np.mean(lambdas[-window:])
        
        # 拍動判定: 変動ありかつ平均安定
        pulsation = local_var > 1e-4 and np.std(lambdas[-window:]) / (lambda_mean + 1e-10) < 0.1
        
        return {
            'pulsation': pulsation,
            'lambda_var': local_var,
            'lambda_mean': lambda_mean,
            'lambda_std': np.std(lambdas[-window:])
        }


class TimeEvolutionEngine:
    """
    時間発展エンジン
    
    Memory-DFT と標準量子力学の統一インターフェース
    """
    
    def __init__(self, 
                 H_kinetic,
                 H_potential,
                 config: Optional[EvolutionConfig] = None,
                 use_gpu: bool = True):
        """
        Args:
            H_kinetic: 運動エネルギーハミルトニアン
            H_potential: ポテンシャルハミルトニアン
            config: 発展設定
            use_gpu: GPU使用フラグ
        """
        self.H_K = H_kinetic
        self.H_V = H_potential
        self.H = H_kinetic + H_potential
        
        self.config = config or EvolutionConfig()
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        
        # Memory-DFTコンポーネント
        if self.config.use_memory:
            self._setup_memory_components()
    
    def _setup_memory_components(self):
        """Memory-DFTコンポーネントの初期化"""
        cfg = self.config
        
        # Memory kernel
        self.kernel = CompositeMemoryKernel(
            gamma_field=cfg.gamma_field,
            beta_phys=cfg.beta_phys,
            tau0_phys=cfg.tau0_phys,
            t_react_chem=cfg.t_react_chem
        )
        
        # History manager
        self.history = HistoryManager(
            max_history=cfg.max_history,
            use_gpu=self.use_gpu
        )
        
        # Solver
        if cfg.adaptive_memory:
            self.solver = AdaptiveMemorySolver(
                memory_kernel=self.kernel,
                history_manager=self.history,
                krylov_dim=cfg.krylov_dim,
                use_gpu=self.use_gpu
            )
        else:
            self.solver = MemoryLanczosSolver(
                memory_kernel=self.kernel,
                history_manager=self.history,
                memory_strength=cfg.memory_strength,
                krylov_dim=cfg.krylov_dim,
                use_gpu=self.use_gpu
            )
    
    def compute_lambda(self, psi) -> float:
        """Λ = K / |V| を計算"""
        xp = self.xp
        
        K = float(xp.real(xp.vdot(psi, self.H_K @ psi)))
        V = float(xp.real(xp.vdot(psi, self.H_V @ psi)))
        
        return abs(K) / (abs(V) + 1e-10)
    
    def run(self,
            psi_initial,
            observables: Optional[Dict[str, Any]] = None,
            callback: Optional[Callable] = None) -> EvolutionResult:
        """
        時間発展を実行
        
        Args:
            psi_initial: 初期状態
            observables: 測定する物理量 {'name': operator}
            callback: 各ステップで呼ばれる関数
            
        Returns:
            EvolutionResult
        """
        cfg = self.config
        xp = self.xp
        
        n_steps = int((cfg.t_end - cfg.t_start) / cfg.dt)
        times = np.linspace(cfg.t_start, cfg.t_end, n_steps + 1)
        
        # 初期化
        psi = psi_initial.copy()
        
        result = EvolutionResult(
            times=times,
            states=[psi.copy()],
            config=cfg
        )
        
        # 初期値
        result.energies.append(float(xp.real(xp.vdot(psi, self.H @ psi))))
        result.lambdas.append(self.compute_lambda(psi))
        
        if observables:
            for name in observables:
                result.observables[name] = []
            for name, op in observables.items():
                val = float(xp.real(xp.vdot(psi, op @ psi)))
                result.observables[name].append(val)
        
        if cfg.verbose:
            print(f"⏱️ Time Evolution: {n_steps} steps")
            print(f"   Mode: {'Memory-DFT' if cfg.use_memory else 'Standard QM'}")
            if cfg.use_memory:
                print(f"   Memory strength: η={cfg.memory_strength}")
        
        t0_wall = time.time()
        
        # 時間発展ループ
        for i, t in enumerate(times[:-1]):
            dt = cfg.dt
            
            if cfg.use_memory:
                # Memory-DFT発展
                psi = self.solver.evolve(
                    self.H, psi, t, dt,
                    lambda_calculator=self.compute_lambda
                )
                
                # 診断情報
                if hasattr(self.solver, 'eta'):
                    result.eta_history.append(self.solver.eta)
                if self.solver.diagnostics['overlap_with_memory']:
                    result.memory_overlaps.append(
                        self.solver.diagnostics['overlap_with_memory'][-1]
                    )
            else:
                # 標準量子力学
                psi = lanczos_expm_multiply(self.H, psi, dt, cfg.krylov_dim)
            
            # 状態保存
            result.states.append(psi.copy())
            
            # 物理量
            result.energies.append(float(xp.real(xp.vdot(psi, self.H @ psi))))
            result.lambdas.append(self.compute_lambda(psi))
            
            if observables:
                for name, op in observables.items():
                    val = float(xp.real(xp.vdot(psi, op @ psi)))
                    result.observables[name].append(val)
            
            # コールバック
            if callback:
                callback(i, t, psi, result)
            
            # 進捗
            if cfg.verbose and (i + 1) % max(n_steps // 4, 1) == 0:
                elapsed = time.time() - t0_wall
                Lambda = result.lambdas[-1]
                print(f"   Step {i+1}/{n_steps}: Λ={Lambda:.4f}, t={elapsed:.2f}s")
        
        result.wall_time = time.time() - t0_wall
        
        if cfg.verbose:
            print(f"   ✅ Done in {result.wall_time:.2f}s")
            puls = result.check_pulsation()
            if puls['pulsation']:
                print(f"   🫀 Pulsation detected! (var={puls['lambda_var']:.4f})")
        
        return result
    
    def compare_with_standard(self, psi_initial, observables=None) -> Tuple[EvolutionResult, EvolutionResult]:
        """
        Memory-DFT と標準量子力学を比較
        
        Returns:
            (memory_result, standard_result)
        """
        # Memory-DFT
        result_memory = self.run(psi_initial, observables)
        
        # 標準（Memoryなし）
        cfg_std = EvolutionConfig(
            t_start=self.config.t_start,
            t_end=self.config.t_end,
            dt=self.config.dt,
            use_memory=False,
            verbose=self.config.verbose
        )
        
        engine_std = TimeEvolutionEngine(
            self.H_K, self.H_V, cfg_std, self.use_gpu
        )
        result_std = engine_std.run(psi_initial, observables)
        
        return result_memory, result_std


# =============================================================================
# Utility Functions
# =============================================================================

def quick_evolve(H, psi0, t_end: float = 10.0, dt: float = 0.1,
                 memory: bool = True, verbose: bool = True):
    """
    簡易時間発展
    
    H = H_K + H_V の分離がない場合、全体をHとして扱う
    """
    # Hを運動エネルギーとポテンシャルに分離できない場合
    # 全体を「運動エネルギー」として扱う（Λ計算は意味をなさない）
    
    config = EvolutionConfig(
        t_start=0,
        t_end=t_end,
        dt=dt,
        use_memory=memory,
        verbose=verbose
    )
    
    # ダミーのH_V（ゼロ行列）
    if hasattr(H, 'shape'):
        import scipy.sparse as sp
        H_V = sp.csr_matrix(H.shape, dtype=H.dtype)
    else:
        H_V = H * 0
    
    engine = TimeEvolutionEngine(H, H_V, config, use_gpu=HAS_CUPY)
    return engine.run(psi0)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Time Evolution Engine Test")
    print("="*70)
    
    import sys
    sys.path.insert(0, '/home/claude/memory_dft')
    from core.sparse_engine import SparseHamiltonianEngine
    
    # 4サイト鎖
    engine = SparseHamiltonianEngine(n_sites=4, use_gpu=False, verbose=False)
    geom = engine.build_chain_geometry(L=4)
    
    H_K, H_V = engine.build_heisenberg_hamiltonian(geom.bonds, J=1.0, Jz=0.5)
    
    # 初期状態（ランダム）
    xp = engine.xp
    psi0 = xp.random.randn(engine.dim) + 1j * xp.random.randn(engine.dim)
    psi0 = psi0 / xp.linalg.norm(psi0)
    
    # Memory-DFT発展
    config = EvolutionConfig(
        t_end=5.0,
        dt=0.1,
        use_memory=True,
        memory_strength=0.1,
        verbose=True
    )
    
    evol_engine = TimeEvolutionEngine(H_K, H_V, config, use_gpu=False)
    result = evol_engine.run(psi0)
    
    print(f"\nResults:")
    print(f"  Final Λ: {result.lambdas[-1]:.4f}")
    print(f"  Λ range: [{min(result.lambdas):.4f}, {max(result.lambdas):.4f}]")
    print(f"  Energy drift: {abs(result.energies[-1] - result.energies[0]):.6f}")
    
    puls = result.check_pulsation()
    print(f"  Pulsation: {puls}")
    
    print("\n✅ Time Evolution Engine OK!")
