"""
Holographic Measurement Protocol
================================

AdS/CFT 双対性検証のための特殊な測定プロトコル。

通常の DSE solver (time_evolution.py) は汎用的に使いたいので、
AdS/CFT 検証に必要な PRE/POST 測定はここに分離。

PhaseShift-X で学んだノウハウ:
  - PRE/POST 分離: 因果の方向性を明確化
  - Delayed measurement: zero-lag correlation 回避
  - c_eff cap: 発散防止

Usage:
    from memory_dft.holographic import HolographicMeasurement
    
    # 通常の solver でまず発展
    engine = TimeEvolutionEngine(H_K, H_V, config)
    result = engine.run(psi0)
    
    # Holographic measurement でラップ
    hm = HolographicMeasurement(result)
    records = hm.compute_pre_post_lambda(lambda_func)
    
    # 双対性検証
    duality = hm.verify_duality()

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from .dual import (
    HolographicDual,
    transfer_entropy,
    crosscorr_at_lags,
    spearman_corr,
    verify_duality,
)


@dataclass
class MeasurementRecord:
    """1ステップの測定記録"""
    t: int                    # ステップ番号
    lambda_pre: float         # 更新前のλ
    lambda_post: float        # 更新後のλ
    S_RT: float              # RT entropy
    energy: float            # エネルギー
    delta_lambda: float = 0.0  # λの変化量


@dataclass
class HolographicMeasurementResult:
    """Holographic測定の結果"""
    records: List[MeasurementRecord]
    
    # 時系列データ（便利用）
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_pre_series: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_post_series: np.ndarray = field(default_factory=lambda: np.array([]))
    S_RT_series: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_series: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 双対性検証結果
    duality_results: Optional[Dict[str, Any]] = None
    
    def to_arrays(self):
        """recordsから配列を生成"""
        if not self.records:
            return
        self.times = np.array([r.t for r in self.records])
        self.lambda_pre_series = np.array([r.lambda_pre for r in self.records])
        self.lambda_post_series = np.array([r.lambda_post for r in self.records])
        self.S_RT_series = np.array([r.S_RT for r in self.records])
        self.energy_series = np.array([r.energy for r in self.records])


class HolographicMeasurement:
    """
    AdS/CFT 双対性検証のための測定プロトコル
    
    PRE/POST λ測定により、因果の方向性を明確化する。
    
    使用フロー:
      1. 通常の solver で時間発展 → EvolutionResult
      2. HolographicMeasurement でラップ
      3. PRE/POST λ を計算
      4. 双対性を検証
    """
    
    def __init__(self, 
                 gate_delay: int = 1,
                 measurement_interval: int = 1):
        """
        Args:
            gate_delay: 測定の遅延ステップ数（因果分離用）
            measurement_interval: 何ステップごとに測定するか
        """
        self.gate_delay = gate_delay
        self.measurement_interval = measurement_interval
        
        self.holo = HolographicDual()
        self.records: List[MeasurementRecord] = []
        self.phi_history: List[float] = []
    
    def measure_from_evolution_result(self,
                                      result,  # EvolutionResult
                                      lambda_transform: Optional[Callable] = None
                                      ) -> HolographicMeasurementResult:
        """
        EvolutionResult から Holographic 測定を行う
        
        Args:
            result: TimeEvolutionEngine.run() の結果
            lambda_transform: λ値の変換関数（オプション）
        
        Returns:
            HolographicMeasurementResult
        """
        lambdas = np.array(result.lambdas)
        energies = np.array(result.energies)
        times = np.array(result.times)
        
        if lambda_transform:
            lambdas = np.array([lambda_transform(l) for l in lambdas])
        
        records = []
        phi_history = []
        phi = 0.0
        
        n_steps = len(lambdas)
        
        for i in range(n_steps):
            # PRE: 現在のλ（更新前として扱う）
            lambda_pre = lambdas[i]
            
            # POST: gate_delay 後のλ（更新後として扱う）
            if i + self.gate_delay < n_steps:
                lambda_post = lambdas[i + self.gate_delay]
            else:
                lambda_post = lambdas[-1]
            
            # 位相蓄積
            phi += lambda_pre * 0.1  # dt = 0.1 相当
            phi_history.append(phi)
            
            # Bulk 構築 & S_RT
            if len(phi_history) >= 2:
                self.holo.history_to_bulk(phi_history)
                S_RT = self.holo.rt_entropy()
            else:
                S_RT = 0.0
            
            record = MeasurementRecord(
                t=i,
                lambda_pre=float(lambda_pre),
                lambda_post=float(lambda_post),
                S_RT=float(S_RT),
                energy=float(energies[i]) if i < len(energies) else 0.0,
                delta_lambda=float(lambda_post - lambda_pre)
            )
            records.append(record)
        
        self.records = records
        self.phi_history = phi_history
        
        result_obj = HolographicMeasurementResult(records=records)
        result_obj.to_arrays()
        
        return result_obj
    
    def measure_live(self,
                     engine,  # TimeEvolutionEngine
                     psi_initial,
                     lambda_calculator: Callable,
                     observables: Optional[Dict] = None,
                     ) -> HolographicMeasurementResult:
        """
        リアルタイムで PRE/POST 測定しながら時間発展
        
        より厳密な因果解析が必要な場合に使用。
        
        Args:
            engine: TimeEvolutionEngine インスタンス
            psi_initial: 初期状態
            lambda_calculator: λ計算関数
            observables: 測定する物理量
        """
        # pending queue for delayed measurement
        pending_lambda: List[Optional[float]] = [None] * (self.gate_delay + 1)
        
        records = []
        phi_history = []
        phi = 0.0
        
        # カスタムコールバックで PRE/POST 測定
        def measurement_callback(step, t, psi, result):
            nonlocal phi
            
            # PRE: 現在のλ
            lambda_current = lambda_calculator(psi)
            
            # POST: 遅延キューから取得
            lambda_delayed = pending_lambda.pop(0)
            pending_lambda.append(lambda_current)
            
            if lambda_delayed is None:
                lambda_delayed = lambda_current
            
            # 位相蓄積
            phi += lambda_current * engine.config.dt
            phi_history.append(phi)
            
            # S_RT
            if len(phi_history) >= 2:
                self.holo.history_to_bulk(phi_history)
                S_RT = self.holo.rt_entropy()
            else:
                S_RT = 0.0
            
            record = MeasurementRecord(
                t=step,
                lambda_pre=float(lambda_current),
                lambda_post=float(lambda_delayed),
                S_RT=float(S_RT),
                energy=float(result.energies[-1]) if result.energies else 0.0,
                delta_lambda=float(lambda_delayed - lambda_current)
            )
            records.append(record)
        
        # 発展実行
        engine.run(psi_initial, observables, callback=measurement_callback)
        
        self.records = records
        self.phi_history = phi_history
        
        result_obj = HolographicMeasurementResult(records=records)
        result_obj.to_arrays()
        
        return result_obj
    
    def verify_duality(self, 
                       result: Optional[HolographicMeasurementResult] = None
                       ) -> Dict[str, Any]:
        """
        AdS/CFT 双対性を検証
        
        PRE λ (boundary) と S_RT (bulk) 間の情報フローを解析
        """
        if result is None:
            result = HolographicMeasurementResult(records=self.records)
            result.to_arrays()
        
        if len(result.lambda_pre_series) < 10:
            return {'error': 'Not enough data points'}
        
        # Bulk observable: S_RT
        # Boundary observable: λ_pre
        bulk_series = result.S_RT_series
        boundary_series = result.lambda_pre_series
        
        # 双対性検証
        duality = verify_duality(
            bulk_history=list(bulk_series),
            boundary_history=list(boundary_series),
            n_bins=3,
            maxlag=16
        )
        
        result.duality_results = duality
        
        return duality
    
    def summary(self, result: HolographicMeasurementResult) -> str:
        """結果のサマリーを文字列で返す"""
        if result.duality_results is None:
            self.verify_duality(result)
        
        d = result.duality_results
        
        lines = [
            "=" * 60,
            "HOLOGRAPHIC MEASUREMENT SUMMARY",
            "=" * 60,
            f"  Steps measured: {len(result.records)}",
            f"  Gate delay: {self.gate_delay}",
            "",
            "--- Statistics ---",
            f"  λ_pre  mean: {result.lambda_pre_series.mean():.4f}",
            f"  λ_post mean: {result.lambda_post_series.mean():.4f}",
            f"  S_RT   mean: {result.S_RT_series.mean():.4f}",
            "",
            "--- Duality Verification ---",
            f"  TE(Bulk → Boundary): {d['TE_bulk_to_boundary']:.4f} nats",
            f"  TE(Boundary → Bulk): {d['TE_boundary_to_bulk']:.4f} nats",
            f"  Duality Index: {d['duality_index']:.4f}",
            f"  Best lag: {d['best_lag']}",
            f"  Max correlation: {d['max_corr']:.4f}",
            "",
        ]
        
        di = d['duality_index']
        if di < 0.2:
            lines.append("  ✓ STRONG DUALITY (index < 0.2)")
        elif di < 0.5:
            lines.append("  ○ MODERATE DUALITY (0.2 < index < 0.5)")
        else:
            lines.append("  ✗ WEAK DUALITY (index > 0.5)")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_holographic_measurement(evolution_result,
                                  gate_delay: int = 1
                                  ) -> HolographicMeasurementResult:
    """
    EvolutionResult から簡単に Holographic 測定
    
    Usage:
        result = engine.run(psi0)
        hm_result = quick_holographic_measurement(result)
        print(hm_result.duality_results)
    """
    hm = HolographicMeasurement(gate_delay=gate_delay)
    result = hm.measure_from_evolution_result(evolution_result)
    hm.verify_duality(result)
    print(hm.summary(result))
    return result


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Holographic Measurement Protocol Test")
    print("=" * 60)
    
    # ダミーの EvolutionResult をシミュレート
    class DummyResult:
        def __init__(self, n_steps=100):
            np.random.seed(42)
            self.times = np.linspace(0, 10, n_steps)
            # ダミーのλ: pump + decay
            self.lambdas = []
            lam = 1.0
            for i in range(n_steps):
                if i < 50:
                    lam += 0.02 * np.sin(0.2 * i) + 0.01 * np.random.randn()
                else:
                    lam *= 0.98
                    lam += 0.005 * np.random.randn()
                self.lambdas.append(lam)
            self.lambdas = np.array(self.lambdas)
            self.energies = -self.lambdas + np.random.randn(n_steps) * 0.1
    
    dummy = DummyResult(100)
    
    print(f"\nDummy evolution: {len(dummy.lambdas)} steps")
    print(f"  λ range: [{dummy.lambdas.min():.3f}, {dummy.lambdas.max():.3f}]")
    
    # Holographic measurement
    hm = HolographicMeasurement(gate_delay=1)
    result = hm.measure_from_evolution_result(dummy)
    
    # Duality verification
    duality = hm.verify_duality(result)
    
    # Summary
    print(hm.summary(result))
