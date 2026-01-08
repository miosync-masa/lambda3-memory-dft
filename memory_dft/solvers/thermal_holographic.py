#!/usr/bin/env python3
"""
Thermal Holographic Evolution Module
=====================================

温度変化 × Memory効果 × Holographic測定 × 材料破壊予測

【核心的洞察】
  Energy = topology の結び目
  質量 = topology
  熱 = 結び目を揺らす
  応力 = 結び目を引っ張る
  溶解 = 結び目がほどける
  Coherence = 結び目が揃ってる
  エントロピー = 結び目が散らばる
  
  → 全部 topology で統一！
  → トポロジー = 意味の保存則

【温度変化速度の効果】
  急冷（Quench）: dt小 → Memory効果強 → 非平衡凍結 → 残留応力
  徐冷（Anneal）: dt大 → Memory効果弱 → 平衡接近 → 応力解放

【第5の力との接続】
  τ₀ ≈ 10⁻¹³ s = Debye振動周期
  λ = c × τ₀ = 30 μm = 結晶粒サイズ = 第5の力の到達距離
  
  材料の残留応力 = 第5の力の閉じ込め
  中性子星の異常冷却 = 第5の力の漏れ出し

【スケール階層】
  音速 × τ₀ = 0.5 nm  ← 原子の世界（フォノン）
  光速 × τ₀ = 30 μm   ← 結晶粒の世界（第5の力）
  比率 c/v_s ≈ 60,000

Author: Tamaki & Masamichi Iizumi
Date: 2025-01
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

# =============================================================================
# Imports from existing modules
# =============================================================================

# Core environment operators
from memory_dft.core.environment_operators import (
    ThermalEnsemble,
    ThermalObservable,
    boltzmann_weights,
    T_to_beta,
    K_B_EV,
)

# Memory kernel
from memory_dft.core.memory_kernel import MemoryKernel

# DSE Solver
from memory_dft.solvers.dse_solver import DSESolver

# Holographic measurement
from memory_dft.holographic.measurement import (
    HolographicMeasurement,
    MeasurementRecord,
    HolographicMeasurementResult,
)

# Material failure / Topology analysis
from memory_dft.core.material_failure import (
    ThermalTopologyAnalyzer,
    StressTopologyAnalyzer,
    CombinedFailureAnalyzer,
    TopologyResult,
    ThermalTopologyResult,
    StressTopologyResult,
    FailurePrediction as MaterialFailurePrediction,
)

# =============================================================================
# Physical Constants
# =============================================================================

# Debye振動周期 - 原子の「息づかい」
TAU_0 = 1e-13  # s

# 光速・音速
C_LIGHT = 3e8      # m/s
V_SOUND = 5000     # m/s (金属の典型値)

# 特性長さ
LAMBDA_LIGHT = C_LIGHT * TAU_0    # 30 μm = 結晶粒サイズ = 第5の力到達距離
LAMBDA_PHONON = V_SOUND * TAU_0   # 0.5 nm = フォノン波長 ≈ 格子定数

# スケール比
SCALE_RATIO = C_LIGHT / V_SOUND   # ≈ 60,000


# =============================================================================
# Enums
# =============================================================================

class CoolingMode(Enum):
    """冷却モード"""
    QUENCH = "quench"      # 急冷
    ANNEAL = "anneal"      # 徐冷
    LINEAR = "linear"      # 線形
    EXPONENTIAL = "exp"    # 指数的
    CUSTOM = "custom"      # カスタム


class TopologyState(Enum):
    """Topology状態"""
    COHERENT = "coherent"       # 結び目が揃ってる（固体）
    FLUCTUATING = "fluctuating" # 揺らいでる（臨界付近）
    DISORDERED = "disordered"   # 散らばってる（液体）
    BROKEN = "broken"           # 切れた（破壊）


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ThermalHolographicRecord:
    """1ステップの記録"""
    step: int
    time: float
    temperature: float
    dt: float
    
    # Topology
    lambda_value: float           # λ = K/|V|
    coherence: float              # 位相コヒーレンス
    lindemann_delta: float        # Lindemann パラメータ
    topology_state: TopologyState
    
    # Holographic
    lambda_pre: float             # 更新前λ
    lambda_post: float            # 更新後λ
    S_RT: float                   # Bulk entropy
    phi_accumulated: float        # 蓄積位相
    
    # Energy
    energy: float
    kinetic: float
    potential: float
    
    # Memory
    gamma_memory: float           # Memory強度
    memory_contribution: float    # Memory項の寄与


@dataclass
class ThermalPath:
    """温度パス定義"""
    T_start: float
    T_end: float
    n_steps: int
    mode: CoolingMode = CoolingMode.LINEAR
    
    # Quench/Anneal パラメータ
    quench_rate: float = 100.0    # K/step (急冷)
    anneal_rate: float = 1.0      # K/step (徐冷)
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """温度列と dt 列を生成"""
        if self.mode == CoolingMode.QUENCH:
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.01)  # 小さいdt → Memory強
            
        elif self.mode == CoolingMode.ANNEAL:
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.5)   # 大きいdt → Memory弱
            
        elif self.mode == CoolingMode.LINEAR:
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.1)
            
        elif self.mode == CoolingMode.EXPONENTIAL:
            tau = self.n_steps / 3
            t = np.arange(self.n_steps)
            T_values = self.T_end + (self.T_start - self.T_end) * np.exp(-t / tau)
            dT = np.abs(np.gradient(T_values))
            dt_values = 0.1 / (dT / dT.mean() + 0.1)
            
        else:
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.1)
        
        return T_values, dt_values


@dataclass
class DualityMetrics:
    """双対性メトリクス"""
    TE_bulk_to_boundary: float
    TE_boundary_to_bulk: float
    duality_index: float
    best_lag: int
    max_correlation: float
    
    def is_strong_duality(self) -> bool:
        return self.duality_index < 0.2
    
    def is_moderate_duality(self) -> bool:
        return 0.2 <= self.duality_index < 0.5


@dataclass 
class FailurePrediction:
    """破壊予測"""
    will_fail: bool
    failure_step: Optional[int]
    failure_temperature: Optional[float]
    failure_site: Optional[int]
    failure_mechanism: str
    lambda_at_failure: float
    confidence: float


@dataclass
class ThermalHolographicResult:
    """全体結果"""
    records: List[ThermalHolographicRecord]
    thermal_path: ThermalPath
    
    T_range: Tuple[float, float] = (0.0, 0.0)
    lambda_range: Tuple[float, float] = (0.0, 0.0)
    coherence_range: Tuple[float, float] = (0.0, 0.0)
    
    duality: Optional[DualityMetrics] = None
    failure: Optional[FailurePrediction] = None
    
    # 追加: Topology analysis results
    thermal_topology: Optional[ThermalTopologyResult] = None
    stress_topology: Optional[StressTopologyResult] = None
    
    def compute_summary(self):
        """サマリー統計を計算"""
        if not self.records:
            return
            
        temps = [r.temperature for r in self.records]
        lambdas = [r.lambda_value for r in self.records]
        cohs = [r.coherence for r in self.records]
        
        self.T_range = (min(temps), max(temps))
        self.lambda_range = (min(lambdas), max(lambdas))
        self.coherence_range = (min(cohs), max(cohs))


# =============================================================================
# Main Class: ThermalHolographicEvolution
# =============================================================================

class ThermalHolographicEvolution:
    """
    温度変化 × Memory効果 × Holographic測定 × 材料破壊予測
    
    【統合アーキテクチャ】
      ThermalEnsemble (温度→分布→状態)
          ↓
      DSESolver (Memory付き時間発展)
          ↓
      HolographicMeasurement (PRE/POST λ, S_RT)
          ↓
      ThermalTopologyAnalyzer (Coherence, Lindemann, 破壊予測)
          ↓
      material_failure (計測・予測)
    
    Usage:
        # Hubbard モデルで初期化
        evolution = ThermalHolographicEvolution.from_hubbard(n_sites=4)
        
        # 急冷
        result_quench = evolution.quench(T_start=1000, T_end=100)
        
        # 徐冷
        result_anneal = evolution.anneal(T_start=1000, T_end=100)
        
        # 比較
        evolution.compare(result_quench, result_anneal)
    """
    
    def __init__(self,
                 H_K: np.ndarray,
                 H_V: np.ndarray,
                 ensemble: ThermalEnsemble,
                 solver: DSESolver,
                 measurement: HolographicMeasurement,
                 thermal_analyzer: ThermalTopologyAnalyzer,
                 lindemann_critical: float = 0.1):
        """
        Args:
            H_K: 運動エネルギー項
            H_V: ポテンシャル項
            ensemble: 熱アンサンブル
            solver: DSE ソルバー
            measurement: Holographic 測定器
            thermal_analyzer: 熱トポロジー解析器
            lindemann_critical: Lindemann 臨界値
        """
        self.H_K = H_K
        self.H_V = H_V
        self.H = H_K + H_V
        self.ensemble = ensemble
        self.solver = solver
        self.measurement = measurement
        self.thermal_analyzer = thermal_analyzer
        self.lindemann_critical = lindemann_critical
    
    @classmethod
    def from_hubbard(cls, n_sites: int = 4, t: float = 1.0, U: float = 2.0,
                     gamma_memory: float = 0.1, eta_memory: float = 0.1,
                     gate_delay: int = 1,
                     n_eigenstates: int = 20) -> 'ThermalHolographicEvolution':
        """
        Hubbard モデルから初期化
        """
        H_K, H_V = cls._build_hubbard(n_sites, t, U)
        H = H_K + H_V
        
        # ThermalEnsemble (from environment_operators.py)
        ensemble = ThermalEnsemble(H, n_eigenstates=n_eigenstates)
        
        # DSESolver (from dse_solver.py)
        solver = DSESolver(H_K, H_V, gamma_memory=gamma_memory, eta_memory=eta_memory)
        
        # HolographicMeasurement (from holographic/measurement.py)
        measurement = HolographicMeasurement(gate_delay=gate_delay)
        
        # ThermalTopologyAnalyzer (from material_failure.py)
        thermal_analyzer = ThermalTopologyAnalyzer(ensemble)
        
        return cls(H_K, H_V, ensemble, solver, measurement, thermal_analyzer)
    
    @classmethod
    def from_hamiltonian(cls, H_K: np.ndarray, H_V: np.ndarray,
                         gamma_memory: float = 0.1, eta_memory: float = 0.1,
                         gate_delay: int = 1,
                         n_eigenstates: int = 20) -> 'ThermalHolographicEvolution':
        """
        任意のハミルトニアンから初期化
        """
        H = H_K + H_V
        
        ensemble = ThermalEnsemble(H, n_eigenstates=n_eigenstates)
        solver = DSESolver(H_K, H_V, gamma_memory=gamma_memory, eta_memory=eta_memory)
        measurement = HolographicMeasurement(gate_delay=gate_delay)
        thermal_analyzer = ThermalTopologyAnalyzer(ensemble)
        
        return cls(H_K, H_V, ensemble, solver, measurement, thermal_analyzer)
    
    @staticmethod
    def _build_hubbard(n_sites: int, t: float, U: float) -> Tuple[np.ndarray, np.ndarray]:
        """Hubbard ハミルトニアンを構築"""
        dim = 2 ** n_sites
        bonds = [(i, (i + 1) % n_sites) for i in range(n_sites)]
        
        H_K = np.zeros((dim, dim), dtype=complex)
        H_V = np.zeros((dim, dim), dtype=complex)
        
        for state in range(dim):
            for (i, j) in bonds:
                if (state >> i) & 1 and not ((state >> j) & 1):
                    new_state = state ^ (1 << i) ^ (1 << j)
                    sign = 1
                    for k in range(min(i, j) + 1, max(i, j)):
                        if (state >> k) & 1:
                            sign *= -1
                    H_K[new_state, state] += -t * sign
                    H_K[state, new_state] += -t * sign
            
            for (i, j) in bonds:
                ni = (state >> i) & 1
                nj = (state >> j) & 1
                H_V[state, state] += U * ni * nj
        
        return H_K, H_V
    
    def _determine_topology_state(self, coherence: float, lindemann: float,
                                   lambda_value: float) -> TopologyState:
        """Topology 状態を判定"""
        if lambda_value >= 1.0:
            return TopologyState.BROKEN
        elif lindemann > self.lindemann_critical:
            return TopologyState.DISORDERED
        elif coherence < 0.5:
            return TopologyState.FLUCTUATING
        else:
            return TopologyState.COHERENT
    
    def evolve(self, thermal_path: ThermalPath,
               verbose: bool = True) -> ThermalHolographicResult:
        """
        温度パスに沿って発展
        """
        # リセット
        self.solver.reset()
        self.measurement.reset()
        
        T_values, dt_values = thermal_path.generate()
        
        # 初期状態
        psi = self.ensemble.get_thermal_state(T_values[0])
        
        records = []
        
        if verbose:
            print("=" * 60)
            print(f"THERMAL HOLOGRAPHIC EVOLUTION")
            print(f"  Mode: {thermal_path.mode.value}")
            print(f"  T: {T_values[0]:.0f}K → {T_values[-1]:.0f}K")
            print(f"  Steps: {thermal_path.n_steps}")
            print("=" * 60)
        
        for step, (T, dt) in enumerate(zip(T_values, dt_values)):
            # DSE 発展 (Memory 効果付き)
            psi, solver_info = self.solver.step(psi, dt)
            
            # Holographic 測定
            holo_info = self.measurement.measure(solver_info['lambda'], dt)
            
            # Topology 解析 (material_failure.py)
            thermal_result = self.thermal_analyzer.analyze_temperature(T)
            coherence = thermal_result.coherence
            lindemann = thermal_result.lindemann_delta
            
            topology_state = self._determine_topology_state(
                coherence, lindemann, solver_info['lambda']
            )
            
            # 記録
            record = ThermalHolographicRecord(
                step=step,
                time=self.solver.time,
                temperature=T,
                dt=dt,
                lambda_value=solver_info['lambda'],
                coherence=coherence,
                lindemann_delta=lindemann,
                topology_state=topology_state,
                lambda_pre=holo_info['lambda_pre'],
                lambda_post=holo_info['lambda_post'],
                S_RT=holo_info['S_RT'],
                phi_accumulated=holo_info['phi'],
                energy=solver_info['energy'],
                kinetic=solver_info['kinetic'],
                potential=solver_info['potential'],
                gamma_memory=solver_info['gamma_memory'],
                memory_contribution=solver_info['memory_contribution']
            )
            records.append(record)
            
            if verbose and step % max(1, thermal_path.n_steps // 10) == 0:
                print(f"  Step {step:4d}: T={T:7.1f}K  λ={solver_info['lambda']:.4f}  "
                      f"Coh={coherence:.3f}  δ={lindemann:.4f}  [{topology_state.value}]")
        
        # 結果を構築
        result = ThermalHolographicResult(
            records=records,
            thermal_path=thermal_path
        )
        result.compute_summary()
        
        # 双対性検証
        result.duality = self._verify_duality()
        
        # 破壊予測
        result.failure = self._predict_failure(records)
        
        # 最終温度での Topology 結果
        result.thermal_topology = self.thermal_analyzer.analyze_temperature(T_values[-1])
        
        if verbose:
            self._print_summary(result)
        
        return result
    
    def _verify_duality(self) -> DualityMetrics:
        """双対性を検証"""
        duality_result = self.measurement.verify_duality()
        
        return DualityMetrics(
            TE_bulk_to_boundary=duality_result.get('TE_bulk_to_boundary', 0.0),
            TE_boundary_to_bulk=duality_result.get('TE_boundary_to_bulk', 0.0),
            duality_index=duality_result.get('duality_index', 1.0),
            best_lag=duality_result.get('best_lag', 0),
            max_correlation=duality_result.get('max_corr', 0.0)
        )
    
    def _predict_failure(self, records: List[ThermalHolographicRecord]) -> FailurePrediction:
        """破壊を予測"""
        for record in records:
            if record.topology_state == TopologyState.BROKEN:
                return FailurePrediction(
                    will_fail=True,
                    failure_step=record.step,
                    failure_temperature=record.temperature,
                    failure_site=0,
                    failure_mechanism='mechanical',
                    lambda_at_failure=record.lambda_value,
                    confidence=0.9
                )
            elif record.topology_state == TopologyState.DISORDERED:
                return FailurePrediction(
                    will_fail=True,
                    failure_step=record.step,
                    failure_temperature=record.temperature,
                    failure_site=None,
                    failure_mechanism='thermal',
                    lambda_at_failure=record.lambda_value,
                    confidence=0.7
                )
        
        return FailurePrediction(
            will_fail=False,
            failure_step=None,
            failure_temperature=None,
            failure_site=None,
            failure_mechanism='none',
            lambda_at_failure=records[-1].lambda_value if records else 0.0,
            confidence=0.8
        )
    
    def _print_summary(self, result: ThermalHolographicResult):
        """サマリーを出力"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Temperature: {result.T_range[0]:.0f}K → {result.T_range[1]:.0f}K")
        print(f"  λ range: [{result.lambda_range[0]:.4f}, {result.lambda_range[1]:.4f}]")
        print(f"  Coherence range: [{result.coherence_range[0]:.4f}, {result.coherence_range[1]:.4f}]")
        
        print("\n--- Duality (AdS/CFT) ---")
        d = result.duality
        print(f"  TE(Bulk→Boundary): {d.TE_bulk_to_boundary:.4f}")
        print(f"  TE(Boundary→Bulk): {d.TE_boundary_to_bulk:.4f}")
        print(f"  Duality Index: {d.duality_index:.4f}")
        if d.is_strong_duality():
            print("  ✓ STRONG DUALITY (結び目凍結 = 残留応力)")
        elif d.is_moderate_duality():
            print("  ○ MODERATE DUALITY")
        else:
            print("  ✗ WEAK DUALITY (結び目緩和 = 応力解放)")
        
        print("\n--- Failure Prediction ---")
        f = result.failure
        if f.will_fail:
            print(f"  ⚠ FAILURE PREDICTED")
            print(f"    Step: {f.failure_step}")
            print(f"    Temperature: {f.failure_temperature:.0f}K")
            print(f"    Mechanism: {f.failure_mechanism}")
            print(f"    λ at failure: {f.lambda_at_failure:.4f}")
        else:
            print("  ✓ NO FAILURE")
        
        print("=" * 60)
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def quench(self, T_start: float = 1000, T_end: float = 100,
               n_steps: int = 50, verbose: bool = True) -> ThermalHolographicResult:
        """急冷"""
        path = ThermalPath(T_start, T_end, n_steps, CoolingMode.QUENCH)
        return self.evolve(path, verbose)
    
    def anneal(self, T_start: float = 1000, T_end: float = 100,
               n_steps: int = 50, verbose: bool = True) -> ThermalHolographicResult:
        """徐冷"""
        path = ThermalPath(T_start, T_end, n_steps, CoolingMode.ANNEAL)
        return self.evolve(path, verbose)
    
    def thermal_cycle(self, T_low: float = 100, T_high: float = 1000,
                      n_cycles: int = 3, steps_per_cycle: int = 20,
                      verbose: bool = True) -> List[ThermalHolographicResult]:
        """熱サイクル（加熱・冷却の繰り返し）"""
        results = []
        for cycle in range(n_cycles):
            if verbose:
                print(f"\n🔄 Cycle {cycle + 1}/{n_cycles}")
            
            # 加熱
            path_heat = ThermalPath(T_low, T_high, steps_per_cycle, CoolingMode.LINEAR)
            result_heat = self.evolve(path_heat, verbose=False)
            results.append(result_heat)
            
            # 冷却
            path_cool = ThermalPath(T_high, T_low, steps_per_cycle, CoolingMode.LINEAR)
            result_cool = self.evolve(path_cool, verbose=False)
            results.append(result_cool)
        
        if verbose:
            print(f"\n✅ Completed {n_cycles} thermal cycles")
        
        return results
    
    def compare(self, result1: ThermalHolographicResult,
                result2: ThermalHolographicResult,
                label1: str = "Result 1",
                label2: str = "Result 2"):
        """2つの結果を比較"""
        print("\n" + "🔬" * 30)
        print("COMPARISON")
        print("🔬" * 30)
        
        print(f"\n{'Metric':<25} {label1:<20} {label2:<20}")
        print("-" * 65)
        
        print(f"{'λ min':<25} {result1.lambda_range[0]:<20.4f} {result2.lambda_range[0]:<20.4f}")
        print(f"{'λ max':<25} {result1.lambda_range[1]:<20.4f} {result2.lambda_range[1]:<20.4f}")
        print(f"{'Coherence min':<25} {result1.coherence_range[0]:<20.4f} {result2.coherence_range[0]:<20.4f}")
        print(f"{'Coherence max':<25} {result1.coherence_range[1]:<20.4f} {result2.coherence_range[1]:<20.4f}")
        print(f"{'Duality Index':<25} {result1.duality.duality_index:<20.4f} {result2.duality.duality_index:<20.4f}")
        
        f1 = "YES" if result1.failure.will_fail else "NO"
        f2 = "YES" if result2.failure.will_fail else "NO"
        print(f"{'Failure':<25} {f1:<20} {f2:<20}")
        
        print("-" * 65)
        
        mem1 = np.mean([r.memory_contribution for r in result1.records])
        mem2 = np.mean([r.memory_contribution for r in result2.records])
        print(f"{'Avg Memory Contribution':<25} {mem1:<20.4f} {mem2:<20.4f}")
        
        # 物理的解釈
        print("\n--- Physical Interpretation ---")
        if result1.duality.duality_index < result2.duality.duality_index:
            print(f"  {label1}: 結び目凍結 → 残留応力 大")
            print(f"  {label2}: 結び目緩和 → 残留応力 小")
        else:
            print(f"  {label1}: 結び目緩和 → 残留応力 小")
            print(f"  {label2}: 結び目凍結 → 残留応力 大")
        
        print("\n" + "=" * 65)


# =============================================================================
# Info Function
# =============================================================================

def info():
    """パッケージ情報を表示"""
    print("=" * 60)
    print("Thermal Holographic Evolution")
    print("=" * 60)
    print()
    print("Physical Constants:")
    print(f"  τ₀ (Debye period):     {TAU_0:.0e} s")
    print(f"  c (light speed):       {C_LIGHT:.0e} m/s")
    print(f"  v_s (sound speed):     {V_SOUND:.0e} m/s")
    print()
    print("Characteristic Lengths:")
    print(f"  λ_light = c×τ₀:        {LAMBDA_LIGHT*1e6:.0f} μm (grain size)")
    print(f"  λ_phonon = v_s×τ₀:     {LAMBDA_PHONON*1e9:.1f} nm (lattice)")
    print(f"  Scale ratio c/v_s:     {SCALE_RATIO:.0f}")
    print()
    print("Core Insight:")
    print("  Energy = Topology (結び目)")
    print("  Quench → 残留応力 (結び目凍結) → Strong Duality")
    print("  Anneal → 応力解放 (結び目緩和) → Weak Duality")
    print("  第5の力 = トポロジカル結び目のリコネクション")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    info()
