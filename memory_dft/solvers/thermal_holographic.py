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

【温度変化速度の効果】
  急冷（Quench）: dt小 → Memory効果強 → 非平衡凍結
  徐冷（Anneal）: dt大 → Memory効果弱 → 平衡接近

【アーキテクチャ】
  ThermalEnsemble (温度→分布)
      ↓
  DSESolver (Memory付き時間発展)
      ↓
  HolographicMeasurement (PRE/POST λ, 双対性)
      ↓
  ThermalTopologyAnalyzer (Coherence, Lindemann, 破壊予測)

Author: Tamaki & Masamichi Iizumi
Date: 2025-01
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

# =============================================================================
# Constants
# =============================================================================

K_B_EV = 8.617333262e-5  # eV/K


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
            # 急冷: 温度が急激に下がる、dt は小さい
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.01)  # 小さいdt
            
        elif self.mode == CoolingMode.ANNEAL:
            # 徐冷: 温度がゆっくり下がる、dt は大きい
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.5)   # 大きいdt
            
        elif self.mode == CoolingMode.LINEAR:
            # 線形: 均等
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.1)
            
        elif self.mode == CoolingMode.EXPONENTIAL:
            # 指数的冷却
            tau = self.n_steps / 3  # 特性時間
            t = np.arange(self.n_steps)
            T_values = self.T_end + (self.T_start - self.T_end) * np.exp(-t / tau)
            # dt は温度変化率に反比例
            dT = np.abs(np.gradient(T_values))
            dt_values = 0.1 / (dT / dT.mean() + 0.1)
            
        else:
            # カスタム: 線形フォールバック
            T_values = np.linspace(self.T_start, self.T_end, self.n_steps)
            dt_values = np.full(self.n_steps, 0.1)
        
        return T_values, dt_values


@dataclass
class DualityMetrics:
    """双対性メトリクス"""
    TE_bulk_to_boundary: float    # Transfer Entropy: Bulk → Boundary
    TE_boundary_to_bulk: float    # Transfer Entropy: Boundary → Bulk
    duality_index: float          # |TE_B→b - TE_b→B| / (TE_B→b + TE_b→B)
    best_lag: int                 # 最適ラグ
    max_correlation: float        # 最大相関
    
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
    failure_mechanism: str        # 'thermal', 'mechanical', 'combined'
    lambda_at_failure: float
    confidence: float             # 予測信頼度


@dataclass
class ThermalHolographicResult:
    """全体結果"""
    records: List[ThermalHolographicRecord]
    thermal_path: ThermalPath
    
    # Summary statistics
    T_range: Tuple[float, float] = (0.0, 0.0)
    lambda_range: Tuple[float, float] = (0.0, 0.0)
    coherence_range: Tuple[float, float] = (0.0, 0.0)
    
    # Duality
    duality: Optional[DualityMetrics] = None
    
    # Failure prediction
    failure: Optional[FailurePrediction] = None
    
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
# Lightweight Thermal Ensemble (standalone)
# =============================================================================

class LightweightThermalEnsemble:
    """
    軽量版 ThermalEnsemble
    
    外部依存なしで動作。
    本番では environment_operators.ThermalEnsemble を使用。
    """
    
    def __init__(self, H: np.ndarray, n_eigenstates: int = 20):
        """
        Args:
            H: ハミルトニアン
            n_eigenstates: 固有状態数
        """
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import issparse, csr_matrix
        
        self.H = H
        self.n_eigenstates = min(n_eigenstates, H.shape[0] - 2)
        
        # 固有値・固有ベクトル計算
        if not issparse(H):
            H_sparse = csr_matrix(H)
        else:
            H_sparse = H
            
        self.eigenvalues, self.eigenvectors = eigsh(
            H_sparse, k=self.n_eigenstates, which='SA'
        )
        
        # ソート
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        
        # Observable キャッシュ
        self._obs_cache: Dict[str, np.ndarray] = {}
        self._register_default_observables()
    
    def _register_default_observables(self):
        """デフォルトの observable を登録"""
        # Phase entropy
        def phase_entropy(psi):
            theta = np.angle(psi)
            hist, _ = np.histogram(theta, bins=20, range=(-np.pi, np.pi))
            p = hist / (hist.sum() + 1e-10)
            return -np.sum(p[p > 0] * np.log(p[p > 0]))
        
        # Phase variance (Lindemann proxy)
        def phase_variance(psi):
            return np.var(np.angle(psi))
        
        # Winding number
        def winding(psi):
            theta = np.angle(psi)
            dtheta = np.diff(theta)
            dtheta = ((dtheta + np.pi) % (2 * np.pi)) - np.pi
            return np.sum(dtheta) / (2 * np.pi)
        
        self.register_observable('phase_entropy', phase_entropy)
        self.register_observable('phase_variance', phase_variance)
        self.register_observable('winding', winding)
    
    def register_observable(self, name: str, func: Callable):
        """Observable を登録"""
        values = np.zeros(self.n_eigenstates)
        for n in range(self.n_eigenstates):
            psi = self.eigenvectors[:, n]
            values[n] = func(psi)
        self._obs_cache[name] = values
    
    def get_weights(self, T: float) -> np.ndarray:
        """Boltzmann 重みを取得"""
        if T <= 0:
            weights = np.zeros(self.n_eigenstates)
            weights[0] = 1.0
            return weights
        
        beta = 1.0 / (K_B_EV * T)
        E_shifted = self.eigenvalues - self.eigenvalues[0]
        weights = np.exp(-beta * E_shifted)
        return weights / weights.sum()
    
    def thermal_average(self, observable: str, T: float) -> float:
        """熱平均を計算"""
        if observable not in self._obs_cache:
            raise ValueError(f"Observable '{observable}' not registered")
        
        weights = self.get_weights(T)
        return float(np.sum(weights * self._obs_cache[observable]))
    
    def get_thermal_state(self, T: float) -> np.ndarray:
        """温度 T での熱的状態（混合状態の代表）"""
        weights = self.get_weights(T)
        # 重み付き重ね合わせ（簡易版）
        psi = np.zeros(self.eigenvectors.shape[0], dtype=complex)
        for n in range(self.n_eigenstates):
            psi += np.sqrt(weights[n]) * self.eigenvectors[:, n]
        return psi / np.linalg.norm(psi)
    
    def compute_coherence(self, T: float) -> float:
        """位相コヒーレンスを計算"""
        weights = self.get_weights(T)
        phase_sum = 0.0 + 0.0j
        for n in range(self.n_eigenstates):
            psi = self.eigenvectors[:, n]
            avg_phase = np.angle(np.sum(psi))
            phase_sum += weights[n] * np.exp(1j * avg_phase)
        return float(abs(phase_sum))
    
    def compute_lindemann(self, T: float) -> float:
        """Lindemann パラメータを計算"""
        phase_var = self.thermal_average('phase_variance', T)
        return float(np.sqrt(phase_var) / np.pi)


# =============================================================================
# Lightweight DSE Solver (standalone)
# =============================================================================

class LightweightDSESolver:
    """
    軽量版 DSE Solver
    
    Memory 効果付き時間発展。
    本番では solvers/dse_solver.py を使用。
    """
    
    def __init__(self, H_K: np.ndarray, H_V: np.ndarray, 
                 gamma_memory: float = 0.1,
                 eta_memory: float = 0.1):
        """
        Args:
            H_K: 運動エネルギー項
            H_V: ポテンシャル項
            gamma_memory: Memory カーネル減衰率
            eta_memory: Memory 混合率
        """
        self.H_K = np.asarray(H_K)
        self.H_V = np.asarray(H_V)
        self.H = self.H_K + self.H_V
        self.gamma_memory = gamma_memory
        self.eta_memory = eta_memory
        
        # History
        self.history: List[Dict] = []
        self.time = 0.0
    
    def reset(self):
        """履歴をリセット"""
        self.history = []
        self.time = 0.0
    
    def compute_lambda(self, psi: np.ndarray) -> float:
        """λ = K/|V| を計算"""
        K = np.real(np.vdot(psi, self.H_K @ psi))
        V = np.real(np.vdot(psi, self.H_V @ psi))
        return abs(K) / (abs(V) + 1e-10)
    
    def compute_memory_contribution(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Memory 項の寄与を計算"""
        if len(self.history) < 2:
            return np.zeros_like(psi)
        
        memory_psi = np.zeros_like(psi, dtype=complex)
        
        for i, entry in enumerate(self.history):
            tau = self.time - entry['time']
            if tau > 0:
                # Memory kernel: K(τ) = (dt + ε)^(-γ) × exp(-τ/τ₀)
                # dt が小さい（急冷）→ K 大 → Memory 強
                # dt が大きい（徐冷）→ K 小 → Memory 弱
                K_base = (dt + 0.01) ** (-self.gamma_memory)
                K_decay = np.exp(-tau / 10.0)  # τ₀ = 10
                K_total = K_base * K_decay
                
                memory_psi += K_total * entry['psi']
        
        norm = np.linalg.norm(memory_psi)
        if norm > 1e-10:
            memory_psi /= norm
        
        return memory_psi
    
    def step(self, psi: np.ndarray, dt: float) -> Tuple[np.ndarray, Dict]:
        """1ステップ発展"""
        # Memory 寄与
        memory_psi = self.compute_memory_contribution(psi, dt)
        memory_strength = np.linalg.norm(memory_psi)
        
        # Schrödinger 発展
        # exp(-iHdt) ≈ 1 - iHdt (1次近似)
        psi_evolved = psi - 1j * dt * (self.H @ psi)
        
        # Memory 混合
        if memory_strength > 1e-10:
            psi_new = (1 - self.eta_memory) * psi_evolved + self.eta_memory * memory_psi
        else:
            psi_new = psi_evolved
        
        # 正規化
        psi_new = psi_new / np.linalg.norm(psi_new)
        
        # エネルギー計算
        E = np.real(np.vdot(psi_new, self.H @ psi_new))
        K = np.real(np.vdot(psi_new, self.H_K @ psi_new))
        V = np.real(np.vdot(psi_new, self.H_V @ psi_new))
        
        # 履歴に追加
        self.history.append({
            'time': self.time,
            'psi': psi.copy(),
            'energy': E,
            'lambda': self.compute_lambda(psi_new)
        })
        
        self.time += dt
        
        info = {
            'energy': E,
            'kinetic': K,
            'potential': V,
            'lambda': self.compute_lambda(psi_new),
            'memory_contribution': memory_strength,
            'gamma_memory': self.gamma_memory
        }
        
        return psi_new, info


# =============================================================================
# Lightweight Holographic Measurement (standalone)
# =============================================================================

class LightweightHolographicMeasurement:
    """
    軽量版 Holographic Measurement
    
    PRE/POST λ測定と双対性検証。
    本番では holographic/measurement.py を使用。
    """
    
    def __init__(self, gate_delay: int = 1):
        self.gate_delay = gate_delay
        self.phi_history: List[float] = []
        self.lambda_history: List[float] = []
        self.S_RT_history: List[float] = []
        
    def reset(self):
        """履歴をリセット"""
        self.phi_history = []
        self.lambda_history = []
        self.S_RT_history = []
    
    def measure(self, lambda_value: float, dt: float) -> Dict:
        """1ステップの測定"""
        # PRE λ
        lambda_pre = lambda_value
        
        # POST λ (遅延)
        if len(self.lambda_history) >= self.gate_delay:
            lambda_post = self.lambda_history[-self.gate_delay]
        else:
            lambda_post = lambda_value
        
        self.lambda_history.append(lambda_value)
        
        # 位相蓄積
        if self.phi_history:
            phi = self.phi_history[-1] + lambda_value * dt
        else:
            phi = lambda_value * dt
        self.phi_history.append(phi)
        
        # S_RT (Bulk entropy) - 簡易版
        if len(self.phi_history) >= 2:
            phi_arr = np.array(self.phi_history[-20:])  # 最新20点
            S_RT = np.std(phi_arr) * np.log(len(phi_arr) + 1)
        else:
            S_RT = 0.0
        self.S_RT_history.append(S_RT)
        
        return {
            'lambda_pre': lambda_pre,
            'lambda_post': lambda_post,
            'phi': phi,
            'S_RT': S_RT
        }
    
    def verify_duality(self) -> DualityMetrics:
        """双対性を検証"""
        if len(self.lambda_history) < 10:
            return DualityMetrics(0, 0, 1.0, 0, 0)
        
        boundary = np.array(self.lambda_history)
        bulk = np.array(self.S_RT_history)
        
        # Transfer Entropy (簡易版)
        # TE(X→Y) ≈ correlation(X[:-1], Y[1:])
        TE_b2B = abs(np.corrcoef(boundary[:-1], bulk[1:])[0, 1])
        TE_B2b = abs(np.corrcoef(bulk[:-1], boundary[1:])[0, 1])
        
        # 相互相関でベストラグを探す
        max_corr = 0.0
        best_lag = 0
        for lag in range(-10, 11):
            if lag == 0:
                continue
            if lag > 0:
                corr = abs(np.corrcoef(boundary[:-lag], bulk[lag:])[0, 1])
            else:
                corr = abs(np.corrcoef(boundary[-lag:], bulk[:lag])[0, 1])
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
        
        # Duality index
        denom = TE_b2B + TE_B2b + 1e-10
        duality_index = abs(TE_b2B - TE_B2b) / denom
        
        return DualityMetrics(
            TE_bulk_to_boundary=float(TE_B2b),
            TE_boundary_to_bulk=float(TE_b2B),
            duality_index=float(duality_index),
            best_lag=best_lag,
            max_correlation=float(max_corr)
        )


# =============================================================================
# Main Class: ThermalHolographicEvolution
# =============================================================================

class ThermalHolographicEvolution:
    """
    温度変化 × Memory効果 × Holographic測定 × 材料破壊予測
    
    【統合アーキテクチャ】
      Temperature Path
          ↓
      ThermalEnsemble (温度→分布→状態)
          ↓
      DSESolver (Memory付き時間発展)
          ↓
      HolographicMeasurement (PRE/POST λ, S_RT)
          ↓
      TopologyAnalysis (Coherence, Lindemann, 破壊予測)
    
    Usage:
        # Hubbard モデルで初期化
        evolution = ThermalHolographicEvolution.from_hubbard(n_sites=4, t=1.0, U=2.0)
        
        # 急冷
        result_quench = evolution.quench(T_start=1000, T_end=100, n_steps=50)
        
        # 徐冷
        result_anneal = evolution.anneal(T_start=1000, T_end=100, n_steps=50)
        
        # 比較
        evolution.compare(result_quench, result_anneal)
    """
    
    def __init__(self,
                 ensemble: LightweightThermalEnsemble,
                 solver: LightweightDSESolver,
                 measurement: LightweightHolographicMeasurement,
                 lindemann_critical: float = 0.1):
        """
        Args:
            ensemble: 熱アンサンブル
            solver: DSE ソルバー
            measurement: Holographic 測定器
            lindemann_critical: Lindemann 臨界値
        """
        self.ensemble = ensemble
        self.solver = solver
        self.measurement = measurement
        self.lindemann_critical = lindemann_critical
    
    @classmethod
    def from_hubbard(cls, n_sites: int = 4, t: float = 1.0, U: float = 2.0,
                     gamma_memory: float = 0.1, eta_memory: float = 0.1,
                     gate_delay: int = 1) -> 'ThermalHolographicEvolution':
        """
        Hubbard モデルから初期化
        
        Args:
            n_sites: サイト数
            t: ホッピング
            U: オンサイト相互作用
            gamma_memory: Memory 減衰率
            eta_memory: Memory 混合率
            gate_delay: 測定遅延
        """
        H_K, H_V = cls._build_hubbard(n_sites, t, U)
        H = H_K + H_V
        
        ensemble = LightweightThermalEnsemble(H)
        solver = LightweightDSESolver(H_K, H_V, gamma_memory, eta_memory)
        measurement = LightweightHolographicMeasurement(gate_delay)
        
        return cls(ensemble, solver, measurement)
    
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
        
        Args:
            thermal_path: 温度パス
            verbose: 詳細出力
        
        Returns:
            ThermalHolographicResult
        """
        # リセット
        self.solver.reset()
        self.measurement.reset()
        
        # 温度・dt 列を生成
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
            # 熱的状態を取得（温度変化を反映）
            psi_thermal = self.ensemble.get_thermal_state(T)
            
            # DSE 発展 (Memory 効果付き)
            psi, solver_info = self.solver.step(psi, dt)
            
            # Holographic 測定
            holo_info = self.measurement.measure(solver_info['lambda'], dt)
            
            # Topology 解析
            coherence = self.ensemble.compute_coherence(T)
            lindemann = self.ensemble.compute_lindemann(T)
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
            
            # 進捗表示
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
        result.duality = self.measurement.verify_duality()
        
        # 破壊予測
        result.failure = self._predict_failure(records)
        
        if verbose:
            self._print_summary(result)
        
        return result
    
    def _predict_failure(self, records: List[ThermalHolographicRecord]) -> FailurePrediction:
        """破壊を予測"""
        for record in records:
            if record.topology_state == TopologyState.BROKEN:
                return FailurePrediction(
                    will_fail=True,
                    failure_step=record.step,
                    failure_temperature=record.temperature,
                    failure_site=0,  # TODO: local analysis
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
        
        print("\n--- Duality ---")
        d = result.duality
        print(f"  TE(Bulk→Boundary): {d.TE_bulk_to_boundary:.4f}")
        print(f"  TE(Boundary→Bulk): {d.TE_boundary_to_bulk:.4f}")
        print(f"  Duality Index: {d.duality_index:.4f}")
        if d.is_strong_duality():
            print("  ✓ STRONG DUALITY")
        elif d.is_moderate_duality():
            print("  ○ MODERATE DUALITY")
        else:
            print("  ✗ WEAK DUALITY")
        
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
    
    def linear_cooling(self, T_start: float = 1000, T_end: float = 100,
                       n_steps: int = 50, verbose: bool = True) -> ThermalHolographicResult:
        """線形冷却"""
        path = ThermalPath(T_start, T_end, n_steps, CoolingMode.LINEAR)
        return self.evolve(path, verbose)
    
    def exponential_cooling(self, T_start: float = 1000, T_end: float = 100,
                            n_steps: int = 50, verbose: bool = True) -> ThermalHolographicResult:
        """指数的冷却"""
        path = ThermalPath(T_start, T_end, n_steps, CoolingMode.EXPONENTIAL)
        return self.evolve(path, verbose)
    
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
        
        # λ range
        print(f"{'λ min':<25} {result1.lambda_range[0]:<20.4f} {result2.lambda_range[0]:<20.4f}")
        print(f"{'λ max':<25} {result1.lambda_range[1]:<20.4f} {result2.lambda_range[1]:<20.4f}")
        
        # Coherence
        print(f"{'Coherence min':<25} {result1.coherence_range[0]:<20.4f} {result2.coherence_range[0]:<20.4f}")
        print(f"{'Coherence max':<25} {result1.coherence_range[1]:<20.4f} {result2.coherence_range[1]:<20.4f}")
        
        # Duality
        print(f"{'Duality Index':<25} {result1.duality.duality_index:<20.4f} {result2.duality.duality_index:<20.4f}")
        
        # Failure
        f1 = "YES" if result1.failure.will_fail else "NO"
        f2 = "YES" if result2.failure.will_fail else "NO"
        print(f"{'Failure':<25} {f1:<20} {f2:<20}")
        
        if result1.failure.will_fail:
            print(f"{'  Mechanism':<25} {result1.failure.failure_mechanism:<20}")
        if result2.failure.will_fail:
            print(f"{'  Mechanism':<25} {'':20} {result2.failure.failure_mechanism:<20}")
        
        print("-" * 65)
        
        # Memory 効果の違い
        mem1 = np.mean([r.memory_contribution for r in result1.records])
        mem2 = np.mean([r.memory_contribution for r in result2.records])
        print(f"{'Avg Memory Contribution':<25} {mem1:<20.4f} {mem2:<20.4f}")
        
        print("\n" + "=" * 65)


# =============================================================================
# Test
# =============================================================================

def run_thermal_holographic_test():
    """テストを実行"""
    print("\n" + "🔬" * 30)
    print("THERMAL HOLOGRAPHIC EVOLUTION TEST")
    print("🔬" * 30 + "\n")
    
    # Hubbard モデルで初期化
    evolution = ThermalHolographicEvolution.from_hubbard(
        n_sites=4, t=1.0, U=2.0,
        gamma_memory=0.3, eta_memory=0.15
    )
    
    print("✅ Built 4-site Hubbard system\n")
    
    # 急冷テスト
    print("\n" + "=" * 60)
    print("TEST 1: QUENCH (急冷)")
    print("=" * 60)
    result_quench = evolution.quench(T_start=1000, T_end=100, n_steps=30)
    
    # 徐冷テスト
    print("\n" + "=" * 60)
    print("TEST 2: ANNEAL (徐冷)")
    print("=" * 60)
    result_anneal = evolution.anneal(T_start=1000, T_end=100, n_steps=30)
    
    # 比較
    evolution.compare(result_quench, result_anneal, "QUENCH", "ANNEAL")
    
    # 追加: 指数的冷却
    print("\n" + "=" * 60)
    print("TEST 3: EXPONENTIAL COOLING")
    print("=" * 60)
    result_exp = evolution.exponential_cooling(T_start=1000, T_end=100, n_steps=30)
    
    return {
        'quench': result_quench,
        'anneal': result_anneal,
        'exponential': result_exp
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_thermal_holographic_test()
    
    print("\n" + "✅" * 30)
    print("ALL TESTS COMPLETED")
    print("✅" * 30)
