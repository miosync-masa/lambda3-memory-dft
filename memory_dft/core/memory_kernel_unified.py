"""
Unified Memory Kernel for DSE (Direct Schrödinger Evolution)
============================================================

DSE の核心：履歴依存量子力学のためのメモリカーネル

【設計思想】
  - 1クラスのみ（SimpleMemoryKernel, CompositeMemoryKernel 等を統合）
  - γ_memory から全パラメータを第一原理的に導出
  - GPU対応（CuPy）

【物理】
  K(t, τ) = K_base(t-τ) × D(dr/dτ) × I(τ, ψ)
  
  K_base: 時間減衰（power-law × exponential）
  D:      方向依存因子（伸張 vs 圧縮の非対称性）
  I:      アクティブウェイト（重要度 × コヒーレンス × エントロピー）

【γ_memory の意味】
  Vorticity 解析から：
    γ_total  = 全相関（r=∞）
    γ_local  = 局所相関（r≤2）
    γ_memory = γ_total - γ_local = Non-Markovian 相関
  
  γ_memory > 0 → Memory Kernel が必要！
  
  Ref: Lie & Fullwood, PRL 135, 230204 (2025) - Markovian QSOTs
       This work extends to Non-Markovian regime

【対応する物理現象】
  - Lindemann 融解則: δ → δ_L で E_a → 0
  - Coffin-Manson 疲労則: 繰り返しによるエントロピー蓄積
  - クリープ: 熱活性化 + 履歴効果
  - 応力腐食割れ: 環境による V 低下 + 履歴

Author: Masamichi Iizumi, Tamaki Iizumi
Date: 2025-01
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MemoryKernelConfig:
    """Memory Kernel の設定"""
    # 基本パラメータ（γ_memory から導出可能）
    gamma: float = 1.0          # 時間減衰の power-law 指数
    tau0: float = 10.0          # 基本時定数
    
    # 方向依存パラメータ
    alpha_stretch: float = 0.3  # 伸張時の増強係数
    beta_compress: float = 0.1  # 圧縮時の減衰係数
    
    # アクティブウェイト
    importance_scale: float = 1.0    # δ による重み付け強度
    repetition_boost: float = 0.1    # 繰り返し増強係数
    coherence_weight: float = 0.5    # コヒーレンス重み
    entropy_weight: float = 0.3      # エントロピー重み
    
    # 履歴管理
    max_history: int = 100           # 最大履歴数
    similarity_threshold: float = 0.1  # 類似イベント判定閾値
    
    @classmethod
    def from_gamma_memory(cls, gamma_memory: float) -> 'MemoryKernelConfig':
        """
        γ_memory から全パラメータを導出
        
        γ_memory が大きい → 長距離相関が強い → 記憶効果が強い
        """
        return cls(
            gamma=gamma_memory,
            tau0=10.0 / (1.0 + 0.5 * gamma_memory),  # γ大 → τ小（早く減衰）
            alpha_stretch=0.3 * gamma_memory,
            beta_compress=0.1 * gamma_memory,
            importance_scale=1.0 + 0.5 * gamma_memory,
            coherence_weight=0.3 + 0.2 * gamma_memory,
            entropy_weight=0.2 + 0.1 * gamma_memory,
        )


# =============================================================================
# History Entry
# =============================================================================

@dataclass
class HistoryEntry:
    """履歴の1エントリ"""
    time: float                          # 時刻
    position: float                      # 位置（結合長など）
    velocity: float                      # 速度 dr/dt
    state: Optional[np.ndarray] = None   # 状態ベクトル
    energy: Optional[float] = None       # エネルギー
    entropy: Optional[float] = None      # エントロピー（キャッシュ）
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Unified Memory Kernel
# =============================================================================

class MemoryKernel:
    """
    統一 Memory Kernel（GPU対応）
    
    DSE (Direct Schrödinger Evolution) の核心コンポーネント
    
    Features:
    1. γ_memory から全パラメータを第一原理的に導出
    2. 方向依存性：伸張と圧縮で非対称
    3. アクティブウェイト：重要なイベントを長く記憶
    4. 位相コヒーレンス：量子干渉効果
    5. エントロピー：記憶の「乱れ」
    6. GPU加速（CuPy）
    
    Usage:
        # 方法1: Vorticity から構築（推奨）
        kernel = MemoryKernel.from_vorticity(V_total, V_local, E_xc)
        
        # 方法2: γ_memory を直接指定
        kernel = MemoryKernel(gamma_memory=1.2)
        
        # 方法3: 設定オブジェクト
        config = MemoryKernelConfig(gamma=1.5, tau0=8.0)
        kernel = MemoryKernel(config=config)
        
        # 状態を追加
        kernel.add_state(t=0.0, r=1.0, state=psi)
        kernel.add_state(t=1.0, r=1.1, state=psi2)
        
        # メモリ寄与を計算
        delta_E = kernel.compute_memory_contribution(t=2.0, current_state=psi3)
    """
    
    def __init__(self,
                 gamma_memory: float = 1.0,
                 tau0: float = 10.0,
                 config: Optional[MemoryKernelConfig] = None,
                 use_gpu: bool = True):
        """
        Args:
            gamma_memory: Non-Markovian 相関指数（Vorticity から導出）
            tau0: 基本時定数
            config: 設定オブジェクト（指定時は gamma_memory, tau0 を上書き）
            use_gpu: GPU使用フラグ
        """
        # 設定
        if config is not None:
            self.config = config
        else:
            self.config = MemoryKernelConfig(gamma=gamma_memory, tau0=tau0)
        
        # GPU設定
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        
        # 履歴
        self.history: List[HistoryEntry] = []
        
        # キャッシュ
        self._coherence_cache: Optional[np.ndarray] = None
        self._cache_valid = False
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_vorticity(cls,
                       V_total: float,
                       V_local: float,
                       E_xc: float,
                       use_gpu: bool = True) -> 'MemoryKernel':
        """
        Vorticity から γ_memory を導出してインスタンス化
        
        【理論的背景】
          α = |E_xc| / V ∝ N^(-γ)
          
          V_total: 全相関を含む Vorticity（距離フィルターなし）
          V_local: 局所相関のみの Vorticity（max_range=2）
          
          γ_memory = f(α_total, α_local)
        
        Args:
            V_total: 全相関 Vorticity
            V_local: 局所相関 Vorticity
            E_xc: 相関エネルギー
            use_gpu: GPU使用フラグ
            
        Returns:
            MemoryKernel インスタンス
        """
        # α 計算
        alpha_total = abs(E_xc) / (V_total + 1e-10)
        alpha_local = abs(E_xc) / (V_local + 1e-10)
        
        # γ_memory 導出
        # α_local > α_total のとき、非局所相関が強い
        if V_local > 1e-10 and V_total > 1e-10:
            # Vorticity 比からγを推定
            # V_total < V_local → 非局所がキャンセル → γ_memory 小
            # V_total > V_local → 非局所が加算 → γ_memory 大
            ratio = V_total / V_local
            
            if ratio > 1:
                gamma_memory = np.log(ratio) * 2  # 非局所が支配的
            else:
                gamma_memory = (ratio - 0.5) * 2  # 局所が支配的だが記憶はある
        else:
            gamma_memory = 1.0  # デフォルト
        
        # 範囲制限
        gamma_memory = float(np.clip(gamma_memory, 0.1, 3.0))
        
        # 設定生成
        config = MemoryKernelConfig.from_gamma_memory(gamma_memory)
        
        return cls(config=config, use_gpu=use_gpu)
    
    @classmethod
    def from_rdm2(cls,
                  rdm2: np.ndarray,
                  n_orb: int,
                  E_xc: float,
                  use_gpu: bool = True) -> 'MemoryKernel':
        """
        2-RDM から直接構築
        
        内部で Vorticity を計算して γ_memory を導出
        
        Args:
            rdm2: 2粒子密度行列 (n_orb, n_orb, n_orb, n_orb)
            n_orb: 軌道数
            E_xc: 相関エネルギー
            use_gpu: GPU使用フラグ
        """
        # Vorticity 計算（簡易版）
        V_total = cls._compute_vorticity_simple(rdm2, n_orb, max_range=None)
        V_local = cls._compute_vorticity_simple(rdm2, n_orb, max_range=2)
        
        return cls.from_vorticity(V_total, V_local, E_xc, use_gpu)
    
    @staticmethod
    def _compute_vorticity_simple(rdm2: np.ndarray, 
                                   n_orb: int, 
                                   max_range: Optional[int] = None,
                                   svd_cut: float = 0.95) -> float:
        """
        簡易 Vorticity 計算
        
        V = √(Σ ||J - J^T||²)
        J = M_λ @ ∇M_λ
        """
        # 距離フィルター
        if max_range is not None:
            rdm2_filtered = np.zeros_like(rdm2)
            for i in range(n_orb):
                for j in range(n_orb):
                    for k in range(n_orb):
                        for l in range(n_orb):
                            max_d = max(abs(i-j), abs(k-l), abs(i-k), abs(j-l))
                            if max_d <= max_range:
                                rdm2_filtered[i,j,k,l] = rdm2[i,j,k,l]
            rdm2 = rdm2_filtered
        
        # 行列形式
        M = rdm2.reshape(n_orb**2, n_orb**2)
        
        # SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        
        total_var = np.sum(S**2)
        if total_var < 1e-14:
            return 0.0
        
        cumvar = np.cumsum(S**2) / total_var
        k = int(np.searchsorted(cumvar, svd_cut) + 1)
        k = max(2, min(k, len(S)))
        
        # Λ空間射影
        S_proj = U[:, :k]
        M_lambda = S_proj.T @ M @ S_proj
        
        # 勾配
        grad_M = np.zeros_like(M_lambda)
        grad_M[:-1, :] = M_lambda[1:, :] - M_lambda[:-1, :]
        
        # 電流と Vorticity
        J = M_lambda @ grad_M
        curl_J = J - J.T
        V = float(np.sqrt(np.sum(curl_J**2)))
        
        return V
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def add_state(self,
                  t: float,
                  r: float,
                  state: Optional[np.ndarray] = None,
                  energy: Optional[float] = None,
                  **metadata):
        """
        状態を履歴に追加
        
        Args:
            t: 時刻
            r: 位置（結合長、変位など）
            state: 状態ベクトル（オプション）
            energy: エネルギー（オプション）
            **metadata: その他のメタデータ
        """
        # 速度計算
        if len(self.history) > 0:
            prev = self.history[-1]
            dt = t - prev.time
            if dt > 1e-10:
                velocity = (r - prev.position) / dt
            else:
                velocity = 0.0
        else:
            velocity = 0.0
        
        # エントリ作成
        entry = HistoryEntry(
            time=t,
            position=r,
            velocity=velocity,
            state=state.copy() if state is not None else None,
            energy=energy,
            metadata=metadata
        )
        
        # エントロピー計算（状態があれば）
        if state is not None:
            entry.entropy = self.compute_state_entropy(state)
        
        self.history.append(entry)
        
        # 履歴制限
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]
        
        # キャッシュ無効化
        self._cache_valid = False
    
    def clear(self):
        """履歴クリア"""
        self.history.clear()
        self._coherence_cache = None
        self._cache_valid = False
    
    # =========================================================================
    # Core Kernel Functions
    # =========================================================================
    
    def kernel_base(self, dt: float) -> float:
        """
        基本カーネル K_base(dt)
        
        K_base = (dt + ε)^(-γ) × exp(-dt/τ₀)
        
        Power-law: 長時間相関（γ が大きいほど早く減衰）
        Exponential: カットオフ（τ₀ で特徴時間）
        """
        if dt <= 0:
            return 0.0
        
        gamma = self.config.gamma
        tau0 = self.config.tau0
        
        # Power-law（特異点回避）
        power = (dt + 0.1) ** (-gamma)
        
        # Exponential cutoff
        exp_cut = np.exp(-dt / tau0)
        
        return power * exp_cut
    
    def direction_factor(self, velocity: float) -> float:
        """
        方向依存因子 D(v)
        
        伸張（v > 0）: 傷が蓄積 → 係数増加
        圧縮（v < 0）: 回復    → 係数減少
        
        【物理的根拠】
          伸張: 結合が弱まる → 電子が局所化 → 相関変化が大きい
          圧縮: 結合が強まる → 電子が非局所化 → 相関が回復
        """
        alpha = self.config.alpha_stretch
        beta = self.config.beta_compress
        
        if velocity > 0:
            # 伸張：tanh で飽和（無限大にならない）
            return 1.0 + alpha * np.tanh(velocity)
        else:
            # 圧縮：減衰
            return 1.0 - beta * np.tanh(-velocity)
    
    # =========================================================================
    # Active Memory Weights
    # =========================================================================
    
    def compute_delta_weight(self, idx: int) -> float:
        """
        δ（変位）による重み
        
        大きい変位 → 重要 → 長く記憶
        小さい変位 → 忘れやすい
        """
        if idx >= len(self.history):
            return 1.0
        
        entry = self.history[idx]
        scale = self.config.importance_scale
        
        return 1.0 + scale * abs(entry.velocity)
    
    def count_similar_events(self, idx: int) -> int:
        """
        類似イベントの数をカウント（疲労累積）
        
        同じ方向・同じ大きさの変位の繰り返し
        """
        if idx >= len(self.history):
            return 0
        
        target = self.history[idx]
        threshold = self.config.similarity_threshold
        
        count = 0
        for i, entry in enumerate(self.history):
            if i == idx:
                continue
            
            # 同じ方向か
            same_dir = entry.velocity * target.velocity > 0
            
            # 大きさが近いか
            similar_mag = abs(abs(entry.velocity) - abs(target.velocity)) < threshold
            
            if same_dir and similar_mag:
                count += 1
        
        return count
    
    def compute_repetition_weight(self, idx: int) -> float:
        """
        繰り返しによる重み増強（疲労）
        
        同じサイクルが繰り返される → 累積効果 → 重み増加
        """
        n_similar = self.count_similar_events(idx)
        boost = self.config.repetition_boost
        
        return 1.0 + boost * n_similar
    
    # =========================================================================
    # Coherence (Quantum Phase)
    # =========================================================================
    
    def compute_coherence(self, current_state: np.ndarray) -> np.ndarray:
        """
        各履歴点との位相コヒーレンスを計算
        
        coherence[i] = |Re[⟨ψ|ψᵢ⟩]| / |⟨ψ|ψᵢ⟩|
        
        位相が揃っていれば 1 に近い
        位相が π ずれていれば 0 に近い
        
        【物理的意味】
          高コヒーレンス: 量子干渉が効く → 「鮮明な記憶」
          低コヒーレンス: 干渉が消える → 「ぼやけた記憶」
        """
        if current_state is None or len(self.history) == 0:
            return np.ones(len(self.history))
        
        xp = self.xp
        coherences = []
        
        for entry in self.history:
            if entry.state is None:
                coherences.append(1.0)
                continue
            
            # 内積
            if self.use_gpu:
                cs = cp.asarray(current_state)
                hs = cp.asarray(entry.state)
                inner = cp.vdot(cs, hs)
                inner = complex(cp.asnumpy(inner))
            else:
                inner = np.vdot(current_state, entry.state)
            
            magnitude = abs(inner)
            
            if magnitude < 1e-10:
                coherences.append(0.0)
                continue
            
            # 位相の揃い具合: |cos(φ)|
            phase_alignment = abs(inner.real) / magnitude
            coherences.append(float(phase_alignment))
        
        return np.array(coherences)
    
    def compute_collective_coherence(self) -> float:
        """
        履歴全体の集団的コヒーレンス
        
        C = |Σ e^(iφᵢⱼ)|² / N_pairs²
        
        全ての位相が揃っていれば C ≈ 1
        ランダムなら C → 0
        
        【疲労との関係】
          同じサイクルの繰り返し → 位相揃う → C 高 → 累積効果大
          ランダムな変動 → 位相バラバラ → C 低 → 累積効果小
        """
        states = [e.state for e in self.history if e.state is not None]
        if len(states) < 2:
            return 1.0
        
        xp = self.xp
        phase_sum = 0.0 + 0.0j
        n_pairs = 0
        
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                if self.use_gpu:
                    si = cp.asarray(states[i])
                    sj = cp.asarray(states[j])
                    inner = complex(cp.asnumpy(cp.vdot(si, sj)))
                else:
                    inner = np.vdot(states[i], states[j])
                
                if abs(inner) > 1e-10:
                    phase_sum += inner / abs(inner)  # 位相のみ
                    n_pairs += 1
        
        if n_pairs == 0:
            return 1.0
        
        C = abs(phase_sum) ** 2 / (n_pairs ** 2)
        return float(C)
    
    def compute_coherence_weight(self, idx: int, current_state: np.ndarray = None) -> float:
        """
        コヒーレンスに基づく重み
        """
        if current_state is None:
            return 1.0
        
        coherences = self.compute_coherence(current_state)
        if idx < len(coherences):
            coh = coherences[idx]
        else:
            coh = 1.0
        
        w = self.config.coherence_weight
        return 1.0 + w * coh
    
    # =========================================================================
    # Entropy
    # =========================================================================
    
    def compute_state_entropy(self, state: np.ndarray) -> float:
        """
        状態のエンタングルメントエントロピー（Schmidt分解）
        
        |ψ⟩ = Σ λᵢ |aᵢ⟩|bᵢ⟩
        S = -Σ λᵢ² ln(λᵢ²)
        
        【物理的意味】
          S 低い: 状態が「整列」→ 回復可能
          S 高い: 状態が「乱れ」→ 損傷（不可逆）
        """
        if state is None:
            return 0.0
        
        xp = self.xp
        N = len(state)
        n = int(np.sqrt(N))
        
        if n * n != N:
            # 完全平方でなければ Shannon entropy
            if self.use_gpu:
                p = cp.abs(cp.asarray(state)) ** 2
                p = cp.asnumpy(p)
            else:
                p = np.abs(state) ** 2
            p = p / (p.sum() + 1e-10)
            return float(-np.sum(p * np.log(p + 1e-10)))
        
        # Schmidt 分解（SVD）
        if self.use_gpu:
            mat = cp.reshape(cp.asarray(state), (n, n))
            _, s, _ = cp.linalg.svd(mat)
            s = cp.asnumpy(s)
        else:
            mat = np.reshape(state, (n, n))
            _, s, _ = np.linalg.svd(mat)
        
        # 正規化
        p = s ** 2
        p = p / (p.sum() + 1e-10)
        
        # von Neumann entropy
        S = -np.sum(p * np.log(p + 1e-10))
        return float(S)
    
    def compute_history_entropy(self) -> float:
        """
        履歴全体のエントロピー
        
        S = -Σ pᵢ ln(pᵢ)
        pᵢ = |⟨ψᵢ|ψ_avg⟩|²
        
        【疲労との関係】
          サイクル累積 → S 蓄積 → S > S_c で破壊
        """
        states = [e.state for e in self.history if e.state is not None]
        if len(states) < 2:
            return 0.0
        
        xp = self.xp
        
        # 平均状態
        if self.use_gpu:
            states_gpu = [cp.asarray(s) for s in states]
            psi_avg = sum(states_gpu) / len(states_gpu)
            psi_avg = psi_avg / (cp.linalg.norm(psi_avg) + 1e-10)
        else:
            psi_avg = sum(states) / len(states)
            psi_avg = psi_avg / (np.linalg.norm(psi_avg) + 1e-10)
        
        # 各状態との重なり
        overlaps = []
        for state in states:
            if self.use_gpu:
                ov = float(abs(cp.vdot(psi_avg, cp.asarray(state))) ** 2)
            else:
                ov = float(abs(np.vdot(psi_avg, state)) ** 2)
            overlaps.append(ov)
        
        # Shannon entropy
        p = np.array(overlaps)
        p = p / (p.sum() + 1e-10)
        S = -np.sum(p * np.log(p + 1e-10))
        
        return float(S)
    
    def compute_entropy_weight(self, idx: int) -> float:
        """
        エントロピーに基づく重み
        
        S 低い → 重み大（秩序的 = 重要な記憶）
        S 高い → 重み小（無秩序 = 薄れる記憶）
        """
        if idx >= len(self.history):
            return 1.0
        
        entry = self.history[idx]
        
        # キャッシュされたエントロピーを使用
        if entry.entropy is not None:
            S = entry.entropy
        elif entry.state is not None:
            S = self.compute_state_entropy(entry.state)
            entry.entropy = S  # キャッシュ
        else:
            return 1.0
        
        # S_max の推定
        if entry.state is not None:
            S_max = np.log(len(entry.state))
        else:
            S_max = 5.0  # デフォルト
        
        # exp(-S/S_ref) 形式
        S_ref = S_max / 2
        w = self.config.entropy_weight
        
        weight = 1.0 + w * np.exp(-S / S_ref)
        return float(weight)
    
    # =========================================================================
    # Total Importance
    # =========================================================================
    
    def compute_importance(self, idx: int, current_state: np.ndarray = None) -> float:
        """
        総合的な重要度（アクティブウェイト）
        
        I = δ重み × 繰り返し × コヒーレンス × エントロピー
        
        【統一される物理】
          - δ 大きい変位は重要（Lindemann）
          - 繰り返しは累積（Coffin-Manson）
          - 位相揃いは干渉を強める（量子効果）
          - 低エントロピーは秩序（回復可能性）
        """
        w_delta = self.compute_delta_weight(idx)
        w_rep = self.compute_repetition_weight(idx)
        w_coh = self.compute_coherence_weight(idx, current_state)
        w_ent = self.compute_entropy_weight(idx)
        
        return w_delta * w_rep * w_coh * w_ent
    
    # =========================================================================
    # Main API
    # =========================================================================
    
    def __call__(self, t: float, tau: float, velocity: float = 0.0) -> float:
        """
        カーネル評価
        
        K(t, τ, v) = K_base(t-τ) × D(v)
        
        注: アクティブウェイト I は compute_memory_contribution で適用
        """
        dt = t - tau
        return self.kernel_base(dt) * self.direction_factor(velocity)
    
    def compute_memory_contribution(self,
                                     t: float,
                                     current_state: np.ndarray = None) -> float:
        """
        メモリ寄与を計算（DSE の核心）
        
        ΔE_mem = Σᵢ K(t, τᵢ, vᵢ) × I(τᵢ, ψ) × overlap(ψ, ψᵢ)
        
        Args:
            t: 現在時刻
            current_state: 現在の状態ベクトル
            
        Returns:
            delta: メモリによるエネルギー補正
        """
        if len(self.history) == 0:
            return 0.0
        
        delta = 0.0
        
        for idx, entry in enumerate(self.history):
            dt = t - entry.time
            if dt <= 0:
                continue
            
            # 基本カーネル × 方向依存
            K = self(t, entry.time, entry.velocity)
            
            # アクティブウェイト
            I = self.compute_importance(idx, current_state)
            
            # 状態重なり
            if current_state is not None and entry.state is not None:
                if self.use_gpu:
                    cs = cp.asarray(current_state)
                    es = cp.asarray(entry.state)
                    overlap = float(abs(cp.vdot(cs, es)) ** 2)
                else:
                    overlap = float(abs(np.vdot(current_state, entry.state)) ** 2)
            else:
                overlap = 1.0
            
            delta += K * I * overlap
        
        return delta
    
    def integrate(self, t: float, history_times: np.ndarray) -> np.ndarray:
        """
        任意の時刻配列に対するカーネル重みを計算（バッチ処理）
        
        外部の履歴を使う場合用
        """
        xp = self.xp
        
        if self.use_gpu:
            times = cp.asarray(history_times)
        else:
            times = np.asarray(history_times)
        
        dt = t - times
        
        # 基本カーネル
        gamma = self.config.gamma
        tau0 = self.config.tau0
        
        valid = dt > 0
        weights = xp.zeros_like(dt, dtype=float)
        
        if self.use_gpu:
            weights[valid] = (dt[valid] + 0.1) ** (-gamma) * cp.exp(-dt[valid] / tau0)
        else:
            weights[valid] = (dt[valid] + 0.1) ** (-gamma) * np.exp(-dt[valid] / tau0)
        
        # 正規化
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        
        if self.use_gpu:
            return cp.asnumpy(weights)
        return weights
    
    # =========================================================================
    # Diagnostics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """診断情報"""
        stats = {
            'history_length': len(self.history),
            'config': {
                'gamma': self.config.gamma,
                'tau0': self.config.tau0,
                'alpha_stretch': self.config.alpha_stretch,
                'beta_compress': self.config.beta_compress,
            },
            'use_gpu': self.use_gpu,
        }
        
        if len(self.history) > 0:
            velocities = [e.velocity for e in self.history]
            stats['velocity_mean'] = float(np.mean(velocities))
            stats['velocity_std'] = float(np.std(velocities))
            stats['n_stretch'] = sum(1 for v in velocities if v > 0)
            stats['n_compress'] = sum(1 for v in velocities if v < 0)
            
            entropies = [e.entropy for e in self.history if e.entropy is not None]
            if entropies:
                stats['entropy_mean'] = float(np.mean(entropies))
                stats['entropy_max'] = float(np.max(entropies))
            
            stats['collective_coherence'] = self.compute_collective_coherence()
            stats['history_entropy'] = self.compute_history_entropy()
        
        return stats
    
    def __repr__(self) -> str:
        return (f"MemoryKernel(γ={self.config.gamma:.2f}, τ₀={self.config.tau0:.1f}, "
                f"GPU={self.use_gpu}, history={len(self.history)})")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Unified Memory Kernel Test")
    print("=" * 70)
    
    # 基本テスト
    print("\n[1] Basic Kernel Test")
    kernel = MemoryKernel(gamma_memory=1.2, tau0=10.0, use_gpu=False)
    print(f"  {kernel}")
    
    # 状態追加
    np.random.seed(42)
    dim = 16
    
    for t in range(10):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state = state / np.linalg.norm(state)
        r = 1.0 + 0.1 * np.sin(t)  # 振動
        kernel.add_state(t=float(t), r=r, state=state)
    
    print(f"  History: {len(kernel.history)} entries")
    
    # メモリ寄与
    current = np.random.randn(dim) + 1j * np.random.randn(dim)
    current = current / np.linalg.norm(current)
    delta = kernel.compute_memory_contribution(t=10.0, current_state=current)
    print(f"  Memory contribution: {delta:.6f}")
    
    # 統計
    stats = kernel.get_statistics()
    print(f"  Collective coherence: {stats.get('collective_coherence', 0):.4f}")
    print(f"  History entropy: {stats.get('history_entropy', 0):.4f}")
    
    # 方向依存テスト
    print("\n[2] Direction Dependence Test")
    print("  velocity    D(v)")
    print("  " + "-" * 25)
    for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        D = kernel.direction_factor(v)
        direction = "compress" if v < 0 else ("stretch" if v > 0 else "static")
        print(f"  {v:+6.2f}    {D:.4f}  ({direction})")
    
    # γ_memory からの構築テスト
    print("\n[3] from_gamma_memory Test")
    for gamma in [0.5, 1.0, 1.5, 2.0]:
        config = MemoryKernelConfig.from_gamma_memory(gamma)
        print(f"  γ={gamma:.1f}: tau0={config.tau0:.2f}, "
              f"α_stretch={config.alpha_stretch:.2f}, "
              f"coh_w={config.coherence_weight:.2f}")
    
    # Vorticity からの構築テスト
    print("\n[4] from_vorticity Test")
    # ダミー値
    V_total, V_local = 2.0, 1.5
    E_xc = -0.5
    kernel2 = MemoryKernel.from_vorticity(V_total, V_local, E_xc, use_gpu=False)
    print(f"  V_total={V_total}, V_local={V_local}, E_xc={E_xc}")
    print(f"  → {kernel2}")
    
    # カーネル減衰テスト
    print("\n[5] Kernel Decay Test")
    print("  dt        K_base(dt)")
    print("  " + "-" * 25)
    for dt in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        K = kernel.kernel_base(dt)
        print(f"  {dt:6.1f}    {K:.6f}")
    
    # GPU テスト
    print("\n[6] GPU Test")
    if HAS_CUPY:
        kernel_gpu = MemoryKernel(gamma_memory=1.0, use_gpu=True)
        
        for t in range(5):
            state = np.random.randn(dim) + 1j * np.random.randn(dim)
            state = state / np.linalg.norm(state)
            kernel_gpu.add_state(t=float(t), r=1.0 + 0.1*t, state=state)
        
        delta_gpu = kernel_gpu.compute_memory_contribution(t=5.0, current_state=current)
        print(f"  GPU kernel works! delta={delta_gpu:.6f}")
    else:
        print("  CuPy not available, skipping GPU test")
    
    print("\n" + "=" * 70)
    print("✅ Unified Memory Kernel Test Complete!")
    print("=" * 70)
