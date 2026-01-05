"""
DSE Holographic Interpretation
==============================

DSEの履歴依存構造をAdS/CFT的に解釈するモジュール。

核心的アイデア:
    - φ_history (位相蓄積履歴) → Bulk geometry
    - 非マルコフ性 → Bulk の深さ方向 (z)
    - RT entropy → 履歴の複雑さ/エンタングルメント

DSEのcore/solverは一切変更せず、「解釈層」として追加。

Author: Masamichi Iizumi & Tamaki (Miosync, Inc.)
Date: 2025-01-06
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt


class HolographicDual:
    """
    DSEの履歴をAdS/CFT bulk として解釈するクラス
    
    AdS/CFT対応:
        Boundary (CFT)  ←→  現在の状態 ψ(t)
        Bulk (AdS)      ←→  φ_history (位相蓄積の履歴)
        z座標 (radial)  ←→  「どれだけ過去か」
        RT entropy      ←→  履歴の絡まり具合
    
    Parameters
    ----------
    L_ads : float
        AdS radius (スケール因子)
    Z_depth : int
        Bulk の z 方向離散化数
    G_N : float
        Newton定数 (RT entropy の正規化)
    """
    
    def __init__(self, L_ads: float = 1.0, Z_depth: int = 16, G_N: float = 1.0):
        self.L_ads = L_ads
        self.Z_depth = Z_depth
        self.G_N = G_N
        
        # Bulk storage
        self._bulk = None
        self._z_coords = None
        self._warp_factors = None
        
        # 履歴キャッシュ
        self._phi_history = None
        
        self._setup_z_coords()
    
    def _setup_z_coords(self):
        """z座標とwarp factorを事前計算"""
        # z ∈ (0, 1] で離散化 (z=0 が boundary, z→∞ が IR)
        # 実際は z_k = (k+1)/Z で k=0,...,Z-1
        self._z_coords = (np.arange(self.Z_depth) + 1) / self.Z_depth
        
        # AdS warp factor: ds² = (L/z)² (dz² + dx²)
        # 場の値は (L/z)² でスケール
        self._warp_factors = (self.L_ads / self._z_coords) ** 2
    
    # =========================================================================
    # Bulk Geometry
    # =========================================================================
    
    def history_to_bulk(self, phi_history: List[float]) -> np.ndarray:
        """
        位相蓄積履歴をbulk geometryに変換
        
        Parameters
        ----------
        phi_history : List[float]
            DSEの位相蓄積履歴 φ(t_0), φ(t_1), ..., φ(t_n)
            最新が末尾
        
        Returns
        -------
        bulk : np.ndarray, shape (Z_depth,)
            Bulk geometry (z方向の場の値)
        """
        self._phi_history = phi_history
        
        n_history = len(phi_history)
        self._bulk = np.zeros(self.Z_depth)
        
        # 最新 → z=0近く (UV), 過去 → z大きい (IR)
        for k in range(self.Z_depth):
            # 履歴のどの時点に対応するか
            history_idx = n_history - 1 - k
            
            if history_idx >= 0:
                phi_k = phi_history[history_idx]
                self._bulk[k] = self._warp_factors[k] * phi_k
            else:
                # 履歴が足りない場合は最古の値で埋める
                self._bulk[k] = self._warp_factors[k] * phi_history[0] if n_history > 0 else 0.0
        
        return self._bulk
    
    def history_to_bulk_2d(self, phi_history_2d: np.ndarray) -> np.ndarray:
        """
        2D boundary の履歴を 3D bulk に変換
        
        Parameters
        ----------
        phi_history_2d : np.ndarray, shape (T, Ly, Lx)
            時系列の2D場
        
        Returns
        -------
        bulk : np.ndarray, shape (Z_depth, Ly, Lx)
        """
        T, Ly, Lx = phi_history_2d.shape
        bulk = np.zeros((self.Z_depth, Ly, Lx))
        
        for k in range(self.Z_depth):
            history_idx = T - 1 - k
            if history_idx >= 0:
                bulk[k] = self._warp_factors[k] * phi_history_2d[history_idx]
            else:
                bulk[k] = self._warp_factors[k] * phi_history_2d[0]
        
        return bulk
    
    @property
    def bulk(self) -> Optional[np.ndarray]:
        """現在のbulk geometry"""
        return self._bulk
    
    @property
    def z_coords(self) -> np.ndarray:
        """z座標"""
        return self._z_coords
    
    # =========================================================================
    # RT Entropy
    # =========================================================================
    
    def rt_entropy(self, bulk: Optional[np.ndarray] = None) -> float:
        """
        Ryu-Takayanagi entropy を計算
        
        S_RT = Area(minimal surface) / (4 G_N)
        
        1D bulk では "area" = bulk profile の変動量
        """
        if bulk is None:
            bulk = self._bulk
        
        if bulk is None:
            return 0.0
        
        # 1D: "minimal surface" = z方向の全変動
        # 2D以上: 適切な minimal surface 探索が必要
        if bulk.ndim == 1:
            area = np.sum(np.abs(np.diff(bulk)))
        else:
            # 多次元の場合は gradient の総和
            grads = [np.abs(np.diff(bulk, axis=i)).sum() for i in range(bulk.ndim)]
            area = np.sum(grads)
        
        return area / (4 * self.G_N)
    
    def rt_entropy_bipartite(self, bulk_2d: np.ndarray, region_A: np.ndarray) -> float:
        """
        2D system での bipartite RT entropy
        
        Parameters
        ----------
        bulk_2d : np.ndarray, shape (Z, Ly, Lx)
        region_A : np.ndarray, shape (Ly, Lx), bool
            boundary での region A
        
        Returns
        -------
        S_RT : float
        """
        # region A の boundary での perimeter
        A = region_A.astype(np.int8)
        dx = np.abs(A - np.roll(A, -1, axis=1))
        dy = np.abs(A - np.roll(A, -1, axis=0))
        boundary_perimeter = dx.sum() + dy.sum()
        
        # bulk direction への拡張 (簡易版: boundary perimeter × bulk depth factor)
        depth_factor = np.mean(self._warp_factors)
        
        area = boundary_perimeter * depth_factor
        
        return area / (4 * self.G_N)
    
    # =========================================================================
    # Holographic C-function
    # =========================================================================
    
    def c_function(self, bulk: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Holographic c-function (RGフローのモニター)
        
        c(z) ∝ 1 / (z^d × |∂_z φ|)
        
        UV (z→0): 大きい
        IR (z→∞): 小さい (情報が粗視化)
        """
        if bulk is None:
            bulk = self._bulk
        
        if bulk is None or len(bulk) < 2:
            return np.array([1.0])
        
        # ∂_z φ (numerical derivative)
        d_bulk = np.gradient(bulk, self._z_coords)
        
        # c(z) = L^(d-1) / (z^(d-1) × G_N × |∂_z φ|)
        # d=1+1 の場合、d-1=1
        d_minus_1 = 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            c = self.L_ads ** d_minus_1 / (
                self._z_coords ** d_minus_1 * self.G_N * (np.abs(d_bulk) + 1e-10)
            )
        
        return np.clip(c, 0, 1e10)
    
    # =========================================================================
    # Complexity
    # =========================================================================
    
    def complexity_volume(self, bulk: Optional[np.ndarray] = None) -> float:
        """
        Holographic complexity (CV conjecture)
        
        C_V = V(maximal slice) / (G_N × L)
        
        1D では単に bulk の積分
        """
        if bulk is None:
            bulk = self._bulk
        
        if bulk is None:
            return 0.0
        
        # Volume = ∫ dz × warp × |φ|
        volume = np.trapezoid(np.abs(bulk), self._z_coords)
        
        return volume / (self.G_N * self.L_ads)
    
    def complexity_action(self, bulk: Optional[np.ndarray] = None) -> float:
        """
        Holographic complexity (CA conjecture)
        
        C_A = S_WDW / π
        
        Wheeler-DeWitt patch の action
        """
        if bulk is None:
            bulk = self._bulk
        
        if bulk is None:
            return 0.0
        
        # 簡易版: kinetic + potential
        kinetic = np.sum(np.diff(bulk) ** 2)
        potential = np.sum(bulk ** 2 / self._warp_factors)
        
        action = kinetic + potential
        
        return action / np.pi
    
    # =========================================================================
    # Thermal interpretation
    # =========================================================================
    
    def hawking_temperature(self, bulk: Optional[np.ndarray] = None) -> float:
        """
        Effective Hawking temperature
        
        bulk の最深部 (IR) での gradient から推定
        """
        if bulk is None:
            bulk = self._bulk
        
        if bulk is None or len(bulk) < 2:
            return 0.0
        
        # IR での surface gravity ∝ |∂_z φ|_{z→∞}
        surface_gravity = np.abs(bulk[-1] - bulk[-2])
        
        # T_H = κ / (2π)
        return surface_gravity / (2 * np.pi)
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot_bulk(self, bulk: Optional[np.ndarray] = None, 
                  ax: Optional[plt.Axes] = None,
                  title: str = "Holographic Bulk Geometry") -> plt.Figure:
        """
        Bulk geometry の可視化
        """
        if bulk is None:
            bulk = self._bulk
        
        if bulk is None:
            print("No bulk data to plot")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        
        if bulk.ndim == 1:
            ax.plot(self._z_coords, bulk, 'b-', linewidth=2, label='φ(z)')
            ax.fill_between(self._z_coords, 0, bulk, alpha=0.3)
            ax.set_xlabel('z (radial / depth)')
            ax.set_ylabel('φ(z) × warp factor')
            ax.set_title(title)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # UV/IR labels
            ax.annotate('UV\n(boundary)', xy=(self._z_coords[0], bulk[0]), 
                       fontsize=9, ha='center')
            ax.annotate('IR\n(deep bulk)', xy=(self._z_coords[-1], bulk[-1]), 
                       fontsize=9, ha='center')
        
        elif bulk.ndim == 2:
            # (Z, X) の2D bulk
            im = ax.imshow(bulk, aspect='auto', origin='lower',
                          extent=[0, bulk.shape[1], self._z_coords[0], self._z_coords[-1]])
            ax.set_xlabel('x (boundary)')
            ax.set_ylabel('z (depth)')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='φ(z,x)')
        
        elif bulk.ndim == 3:
            # (Z, Y, X) → z=0 slice を表示
            im = ax.imshow(bulk[0], aspect='auto', origin='lower')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f"{title} (z=0 slice, UV)")
            plt.colorbar(im, ax=ax, label='φ')
        
        return fig
    
    def plot_diagnostics(self, bulk: Optional[np.ndarray] = None,
                         phi_history: Optional[List[float]] = None) -> plt.Figure:
        """
        Holographic diagnostics の総合プロット
        """
        if bulk is None:
            bulk = self._bulk
        if phi_history is None:
            phi_history = self._phi_history
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # (0,0) Bulk geometry
        if bulk is not None and bulk.ndim == 1:
            axes[0, 0].plot(self._z_coords, bulk, 'b-', linewidth=2)
            axes[0, 0].fill_between(self._z_coords, 0, bulk, alpha=0.3)
            axes[0, 0].set_xlabel('z')
            axes[0, 0].set_ylabel('φ(z)')
            axes[0, 0].set_title('Bulk Geometry')
            axes[0, 0].grid(True, alpha=0.3)
        
        # (0,1) C-function
        if bulk is not None:
            c_func = self.c_function(bulk)
            axes[0, 1].semilogy(self._z_coords, c_func, 'r-', linewidth=2)
            axes[0, 1].set_xlabel('z')
            axes[0, 1].set_ylabel('c(z)')
            axes[0, 1].set_title('Holographic C-function')
            axes[0, 1].grid(True, alpha=0.3)
        
        # (1,0) Phase history (if available)
        if phi_history is not None:
            t = np.arange(len(phi_history))
            axes[1, 0].plot(t, phi_history, 'g-', linewidth=1.5)
            axes[1, 0].set_xlabel('time step')
            axes[1, 0].set_ylabel('φ')
            axes[1, 0].set_title('Phase History (Boundary)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # (1,1) Summary metrics
        if bulk is not None:
            S_RT = self.rt_entropy(bulk)
            C_V = self.complexity_volume(bulk)
            C_A = self.complexity_action(bulk)
            T_H = self.hawking_temperature(bulk)
            
            text = f"""Holographic Observables
─────────────────────────
RT Entropy:     S_RT = {S_RT:.4f}
Complexity (V): C_V  = {C_V:.4f}
Complexity (A): C_A  = {C_A:.4f}
Hawking Temp:   T_H  = {T_H:.4f}
─────────────────────────
Bulk depth:     Z    = {self.Z_depth}
AdS radius:     L    = {self.L_ads}
Newton const:   G_N  = {self.G_N}
"""
            axes[1, 1].text(0.1, 0.5, text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='center',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Summary')
        
        plt.suptitle('DSE Holographic Dual Analysis', fontsize=14)
        plt.tight_layout()
        
        return fig


# =============================================================================
# Utility functions
# =============================================================================

def quick_holographic_analysis(phi_history: List[float],
                                L_ads: float = 1.0,
                                Z_depth: int = 16,
                                G_N: float = 1.0,
                                plot: bool = True) -> Dict[str, Any]:
    """
    DSE履歴の簡易ホログラフィック解析
    
    Parameters
    ----------
    phi_history : List[float]
        位相蓄積の履歴
    L_ads, Z_depth, G_N : float, int, float
        ホログラフィックパラメータ
    plot : bool
        可視化するかどうか
    
    Returns
    -------
    results : Dict
        S_RT, C_V, C_A, T_H, bulk, c_function
    """
    holo = HolographicDual(L_ads=L_ads, Z_depth=Z_depth, G_N=G_N)
    bulk = holo.history_to_bulk(phi_history)
    
    results = {
        'bulk': bulk,
        'z_coords': holo.z_coords,
        'S_RT': holo.rt_entropy(bulk),
        'C_V': holo.complexity_volume(bulk),
        'C_A': holo.complexity_action(bulk),
        'T_H': holo.hawking_temperature(bulk),
        'c_function': holo.c_function(bulk),
    }
    
    if plot:
        fig = holo.plot_diagnostics(bulk, phi_history)
        fig.savefig('dse_holographic_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("[Saved] dse_holographic_analysis.png")
    
    return results


# =============================================================================
# Causality Analysis (from PhaseShift-X)
# =============================================================================

def discretize_quantiles(x: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Quantile-based discretization for Transfer Entropy"""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        mn, mx = float(np.min(x)), float(np.max(x) + 1e-12)
        edges = np.linspace(mn, mx, n_bins + 1)
    d = np.digitize(x, edges[1:-1], right=False)
    return d.astype(np.int32)


def transfer_entropy(x: np.ndarray, y: np.ndarray, n_bins: int = 3) -> float:
    """
    Transfer Entropy: TE(X → Y)
    
    Measures information flow from X to Y.
    TE(X→Y) = H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)
    
    Parameters
    ----------
    x : np.ndarray
        Source time series
    y : np.ndarray
        Target time series
    n_bins : int
        Number of bins for discretization
    
    Returns
    -------
    te : float
        Transfer entropy in nats
    """
    xq = discretize_quantiles(x, n_bins)
    yq = discretize_quantiles(y, n_bins)
    
    yt = yq[:-1]  # Y at time t
    xt = xq[:-1]  # X at time t
    y1 = yq[1:]   # Y at time t+1
    
    n = n_bins
    p = np.zeros((n, n, n), dtype=float)
    
    for i in range(len(y1)):
        p[y1[i], yt[i], xt[i]] += 1.0
    p /= max(1.0, p.sum())
    
    p_yt_xt = p.sum(axis=0) + 1e-12
    p_y1_yt = p.sum(axis=2) + 1e-12
    p_yt = p_yt_xt.sum(axis=1) + 1e-12
    
    te = 0.0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                pij = p[a, b, c]
                if pij <= 0:
                    continue
                te += pij * np.log(
                    (p[a, b, c] / p_yt_xt[b, c]) / (p_y1_yt[a, b] / p_yt[b])
                )
    
    return float(te)


def crosscorr_at_lags(a: np.ndarray, b: np.ndarray, maxlag: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation at various lags.
    
    Positive lag: a leads b
    Negative lag: b leads a
    """
    lags = np.arange(-maxlag, maxlag + 1, 1)
    corrs = []
    
    a0 = (a - a.mean()) / (a.std() + 1e-12)
    b0 = (b - b.mean()) / (b.std() + 1e-12)
    
    for L in lags:
        if L == 0:
            aa, bb = a0, b0
        elif L > 0:
            aa, bb = a0[L:], b0[:-L]
        else:
            aa, bb = a0[:L], b0[-L:]
        
        if len(aa) > 1:
            corrs.append(float(np.mean(aa * bb)))
        else:
            corrs.append(np.nan)
    
    return lags, np.array(corrs, dtype=float)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation"""
    def rankdata(a):
        temp = a.argsort(kind='mergesort')
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(a))
        return ranks
    
    xr = rankdata(x)
    yr = rankdata(y)
    xr = (xr - xr.mean()) / (xr.std() + 1e-12)
    yr = (yr - yr.mean()) / (yr.std() + 1e-12)
    
    return float(np.mean(xr * yr))


def verify_duality(bulk_history: List[float], 
                   boundary_history: List[float],
                   n_bins: int = 3,
                   maxlag: int = 16) -> Dict[str, Any]:
    """
    Verify AdS/CFT duality through information flow analysis.
    
    Parameters
    ----------
    bulk_history : List[float]
        Time series of bulk observable (e.g., S_RT, complexity)
    boundary_history : List[float]
        Time series of boundary observable (e.g., λ, energy)
    
    Returns
    -------
    results : Dict
        TE_bulk_to_boundary : float
        TE_boundary_to_bulk : float
        duality_index : float  (|TE_B→b - TE_b→B| / max, closer to 0 = better duality)
        best_lag : int
        max_corr : float
        spearman : float
    """
    bulk = np.array(bulk_history)
    boundary = np.array(boundary_history)
    
    # Transfer Entropy both directions
    TE_bulk_to_boundary = transfer_entropy(bulk, boundary, n_bins)
    TE_boundary_to_bulk = transfer_entropy(boundary, bulk, n_bins)
    
    # Duality index: perfect duality → balanced information flow
    max_te = max(TE_bulk_to_boundary, TE_boundary_to_bulk, 1e-12)
    duality_index = abs(TE_bulk_to_boundary - TE_boundary_to_bulk) / max_te
    
    # Cross-correlation
    lags, corrs = crosscorr_at_lags(bulk, boundary, maxlag)
    best_idx = int(np.nanargmax(np.abs(corrs)))
    best_lag = int(lags[best_idx])
    max_corr = float(corrs[best_idx])
    
    # Spearman
    rho = spearman_corr(bulk, boundary)
    
    return {
        'TE_bulk_to_boundary': TE_bulk_to_boundary,
        'TE_boundary_to_bulk': TE_boundary_to_bulk,
        'duality_index': duality_index,
        'best_lag': best_lag,
        'max_corr': max_corr,
        'spearman': rho,
        'lags': lags,
        'corrs': corrs,
    }


def plot_duality_analysis(results: Dict[str, Any], 
                          title: str = "AdS/CFT Duality Verification",
                          save: bool = True) -> plt.Figure:
    """
    Visualize duality verification results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # (0) Transfer Entropy comparison
    ax = axes[0]
    te_vals = [results['TE_bulk_to_boundary'], results['TE_boundary_to_bulk']]
    colors = ['#2E86AB', '#A23B72']
    bars = ax.bar(['Bulk → Boundary', 'Boundary → Bulk'], te_vals, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Transfer Entropy (nats)')
    ax.set_title(f"Information Flow\nDuality Index = {results['duality_index']:.3f}")
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, te_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # (1) Cross-correlation
    ax = axes[1]
    ax.plot(results['lags'], results['corrs'], 'o-', linewidth=2, markersize=5, color='navy')
    ax.axvline(results['best_lag'], color='red', linestyle='--', linewidth=2, 
               label=f'Best lag = {results["best_lag"]}')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Lag (steps)')
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Cross-correlation\nMax r = {results["max_corr"]:.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (2) Summary
    ax = axes[2]
    summary_text = f"""
    ══════════════════════════════════
         DUALITY VERIFICATION
    ══════════════════════════════════
    
    Transfer Entropy:
      Bulk → Boundary: {results['TE_bulk_to_boundary']:.4f} nats
      Boundary → Bulk: {results['TE_boundary_to_bulk']:.4f} nats
    
    Duality Index: {results['duality_index']:.4f}
      (0 = perfect duality)
    
    Correlation:
      Pearson (best lag): {results['max_corr']:.4f}
      Spearman: {results['spearman']:.4f}
      Best lag: {results['best_lag']} steps
    
    ══════════════════════════════════
    """
    
    # Interpretation
    di = results['duality_index']
    if di < 0.2:
        verdict = "✓ STRONG DUALITY"
        color = 'green'
    elif di < 0.5:
        verdict = "○ MODERATE DUALITY"
        color = 'orange'
    else:
        verdict = "✗ WEAK DUALITY"
        color = 'red'
    
    summary_text += f"\n    Verdict: {verdict}"
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Summary')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        fname = 'duality_verification.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"[Saved] {fname}")
    
    return fig


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DSE Holographic Module Test + Duality Verification")
    print("=" * 60)
    
    # ダミーの位相履歴を生成（OAM pump + free evolution 的な）
    np.random.seed(42)
    
    N_pump = 50
    N_free = 50
    N_total = N_pump + N_free
    
    phi_history = []
    boundary_history = []  # Boundary observable (e.g., energy)
    phi = 0.0
    E = 1.0
    
    for n in range(N_total):
        if n < N_pump:
            # Pump phase: 位相が蓄積
            d_phi = 0.1 * np.sin(0.3 * n) + 0.02 * np.random.randn()
            d_E = 0.05 * np.cos(0.3 * n) + 0.01 * np.random.randn()
        else:
            # Free evolution: 緩やかな減衰 + 揺らぎ
            d_phi = -0.01 * phi + 0.01 * np.random.randn()
            d_E = -0.02 * (E - 1.0) + 0.01 * np.random.randn()
        
        phi += d_phi
        E += d_E
        phi_history.append(phi)
        boundary_history.append(E)
    
    print(f"\nGenerated {len(phi_history)} steps of history")
    
    # Holographic analysis
    print("\n--- Holographic Analysis ---")
    holo = HolographicDual(L_ads=1.0, Z_depth=16)
    bulk = holo.history_to_bulk(phi_history)
    
    S_RT = holo.rt_entropy()
    C_V = holo.complexity_volume()
    
    print(f"  S_RT = {S_RT:.4f}")
    print(f"  C_V  = {C_V:.4f}")
    
    # Duality verification
    print("\n--- Duality Verification ---")
    
    # Bulk observable: RT entropy時系列を構築
    bulk_series = []
    for t in range(10, N_total):
        holo_t = HolographicDual(Z_depth=min(t, 16))
        holo_t.history_to_bulk(phi_history[:t])
        bulk_series.append(holo_t.rt_entropy())
    
    boundary_series = boundary_history[10:]
    
    results = verify_duality(bulk_series, boundary_series)
    
    print(f"\n  TE(Bulk → Boundary): {results['TE_bulk_to_boundary']:.4f} nats")
    print(f"  TE(Boundary → Bulk): {results['TE_boundary_to_bulk']:.4f} nats")
    print(f"  Duality Index: {results['duality_index']:.4f}")
    print(f"  Best lag: {results['best_lag']}")
    print(f"  Max correlation: {results['max_corr']:.4f}")
    print(f"  Spearman: {results['spearman']:.4f}")
    
    # 可視化
    fig = plot_duality_analysis(results, save=True)
    plt.close()
    
    print("\n" + "=" * 60)
    print("DSE Holographic + Duality Test Complete")
    print("=" * 60)
