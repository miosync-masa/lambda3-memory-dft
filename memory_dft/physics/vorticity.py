"""
Vorticity Calculator for Memory-DFT
====================================

2-RDM から Vorticity を計算し、γ_memory を抽出する。
Unified Memory Kernel と統合して使用。

【理論的背景】
  V = √(Σ ||J - J^T||²)   where J = M_λ @ ∇M_λ
  
  α = |E_xc| / V ∝ N^(-γ)
  
  γ_total  = 全相関（距離制限なし）
  γ_local  = 局所相関（max_range=2）
  γ_memory = γ_total - γ_local = Non-Markovian 相関

【実験結果 (1D Hubbard, U/t=2.0)】
  γ_total  = 2.604
  γ_local  = 1.388
  γ_memory = 1.216 (46.7%)
  
  → γ_memory > 0 は Memory Kernel の存在証明！

【距離フィルター - 重要な修正】
  - Hubbard 模型: サイトインデックス = 空間位置 → そのまま使える
  - 分子（PySCF）: 軌道インデックス ≠ 空間位置 → 空間距離行列が必要
  
  PySCF での問題：
    軌道の並び: Fe1(1s,2s,...,3d), Fe2(1s,2s,...,3d)
    インデックス近接 ≠ 空間近接
    
  解決策：
    compute_orbital_distance_matrix(mol) で空間距離行列を作成
    distance_matrix 引数で渡す

【Memory Kernel との統合】
  from vorticity import VorticityCalculator, GammaExtractor
  from memory_kernel_unified import MemoryKernel
  
  # Vorticity 計算
  calc = VorticityCalculator()
  decomp = calc.compute_gamma_decomposition(rdm2, n_orb, E_xc)
  
  # Memory Kernel 構築
  kernel = MemoryKernel.from_vorticity(
      decomp['V_total'], decomp['V_local'], E_xc
  )

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025) - Markovian QSOTs
  This work extends to Non-Markovian regime via γ_memory

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class VorticityResult:
    """Vorticity 計算結果"""
    vorticity: float           # V
    effective_rank: int        # SVD の有効ランク k
    alpha: float               # α = |E_xc| / V
    gamma: Optional[float] = None
    method: str = 'default'
    
    def __repr__(self):
        return (f"VorticityResult(V={self.vorticity:.4f}, k={self.effective_rank}, "
                f"α={self.alpha:.4f})")


# =============================================================================
# Vorticity Calculator
# =============================================================================

class VorticityCalculator:
    """
    2-RDM から Vorticity を計算
    
    V = √(Σ ||J - J^T||²)
    
    where J = M_λ @ ∇M_λ
    
    Features:
    - 距離フィルターで局所/全体相関を分離
    - 空間距離行列に対応（分子系用）
    - GPU 加速（CuPy）
    
    Usage:
        calc = VorticityCalculator()
        
        # Hubbard 模型（サイトインデックス = 位置）
        result = calc.compute_with_energy(rdm2, n_orb, E_xc, max_range=2)
        
        # 分子系（空間距離行列を指定）
        result = calc.compute_with_energy(rdm2, n_orb, E_xc, 
                                          max_range=3.0,  # Å
                                          distance_matrix=dist_mat)
    """
    
    def __init__(self, svd_cut: float = 0.95, use_gpu: bool = True):
        """
        Args:
            svd_cut: SVD カットオフ（累積分散の何%まで保持）
            use_gpu: GPU 使用フラグ
        """
        self.svd_cut = svd_cut
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
    
    def compute_vorticity(self,
                          rdm2: np.ndarray,
                          n_orb: int,
                          max_range: Optional[float] = None,
                          distance_matrix: Optional[np.ndarray] = None) -> VorticityResult:
        """
        2-RDM から Vorticity を計算
        
        Args:
            rdm2: 2粒子密度行列 (n_orb, n_orb, n_orb, n_orb)
            n_orb: 軌道数
            max_range: 相関距離の上限
                       None = 全相関
                       int  = インデックス差（Hubbard 用）
                       float = 空間距離 Å（分子用、distance_matrix 必要）
            distance_matrix: 軌道間の空間距離行列 (n_orb, n_orb)
                             分子系で max_range を空間距離として使う場合に必要
        
        Returns:
            VorticityResult
        """
        xp = self.xp
        
        # 距離フィルター適用
        if max_range is not None:
            rdm2_filtered = self._apply_distance_filter(
                rdm2, n_orb, max_range, distance_matrix
            )
        else:
            rdm2_filtered = rdm2
        
        # 行列形式に変形
        if self.use_gpu:
            M = cp.asarray(rdm2_filtered.reshape(n_orb**2, n_orb**2))
        else:
            M = rdm2_filtered.reshape(n_orb**2, n_orb**2)
        
        # SVD
        if self.use_gpu:
            U, S, Vt = cp.linalg.svd(M, full_matrices=False)
            S_np = cp.asnumpy(S)
        else:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            S_np = S
        
        # 動的 k 選択
        total_var = np.sum(S_np**2)
        
        if total_var < 1e-14:
            return VorticityResult(vorticity=0.0, effective_rank=0, alpha=0.0)
        
        cumvar = np.cumsum(S_np**2) / total_var
        k = int(np.searchsorted(cumvar, self.svd_cut) + 1)
        k = max(2, min(k, len(S_np)))
        
        # Λ空間への射影
        S_proj = U[:, :k]
        M_lambda = S_proj.conj().T @ M @ S_proj
        
        # 勾配計算
        grad_M = xp.zeros_like(M_lambda)
        grad_M[:-1, :] = M_lambda[1:, :] - M_lambda[:-1, :]
        
        # 電流: J = M_λ @ ∇M_λ
        J_lambda = M_lambda @ grad_M
        
        # Vorticity: ||J - J^T||
        curl_J = J_lambda - J_lambda.T
        V = float(xp.sqrt(xp.sum(xp.abs(curl_J)**2)))
        
        if self.use_gpu:
            V = float(V)
        
        return VorticityResult(
            vorticity=V,
            effective_rank=k,
            alpha=0.0,
            method='svd_projection'
        )
    
    def _apply_distance_filter(self,
                                rdm2: np.ndarray,
                                n_orb: int,
                                max_range: float,
                                distance_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        距離フィルターを適用
        
        2つのモード：
        1. distance_matrix = None: インデックス差（Hubbard 模型用）
        2. distance_matrix あり: 空間距離（分子系用）
        
        Args:
            rdm2: 2-RDM
            n_orb: 軌道数
            max_range: 最大距離
            distance_matrix: 空間距離行列（オプション）
        """
        rdm2_filtered = np.zeros_like(rdm2)
        
        for i in range(n_orb):
            for j in range(n_orb):
                for k in range(n_orb):
                    for ll in range(n_orb):
                        if distance_matrix is not None:
                            # 空間距離を使用
                            d1 = distance_matrix[i, j]
                            d2 = distance_matrix[k, ll]
                            d3 = distance_matrix[i, k]
                            d4 = distance_matrix[j, ll]
                        else:
                            # インデックス差を使用
                            d1 = abs(i - j)
                            d2 = abs(k - ll)
                            d3 = abs(i - k)
                            d4 = abs(j - ll)
                        
                        max_d = max(d1, d2, d3, d4)
                        
                        if max_d <= max_range:
                            rdm2_filtered[i, j, k, ll] = rdm2[i, j, k, ll]
        
        return rdm2_filtered
    
    def compute_with_energy(self,
                            rdm2: np.ndarray,
                            n_orb: int,
                            E_xc: float,
                            max_range: Optional[float] = None,
                            distance_matrix: Optional[np.ndarray] = None) -> VorticityResult:
        """
        E_xc 付きで Vorticity と α を計算
        
        α = |E_xc| / V
        """
        result = self.compute_vorticity(rdm2, n_orb, max_range, distance_matrix)
        
        if result.vorticity > 1e-10:
            alpha = abs(E_xc) / result.vorticity
        else:
            alpha = 0.0
        
        return VorticityResult(
            vorticity=result.vorticity,
            effective_rank=result.effective_rank,
            alpha=alpha,
            method=result.method
        )
    
    def compute_gamma_decomposition(self,
                                     rdm2: np.ndarray,
                                     n_orb: int,
                                     E_xc: float,
                                     local_range: int = 2,
                                     distance_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        γ を局所/非局所に分解
        
        γ_total  = 全相関
        γ_local  = 局所相関（max_range=local_range）
        γ_memory = γ_total - γ_local
        
        Args:
            rdm2: 2-RDM
            n_orb: 軌道数
            E_xc: 相関エネルギー
            local_range: 「局所」の定義（デフォルト: 2サイト）
            distance_matrix: 空間距離行列（分子系用）
        
        Returns:
            γ 分解の辞書
        """
        # 全相関
        result_total = self.compute_with_energy(rdm2, n_orb, E_xc, 
                                                 max_range=None,
                                                 distance_matrix=distance_matrix)
        
        # 局所相関
        result_local = self.compute_with_energy(rdm2, n_orb, E_xc,
                                                 max_range=local_range,
                                                 distance_matrix=distance_matrix)
        
        # α から γ を推定（単一系では近似的）
        # 本来は複数サイズでスケーリングが必要
        
        return {
            'V_total': result_total.vorticity,
            'V_local': result_local.vorticity,
            'alpha_total': result_total.alpha,
            'alpha_local': result_local.alpha,
            'ratio': result_total.vorticity / (result_local.vorticity + 1e-10),
            'local_range': local_range,
        }


# =============================================================================
# Gamma Extractor (Multi-size Scaling)
# =============================================================================

class GammaExtractor:
    """
    複数サイズのデータから γ を抽出
    
    α = |E_xc| / V ∝ N^(-γ)
    
    → log(α) = const - γ log(N)
    """
    
    def __init__(self):
        self.data_points: List[Tuple[int, float, float]] = []
    
    def add_data(self, n_electrons: int, E_xc: float, vorticity: float):
        """データ点を追加"""
        self.data_points.append((n_electrons, E_xc, vorticity))
    
    def clear(self):
        """データをクリア"""
        self.data_points.clear()
    
    def extract_gamma(self) -> Dict[str, Any]:
        """
        γ をフィッティングで抽出
        
        Returns:
            gamma, r_squared, interpretation を含む辞書
        """
        if len(self.data_points) < 3:
            return {'gamma': None, 'error': 'Insufficient data points (need >= 3)'}
        
        Ns = np.array([d[0] for d in self.data_points])
        E_xcs = np.array([d[1] for d in self.data_points])
        Vs = np.array([d[2] for d in self.data_points])
        
        # ゼロ除算回避
        valid = Vs > 1e-10
        if np.sum(valid) < 3:
            return {'gamma': None, 'error': 'Too many zero vorticities'}
        
        Ns = Ns[valid]
        alphas = np.abs(E_xcs[valid]) / Vs[valid]
        
        # log-log フィッティング
        log_N = np.log(Ns)
        log_alpha = np.log(alphas + 1e-10)
        
        # 線形回帰
        slope, intercept = np.polyfit(log_N, log_alpha, 1)
        gamma = -slope
        
        # R²
        pred = slope * log_N + intercept
        ss_res = np.sum((log_alpha - pred)**2)
        ss_tot = np.sum((log_alpha - log_alpha.mean())**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        # 解釈
        if gamma < 1.5:
            interpretation = "Short-range dominant (local correlations)"
        elif gamma < 2.5:
            interpretation = "Mixed regime (local + non-local)"
        else:
            interpretation = "Long-range dominant (collective effects)"
        
        return {
            'gamma': float(gamma),
            'r_squared': float(r_squared),
            'intercept': float(intercept),
            'n_points': len(Ns),
            'interpretation': interpretation
        }
    
    @staticmethod
    def decompose_gamma(gamma_total: float, gamma_local: float) -> Dict[str, float]:
        """
        γ を成分分解
        
        γ_memory = γ_total - γ_local
        """
        gamma_memory = gamma_total - gamma_local
        
        if gamma_memory > gamma_local:
            interpretation = "Memory-dominated: Long-range correlations primary"
        elif gamma_memory > 0.5:
            interpretation = "Significant memory: Both local and non-local matter"
        else:
            interpretation = "Local-dominated: Short-range correlations primary"
        
        return {
            'gamma_total': gamma_total,
            'gamma_local': gamma_local,
            'gamma_memory': gamma_memory,
            'memory_fraction': gamma_memory / (gamma_total + 1e-10),
            'interpretation': interpretation
        }


# =============================================================================
# PySCF Integration
# =============================================================================

def compute_orbital_distance_matrix(mol) -> np.ndarray:
    """
    PySCF 分子オブジェクトから軌道間距離行列を計算
    
    各軌道の「重心」を原子位置で近似し、距離を計算
    
    Args:
        mol: PySCF Mole オブジェクト
        
    Returns:
        distance_matrix: (n_orb, n_orb) 距離行列 [Å]
    """
    # 原子座標を取得
    coords = mol.atom_coords()  # Bohr
    coords_ang = coords * 0.529177  # Å に変換
    
    # 各軌道がどの原子に属するか
    ao_labels = mol.ao_labels()
    n_orb = len(ao_labels)
    
    # 軌道の原子インデックスを抽出
    orbital_atoms = []
    for label in ao_labels:
        # label は "0 Fe 1s" のような形式
        atom_idx = int(label.split()[0])
        orbital_atoms.append(atom_idx)
    
    # 距離行列を構築
    distance_matrix = np.zeros((n_orb, n_orb))
    
    for i in range(n_orb):
        for j in range(n_orb):
            atom_i = orbital_atoms[i]
            atom_j = orbital_atoms[j]
            
            if atom_i == atom_j:
                distance_matrix[i, j] = 0.0
            else:
                diff = coords_ang[atom_i] - coords_ang[atom_j]
                distance_matrix[i, j] = np.linalg.norm(diff)
    
    return distance_matrix


def vorticity_from_pyscf(mf, method: str = 'ccsd', local_range: float = 3.0) -> Dict[str, Any]:
    """
    PySCF 計算から γ 分解を実行
    
    Args:
        mf: PySCF SCF オブジェクト（収束済み）
        method: 'ccsd' or 'fci'
        local_range: 局所相関の距離閾値 [Å]
        
    Returns:
        γ 分解結果
    """
    from pyscf import cc, fci
    
    mol = mf.mol
    n_orb = mol.nao
    
    # 距離行列を計算
    dist_mat = compute_orbital_distance_matrix(mol)
    
    # 相関計算
    if method == 'ccsd':
        mycc = cc.CCSD(mf)
        mycc.kernel()
        E_corr = mycc.e_corr
        
        # 2-RDM 取得
        rdm1, rdm2 = mycc.make_rdm1(), mycc.make_rdm2()
        
        # UCCSD の場合は成分を結合
        if isinstance(rdm2, tuple):
            rdm2_aa, rdm2_ab, rdm2_bb = rdm2
            rdm2 = rdm2_aa + rdm2_ab + rdm2_ab.transpose(2,3,0,1) + rdm2_bb
            
    elif method == 'fci':
        cisolver = fci.FCI(mf)
        E_fci, fcivec = cisolver.kernel()
        E_corr = E_fci - mf.e_tot
        
        n_elec = mol.nelectron
        nelec = (n_elec // 2, n_elec // 2)
        rdm1, rdm2 = cisolver.make_rdm12(fcivec, n_orb, nelec)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Vorticity 計算
    calc = VorticityCalculator(use_gpu=False)
    result = calc.compute_gamma_decomposition(
        rdm2, n_orb, E_corr,
        local_range=local_range,
        distance_matrix=dist_mat
    )
    
    result['E_corr'] = E_corr
    result['method'] = method
    result['n_orb'] = n_orb
    
    return result


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Vorticity Calculator Test")
    print("=" * 70)
    
    # ダミー 2-RDM
    n_orb = 6
    np.random.seed(42)
    rdm2 = np.random.randn(n_orb, n_orb, n_orb, n_orb)
    rdm2 = (rdm2 + rdm2.transpose(2, 3, 0, 1)) / 2  # 対称化
    
    calc = VorticityCalculator(svd_cut=0.95, use_gpu=False)
    
    # 基本テスト
    print("\n[1] Basic Vorticity")
    result = calc.compute_with_energy(rdm2, n_orb, E_xc=-0.5)
    print(f"  {result}")
    
    # 距離フィルター（インデックス差）
    print("\n[2] Distance Filter (Index)")
    for max_r in [1, 2, 3, None]:
        result = calc.compute_with_energy(rdm2, n_orb, E_xc=-0.5, max_range=max_r)
        r_label = "∞" if max_r is None else max_r
        print(f"  r≤{r_label}: V={result.vorticity:.4f}, α={result.alpha:.4f}")
    
    # 空間距離フィルター
    print("\n[3] Distance Filter (Spatial)")
    # ダミー距離行列（線形配置）
    dist_mat = np.zeros((n_orb, n_orb))
    positions = np.linspace(0, 5, n_orb)  # 0〜5 Å に配置
    for i in range(n_orb):
        for j in range(n_orb):
            dist_mat[i, j] = abs(positions[i] - positions[j])
    
    for max_r in [1.0, 2.0, 3.0, None]:
        result = calc.compute_with_energy(rdm2, n_orb, E_xc=-0.5,
                                          max_range=max_r,
                                          distance_matrix=dist_mat)
        r_label = "∞" if max_r is None else f"{max_r:.1f}Å"
        print(f"  r≤{r_label}: V={result.vorticity:.4f}, α={result.alpha:.4f}")
    
    # γ 分解
    print("\n[4] Gamma Decomposition")
    decomp = calc.compute_gamma_decomposition(rdm2, n_orb, E_xc=-0.5, local_range=2)
    for k, v in decomp.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # γ 抽出（マルチサイズ）
    print("\n[5] Gamma Extraction (Multi-size)")
    extractor = GammaExtractor()
    
    for N in [4, 6, 8, 10, 12]:
        V = N ** 2.5
        E_xc = -0.1 * N
        extractor.add_data(N, E_xc, V)
    
    gamma_result = extractor.extract_gamma()
    for k, v in gamma_result.items():
        print(f"  {k}: {v}")
    
    # 1D Hubbard の結果
    print("\n[6] Hubbard Model Results (Literature)")
    decomp = GammaExtractor.decompose_gamma(gamma_total=2.604, gamma_local=1.388)
    for k, v in decomp.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("✅ Vorticity Calculator Test Complete!")
    print("=" * 70)
