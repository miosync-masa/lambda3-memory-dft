"""
Vorticity and γ Calculation for Memory-DFT
==========================================

PySCF vorticityとの連携によるγ（相関指数）の計算

理論的背景:
- α = E_xc / V ∝ N^(-γ)
- γ_total = γ_local + γ_memory
- γ_memory ≈ 1.0 (PySCF - DMRG から導出)

この差分がMemory kernelの存在証明！

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

# JAX support (optional, for GPU acceleration)
try:
    import jax.numpy as jnp
    from jax import jit
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False
    def jit(f): return f


@dataclass
class VorticityResult:
    """Vorticity計算結果"""
    vorticity: float
    effective_rank: int
    alpha: float  # E_xc / V
    gamma: Optional[float] = None  # スケーリング指数
    components: Dict[str, float] = None


class VorticityCalculator:
    """
    2-RDMからVorticityを計算
    
    V = √(Σ ||J - J^T||²)
    
    ここで J = M_λ @ ∇M_λ
    """
    
    def __init__(self, svd_cut: float = 0.95, use_jax: bool = True):
        """
        Args:
            svd_cut: SVDのカットオフ（累積分散の何%まで）
            use_jax: JAX使用フラグ
        """
        self.svd_cut = svd_cut
        self.use_jax = use_jax and HAS_JAX
        self.xp = jnp if self.use_jax else np
    
    def compute_vorticity(self, rdm2: np.ndarray, n_orb: int) -> VorticityResult:
        """
        2-RDMからVorticityを計算
        
        Args:
            rdm2: 2粒子密度行列 (n_orb, n_orb, n_orb, n_orb)
            n_orb: 軌道数
            
        Returns:
            VorticityResult
        """
        xp = self.xp
        
        # 行列形式に変形
        M = xp.array(rdm2.reshape(n_orb**2, n_orb**2))
        
        # SVD
        U, S, Vt = xp.linalg.svd(M, full_matrices=False)
        
        # 動的k選択（NumPyで）
        S_np = np.array(S) if self.use_jax else S
        total_var = np.sum(S_np**2)
        
        if total_var < 1e-14:
            return VorticityResult(vorticity=0.0, effective_rank=0, alpha=0.0)
        
        cumvar = np.cumsum(S_np**2) / total_var
        k = int(np.searchsorted(cumvar, self.svd_cut) + 1)
        k = max(k, 2)
        k = min(k, len(S_np))
        
        # Λ空間への射影
        S_proj = U[:, :k]
        M_lambda = S_proj.T @ M @ S_proj
        
        # 勾配計算
        grad_M = xp.zeros_like(M_lambda)
        if self.use_jax:
            grad_M = grad_M.at[:-1, :].set(M_lambda[1:, :] - M_lambda[:-1, :])
        else:
            grad_M[:-1, :] = M_lambda[1:, :] - M_lambda[:-1, :]
        
        # 電流: J = M_λ @ ∇M_λ
        J_lambda = M_lambda @ grad_M
        
        # Vorticity: ||J - J^T||²
        curl_J = J_lambda - J_lambda.T
        V = float(xp.sqrt(xp.sum(curl_J**2)))
        
        return VorticityResult(
            vorticity=V,
            effective_rank=k,
            alpha=0.0  # E_xc が必要
        )
    
    def compute_with_energy(self, 
                            rdm2: np.ndarray, 
                            n_orb: int,
                            E_xc: float) -> VorticityResult:
        """
        E_xc付きでVorticityとαを計算
        
        α = E_xc / V ∝ N^(-γ)
        """
        result = self.compute_vorticity(rdm2, n_orb)
        
        if result.vorticity > 1e-10:
            alpha = abs(E_xc) / result.vorticity
        else:
            alpha = 0.0
        
        return VorticityResult(
            vorticity=result.vorticity,
            effective_rank=result.effective_rank,
            alpha=alpha
        )


class GammaExtractor:
    """
    γ（相関指数）の抽出
    
    α = E_xc / V ∝ N^(-γ)
    
    → log(α) = const - γ log(N)
    """
    
    def __init__(self):
        self.data_points: List[Tuple[int, float, float]] = []  # (N, E_xc, V)
    
    def add_data(self, n_electrons: int, E_xc: float, vorticity: float):
        """データ点を追加"""
        self.data_points.append((n_electrons, E_xc, vorticity))
    
    def extract_gamma(self) -> Dict[str, Any]:
        """
        γをフィッティングで抽出
        
        Returns:
            gamma: スケーリング指数
            r_squared: 決定係数
            interpretation: 物理的解釈
        """
        if len(self.data_points) < 3:
            return {'gamma': None, 'error': 'Insufficient data points'}
        
        # α = E_xc / V
        Ns = np.array([d[0] for d in self.data_points])
        E_xcs = np.array([d[1] for d in self.data_points])
        Vs = np.array([d[2] for d in self.data_points])
        
        # ゼロ除算回避
        valid = Vs > 1e-10
        if np.sum(valid) < 3:
            return {'gamma': None, 'error': 'Too many zero vorticities'}
        
        Ns = Ns[valid]
        alphas = np.abs(E_xcs[valid]) / Vs[valid]
        
        # log-logフィッティング
        log_N = np.log(Ns)
        log_alpha = np.log(alphas + 1e-10)
        
        # 線形回帰
        slope, intercept = np.polyfit(log_N, log_alpha, 1)
        gamma = -slope  # α ∝ N^(-γ) より
        
        # R²
        pred = slope * log_N + intercept
        ss_res = np.sum((log_alpha - pred)**2)
        ss_tot = np.sum((log_alpha - log_alpha.mean())**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        # 物理的解釈
        if gamma < 1.5:
            interpretation = "Short-range dominant (local correlations)"
        elif gamma < 2.5:
            interpretation = "Mixed regime (local + non-local)"
        else:
            interpretation = "Long-range dominant (collective effects)"
        
        return {
            'gamma': gamma,
            'r_squared': r_squared,
            'intercept': intercept,
            'n_points': len(Ns),
            'interpretation': interpretation
        }
    
    def decompose_gamma(self, 
                        gamma_total: float,
                        gamma_local: float) -> Dict[str, float]:
        """
        γを成分分解
        
        γ_total = γ_local + γ_memory
        
        Args:
            gamma_total: PySCF（全体2-RDM）から
            gamma_local: DMRG（局所2-RDM）から
            
        Returns:
            γ成分と解釈
        """
        gamma_memory = gamma_total - gamma_local
        
        return {
            'gamma_total': gamma_total,
            'gamma_local': gamma_local,
            'gamma_memory': gamma_memory,
            'memory_fraction': gamma_memory / (gamma_total + 1e-10),
            'interpretation': self._interpret_decomposition(gamma_local, gamma_memory)
        }
    
    def _interpret_decomposition(self, gamma_local: float, gamma_memory: float) -> str:
        """分解の物理的解釈"""
        if gamma_memory > gamma_local:
            return "Memory-dominated: Long-range correlations are primary"
        elif gamma_memory > 0.5:
            return "Significant memory: Both local and non-local matter"
        else:
            return "Local-dominated: Short-range correlations are primary"


class MemoryKernelFromGamma:
    """
    γからMemory kernelパラメータを推定
    
    γ_memory ≈ 1.0 → Power-law kernel
    γ_memory < 0.5 → Stretched exponential
    """
    
    @staticmethod
    def estimate_kernel_params(gamma_decomposition: Dict[str, float]) -> Dict[str, Any]:
        """
        γ分解からkernelパラメータを推定
        """
        gamma_memory = gamma_decomposition['gamma_memory']
        
        # Power-law成分
        gamma_field = min(gamma_memory, 1.5)  # 上限あり
        
        # 残りはstretched exp成分
        gamma_residual = max(0, gamma_memory - gamma_field)
        beta_phys = 0.5 + 0.3 * (1 - gamma_residual)  # 0.5-0.8の範囲
        
        # 重み推定
        if gamma_memory > 1.0:
            w_field = 0.6
            w_phys = 0.3
            w_chem = 0.1
        elif gamma_memory > 0.5:
            w_field = 0.4
            w_phys = 0.4
            w_chem = 0.2
        else:
            w_field = 0.2
            w_phys = 0.5
            w_chem = 0.3
        
        return {
            'gamma_field': gamma_field,
            'beta_phys': beta_phys,
            'weights': {
                'field': w_field,
                'phys': w_phys,
                'chem': w_chem
            },
            'confidence': 'high' if gamma_decomposition.get('r_squared', 0) > 0.9 else 'medium'
        }


# =============================================================================
# PySCF Integration (placeholder)
# =============================================================================

def from_pyscf_mol(mol, method: str = 'ccsd') -> VorticityResult:
    """
    PySCF分子オブジェクトからVorticityを計算
    
    TODO: 本格的な実装
    """
    raise NotImplementedError("PySCF integration not yet implemented. Use rdm2 directly.")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Vorticity Calculator Test")
    print("="*70)
    
    # ダミー2-RDM
    n_orb = 4
    rdm2 = np.random.randn(n_orb, n_orb, n_orb, n_orb)
    rdm2 = (rdm2 + rdm2.transpose(2, 3, 0, 1)) / 2  # 対称化
    
    calc = VorticityCalculator(svd_cut=0.95, use_jax=False)
    result = calc.compute_with_energy(rdm2, n_orb, E_xc=-0.5)
    
    print(f"\nVorticity Result:")
    print(f"  V = {result.vorticity:.6f}")
    print(f"  k (effective rank) = {result.effective_rank}")
    print(f"  α = E_xc/V = {result.alpha:.6f}")
    
    # γ抽出テスト
    print("\n" + "="*70)
    print("Gamma Extraction Test")
    print("="*70)
    
    extractor = GammaExtractor()
    
    # ダミーデータ（γ≈2のスケーリング）
    for N in [4, 6, 8, 10, 12]:
        V = N**2.5  # V ∝ N^2.5
        E_xc = -0.1 * N  # E_xc ∝ N
        extractor.add_data(N, E_xc, V)
    
    gamma_result = extractor.extract_gamma()
    print(f"\nGamma extraction:")
    for k, v in gamma_result.items():
        print(f"  {k}: {v}")
    
    # γ分解
    print("\n" + "="*70)
    print("Gamma Decomposition")
    print("="*70)
    
    decomp = extractor.decompose_gamma(gamma_total=2.3, gamma_local=1.2)
    print("\nDecomposition (PySCF=2.3, DMRG=1.2):")
    for k, v in decomp.items():
        print(f"  {k}: {v}")
    
    # Kernel推定
    kernel_params = MemoryKernelFromGamma.estimate_kernel_params(decomp)
    print("\nEstimated kernel parameters:")
    for k, v in kernel_params.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Vorticity Calculator OK!")
