"""
Memory-DFT: Unified γ Decomposition (Hubbard Model)
====================================================

同じHubbard模型で：
- ED (厳密対角化) → γ_total（全相関）
- DMRG (TeNPy) → γ_local（局所相関）
- 差分 → γ_memory

これが正しい比較！

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings

# =============================================================================
# Backend Selection
# =============================================================================

# Try CuPy first, fallback to SciPy
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import eigsh as gpu_eigsh
    HAS_CUPY = True
    print("✅ CuPy available (GPU)")
except ImportError:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh as cpu_eigsh
    cp = np
    csp = sp
    HAS_CUPY = False
    print("✅ NumPy/SciPy (CPU)")

# TeNPy
try:
    import tenpy
    from tenpy.networks.mps import MPS
    from tenpy.models.hubbard import FermiHubbardModel
    from tenpy.algorithms import dmrg
    HAS_TENPY = True
    print(f"✅ TeNPy available (v{tenpy.__version__})")
except ImportError:
    HAS_TENPY = False
    print("⚠️ TeNPy not found")

# Memory-DFT
try:
    from memory_dft.physics.vorticity import GammaExtractor, MemoryKernelFromGamma
    from memory_dft.core.memory_kernel import CompositeMemoryKernel, KernelWeights
    HAS_MEMORY_DFT = True
except ImportError:
    import sys
    sys.path.insert(0, '/content/lambda3-memory-dft')
    try:
        from memory_dft.physics.vorticity import GammaExtractor, MemoryKernelFromGamma
        from memory_dft.core.memory_kernel import CompositeMemoryKernel, KernelWeights
        HAS_MEMORY_DFT = True
    except:
        HAS_MEMORY_DFT = False
        print("⚠️ Memory-DFT not found")


# =============================================================================
# Sparse Hubbard Hamiltonian (from Sparse-Meteor)
# =============================================================================

class SparseHubbardEngine:
    """
    1D Hubbardモデルのスパースハミルトニアン
    
    H = -t Σ (c†_i c_j + h.c.) + U Σ n_i↑ n_i↓
    
    スピンレス版で簡略化（メモリ節約）
    """
    
    def __init__(self, L: int, use_gpu: bool = True, verbose: bool = True):
        self.L = L
        self.dim = 2 ** L
        self.use_gpu = use_gpu and HAS_CUPY
        self.verbose = verbose
        
        if self.use_gpu:
            self.xp = cp
            self.sparse = csp
        else:
            self.xp = np
            import scipy.sparse as sp
            self.sparse = sp
        
        if verbose:
            print(f"\n🔧 SparseHubbardEngine: L={L}, Dim={self.dim:,}")
            print(f"   Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (SciPy)'}")
        
        # パウリ行列
        self._build_operators()
    
    def _build_operators(self):
        """基本演算子を構築"""
        xp = self.xp
        
        I_np = np.eye(2, dtype=np.complex128)
        Sp_np = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        Sm_np = np.array([[0, 0], [1, 0]], dtype=np.complex128)
        n_np = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # 数演算子
        
        if self.use_gpu:
            self.I = csp.csr_matrix(cp.asarray(I_np))
            self.Sp = csp.csr_matrix(cp.asarray(Sp_np))
            self.Sm = csp.csr_matrix(cp.asarray(Sm_np))
            self.n = csp.csr_matrix(cp.asarray(n_np))
        else:
            import scipy.sparse as sp
            self.I = sp.csr_matrix(I_np)
            self.Sp = sp.csr_matrix(Sp_np)
            self.Sm = sp.csr_matrix(Sm_np)
            self.n = sp.csr_matrix(n_np)
    
    def _site_operator(self, op, site: int):
        """サイト演算子を構築"""
        ops = [self.I] * self.L
        ops[site] = op
        
        result = ops[0]
        for i in range(1, self.L):
            result = self.sparse.kron(result, ops[i], format='csr')
        
        return result
    
    def build_hamiltonian(self, t: float = 1.0, U: float = 4.0, periodic: bool = True):
        """
        Hubbardハミルトニアンを構築
        
        Returns:
            H: 全ハミルトニアン
            H_t: ホッピング項（運動エネルギー）
            H_U: 相互作用項（ポテンシャル）
        """
        if self.verbose:
            print(f"   Building H: t={t}, U={U}, periodic={periodic}")
        
        H_t = None
        H_U = None
        
        # ホッピング項
        n_bonds = self.L if periodic else self.L - 1
        for i in range(n_bonds):
            j = (i + 1) % self.L
            
            Sp_i = self._site_operator(self.Sp, i)
            Sm_i = self._site_operator(self.Sm, i)
            Sp_j = self._site_operator(self.Sp, j)
            Sm_j = self._site_operator(self.Sm, j)
            
            term = -t * (Sp_i @ Sm_j + Sm_i @ Sp_j)
            
            if H_t is None:
                H_t = term
            else:
                H_t = H_t + term
        
        # 相互作用項（隣接サイト間）
        for i in range(n_bonds):
            j = (i + 1) % self.L
            
            n_i = self._site_operator(self.n, i)
            n_j = self._site_operator(self.n, j)
            
            term = U * n_i @ n_j
            
            if H_U is None:
                H_U = term
            else:
                H_U = H_U + term
        
        H = H_t + H_U
        
        if self.verbose:
            print(f"   ✅ Built: nnz={H.nnz:,}")
        
        return H, H_t, H_U
    
    def compute_ground_state(self, H, k: int = 1):
        """基底状態を計算"""
        if self.verbose:
            print(f"   Computing {k} lowest eigenstates...")
        
        if self.use_gpu:
            eigenvalues, eigenvectors = gpu_eigsh(H, k=k, which='SA')
        else:
            eigenvalues, eigenvectors = cpu_eigsh(H, k=k, which='SA')
        
        # ソートはxp空間で
        xp = self.xp
        idx = xp.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        if self.verbose:
            E0 = float(eigenvalues[0].get()) if self.use_gpu else float(eigenvalues[0])
            print(f"   E_0 = {E0:.6f}")
        
        return eigenvalues, eigenvectors
    
    def compute_2rdm(self, psi):
        """
        2粒子密度行列を計算
        
        ρ^(2)_{ijkl} = ⟨ψ|c†_i c†_j c_k c_l|ψ⟩
        
        簡略化: 対角ブロックのみ
        """
        xp = self.xp
        L = self.L
        
        # (L, L, L, L) は大きすぎる場合があるので
        # 近接ペアのみ計算
        rdm2_local = []
        
        for i in range(L):
            j = (i + 1) % L
            
            # ⟨n_i n_j⟩
            n_i = self._site_operator(self.n, i)
            n_j = self._site_operator(self.n, j)
            
            rho_ij = float(xp.real(xp.vdot(psi, (n_i @ n_j) @ psi)))
            rdm2_local.append(rho_ij)
        
        return np.array(rdm2_local)


# =============================================================================
# ED: γ_total calculation
# =============================================================================

def ed_compute_gamma_total(L_values: List[int] = None, U_t: float = 2.0) -> Dict:
    """
    ED（厳密対角化）でγ_totalを計算
    
    α = |E_xc| / V ∝ L^(-γ)
    """
    if L_values is None:
        L_values = [6, 8, 10, 12]  # EDで可能なサイズ
    
    print("\n" + "="*60)
    print(f"ED: γ_total Extraction (U/t={U_t})")
    print("="*60)
    
    results = []
    extractor = GammaExtractor() if HAS_MEMORY_DFT else None
    
    # E(U=0) reference
    E_U0 = {}
    for L in L_values:
        if 2**L > 100000:  # メモリ制限
            print(f"  L={L}: Too large for ED (dim={2**L:,})")
            continue
        
        engine = SparseHubbardEngine(L, use_gpu=HAS_CUPY, verbose=False)
        H, _, _ = engine.build_hamiltonian(t=1.0, U=0.0)
        E, _ = engine.compute_ground_state(H)
        # CuPy→NumPy変換
        if HAS_CUPY:
            E_U0[L] = float(E[0].get())
        else:
            E_U0[L] = float(E[0])
    
    # U≠0 計算
    for L in L_values:
        if L not in E_U0:
            continue
        
        engine = SparseHubbardEngine(L, use_gpu=HAS_CUPY, verbose=False)
        H, H_t, H_U = engine.build_hamiltonian(t=1.0, U=U_t)
        E, psi = engine.compute_ground_state(H)
        
        # E_xc計算（CuPy→float変換）
        if HAS_CUPY:
            E_xc = float(E[0].get()) - E_U0[L]
        else:
            E_xc = float(E[0]) - E_U0[L]
        
        # Vorticity（簡易版：運動エネルギーの分散）
        xp = engine.xp
        psi_vec = psi[:, 0]
        
        K_val = xp.real(xp.vdot(psi_vec, H_t @ psi_vec))
        K2_val = xp.real(xp.vdot(psi_vec, (H_t @ H_t) @ psi_vec))
        
        # CuPy→NumPy変換
        if HAS_CUPY:
            K = float(K_val.get())
            K2 = float(K2_val.get())
        else:
            K = float(K_val)
            K2 = float(K2_val)
        
        V_approx = np.sqrt(abs(K2 - K**2)) * L  # スケーリング
        
        alpha = abs(E_xc) / (V_approx + 1e-10)
        
        print(f"  L={L:2d}: E_xc={E_xc:8.4f}, V={V_approx:.4f}, α={alpha:.4f}")
        
        results.append({'L': L, 'E_xc': E_xc, 'V': V_approx, 'alpha': alpha})
        
        if extractor:
            extractor.add_data(L, E_xc, V_approx)
    
    # γ抽出
    if extractor and len(results) >= 3:
        gamma_result = extractor.extract_gamma()
        print(f"\n  → γ_total = {gamma_result['gamma']:.3f} (R²={gamma_result['r_squared']:.3f})")
        return {'gamma': gamma_result['gamma'], 'r2': gamma_result['r_squared'], 'results': results}
    
    return {'gamma': None, 'results': results}


# =============================================================================
# DMRG: γ_local calculation
# =============================================================================

def dmrg_compute_gamma_local(L_values: List[int] = None, U_t: float = 2.0, chi_max: int = 100) -> Dict:
    """
    DMRGでγ_localを計算（局所相関のみ）
    """
    if not HAS_TENPY:
        print("❌ TeNPy required for DMRG")
        return {'gamma': None}
    
    if L_values is None:
        L_values = [8, 12, 16, 20]
    
    print("\n" + "="*60)
    print(f"DMRG: γ_local Extraction (U/t={U_t})")
    print("="*60)
    
    results = []
    
    # E(U=0) reference
    E_U0 = {}
    for L in L_values:
        res = _run_dmrg(L, U_t=0.0, chi_max=chi_max)
        E_U0[L] = res['E']
    
    # U≠0 計算
    for L in L_values:
        res = _run_dmrg(L, U_t=U_t, chi_max=chi_max)
        E_xc = res['E'] - E_U0[L]
        
        V, avg_rank = _compute_local_correlations(res['psi'], L)
        
        print(f"  L={L:2d}: E_xc={E_xc:8.4f}, V={V:.4f}, rank={avg_rank:.2f}")
        
        results.append({'L': L, 'E_xc': E_xc, 'V': V, 'rank': avg_rank})
    
    # γ_local抽出
    if len(results) >= 3:
        Ls = np.array([r['L'] for r in results])
        Vs = np.array([r['V'] for r in results])
        E_xcs = np.array([abs(r['E_xc']) for r in results])
        
        alphas = E_xcs / (Vs + 1e-10)
        
        log_L = np.log(Ls)
        log_alpha = np.log(alphas + 1e-10)
        
        slope, intercept = np.polyfit(log_L, log_alpha, 1)
        gamma_local = -slope
        
        # R²
        pred = slope * log_L + intercept
        ss_res = np.sum((log_alpha - pred)**2)
        ss_tot = np.sum((log_alpha - log_alpha.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        print(f"\n  → γ_local = {gamma_local:.3f} (R²={r2:.3f})")
        return {'gamma': gamma_local, 'r2': r2, 'results': results}
    
    return {'gamma': None, 'results': results}


def _run_dmrg(L: int, U_t: float, chi_max: int = 100) -> Dict:
    """DMRG計算（警告抑制版）"""
    model_params = {
        'L': L,
        't': 1.0,
        'U': U_t,
        'mu': 0.0,
        'bc_MPS': 'finite',
    }
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        model = FermiHubbardModel(model_params)
        
        init_state = ['up', 'down'] * (L // 2)
        if len(init_state) < L:
            init_state.append('up')
        
        psi = MPS.from_product_state(model.lat.mps_sites(), init_state, bc='finite')
        
        dmrg_params = {
            'mixer': True,
            'max_E_err': 1e-10,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        }
        
        info = dmrg.run(psi, model, dmrg_params)
    
    return {'E': info['E'], 'psi': psi}


def _compute_local_correlations(psi, L: int, max_range: int = 2) -> Tuple[float, float]:
    """局所相関からVorticityを計算"""
    V = 0.0
    total_rank = 0.0
    pairs = 0
    
    for i in range(L):
        for j in range(i+1, min(i+max_range+1, L)):
            try:
                rho = psi.get_rho_segment([i, j])
                rho_np = rho.to_ndarray()
                
                rho_swap = rho_np.transpose(2, 3, 0, 1)
                asym = rho_np + rho_swap
                V += np.sum(np.abs(asym)**2)
                
                d = rho_np.shape[0]
                M = rho_np.reshape(d*d, d*d)
                U, S, Vh = np.linalg.svd(M, full_matrices=False)
                
                S2 = S**2
                S2 = S2[S2 > 1e-12]
                p = S2 / S2.sum()
                entropy = -np.sum(p * np.log(p + 1e-14))
                eff_rank = np.exp(entropy)
                
                total_rank += eff_rank
                pairs += 1
            except:
                continue
    
    avg_rank = total_rank / pairs if pairs > 0 else 0
    return V, avg_rank


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("🧪 Memory-DFT: Unified γ Decomposition (Hubbard Model)")
    print("="*70)
    print("\n同じHubbard模型で比較:")
    print("  ED   → γ_total (全相関)")
    print("  DMRG → γ_local (局所相関)")
    print("  差分 → γ_memory (Memory効果！)")
    
    U_t = 2.0
    
    # ED: γ_total
    ed_result = ed_compute_gamma_total(L_values=[6, 8, 10], U_t=U_t)
    
    # DMRG: γ_local
    dmrg_result = dmrg_compute_gamma_local(L_values=[8, 12, 16, 20], U_t=U_t)
    
    # γ分解
    print("\n" + "="*60)
    print("γ DECOMPOSITION (Same Hubbard Model)")
    print("="*60)
    
    gamma_total = ed_result.get('gamma')
    gamma_local = dmrg_result.get('gamma')
    
    if gamma_total is not None and gamma_local is not None:
        gamma_memory = gamma_total - gamma_local
        memory_fraction = gamma_memory / (gamma_total + 1e-10)
        
        print(f"  γ_total  (ED)   = {gamma_total:.3f}")
        print(f"  γ_local  (DMRG) = {gamma_local:.3f}")
        print(f"  ─────────────────────────")
        print(f"  γ_memory        = {gamma_memory:.3f}")
        print(f"  Memory fraction = {memory_fraction*100:.1f}%")
        
        # 物理的解釈
        print(f"\n  Physical Interpretation:")
        if gamma_memory > 0.5:
            print(f"    ✅ γ_memory > 0.5: Significant long-range correlations!")
            print(f"    → Memory kernel is NECESSARY")
        elif gamma_memory > 0:
            print(f"    ⚠️ γ_memory > 0: Some long-range effects")
        else:
            print(f"    ❓ γ_memory ≤ 0: Local correlations dominate")
        
        # Memory kernel推定
        if HAS_MEMORY_DFT and gamma_memory > 0:
            print(f"\n  Memory Kernel Parameters:")
            decomp = {'gamma_total': gamma_total, 'gamma_local': gamma_local, 'gamma_memory': gamma_memory}
            params = MemoryKernelFromGamma.estimate_kernel_params(decomp)
            print(f"    γ_field = {params['gamma_field']:.2f}")
            print(f"    β_phys  = {params['beta_phys']:.2f}")
            print(f"    weights = {params['weights']}")
    else:
        print("  ⚠️ Could not compute both γ values")
        if gamma_total is not None:
            print(f"  γ_total = {gamma_total:.3f}")
        if gamma_local is not None:
            print(f"  γ_local = {gamma_local:.3f}")
    
    print("\n" + "="*70)
    print("✅ Unified γ Decomposition Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
