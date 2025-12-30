"""
Memory-DFT: γ(r) Distance Scan (ED Only!)
==========================================

DMRGを待たずにEDから直接γ_localとγ_memoryを分解！

原理:
  ED + 距離制限なし → γ_total
  ED + r≤2         → γ_local
  差分             → γ_memory

これでPRLの「Non-Markovian extension」が5分で検証できる！

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Dict, List, Optional
import time

# Backend
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import eigsh as gpu_eigsh
    HAS_CUPY = True
    print("✅ CuPy (GPU)")
except ImportError:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh as cpu_eigsh
    cp = np
    csp = sp
    HAS_CUPY = False
    print("✅ NumPy/SciPy (CPU)")

# Memory-DFT
try:
    from memory_dft.physics.vorticity import VorticityCalculator, GammaExtractor
    print("✅ Memory-DFT")
except ImportError:
    import sys
    sys.path.insert(0, '/content/lambda3-memory-dft')
    from memory_dft.physics.vorticity import VorticityCalculator, GammaExtractor
    print("✅ Memory-DFT (from path)")


class EDGammaScanner:
    """
    EDから相関距離ごとのγを抽出
    
    DMRG不要！全てEDデータから！
    """
    
    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        self.use_gpu = use_gpu and HAS_CUPY
        self.verbose = verbose
        
        if self.use_gpu:
            self.xp = cp
            self.sparse = csp
        else:
            self.xp = np
            import scipy.sparse as sp
            self.sparse = sp
        
        # Pauli operators
        self._build_operators()
    
    def _build_operators(self):
        """スピン演算子"""
        I = np.eye(2, dtype=np.complex128)
        Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
        n = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        
        if self.use_gpu:
            self.I = csp.csr_matrix(cp.asarray(I))
            self.Sp = csp.csr_matrix(cp.asarray(Sp))
            self.Sm = csp.csr_matrix(cp.asarray(Sm))
            self.n = csp.csr_matrix(cp.asarray(n))
        else:
            self.I = sp.csr_matrix(I)
            self.Sp = sp.csr_matrix(Sp)
            self.Sm = sp.csr_matrix(Sm)
            self.n = sp.csr_matrix(n)
    
    def _site_op(self, op, site: int, L: int):
        """サイト演算子"""
        ops = [self.I] * L
        ops[site] = op
        result = ops[0]
        for i in range(1, L):
            result = self.sparse.kron(result, ops[i], format='csr')
        return result
    
    def build_hubbard(self, L: int, t: float = 1.0, U: float = 2.0):
        """Hubbardハミルトニアン（H_t, H_U分離）"""
        if self.verbose:
            print(f"  Building Hubbard: L={L}, t={t}, U={U}")
        
        H_t = None
        H_U = None
        
        # ホッピング
        for i in range(L - 1):
            j = i + 1
            Sp_i = self._site_op(self.Sp, i, L)
            Sm_i = self._site_op(self.Sm, i, L)
            Sp_j = self._site_op(self.Sp, j, L)
            Sm_j = self._site_op(self.Sm, j, L)
            term = -t * (Sp_i @ Sm_j + Sm_i @ Sp_j)
            H_t = term if H_t is None else H_t + term
        
        # 相互作用
        for i in range(L - 1):
            j = i + 1
            n_i = self._site_op(self.n, i, L)
            n_j = self._site_op(self.n, j, L)
            term = U * n_i @ n_j
            H_U = term if H_U is None else H_U + term
        
        return H_t, H_U
    
    def compute_ground_state(self, H):
        """基底状態"""
        if self.use_gpu:
            E, psi = gpu_eigsh(H, k=1, which='SA')
            E = float(E[0].get())
            psi = psi[:, 0].get()
        else:
            E, psi = cpu_eigsh(H, k=1, which='SA')
            E = float(E[0])
            psi = psi[:, 0]
        return E, psi
    
    def compute_2rdm(self, psi, L: int) -> np.ndarray:
        """
        波動関数から2-RDMを計算
        
        ρ^(2)_{ijkl} = ⟨ψ|c†_i c†_j c_k c_l|ψ⟩
        
        簡略版：対角成分 ⟨n_i n_j⟩ を格納
        """
        if self.verbose:
            print(f"  Computing 2-RDM...")
        
        xp = self.xp
        rdm2 = np.zeros((L, L, L, L), dtype=np.complex128)
        
        psi_gpu = xp.asarray(psi) if self.use_gpu else psi
        
        for i in range(L):
            for j in range(L):
                n_i = self._site_op(self.n, i, L)
                n_j = self._site_op(self.n, j, L)
                
                # ⟨n_i n_j⟩
                val = xp.vdot(psi_gpu, (n_i @ n_j) @ psi_gpu)
                if self.use_gpu:
                    val = float(val.get().real)
                else:
                    val = float(val.real)
                
                # 2-RDMに格納（対角近似）
                rdm2[i, i, j, j] = val
                rdm2[i, j, i, j] = val * 0.5
                rdm2[i, j, j, i] = -val * 0.5  # 反対称性
        
        return rdm2
    
    def scan_gamma_vs_range(self, L_values: List[int], U_t: float = 2.0,
                             ranges: List[int] = None) -> Dict:
        """
        γ(r)スキャン
        
        Args:
            L_values: システムサイズのリスト
            U_t: 相互作用強度
            ranges: スキャンする相関距離
            
        Returns:
            γ(r) データ
        """
        if ranges is None:
            ranges = [2, 4, None]  # None = 全相関
        
        print("\n" + "="*60)
        print("ED γ(r) Distance Scan")
        print("="*60)
        print(f"L values: {L_values}")
        print(f"Ranges: {ranges}")
        
        calc = VorticityCalculator(svd_cut=0.95, use_jax=False)
        results = {r: [] for r in ranges}
        
        # E(U=0) reference
        E_U0 = {}
        for L in L_values:
            if 2**L > 50000:
                print(f"  L={L}: Too large, skipping")
                continue
            H_t, H_U = self.build_hubbard(L, t=1.0, U=0.0)
            H = H_t + H_U
            E, _ = self.compute_ground_state(H)
            E_U0[L] = E
        
        # 各サイズで計算
        for L in L_values:
            if L not in E_U0:
                continue
            
            print(f"\n--- L = {L} ---")
            
            H_t, H_U = self.build_hubbard(L, t=1.0, U=U_t)
            H = H_t + H_U
            E, psi = self.compute_ground_state(H)
            E_xc = E - E_U0[L]
            
            print(f"  E_xc = {E_xc:.6f}")
            
            # 2-RDM計算
            rdm2 = self.compute_2rdm(psi, L)
            
            # 各距離でVorticity計算
            for r in ranges:
                if r is not None and r > L:
                    continue
                
                result = calc.compute_with_energy(rdm2, L, E_xc, max_range=r)
                
                r_label = r if r is not None else L
                print(f"  r≤{r_label}: V={result.vorticity:.4f}, α={result.alpha:.4f}")
                
                results[r].append({
                    'L': L,
                    'E_xc': E_xc,
                    'V': result.vorticity,
                    'alpha': result.alpha
                })
        
        return results
    
    def extract_gammas(self, scan_results: Dict) -> Dict:
        """
        γ(r)を抽出
        
        Returns:
            各距離でのγとγ_memory
        """
        print("\n" + "="*60)
        print("γ Extraction by Range")
        print("="*60)
        
        gammas = {}
        
        for r, data in scan_results.items():
            if len(data) < 3:
                continue
            
            Ls = np.array([d['L'] for d in data])
            alphas = np.array([d['alpha'] for d in data])
            
            # log-log fit
            valid = alphas > 1e-10
            if np.sum(valid) < 3:
                continue
            
            log_L = np.log(Ls[valid])
            log_alpha = np.log(alphas[valid])
            
            slope, intercept = np.polyfit(log_L, log_alpha, 1)
            gamma = -slope
            
            # R²
            pred = slope * log_L + intercept
            ss_res = np.sum((log_alpha - pred)**2)
            ss_tot = np.sum((log_alpha - log_alpha.mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            r_label = r if r is not None else "∞"
            print(f"  r ≤ {r_label}: γ = {gamma:.3f} (R² = {r2:.3f})")
            
            gammas[r] = {'gamma': gamma, 'r2': r2}
        
        # γ分解
        if None in gammas and 2 in gammas:
            gamma_total = gammas[None]['gamma']
            gamma_local = gammas[2]['gamma']
            gamma_memory = gamma_total - gamma_local
            
            print(f"\n" + "="*60)
            print("γ DECOMPOSITION")
            print("="*60)
            print(f"  γ_total  (r=∞) = {gamma_total:.3f}")
            print(f"  γ_local  (r≤2) = {gamma_local:.3f}")
            print(f"  ─────────────────────────")
            print(f"  γ_memory       = {gamma_memory:.3f}")
            print(f"  Memory %       = {gamma_memory/gamma_total*100:.1f}%")
            
            gammas['decomposition'] = {
                'gamma_total': gamma_total,
                'gamma_local': gamma_local,
                'gamma_memory': gamma_memory,
                'memory_fraction': gamma_memory / gamma_total
            }
        
        return gammas


def main():
    print("="*70)
    print("🚀 ED γ(r) Distance Scan - No DMRG Required!")
    print("="*70)
    print("\nThis proves γ_memory without waiting 30 min for DMRG!")
    
    t0 = time.time()
    
    scanner = EDGammaScanner(use_gpu=HAS_CUPY, verbose=True)
    
    # スキャン実行
    # L=12まではGPUなら数秒、CPUでも1分以内
    scan_results = scanner.scan_gamma_vs_range(
        L_values=[6, 8, 10, 12],
        U_t=2.0,
        ranges=[2, 4, None]  # r≤2, r≤4, 全相関
    )
    
    # γ抽出
    gammas = scanner.extract_gammas(scan_results)
    
    print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")
    
    # 結論
    if 'decomposition' in gammas:
        decomp = gammas['decomposition']
        print("\n" + "="*70)
        print("📊 CONCLUSION")
        print("="*70)
        
        if decomp['gamma_memory'] > 0.5:
            print(f"""
    γ_memory = {decomp['gamma_memory']:.3f} > 0.5
    
    ✅ Non-Markovian correlations exist!
    ✅ Memory kernel is NECESSARY!
    ✅ This extends Lie & Fullwood PRL 2025!
    
    "We implemented one. てへぺろ (・ω<)"
            """)
        else:
            print(f"""
    γ_memory = {decomp['gamma_memory']:.3f} ≤ 0.5
    
    → Local correlations dominate
    → Markovian QSOT may be sufficient
            """)
    
    print("\n✅ Done!")
    return gammas


if __name__ == "__main__":
    main()
