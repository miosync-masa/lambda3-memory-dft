"""
Sparse Hamiltonian Engine for Memory-DFT
========================================

CuPy + Sparse 行列でハミルトニアンを効率的に構築

Sparse-Meteor v3 をベースに Memory-DFT 用に拡張

Features:
- スパース行列でメモリ効率化
- GPU加速（CuPy利用可能時）
- 各種モデルハミルトニアン生成
- 2-body項のサポート

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# GPU support (optional)
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    HAS_CUPY = True
except ImportError:
    import scipy.sparse as sp
    cp = np
    csp = sp
    HAS_CUPY = False


@dataclass
class SystemGeometry:
    """系のジオメトリ情報"""
    n_sites: int
    bonds: List[Tuple[int, int]]
    plaquettes: List[Tuple[int, ...]] = None
    positions: np.ndarray = None  # (n_sites, 3) optional
    
    @property
    def dim(self) -> int:
        """ヒルベルト空間の次元"""
        return 2 ** self.n_sites


class SparseHamiltonianEngine:
    """
    スパース行列でハミルトニアンを構築するエンジン
    
    Memory-DFT用に拡張:
    - 運動エネルギー/ポテンシャルの分離
    - Λ計算用の演算子生成
    """
    
    def __init__(self, n_sites: int, use_gpu: bool = True, verbose: bool = True):
        """
        Args:
            n_sites: サイト数
            use_gpu: GPU使用フラグ
            verbose: 進捗表示
        """
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.verbose = verbose
        
        # Backend選択
        if self.use_gpu:
            self.xp = cp
            self.sparse = csp
        else:
            self.xp = np
            import scipy.sparse as sp
            self.sparse = sp
        
        if verbose:
            print(f"🚀 Sparse Engine: N={n_sites}, Dim={self.dim:,}")
            print(f"   Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (SciPy)'}")
            mem_dense = self.dim * self.dim * 16 / 1e9
            print(f"   Dense would need: {mem_dense:.1f} GB")
        
        # パウリ行列（スパース）
        self._build_pauli_matrices()
        
    def _build_pauli_matrices(self):
        """パウリ行列をスパース形式で構築"""
        xp = self.xp
        
        # NumPyで作成
        I_np = np.eye(2, dtype=np.complex128)
        X_np = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)  # Sx
        Y_np = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)  # Sy
        Z_np = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)  # Sz
        Sp_np = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # S+
        Sm_np = np.array([[0, 0], [1, 0]], dtype=np.complex128)  # S-
        
        # スパース行列に変換
        if self.use_gpu:
            self.I = csp.csr_matrix(cp.asarray(I_np))
            self.X = csp.csr_matrix(cp.asarray(X_np))
            self.Y = csp.csr_matrix(cp.asarray(Y_np))
            self.Z = csp.csr_matrix(cp.asarray(Z_np))
            self.Sp = csp.csr_matrix(cp.asarray(Sp_np))
            self.Sm = csp.csr_matrix(cp.asarray(Sm_np))
        else:
            import scipy.sparse as sp
            self.I = sp.csr_matrix(I_np)
            self.X = sp.csr_matrix(X_np)
            self.Y = sp.csr_matrix(Y_np)
            self.Z = sp.csr_matrix(Z_np)
            self.Sp = sp.csr_matrix(Sp_np)
            self.Sm = sp.csr_matrix(Sm_np)
    
    def get_site_operator(self, op_type: str, site: int):
        """指定サイトに演算子を作用させるスパース行列"""
        ops = [self.I] * self.n_sites
        
        if op_type == 'X': ops[site] = self.X
        elif op_type == 'Y': ops[site] = self.Y
        elif op_type == 'Z': ops[site] = self.Z
        elif op_type == '+': ops[site] = self.Sp
        elif op_type == '-': ops[site] = self.Sm
        elif op_type == 'I': pass
        else:
            raise ValueError(f"Unknown operator type: {op_type}")
        
        # クロネッカー積
        full_op = ops[0]
        for i in range(1, self.n_sites):
            full_op = self.sparse.kron(full_op, ops[i], format='csr')
        
        return full_op
    
    def build_heisenberg_hamiltonian(self, 
                                      bonds: List[Tuple[int, int]],
                                      J: float = 1.0,
                                      Jz: Optional[float] = None):
        """
        ハイゼンベルクハミルトニアン
        
        H = J Σ (Sx_i Sx_j + Sy_i Sy_j) + Jz Σ Sz_i Sz_j
        
        Returns:
            H_kinetic: XY項（運動エネルギー的）
            H_potential: ZZ項（ポテンシャル的）
        """
        if Jz is None:
            Jz = J
            
        if self.verbose:
            print(f"🔨 Building Heisenberg: {len(bonds)} bonds, J={J}, Jz={Jz}")
        
        H_kinetic = None
        H_potential = None
        
        for (i, j) in bonds:
            # XY項（運動エネルギー的：スピンのホッピング）
            Sx_i = self.get_site_operator('X', i)
            Sx_j = self.get_site_operator('X', j)
            Sy_i = self.get_site_operator('Y', i)
            Sy_j = self.get_site_operator('Y', j)
            
            term_xy = J * (Sx_i @ Sx_j + Sy_i @ Sy_j)
            
            if H_kinetic is None:
                H_kinetic = term_xy
            else:
                H_kinetic = H_kinetic + term_xy
            
            # ZZ項（ポテンシャル的：Ising相互作用）
            Sz_i = self.get_site_operator('Z', i)
            Sz_j = self.get_site_operator('Z', j)
            
            term_zz = Jz * Sz_i @ Sz_j
            
            if H_potential is None:
                H_potential = term_zz
            else:
                H_potential = H_potential + term_zz
        
        if self.verbose:
            H_total = H_kinetic + H_potential
            print(f"   ✅ Built: nnz={H_total.nnz:,}")
        
        return H_kinetic, H_potential
    
    def build_hubbard_hamiltonian(self,
                                   bonds: List[Tuple[int, int]],
                                   t: float = 1.0,
                                   U: float = 4.0):
        """
        Hubbardハミルトニアン（スピンレス簡易版）
        
        H = -t Σ (c†_i c_j + h.c.) + U Σ n_i n_j
        
        Returns:
            H_kinetic: ホッピング項
            H_potential: 相互作用項
        """
        if self.verbose:
            print(f"🔨 Building Hubbard: {len(bonds)} bonds, t={t}, U={U}")
        
        H_kinetic = None
        H_potential = None
        
        for (i, j) in bonds:
            # ホッピング（XY型に対応）
            Sp_i = self.get_site_operator('+', i)
            Sm_i = self.get_site_operator('-', i)
            Sp_j = self.get_site_operator('+', j)
            Sm_j = self.get_site_operator('-', j)
            
            term_hop = -t * (Sp_i @ Sm_j + Sm_i @ Sp_j)
            
            if H_kinetic is None:
                H_kinetic = term_hop
            else:
                H_kinetic = H_kinetic + term_hop
            
            # 密度-密度相互作用
            n_i = self.get_site_operator('Z', i) + 0.5 * self.get_site_operator('I', i)
            n_j = self.get_site_operator('Z', j) + 0.5 * self.get_site_operator('I', j)
            
            term_U = U * n_i @ n_j
            
            if H_potential is None:
                H_potential = term_U
            else:
                H_potential = H_potential + term_U
        
        if self.verbose:
            H_total = H_kinetic + H_potential
            print(f"   ✅ Built: nnz={H_total.nnz:,}")
        
        return H_kinetic, H_potential
    
    def build_chain_geometry(self, L: int, periodic: bool = True) -> SystemGeometry:
        """1D鎖のジオメトリ"""
        bonds = [(i, (i + 1) % L) for i in range(L)]
        if not periodic:
            bonds = bonds[:-1]
        return SystemGeometry(n_sites=L, bonds=bonds)
    
    def build_ladder_geometry(self, L: int, periodic: bool = True) -> SystemGeometry:
        """ラダー系のジオメトリ"""
        N = 2 * L
        
        # Leg bonds
        leg0 = [(i, (i + 1) % L) for i in range(L)]
        leg1 = [(L + i, L + (i + 1) % L) for i in range(L)]
        
        # Rung bonds
        rungs = [(i, L + i) for i in range(L)]
        
        if not periodic:
            leg0 = leg0[:-1]
            leg1 = leg1[:-1]
        
        bonds = leg0 + leg1 + rungs
        
        # プラケット（Λ計算用）
        plaquettes = []
        for i in range(L if periodic else L-1):
            bl, br = i, (i + 1) % L
            tl, tr = L + i, L + (i + 1) % L
            plaquettes.append((bl, br, tr, tl))
        
        return SystemGeometry(n_sites=N, bonds=bonds, plaquettes=plaquettes)
    
    def build_current_operator(self, bonds: List[Tuple[int, int]]):
        """
        スピン流演算子
        
        J = Σ 2(Sx_i Sy_j - Sy_i Sx_j)
        
        Λ³理論での進行ベクトル Λ_F に対応
        """
        if self.verbose:
            print("🔨 Building Current Operator...")
        
        J_op = None
        
        for (i, j) in bonds:
            Sx_i = self.get_site_operator('X', i)
            Sy_i = self.get_site_operator('Y', i)
            Sx_j = self.get_site_operator('X', j)
            Sy_j = self.get_site_operator('Y', j)
            
            term = 2.0 * (Sx_i @ Sy_j - Sy_i @ Sx_j)
            
            if J_op is None:
                J_op = term
            else:
                J_op = J_op + term
        
        return J_op
    
    def compute_lambda(self, psi, H_kinetic, H_potential, epsilon: float = 1e-10) -> float:
        """
        Λ = K / |V|_eff を計算
        
        H-CSP/Λ³理論の核心！
        
        Args:
            psi: 状態ベクトル
            H_kinetic: 運動エネルギー演算子
            H_potential: ポテンシャル演算子
            
        Returns:
            Lambda: 安定性指標
                Λ < 1: 安定
                Λ = 1: 臨界
                Λ > 1: 不安定
        """
        xp = self.xp
        
        # ⟨K⟩
        K_psi = H_kinetic @ psi
        K = float(xp.real(xp.vdot(psi, K_psi)))
        
        # ⟨V⟩
        V_psi = H_potential @ psi
        V = float(xp.real(xp.vdot(psi, V_psi)))
        
        # Λ = K / |V|
        Lambda = abs(K) / (abs(V) + epsilon)
        
        return Lambda
    
    def get_info(self) -> Dict[str, Any]:
        """エンジン情報"""
        return {
            'n_sites': self.n_sites,
            'dim': self.dim,
            'use_gpu': self.use_gpu,
            'backend': 'CuPy' if self.use_gpu else 'SciPy'
        }


# =============================================================================
# Molecular Hamiltonian Builder (for PySCF integration)
# =============================================================================

class MolecularHamiltonianBuilder:
    """
    分子ハミルトニアンビルダー
    
    PySCFとの連携用（将来拡張）
    """
    
    @staticmethod
    def from_integrals(h1e: np.ndarray, h2e: np.ndarray, n_orb: int):
        """
        1電子/2電子積分からハミルトニアンを構築
        
        H = Σ h_pq a†_p a_q + 1/2 Σ g_pqrs a†_p a†_r a_s a_q
        
        TODO: 本格的な実装
        """
        raise NotImplementedError("Full molecular Hamiltonian not yet implemented")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Sparse Engine Test")
    print("="*70)
    
    # 4サイト鎖
    engine = SparseHamiltonianEngine(n_sites=4, use_gpu=False)
    
    # ジオメトリ
    geom = engine.build_chain_geometry(L=4, periodic=True)
    print(f"\nGeometry: {geom.n_sites} sites, {len(geom.bonds)} bonds")
    print(f"Bonds: {geom.bonds}")
    
    # ハイゼンベルク
    H_K, H_V = engine.build_heisenberg_hamiltonian(geom.bonds, J=1.0, Jz=0.5)
    H = H_K + H_V
    
    # ランダム状態でΛ計算
    xp = engine.xp
    psi = xp.random.randn(engine.dim) + 1j * xp.random.randn(engine.dim)
    psi = psi / xp.linalg.norm(psi)
    
    Lambda = engine.compute_lambda(psi, H_K, H_V)
    print(f"\nRandom state Λ = {Lambda:.4f}")
    
    # 電流演算子
    J_op = engine.build_current_operator(geom.bonds)
    J_exp = float(xp.real(xp.vdot(psi, J_op @ psi)))
    print(f"Current ⟨J⟩ = {J_exp:.4f}")
    
    # ラダー
    print("\n" + "="*70)
    print("Ladder Test")
    print("="*70)
    
    engine2 = SparseHamiltonianEngine(n_sites=6, use_gpu=False)
    geom2 = engine2.build_ladder_geometry(L=3, periodic=True)
    print(f"Ladder: {geom2.n_sites} sites, {len(geom2.bonds)} bonds")
    print(f"Plaquettes: {geom2.plaquettes}")
    
    print("\n✅ Sparse Engine OK!")
