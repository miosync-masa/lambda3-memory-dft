"""
Two-Particle Reduced Density Matrix (2-RDM) for Memory-DFT
==========================================================

各種 Hamiltonian から 2-RDM を計算する統一インターフェース

【対応する系】
  - Hubbard 模型（フェルミオン）
  - Heisenberg 模型（スピン系）
  - PySCF 分子計算（CCSD/FCI）
  - 周期系（TODO）

【理論的背景】
  2-RDM: ρ^(2)_{ijkl} = ⟨ψ|c†_i c†_j c_l c_k|ψ⟩
  
  全ての2体相関を記述し、Vorticity 計算の基礎となる

【統合フロー】
  Hamiltonian → RDMCalculator → rdm2 → VorticityCalculator → γ_memory → MemoryKernel

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, List, Tuple, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

if TYPE_CHECKING:
    pass


# =============================================================================
# System Types
# =============================================================================

class SystemType(Enum):
    """系の種類"""
    HUBBARD = "hubbard"
    HEISENBERG = "heisenberg"
    TJ = "t-j"
    PYSCF = "pyscf"
    PERIODIC = "periodic"
    CUSTOM = "custom"


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class RDM2Result:
    """2-RDM 計算結果"""
    rdm2: np.ndarray
    n_orb: int
    system_type: SystemType
    method: str
    trace: Optional[float] = None
    n_particles: Optional[int] = None
    distance_matrix: Optional[np.ndarray] = None  # 空間距離行列
    
    def __post_init__(self):
        if self.trace is None:
            self.trace = float(np.real(np.einsum('iijj->', self.rdm2)))
    
    def validate(self, tol: float = 1e-6) -> bool:
        """2-RDM の性質を検証"""
        # Hermiticity: ρ_{ijkl} = ρ_{klij}^*
        rdm2_swap = self.rdm2.transpose(2, 3, 0, 1).conj()
        hermitian_err = np.max(np.abs(self.rdm2 - rdm2_swap))
        return hermitian_err < tol


# =============================================================================
# Abstract Base Class
# =============================================================================

class RDMCalculator(ABC):
    """
    RDM 計算の抽象基底クラス
    
    各 Hamiltonian タイプごとにサブクラスを実装
    """
    
    @abstractmethod
    def compute_rdm2(self, psi: np.ndarray, **kwargs) -> RDM2Result:
        """2-RDM を計算"""
        pass
    
    @abstractmethod
    def compute_distance_matrix(self, **kwargs) -> np.ndarray:
        """空間距離行列を計算"""
        pass
    
    @property
    @abstractmethod
    def system_type(self) -> SystemType:
        """系の種類"""
        pass


# =============================================================================
# Hubbard Model
# =============================================================================

class HubbardRDM(RDMCalculator):
    """
    Hubbard 模型用 RDM 計算
    
    H = -t Σ (c†_iσ c_jσ + h.c.) + U Σ n_i↑ n_i↓
    
    特徴：
    - サイトインデックス = 空間位置（1D/2D/3D）
    - フェルミオン演算子から構築
    """
    
    def __init__(self, n_sites: int, lattice: str = '1d', 
                 lattice_constant: float = 1.0):
        """
        Args:
            n_sites: サイト数
            lattice: 格子タイプ ('1d', '2d_square', '2d_triangular', ...)
            lattice_constant: 格子定数
        """
        self.n_sites = n_sites
        self.lattice = lattice
        self.lattice_constant = lattice_constant
        self._number_ops = None
    
    @property
    def system_type(self) -> SystemType:
        return SystemType.HUBBARD
    
    def compute_rdm2(self, psi: np.ndarray, 
                     method: str = 'diagonal') -> RDM2Result:
        """
        波動関数から 2-RDM を計算
        
        Args:
            psi: 正規化された波動関数
            method: 'diagonal' (高速、密度-密度のみ) or 'full' (完全)
        """
        if method == 'diagonal':
            rdm2 = self._compute_diagonal(psi)
        else:
            rdm2 = self._compute_full(psi)
        
        return RDM2Result(
            rdm2=rdm2,
            n_orb=self.n_sites,
            system_type=self.system_type,
            method=method,
            distance_matrix=self.compute_distance_matrix()
        )
    
    def _compute_diagonal(self, psi: np.ndarray) -> np.ndarray:
        """
        対角近似: ⟨n_i n_j⟩ のみ計算
        """
        n_sites = self.n_sites
        
        # 数演算子を構築
        if self._number_ops is None:
            self._number_ops = self._build_number_operators()
        
        rdm2 = np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=np.complex128)
        
        for i in range(n_sites):
            for j in range(n_sites):
                n_i = self._number_ops[i]
                n_j = self._number_ops[j]
                
                # ⟨n_i n_j⟩
                n_i_n_j = n_i @ n_j
                val = np.vdot(psi, n_i_n_j @ psi)
                
                rdm2[i, i, j, j] = val
                rdm2[i, j, i, j] = val * 0.5
                rdm2[i, j, j, i] = -val * 0.5  # フェルミオン反対称性
        
        return rdm2
    
    def _compute_full(self, psi: np.ndarray) -> np.ndarray:
        """完全 2-RDM（未実装 → 対角にフォールバック）"""
        import warnings
        warnings.warn("Full 2-RDM not implemented, using diagonal", UserWarning)
        return self._compute_diagonal(psi)
    
    def _build_number_operators(self) -> List[sp.csr_matrix]:
        """数演算子 n_i = |1⟩⟨1| を構築"""
        I = sp.eye(2, format='csr', dtype=np.complex128)
        n_single = sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)
        
        number_ops = []
        for site in range(self.n_sites):
            ops = [I] * self.n_sites
            ops[site] = n_single
            
            result = ops[0]
            for i in range(1, self.n_sites):
                result = sp.kron(result, ops[i], format='csr')
            
            number_ops.append(result)
        
        return number_ops
    
    def compute_distance_matrix(self) -> np.ndarray:
        """格子上の距離行列"""
        n = self.n_sites
        a = self.lattice_constant
        
        if self.lattice == '1d':
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist[i, j] = abs(i - j) * a
            return dist
        
        elif self.lattice == '2d_square':
            # 正方格子（n = L×L を仮定）
            L = int(np.sqrt(n))
            dist = np.zeros((n, n))
            for i in range(n):
                xi, yi = i % L, i // L
                for j in range(n):
                    xj, yj = j % L, j // L
                    dist[i, j] = np.sqrt((xi-xj)**2 + (yi-yj)**2) * a
            return dist
        
        else:
            # デフォルト: インデックス差
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist[i, j] = abs(i - j) * a
            return dist


# =============================================================================
# Heisenberg Model
# =============================================================================

class HeisenbergRDM(RDMCalculator):
    """
    Heisenberg スピン模型用 RDM 計算
    
    H = J Σ S_i · S_j = J Σ (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)
    
    特徴：
    - スピン演算子から構築
    - 2-RDM はスピン相関 ⟨S_i · S_j⟩ に対応
    """
    
    def __init__(self, n_sites: int, lattice: str = '1d',
                 lattice_constant: float = 1.0):
        self.n_sites = n_sites
        self.lattice = lattice
        self.lattice_constant = lattice_constant
        self._spin_ops = None
    
    @property
    def system_type(self) -> SystemType:
        return SystemType.HEISENBERG
    
    def compute_rdm2(self, psi: np.ndarray,
                     method: str = 'spin_correlation') -> RDM2Result:
        """
        スピン相関から 2-RDM を構築
        
        ρ^(2)_{iijj} ∝ ⟨S_i · S_j⟩
        """
        rdm2 = self._compute_spin_correlation(psi)
        
        return RDM2Result(
            rdm2=rdm2,
            n_orb=self.n_sites,
            system_type=self.system_type,
            method=method,
            distance_matrix=self.compute_distance_matrix()
        )
    
    def _compute_spin_correlation(self, psi: np.ndarray) -> np.ndarray:
        """スピン-スピン相関 ⟨S_i · S_j⟩"""
        n = self.n_sites
        
        if self._spin_ops is None:
            self._spin_ops = self._build_spin_operators()
        
        Sx, Sy, Sz = self._spin_ops
        
        rdm2 = np.zeros((n, n, n, n), dtype=np.complex128)
        
        for i in range(n):
            for j in range(n):
                # ⟨S_i · S_j⟩ = ⟨Sx_i Sx_j⟩ + ⟨Sy_i Sy_j⟩ + ⟨Sz_i Sz_j⟩
                SiSj = Sx[i] @ Sx[j] + Sy[i] @ Sy[j] + Sz[i] @ Sz[j]
                val = np.vdot(psi, SiSj @ psi)
                
                rdm2[i, i, j, j] = val
        
        return rdm2
    
    def _build_spin_operators(self) -> Tuple[List, List, List]:
        """スピン-1/2 演算子を構築"""
        n = self.n_sites
        I = sp.eye(2, format='csr', dtype=np.complex128)
        
        sx_single = sp.csr_matrix([[0, 0.5], [0.5, 0]], dtype=np.complex128)
        sy_single = sp.csr_matrix([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)
        sz_single = sp.csr_matrix([[0.5, 0], [0, -0.5]], dtype=np.complex128)
        
        Sx, Sy, Sz = [], [], []
        
        for site in range(n):
            ops_x = [I] * n
            ops_y = [I] * n
            ops_z = [I] * n
            
            ops_x[site] = sx_single
            ops_y[site] = sy_single
            ops_z[site] = sz_single
            
            # テンソル積
            result_x = ops_x[0]
            result_y = ops_y[0]
            result_z = ops_z[0]
            
            for i in range(1, n):
                result_x = sp.kron(result_x, ops_x[i], format='csr')
                result_y = sp.kron(result_y, ops_y[i], format='csr')
                result_z = sp.kron(result_z, ops_z[i], format='csr')
            
            Sx.append(result_x)
            Sy.append(result_y)
            Sz.append(result_z)
        
        return Sx, Sy, Sz
    
    def compute_distance_matrix(self) -> np.ndarray:
        """Hubbard と同じ"""
        n = self.n_sites
        a = self.lattice_constant
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = abs(i - j) * a
        return dist


# =============================================================================
# PySCF Integration
# =============================================================================

class PySCFRDM(RDMCalculator):
    """
    PySCF 分子計算用 RDM 取得
    
    CCSD または FCI から 2-RDM を取得
    
    特徴：
    - 軌道インデックス ≠ 空間位置
    - 空間距離行列が必要
    """
    
    def __init__(self, mol=None):
        """
        Args:
            mol: PySCF Mole オブジェクト
        """
        self.mol = mol
        self._mf = None
    
    @property
    def system_type(self) -> SystemType:
        return SystemType.PYSCF
    
    def compute_rdm2(self, mf, method: str = 'ccsd') -> RDM2Result:
        """
        PySCF 計算から 2-RDM を取得
        
        Args:
            mf: 収束済み SCF オブジェクト
            method: 'ccsd' or 'fci'
        """
        self.mol = mf.mol
        self._mf = mf
        n_orb = self.mol.nao
        
        if method == 'ccsd':
            rdm2, E_corr = self._compute_ccsd(mf)
        elif method == 'fci':
            rdm2, E_corr = self._compute_fci(mf)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return RDM2Result(
            rdm2=rdm2,
            n_orb=n_orb,
            system_type=self.system_type,
            method=method,
            distance_matrix=self.compute_distance_matrix()
        )
    
    def _compute_ccsd(self, mf) -> Tuple[np.ndarray, float]:
        """CCSD から 2-RDM を取得"""
        from pyscf import cc
        
        mycc = cc.CCSD(mf)
        mycc.kernel()
        E_corr = mycc.e_corr
        
        rdm2 = mycc.make_rdm2()
        
        # UCCSD の場合は成分を結合
        if isinstance(rdm2, tuple):
            rdm2_aa, rdm2_ab, rdm2_bb = rdm2
            rdm2 = rdm2_aa + rdm2_ab + rdm2_ab.transpose(2,3,0,1) + rdm2_bb
        
        return rdm2, E_corr
    
    def _compute_fci(self, mf) -> Tuple[np.ndarray, float]:
        """FCI から 2-RDM を取得"""
        from pyscf import fci
        
        mol = mf.mol
        n_orb = mol.nao
        
        cisolver = fci.FCI(mf)
        E_fci, fcivec = cisolver.kernel()
        E_corr = E_fci - mf.e_tot
        
        n_elec = mol.nelectron
        nelec = (n_elec // 2, n_elec // 2)
        rdm1, rdm2 = cisolver.make_rdm12(fcivec, n_orb, nelec)
        
        return rdm2, E_corr
    
    def compute_distance_matrix(self) -> np.ndarray:
        """
        軌道間の空間距離行列を計算
        
        各軌道の「位置」を所属原子の座標で近似
        """
        if self.mol is None:
            raise ValueError("Mole object not set")
        
        # 原子座標
        coords = self.mol.atom_coords()  # Bohr
        coords_ang = coords * 0.529177   # Å
        
        # 軌道→原子マッピング
        ao_labels = self.mol.ao_labels()
        n_orb = len(ao_labels)
        
        orbital_atoms = []
        for label in ao_labels:
            atom_idx = int(label.split()[0])
            orbital_atoms.append(atom_idx)
        
        # 距離行列
        dist = np.zeros((n_orb, n_orb))
        for i in range(n_orb):
            for j in range(n_orb):
                ai, aj = orbital_atoms[i], orbital_atoms[j]
                if ai == aj:
                    dist[i, j] = 0.0
                else:
                    diff = coords_ang[ai] - coords_ang[aj]
                    dist[i, j] = np.linalg.norm(diff)
        
        return dist


# =============================================================================
# Factory Function
# =============================================================================

def get_rdm_calculator(system_type: Union[str, SystemType], 
                       **kwargs) -> RDMCalculator:
    """
    系の種類に応じた RDM 計算器を取得
    
    Args:
        system_type: 'hubbard', 'heisenberg', 'pyscf', ...
        **kwargs: 各計算器への引数
    
    Returns:
        RDMCalculator インスタンス
    """
    if isinstance(system_type, str):
        system_type = SystemType(system_type.lower())
    
    if system_type == SystemType.HUBBARD:
        return HubbardRDM(**kwargs)
    elif system_type == SystemType.HEISENBERG:
        return HeisenbergRDM(**kwargs)
    elif system_type == SystemType.PYSCF:
        return PySCFRDM(**kwargs)
    else:
        raise ValueError(f"Unsupported system type: {system_type}")


def compute_rdm2(psi_or_mf, 
                 system_type: Union[str, SystemType],
                 n_sites: Optional[int] = None,
                 **kwargs) -> RDM2Result:
    """
    便利関数: 2-RDM を計算
    
    Args:
        psi_or_mf: 波動関数 or PySCF SCF オブジェクト
        system_type: 系の種類
        n_sites: サイト数（格子系の場合）
        **kwargs: 追加引数
    
    Returns:
        RDM2Result
    
    Example:
        # Hubbard
        result = compute_rdm2(psi, 'hubbard', n_sites=6)
        
        # PySCF
        result = compute_rdm2(mf, 'pyscf', method='ccsd')
    """
    if isinstance(system_type, str):
        system_type = SystemType(system_type.lower())
    
    if system_type == SystemType.PYSCF:
        calc = PySCFRDM()
        method = kwargs.get('method', 'ccsd')
        return calc.compute_rdm2(psi_or_mf, method=method)
    else:
        if n_sites is None:
            raise ValueError("n_sites required for lattice models")
        
        calc = get_rdm_calculator(system_type, n_sites=n_sites, **kwargs)
        return calc.compute_rdm2(psi_or_mf, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RDM Module Test")
    print("=" * 70)
    
    # Hubbard テスト
    print("\n[1] Hubbard Model")
    n_sites = 4
    dim = 2 ** n_sites
    
    np.random.seed(42)
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi = psi / np.linalg.norm(psi)
    
    calc_hub = HubbardRDM(n_sites=n_sites, lattice='1d')
    result_hub = calc_hub.compute_rdm2(psi)
    
    print(f"  Shape: {result_hub.rdm2.shape}")
    print(f"  Trace: {result_hub.trace:.4f}")
    print(f"  System: {result_hub.system_type.value}")
    print(f"  Distance matrix:\n{result_hub.distance_matrix}")
    
    # Heisenberg テスト
    print("\n[2] Heisenberg Model")
    calc_heis = HeisenbergRDM(n_sites=n_sites, lattice='1d')
    result_heis = calc_heis.compute_rdm2(psi)
    
    print(f"  Shape: {result_heis.rdm2.shape}")
    print(f"  Trace: {result_heis.trace:.4f}")
    print(f"  System: {result_heis.system_type.value}")
    
    # Factory function テスト
    print("\n[3] Factory Function")
    result = compute_rdm2(psi, 'hubbard', n_sites=n_sites)
    print(f"  Shape: {result.rdm2.shape}")
    print(f"  Valid: {result.validate()}")
    
    # 距離行列テスト（2D）
    print("\n[4] 2D Square Lattice")
    calc_2d = HubbardRDM(n_sites=9, lattice='2d_square', lattice_constant=1.0)
    dist_2d = calc_2d.compute_distance_matrix()
    print(f"  Shape: {dist_2d.shape}")
    print(f"  Max distance: {dist_2d.max():.2f}")
    print(f"  Corner to corner: {dist_2d[0, 8]:.2f} (expected: {np.sqrt(8):.2f})")
    
    print("\n" + "=" * 70)
    print("✅ RDM Module Test Complete!")
    print("=" * 70)
