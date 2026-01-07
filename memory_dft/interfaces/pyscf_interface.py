"""
PySCF Interface for DSE
=======================

PySCF と DSE Memory Framework の統合インターフェース

【主要機能】
  1. PySCF DFT 計算 + DSE メモリ効果
  2. 反応経路の計算と比較
  3. 履歴依存エネルギーの抽出

【使用例】
  # 基本的な使い方
  calc = DSECalculator(basis='cc-pvdz', xc='B3LYP')
  
  # 単点計算
  E_dft, E_dse = calc.compute_single(atoms='H 0 0 0; H 0 0 0.74')
  
  # 経路計算
  path = create_h2_stretch_path()
  result = calc.compute_path(path)
  
  # 経路比較（DSE の検証！）
  comparison = calc.compare_paths(path1, path2)
  print(f'DFT差: {comparison.delta_dft:.6f}')   # ≈ 0
  print(f'DSE差: {comparison.delta_dse:.6f}')   # ≠ 0（履歴効果！）

【依存関係】
  - memory_kernel.py (MemoryKernel)
  - vorticity.py (VorticityCalculator) [オプション]
  - rdm.py (PySCFRDM) [オプション]

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

# PySCF
try:
    from pyscf import gto, dft, scf, cc
    HAS_PYSCF = True
except ImportError:
    gto = None
    dft = None
    scf = None
    cc = None
    HAS_PYSCF = False

# Memory Kernel（新しい統一版）
try:
    from memory_kernel import MemoryKernel, MemoryKernelConfig
    HAS_MEMORY_KERNEL = True
except ImportError:
    try:
        from memory_dft.core.memory_kernel import MemoryKernel, MemoryKernelConfig
        HAS_MEMORY_KERNEL = True
    except ImportError:
        MemoryKernel = None
        MemoryKernelConfig = None
        HAS_MEMORY_KERNEL = False

# Vorticity（γ_memory 計算用）
try:
    from vorticity import VorticityCalculator
    HAS_VORTICITY = True
except ImportError:
    try:
        from memory_dft.physics.vorticity import VorticityCalculator
        HAS_VORTICITY = True
    except ImportError:
        VorticityCalculator = None
        HAS_VORTICITY = False

# RDM（2-RDM 計算用）
try:
    from rdm import PySCFRDM
    HAS_RDM = True
except ImportError:
    try:
        from memory_dft.physics.rdm import PySCFRDM
        HAS_RDM = True
    except ImportError:
        PySCFRDM = None
        HAS_RDM = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeometryStep:
    """反応経路の1ステップ"""
    atoms: str              # PySCF 形式の原子文字列
    time: float             # 疑似時間
    label: str = ""         # ラベル（オプション）
    
    @property
    def r(self) -> Optional[float]:
        """結合長を抽出（2原子分子用）"""
        try:
            lines = self.atoms.strip().split(';')
            if len(lines) >= 2:
                coords1 = [float(x) for x in lines[0].split()[1:4]]
                coords2 = [float(x) for x in lines[1].split()[1:4]]
                return np.linalg.norm(np.array(coords2) - np.array(coords1))
        except:
            pass
        return None


@dataclass
class SinglePointResult:
    """単点計算の結果"""
    E_dft: float                       # DFT エネルギー
    E_dse: float                       # DSE エネルギー (DFT + memory)
    delta_memory: float                # メモリ寄与
    coords: np.ndarray                 # 座標
    gamma_memory: Optional[float] = None  # γ_memory（計算した場合）
    
    @property
    def memory_fraction(self) -> float:
        """メモリ効果の割合"""
        if abs(self.E_dft) > 1e-10:
            return self.delta_memory / abs(self.E_dft)
        return 0.0


@dataclass
class PathResult:
    """経路計算の結果"""
    E_dft: List[float]          # DFT エネルギーリスト
    E_dse: List[float]          # DSE エネルギーリスト
    delta_memory: List[float]   # メモリ寄与リスト
    coords_list: List[np.ndarray]  # 座標リスト
    path_label: str             # 経路ラベル
    
    @property
    def E_dft_final(self) -> float:
        return self.E_dft[-1]
    
    @property
    def E_dse_final(self) -> float:
        return self.E_dse[-1]
    
    @property
    def n_steps(self) -> int:
        return len(self.E_dft)
    
    @property
    def total_memory_effect(self) -> float:
        """累積メモリ効果"""
        return sum(self.delta_memory)
    
    @property
    def max_memory_effect(self) -> float:
        """最大メモリ効果"""
        return max(self.delta_memory) if self.delta_memory else 0.0


@dataclass
class ComparisonResult:
    """経路比較の結果"""
    path1: PathResult
    path2: PathResult
    
    @property
    def delta_dft(self) -> float:
        """DFT エネルギー差（同じ終状態なら ≈ 0）"""
        return abs(self.path1.E_dft_final - self.path2.E_dft_final)
    
    @property
    def delta_dse(self) -> float:
        """DSE エネルギー差（履歴効果！）"""
        return abs(self.path1.E_dse_final - self.path2.E_dse_final)
    
    @property
    def history_dependence(self) -> float:
        """履歴依存性 = DSE差 - DFT差"""
        return self.delta_dse - self.delta_dft
    
    def summary(self) -> str:
        """サマリー文字列"""
        return f"""
Path Comparison Result
======================
Path 1: {self.path1.path_label} ({self.path1.n_steps} steps)
Path 2: {self.path2.path_label} ({self.path2.n_steps} steps)

Final Energies:
  DFT Path 1: {self.path1.E_dft_final:.6f} Ha
  DFT Path 2: {self.path2.E_dft_final:.6f} Ha
  DFT Δ: {self.delta_dft:.6f} Ha (should be ~0)

  DSE Path 1: {self.path1.E_dse_final:.6f} Ha
  DSE Path 2: {self.path2.E_dse_final:.6f} Ha
  DSE Δ: {self.delta_dse:.6f} Ha (history effect!)

History Dependence: {self.history_dependence:.6f} Ha
"""


# =============================================================================
# DSE Calculator
# =============================================================================

class DSECalculator:
    """
    DSE (Direct Schrödinger Evolution) 計算器
    
    PySCF DFT 計算と履歴依存メモリ効果を統合
    
    Features:
    - 標準 DFT 計算
    - DSE メモリ効果の計算
    - 経路計算と比較
    - γ_memory の自動抽出（オプション）
    """
    
    def __init__(self,
                 basis: str = 'cc-pvdz',
                 xc: str = 'B3LYP',
                 gamma_memory: float = 1.0,
                 memory_strength: float = 0.1,
                 compute_gamma: bool = False,
                 verbose: int = 0):
        """
        Args:
            basis: 基底関数
            xc: 交換相関汎関数
            gamma_memory: メモリ指数（compute_gamma=False の場合）
            memory_strength: メモリ効果の強度係数
            compute_gamma: γ_memory を Vorticity から計算するか
            verbose: PySCF の詳細度
        """
        if not HAS_PYSCF:
            raise ImportError("PySCF required. Install: pip install pyscf")
        
        if not HAS_MEMORY_KERNEL:
            raise ImportError("MemoryKernel required. Check memory_kernel.py")
        
        self.basis = basis
        self.xc = xc
        self.gamma_memory = gamma_memory
        self.memory_strength = memory_strength
        self.compute_gamma = compute_gamma
        self.verbose = verbose
        
        # MemoryKernel インスタンス
        self.kernel = MemoryKernel(gamma_memory=gamma_memory, use_gpu=False)
        
        # 内部状態
        self._last_mf = None
        self._history_coords: List[np.ndarray] = []
    
    def reset(self):
        """履歴をリセット"""
        self.kernel = MemoryKernel(gamma_memory=self.gamma_memory, use_gpu=False)
        self._history_coords.clear()
    
    def _run_dft(self, atoms: str, charge: int = 0, spin: int = 0) -> Tuple[float, np.ndarray, Any]:
        """DFT 計算を実行"""
        mol = gto.M(
            atom=atoms,
            basis=self.basis,
            charge=charge,
            spin=spin,
            verbose=self.verbose
        )
        
        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        
        mf.xc = self.xc
        E = mf.kernel()
        coords = mol.atom_coords()
        
        self._last_mf = mf
        
        return E, coords, mf
    
    def _compute_gamma_from_vorticity(self, mf) -> float:
        """Vorticity から γ_memory を計算"""
        if not HAS_VORTICITY or not HAS_RDM:
            return self.gamma_memory
        
        try:
            # CCSD で 2-RDM を計算
            mycc = cc.CCSD(mf)
            mycc.kernel()
            E_corr = mycc.e_corr
            
            # 2-RDM 取得
            rdm2 = mycc.make_rdm2()
            if isinstance(rdm2, tuple):
                rdm2_aa, rdm2_ab, rdm2_bb = rdm2
                rdm2 = rdm2_aa + rdm2_ab + rdm2_ab.transpose(2,3,0,1) + rdm2_bb
            
            n_orb = mf.mol.nao
            
            # Vorticity 計算
            calc = VorticityCalculator(use_gpu=False)
            decomp = calc.compute_gamma_decomposition(rdm2, n_orb, E_corr)
            
            # γ_memory 推定
            ratio = decomp['ratio']
            if ratio > 1:
                gamma = np.log(ratio) * 2
            else:
                gamma = (ratio - 0.5) * 2
            
            return float(np.clip(gamma, 0.1, 3.0))
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: γ_memory calculation failed: {e}")
            return self.gamma_memory
    
    def _compute_memory_contribution(self, t: float, E: float, coords: np.ndarray) -> float:
        """メモリ寄与を計算"""
        if len(self._history_coords) == 0:
            return 0.0
        
        # 幾何学的な寄与
        geom_factor = 0.0
        for past_coords in self._history_coords:
            if past_coords.shape == coords.shape:
                delta_r = np.linalg.norm(coords - past_coords)
                geom_factor += np.exp(-delta_r / 0.5)
        
        geom_factor /= len(self._history_coords)
        
        # MemoryKernel からの寄与
        # 状態ベクトルとして座標のフラット化を使用
        state = coords.flatten().astype(np.complex128)
        state = state / (np.linalg.norm(state) + 1e-10)
        
        kernel_contrib = self.kernel.compute_memory_contribution(t, state)
        
        # 合成
        return self.memory_strength * (geom_factor * 0.5 + kernel_contrib * 0.5)
    
    def compute_single(self,
                       atoms: str,
                       time: float = 0.0,
                       charge: int = 0,
                       spin: int = 0) -> SinglePointResult:
        """
        単点計算（DFT + DSE）
        
        Args:
            atoms: 原子文字列
            time: 疑似時間
            charge: 電荷
            spin: スピン多重度
            
        Returns:
            SinglePointResult
        """
        # DFT 計算
        E_dft, coords, mf = self._run_dft(atoms, charge, spin)
        
        # γ_memory 計算（オプション）
        gamma = None
        if self.compute_gamma:
            gamma = self._compute_gamma_from_vorticity(mf)
            # カーネル更新
            self.kernel = MemoryKernel(gamma_memory=gamma, use_gpu=False)
            # 履歴を復元
            for i, past_coords in enumerate(self._history_coords):
                state = past_coords.flatten().astype(np.complex128)
                state = state / (np.linalg.norm(state) + 1e-10)
                self.kernel.add_state(t=float(i), r=0.0, state=state)
        
        # メモリ寄与
        delta_mem = self._compute_memory_contribution(time, E_dft, coords)
        
        # 履歴に追加
        state = coords.flatten().astype(np.complex128)
        state = state / (np.linalg.norm(state) + 1e-10)
        self.kernel.add_state(t=time, r=0.0, state=state, energy=E_dft)
        self._history_coords.append(coords.copy())
        
        return SinglePointResult(
            E_dft=E_dft,
            E_dse=E_dft + delta_mem,
            delta_memory=delta_mem,
            coords=coords,
            gamma_memory=gamma
        )
    
    def compute_path(self,
                     path: List[GeometryStep],
                     charge: int = 0,
                     spin: int = 0,
                     label: str = "") -> PathResult:
        """
        経路計算
        
        Args:
            path: GeometryStep のリスト
            charge: 電荷
            spin: スピン多重度
            label: 経路ラベル
            
        Returns:
            PathResult
        """
        self.reset()
        
        E_dft_list = []
        E_dse_list = []
        delta_mem_list = []
        coords_list = []
        
        for step in path:
            result = self.compute_single(step.atoms, step.time, charge, spin)
            
            E_dft_list.append(result.E_dft)
            E_dse_list.append(result.E_dse)
            delta_mem_list.append(result.delta_memory)
            coords_list.append(result.coords)
        
        return PathResult(
            E_dft=E_dft_list,
            E_dse=E_dse_list,
            delta_memory=delta_mem_list,
            coords_list=coords_list,
            path_label=label or "Path"
        )
    
    def compare_paths(self,
                      path1: List[GeometryStep],
                      path2: List[GeometryStep],
                      charge: int = 0,
                      spin: int = 0,
                      label1: str = "Path 1",
                      label2: str = "Path 2") -> ComparisonResult:
        """
        2つの経路を比較
        
        同じ終状態に至る異なる経路で:
        - DFT: 同じエネルギー（経路非依存）
        - DSE: 異なるエネルギー（履歴依存！）
        """
        result1 = self.compute_path(path1, charge, spin, label1)
        result2 = self.compute_path(path2, charge, spin, label2)
        
        return ComparisonResult(path1=result1, path2=result2)


# =============================================================================
# Path Generators
# =============================================================================

def create_h2_stretch_path(r_start: float = 0.74,
                            r_end: float = 1.5,
                            r_return: float = 0.74,
                            n_steps: int = 10) -> List[GeometryStep]:
    """H2 伸張→復帰 経路"""
    path = []
    
    # 伸張
    for i, r in enumerate(np.linspace(r_start, r_end, n_steps)):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(i),
            label=f"stretch_{i}"
        ))
    
    # 復帰
    for i, r in enumerate(np.linspace(r_end, r_return, n_steps)[1:]):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(n_steps + i),
            label=f"return_{i}"
        ))
    
    return path


def create_h2_compress_path(r_start: float = 0.74,
                             r_end: float = 0.5,
                             r_return: float = 0.74,
                             n_steps: int = 10) -> List[GeometryStep]:
    """H2 圧縮→復帰 経路"""
    path = []
    
    # 圧縮
    for i, r in enumerate(np.linspace(r_start, r_end, n_steps)):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(i),
            label=f"compress_{i}"
        ))
    
    # 復帰
    for i, r in enumerate(np.linspace(r_end, r_return, n_steps)[1:]):
        path.append(GeometryStep(
            atoms=f"H 0 0 0; H 0 0 {r}",
            time=float(n_steps + i),
            label=f"return_{i}"
        ))
    
    return path


def create_cyclic_path(r_eq: float = 0.74,
                        r_max: float = 1.2,
                        r_min: float = 0.5,
                        n_cycles: int = 3,
                        n_steps_per_phase: int = 5) -> List[GeometryStep]:
    """
    サイクリック経路（疲労シミュレーション用）
    
    eq → max → eq → min → eq → ... (n_cycles 回)
    """
    path = []
    t = 0
    
    for cycle in range(n_cycles):
        # eq → max
        for i, r in enumerate(np.linspace(r_eq, r_max, n_steps_per_phase)):
            path.append(GeometryStep(
                atoms=f"H 0 0 0; H 0 0 {r}",
                time=float(t),
                label=f"cycle{cycle}_stretch_{i}"
            ))
            t += 1
        
        # max → eq
        for i, r in enumerate(np.linspace(r_max, r_eq, n_steps_per_phase)[1:]):
            path.append(GeometryStep(
                atoms=f"H 0 0 0; H 0 0 {r}",
                time=float(t),
                label=f"cycle{cycle}_return1_{i}"
            ))
            t += 1
        
        # eq → min
        for i, r in enumerate(np.linspace(r_eq, r_min, n_steps_per_phase)[1:]):
            path.append(GeometryStep(
                atoms=f"H 0 0 0; H 0 0 {r}",
                time=float(t),
                label=f"cycle{cycle}_compress_{i}"
            ))
            t += 1
        
        # min → eq
        for i, r in enumerate(np.linspace(r_min, r_eq, n_steps_per_phase)[1:]):
            path.append(GeometryStep(
                atoms=f"H 0 0 0; H 0 0 {r}",
                time=float(t),
                label=f"cycle{cycle}_return2_{i}"
            ))
            t += 1
    
    return path


# =============================================================================
# Demo Functions
# =============================================================================

def demo_single_point():
    """単点計算のデモ"""
    print("=" * 60)
    print("Single Point Demo")
    print("=" * 60)
    
    calc = DSECalculator(basis='sto-3g', xc='LDA', verbose=0)
    
    result = calc.compute_single(atoms='H 0 0 0; H 0 0 0.74')
    
    print(f"\nH2 at r = 0.74 Å:")
    print(f"  E_DFT = {result.E_dft:.6f} Ha")
    print(f"  E_DSE = {result.E_dse:.6f} Ha")
    print(f"  ΔE_mem = {result.delta_memory:.6f} Ha")
    
    return result


def demo_path_comparison():
    """経路比較のデモ（DSE の検証！）"""
    print("=" * 60)
    print("Path Comparison Demo")
    print("=" * 60)
    
    calc = DSECalculator(basis='sto-3g', xc='LDA', memory_strength=0.05)
    
    path_stretch = create_h2_stretch_path(n_steps=5)
    path_compress = create_h2_compress_path(n_steps=5)
    
    print("\nComputing paths...")
    comparison = calc.compare_paths(
        path_stretch, path_compress,
        label1="Stretch→Return",
        label2="Compress→Return"
    )
    
    print(comparison.summary())
    
    return comparison


def demo_cyclic_fatigue():
    """サイクリック負荷（疲労）のデモ"""
    print("=" * 60)
    print("Cyclic Fatigue Demo")
    print("=" * 60)
    
    calc = DSECalculator(basis='sto-3g', xc='LDA', memory_strength=0.1)
    
    path = create_cyclic_path(n_cycles=2, n_steps_per_phase=3)
    result = calc.compute_path(path, label="Cyclic")
    
    print(f"\nCyclic path: {result.n_steps} steps")
    print(f"Total memory effect: {result.total_memory_effect:.6f} Ha")
    print(f"Max memory effect: {result.max_memory_effect:.6f} Ha")
    
    # サイクルごとのメモリ効果
    print("\nMemory per step:")
    for i, (E_dft, E_dse, delta) in enumerate(zip(result.E_dft, result.E_dse, result.delta_memory)):
        if i % 5 == 0:
            print(f"  Step {i:2d}: E_DFT={E_dft:.4f}, ΔE_mem={delta:.6f}")
    
    return result


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PySCF Interface Test")
    print("=" * 70)
    
    print(f"\nPySCF available: {HAS_PYSCF}")
    print(f"MemoryKernel available: {HAS_MEMORY_KERNEL}")
    print(f"VorticityCalculator available: {HAS_VORTICITY}")
    print(f"PySCFRDM available: {HAS_RDM}")
    
    if not HAS_PYSCF:
        print("\n⚠️ PySCF not installed. Install with: pip install pyscf")
        print("Running mock test...")
        
        # モックテスト
        print("\n[Mock Test] GeometryStep")
        step = GeometryStep(atoms="H 0 0 0; H 0 0 0.74", time=0.0)
        print(f"  r = {step.r:.4f} Å")
        
        print("\n[Mock Test] Path generators")
        path = create_h2_stretch_path(n_steps=3)
        print(f"  Stretch path: {len(path)} steps")
        for s in path:
            print(f"    t={s.time}, r={s.r:.4f}")
    else:
        # 実際のテスト
        try:
            demo_single_point()
            print()
            demo_path_comparison()
            print()
            demo_cyclic_fatigue()
        except Exception as e:
            print(f"\n⚠️ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ PySCF Interface Test Complete!")
    print("=" * 70)
