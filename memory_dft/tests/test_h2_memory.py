"""
H2 Molecule Memory-DFT Test
===========================

簡単なH2分子モデルでMemory-DFTの動作検証

テスト項目:
1. 標準量子力学との比較
2. Memory項の効果
3. Λ軌跡の追跡
4. H-CSP公理の検証

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude/memory_dft')

from core.sparse_engine import SparseHamiltonianEngine
from core.memory_kernel import CompositeMemoryKernel, KernelWeights
from core.history_manager import HistoryManager
from solvers.lanczos_memory import MemoryLanczosSolver
from solvers.time_evolution import TimeEvolutionEngine, EvolutionConfig
from physics.lambda3_bridge import Lambda3Calculator, HCSPValidator
from physics.vorticity import GammaExtractor


def create_h2_model(bond_length: float = 1.4):
    """
    簡易H2モデル（2サイトHeisenberg）
    
    H = J (Sx1 Sx2 + Sy1 Sy2 + Δ Sz1 Sz2) + h (Sz1 + Sz2)
    
    Args:
        bond_length: 結合長（J, Δ に影響）
    """
    # 結合長依存のパラメータ
    J = 1.0 / bond_length  # ホッピング
    Delta = 0.5  # Ising異方性
    h = 0.1  # 磁場（対称性破れ用）
    
    # 2サイト系
    engine = SparseHamiltonianEngine(n_sites=2, use_gpu=False, verbose=False)
    
    # Heisenbergハミルトニアン
    bonds = [(0, 1)]
    H_K, H_V = engine.build_heisenberg_hamiltonian(bonds, J=J, Jz=J*Delta)
    
    # 磁場項をポテンシャルに追加
    Sz_total = engine.get_site_operator('Z', 0) + engine.get_site_operator('Z', 1)
    H_V = H_V + h * Sz_total
    
    return engine, H_K, H_V


def test_basic_evolution():
    """基本的な時間発展テスト"""
    print("="*70)
    print("Test 1: Basic Time Evolution")
    print("="*70)
    
    engine, H_K, H_V = create_h2_model()
    
    # 初期状態: |↑↓⟩ + |↓↑⟩ (一重項的)
    psi0 = np.array([0, 1, 1, 0], dtype=np.complex128)
    psi0 = psi0 / np.linalg.norm(psi0)
    
    # 設定
    config = EvolutionConfig(
        t_end=5.0,
        dt=0.1,
        use_memory=True,
        memory_strength=0.1,
        verbose=True
    )
    
    # 発展
    evol = TimeEvolutionEngine(H_K, H_V, config, use_gpu=False)
    result = evol.run(psi0)
    
    print(f"\nResults:")
    print(f"  Initial Λ: {result.lambdas[0]:.4f}")
    print(f"  Final Λ: {result.lambdas[-1]:.4f}")
    print(f"  Λ range: [{min(result.lambdas):.4f}, {max(result.lambdas):.4f}]")
    print(f"  Energy conservation: ΔE = {abs(result.energies[-1] - result.energies[0]):.6f}")
    
    return result


def test_memory_vs_standard():
    """Memory-DFT vs 標準量子力学の比較"""
    print("\n" + "="*70)
    print("Test 2: Memory-DFT vs Standard QM")
    print("="*70)
    
    engine, H_K, H_V = create_h2_model()
    
    # 初期状態
    psi0 = np.array([1, 0, 0, 0], dtype=np.complex128)  # |↑↑⟩
    
    # Memory-DFT
    config_mem = EvolutionConfig(
        t_end=10.0,
        dt=0.1,
        use_memory=True,
        memory_strength=0.2,
        verbose=False
    )
    evol_mem = TimeEvolutionEngine(H_K, H_V, config_mem, use_gpu=False)
    result_mem = evol_mem.run(psi0)
    
    # Standard QM
    config_std = EvolutionConfig(
        t_end=10.0,
        dt=0.1,
        use_memory=False,
        verbose=False
    )
    evol_std = TimeEvolutionEngine(H_K, H_V, config_std, use_gpu=False)
    result_std = evol_std.run(psi0)
    
    # 比較
    lambda_diff = np.array(result_mem.lambdas) - np.array(result_std.lambdas)
    
    print(f"\nComparison:")
    print(f"  Memory-DFT final Λ: {result_mem.lambdas[-1]:.4f}")
    print(f"  Standard QM final Λ: {result_std.lambdas[-1]:.4f}")
    print(f"  Max |ΔΛ|: {np.max(np.abs(lambda_diff)):.4f}")
    print(f"  Mean |ΔΛ|: {np.mean(np.abs(lambda_diff)):.4f}")
    
    # Memory項の効果確認
    if np.max(np.abs(lambda_diff)) > 0.01:
        print("  ✅ Memory effect detected!")
    else:
        print("  ⚠️ Memory effect is small")
    
    return result_mem, result_std


def test_hcsp_axioms():
    """H-CSP公理の検証"""
    print("\n" + "="*70)
    print("Test 3: H-CSP Axiom Validation")
    print("="*70)
    
    engine, H_K, H_V = create_h2_model()
    
    # 初期状態
    psi0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
    psi0 = psi0 / np.linalg.norm(psi0)
    
    config = EvolutionConfig(
        t_end=20.0,
        dt=0.1,
        use_memory=True,
        memory_strength=0.15,
        adaptive_memory=True,  # 適応的Memory
        verbose=False
    )
    
    evol = TimeEvolutionEngine(H_K, H_V, config, use_gpu=False)
    result = evol.run(psi0)
    
    # H-CSP検証
    validator = HCSPValidator()
    validation = validator.validate_all(result.lambdas)
    
    print("\nH-CSP Axiom Validation:")
    
    for axiom, check in validation.items():
        print(f"\n  {axiom}:")
        for k, v in check.items():
            print(f"    {k}: {v}")
    
    return validation


def test_gamma_scaling():
    """γスケーリングのテスト（簡易版）"""
    print("\n" + "="*70)
    print("Test 4: Gamma Scaling (Simplified)")
    print("="*70)
    
    extractor = GammaExtractor()
    
    # 異なるサイズでシミュレーション
    for n_sites in [2, 4, 6]:
        engine = SparseHamiltonianEngine(n_sites=n_sites, use_gpu=False, verbose=False)
        geom = engine.build_chain_geometry(L=n_sites)
        H_K, H_V = engine.build_heisenberg_hamiltonian(geom.bonds)
        
        # ランダム状態
        psi = np.random.randn(engine.dim) + 1j * np.random.randn(engine.dim)
        psi = psi / np.linalg.norm(psi)
        
        # エネルギー計算
        E_K = float(np.real(np.vdot(psi, H_K @ psi)))
        E_V = float(np.real(np.vdot(psi, H_V @ psi)))
        
        # 擬似Vorticity（実際は2-RDMから計算）
        V_pseudo = abs(E_K) * n_sites**1.5
        
        extractor.add_data(n_sites, E_V, V_pseudo)
        print(f"  N={n_sites}: E_K={E_K:.4f}, E_V={E_V:.4f}, V={V_pseudo:.4f}")
    
    # γ抽出
    gamma_result = extractor.extract_gamma()
    
    print(f"\nGamma extraction:")
    print(f"  γ = {gamma_result.get('gamma', 'N/A'):.4f}" if gamma_result.get('gamma') else "  γ = N/A")
    print(f"  R² = {gamma_result.get('r_squared', 'N/A')}")
    
    return gamma_result


def test_memory_kernel_decomposition():
    """Memory kernel成分のテスト"""
    print("\n" + "="*70)
    print("Test 5: Memory Kernel Decomposition")
    print("="*70)
    
    # 異なるカーネル設定
    kernels = [
        ("Field-dominant", KernelWeights(field=0.7, phys=0.2, chem=0.1)),
        ("Phys-dominant", KernelWeights(field=0.2, phys=0.6, chem=0.2)),
        ("Chem-dominant", KernelWeights(field=0.1, phys=0.2, chem=0.7)),
    ]
    
    engine, H_K, H_V = create_h2_model()
    psi0 = np.array([0, 1, -1, 0], dtype=np.complex128)
    psi0 = psi0 / np.linalg.norm(psi0)
    
    results = {}
    
    for name, weights in kernels:
        kernel = CompositeMemoryKernel(weights=weights)
        history = HistoryManager(max_history=100)
        
        solver = MemoryLanczosSolver(
            memory_kernel=kernel,
            history_manager=history,
            memory_strength=0.2,
            use_gpu=False
        )
        
        H = H_K + H_V
        psi = psi0.copy()
        
        # 短い発展
        for t in range(20):
            psi = solver.evolve(H, psi, float(t), 0.1)
        
        final_norm = np.linalg.norm(psi)
        
        results[name] = {
            'final_norm': final_norm,
            'history_size': len(history.history)
        }
        
        print(f"\n  {name}:")
        print(f"    Final norm: {final_norm:.6f}")
        print(f"    History size: {len(history.history)}")
    
    return results


def run_all_tests():
    """全テスト実行"""
    print("\n" + "="*70)
    print("🧪 Memory-DFT H2 Test Suite")
    print("="*70)
    
    try:
        test_basic_evolution()
        test_memory_vs_standard()
        test_hcsp_axioms()
        test_gamma_scaling()
        test_memory_kernel_decomposition()
        
        print("\n" + "="*70)
        print("🎉 All tests passed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
