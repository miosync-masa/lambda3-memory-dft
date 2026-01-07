"""Smoke tests - 最低限の動作確認"""

import numpy as np
import pytest


def test_import_core():
    """core モジュールがインポートできる"""
    from memory_dft.core import SparseEngine, MemoryKernel, EnvironmentBuilder
    assert SparseEngine is not None
    assert MemoryKernel is not None


def test_import_solvers():
    """solvers モジュールがインポートできる"""
    from memory_dft.solvers import DSESolver, DSEResult, MemoryIndicator
    assert DSESolver is not None


def test_import_physics():
    """physics モジュールがインポートできる"""
    from memory_dft.physics import VorticityCalculator, HubbardRDM
    assert VorticityCalculator is not None


def test_sparse_engine_basic():
    """SparseEngine 基本動作"""
    from memory_dft.core.sparse_engine_unified import SparseEngine
    
    engine = SparseEngine(n_sites=4, use_gpu=False, verbose=False)
    geom = engine.build_chain()
    H_K, H_V = engine.build_heisenberg(geom.bonds)
    E, psi = engine.compute_ground_state(H_K + H_V)
    
    assert E < 0
    assert len(psi) == 16


def test_lambda_calculation():
    """λ = K/|V| 計算"""
    from memory_dft.core.sparse_engine_unified import SparseEngine
    
    engine = SparseEngine(n_sites=4, use_gpu=False, verbose=False)
    geom = engine.build_chain()
    H_K, H_V = engine.build_heisenberg(geom.bonds)
    E, psi = engine.compute_ground_state(H_K + H_V)
    
    lam = engine.compute_lambda(psi, H_K, H_V)
    assert lam > 0


def test_memory_kernel_basic():
    """MemoryKernel 基本動作"""
    from memory_dft.core.memory_kernel import MemoryKernel
    
    kernel = MemoryKernel(gamma_memory=1.0, use_gpu=False)
    
    # 状態を追加
    psi = np.array([1, 0, 0, 0], dtype=complex)
    kernel.add_state(t=0.0, r=0.5, state=psi, energy=-1.0)
    kernel.add_state(t=1.0, r=0.6, state=psi, energy=-0.9)
    
    assert len(kernel.history) == 2


def test_dse_solver_basic():
    """DSESolver 基本動作"""
    from memory_dft.solvers.dse_solver import DSESolver
    import scipy.sparse as sp
    
    # 2準位系
    H_K = sp.csr_matrix([[-1, 0], [0, 1]], dtype=complex)
    H_V = sp.csr_matrix([[0, 0.5], [0.5, 0]], dtype=complex)
    psi0 = np.array([1, 0], dtype=complex)
    
    solver = DSESolver(H_K, H_V, use_memory=False, use_gpu=False)
    result = solver.run(psi0, t_end=1.0, dt=0.1, verbose=False)
    
    assert result.n_steps == 10
    assert len(result.energies) == 11
