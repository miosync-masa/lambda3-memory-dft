"""
Memory-DFT Test Configuration
=============================

pytest の設定とフィクスチャ

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture
def sparse_engine_4site():
    """4-site SparseEngine fixture"""
    from memory_dft.core.sparse_engine import SparseEngine
    return SparseEngine(n_sites=4, use_gpu=False)


@pytest.fixture
def chain_geometry_4site(sparse_engine_4site):
    """4-site chain geometry fixture"""
    return sparse_engine_4site.build_chain_geometry(L=4)


@pytest.fixture
def hubbard_hamiltonian_4site(sparse_engine_4site, chain_geometry_4site):
    """4-site Hubbard Hamiltonian fixture"""
    H_K, H_V = sparse_engine_4site.build_hubbard_hamiltonian(
        chain_geometry_4site.bonds, t=1.0, U=2.0
    )
    return H_K, H_V


# =============================================================================
# Memory Kernel Fixtures
# =============================================================================

@pytest.fixture
def memory_kernel():
    """MemoryKernel fixture (統一版)"""
    from memory_dft.core.memory_kernel import MemoryKernel
    return MemoryKernel(gamma_memory=1.0, use_gpu=False)


@pytest.fixture
def memory_kernel_with_gamma():
    """MemoryKernel with custom γ fixture"""
    from memory_dft.core.memory_kernel import MemoryKernel
    return MemoryKernel(gamma_memory=1.5, tau0=8.0, use_gpu=False)


# =============================================================================
# Solver Fixtures
# =============================================================================

@pytest.fixture
def dse_solver(hubbard_hamiltonian_4site):
    """DSESolver fixture"""
    from memory_dft.solvers.dse_solver import DSESolver
    H_K, H_V = hubbard_hamiltonian_4site
    return DSESolver(H_K, H_V, gamma_memory=1.0, eta=0.1, use_gpu=False)


# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def random_state_16():
    """Random 16-dim quantum state fixture"""
    np.random.seed(42)
    psi = np.random.randn(16) + 1j * np.random.randn(16)
    return psi / np.linalg.norm(psi)


@pytest.fixture
def random_state(sparse_engine_4site):
    """Random quantum state matching engine dimension"""
    np.random.seed(42)
    dim = sparse_engine_4site.dim
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    return psi / np.linalg.norm(psi)


@pytest.fixture
def ground_state(hubbard_hamiltonian_4site):
    """Ground state of 4-site Hubbard"""
    import scipy.sparse.linalg as sla
    H_K, H_V = hubbard_hamiltonian_4site
    H = H_K + H_V
    eigenvalues, eigenvectors = sla.eigsh(H, k=1, which='SA')
    return eigenvectors[:, 0]


# =============================================================================
# Physics Fixtures
# =============================================================================

@pytest.fixture
def vorticity_calculator():
    """VorticityCalculator fixture"""
    from memory_dft.physics.vorticity import VorticityCalculator
    return VorticityCalculator(use_gpu=False)


@pytest.fixture
def hubbard_rdm():
    """HubbardRDM fixture"""
    from memory_dft.physics.rdm import HubbardRDM
    return HubbardRDM(n_sites=4, lattice='1d')


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def environment_builder(sparse_engine_4site):
    """EnvironmentBuilder fixture"""
    from memory_dft.core.environment_operators import EnvironmentBuilder
    return EnvironmentBuilder(sparse_engine_4site, t0=1.0, U0=2.0)


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (CuPy)"
    )
    config.addinivalue_line(
        "markers", "pyscf: marks tests that require PySCF"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# =============================================================================
# Skip Helpers
# =============================================================================

def skip_if_no_gpu():
    """Skip test if CuPy not available"""
    try:
        import cupy
        return False
    except ImportError:
        return True


def skip_if_no_pyscf():
    """Skip test if PySCF not available"""
    try:
        import pyscf
        return False
    except ImportError:
        return True


# Pytest skip decorators
requires_gpu = pytest.mark.skipif(skip_if_no_gpu(), reason="CuPy not available")
requires_pyscf = pytest.mark.skipif(skip_if_no_pyscf(), reason="PySCF not available")
