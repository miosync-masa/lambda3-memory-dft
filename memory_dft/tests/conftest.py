"""Memory-DFT Test Configuration"""

import pytest
import numpy as np

@pytest.fixture
def sparse_engine_4site():
    """4-site SparseEngine fixture"""
    from memory_dft.core.sparse_engine_unified import SparseEngine
    return SparseEngine(n_sites=4, use_gpu=False, verbose=False)

@pytest.fixture
def random_state_16():
    """Random 16-dim quantum state fixture"""
    np.random.seed(42)
    psi = np.random.randn(16) + 1j * np.random.randn(16)
    return psi / np.linalg.norm(psi)

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "pyscf: marks tests that require PySCF")
