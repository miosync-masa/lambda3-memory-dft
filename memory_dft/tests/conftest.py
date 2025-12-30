"""
Memory-DFT Test Configuration
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def hubbard_4site():
    """4-site Hubbard model fixture"""
    from memory_dft.core.hubbard_engine import HubbardEngine
    return HubbardEngine(L=4)


@pytest.fixture
def simple_memory_kernel():
    """Simple memory kernel fixture"""
    from memory_dft.core.memory_kernel import SimpleMemoryKernel
    return SimpleMemoryKernel(eta=0.3, tau=5.0, gamma=0.5)


@pytest.fixture
def repulsive_kernel():
    """Repulsive memory kernel fixture"""
    from memory_dft.core.repulsive_kernel import RepulsiveMemoryKernel
    return RepulsiveMemoryKernel(eta_rep=0.3, tau_rep=3.0, tau_recover=10.0)


@pytest.fixture
def catalyst_kernel():
    """Catalyst memory kernel fixture"""
    from memory_dft.core.memory_kernel import CatalystMemoryKernel
    return CatalystMemoryKernel(eta=0.3, tau_ads=3.0, tau_react=5.0)


@pytest.fixture
def random_state():
    """Random quantum state fixture"""
    np.random.seed(42)
    psi = np.random.randn(16) + 1j * np.random.randn(16)
    return psi / np.linalg.norm(psi)


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "pyscf: marks tests that require PySCF"
    )
