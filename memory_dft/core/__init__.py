# =============================================================================
# Core Components
# =============================================================================

from .memory_kernel import (
    MemoryKernel,
    MemoryKernelConfig,
    HistoryEntry,
)

from .history_manager import (
    HistoryManager,
    HistoryManagerGPU,
    StateSnapshot,
    LambdaDensityCalculator,
)
# =============================================================================
# Unified Sparse Engine (v0.5.0)
# Replaces: sparse_engine.py, hubbard_engine.py, operators.py, hamiltonian.py
# =============================================================================

from .core.sparse_engine_unified import (
    # Main class
    SparseEngine,
    # Data classes
    SystemGeometry,
    ComputeResult,
    # Backward compatibility aliases
    SparseHamiltonianEngine,
    SpinOperatorsCompat,
    HubbardEngineCompat,
)

# Backward compatibility aliases
HubbardEngine = HubbardEngineCompat
HubbardResult = ComputeResult
SpinOperators = SpinOperatorsCompat

# =============================================================================
# Lattice Geometry (now in sparse_engine_unified)
# =============================================================================

from .core.sparse_engine_unified import (
    LatticeGeometry2D,
    LatticeGeometry,
    create_chain,
    create_ladder,
    create_square_lattice,
)

# =============================================================================
# Backward Compatibility - Operators & Hamiltonian
# =============================================================================

def pauli_matrices():
    """Return Pauli matrices (σx, σy, σz)."""
    import numpy as np
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)
    return sx, sy, sz


def create_spin_operators(n_sites, use_gpu=False):
    """Create spin operators for n_sites."""
    return SpinOperatorsCompat(n_sites, use_gpu=use_gpu)


def compute_total_spin(psi, ops):
    """Compute total spin ⟨S²⟩."""
    if hasattr(ops, '_engine'):
        return ops._engine.compute_total_spin(psi)
    raise ValueError("ops must be SpinOperatorsCompat")


def compute_magnetization(psi, ops):
    """Compute magnetization ⟨Sz⟩/N."""
    if hasattr(ops, '_engine'):
        return ops._engine.compute_magnetization(psi)
    raise ValueError("ops must be SpinOperatorsCompat")


def compute_correlation(psi, ops, i, j, component='Z'):
    """Compute spin-spin correlation ⟨Si·Sj⟩."""
    if hasattr(ops, '_engine'):
        return ops._engine.compute_correlation(psi, i, j, component)
    raise ValueError("ops must be SpinOperatorsCompat")


class HamiltonianBuilder:
    """
    Backward compatibility wrapper for HamiltonianBuilder.
    
    New code should use SparseEngine directly:
        engine = SparseEngine(n_sites)
        H = engine.build_heisenberg(bonds, J=1.0)
    """
    def __init__(self, lattice, ops):
        self._engine = ops._engine if hasattr(ops, '_engine') else \
                       SparseEngine(lattice.N_spins, use_gpu=False, verbose=False)
        self._bonds = lattice.bonds_nn if hasattr(lattice, 'bonds_nn') else lattice.bonds
    
    def heisenberg(self, J=1.0, Jz=None):
        return self._engine.build_heisenberg(self._bonds, J=J, Jz=Jz, split_KV=False)
    
    def xy(self, J=1.0):
        return self._engine.build_xy(self._bonds, J=J)
    
    def ising(self, J=1.0, h=0.0):
        return self._engine.build_ising(self._bonds, J=J, h=h, split_KV=False)
    
    def kitaev_rect(self, Kx=1.0, Ky=0.8, Kz_diag=0.5):
        return self._engine.build_kitaev_rect(Kx=Kx, Ky=Ky, Kz_diag=Kz_diag)


def build_hamiltonian(model, lattice, ops, **kwargs):
    """Build Hamiltonian for given model."""
    builder = HamiltonianBuilder(lattice, ops)
    if model == 'heisenberg':
        return builder.heisenberg(**kwargs)
    elif model == 'xy':
        return builder.xy(**kwargs)
    elif model == 'ising':
        return builder.ising(**kwargs)
    elif model == 'kitaev':
        return builder.kitaev_rect(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")

# =============================================================================
# core/environment_operators.py 
# =============================================================================
from memory_dft.core.environment_operators import (
    # Physical Constants
    K_B_EV,
    K_B_J,
    H_EV,
    HBAR_EV,
    
    # Thermodynamic Utilities
    T_to_beta,
    beta_to_T,
    thermal_energy,
    boltzmann_weights,
    partition_function,
    compute_entropy,
    compute_free_energy,
    
    # Dislocation
    Dislocation,
    compute_peach_koehler_force,
    
    # Environment Operators
    EnvironmentOperator,
    TemperatureOperator,
    StressOperator,
    EnvironmentBuilder,
)


__all__ = [
    # Core
    'MemoryKernel',
    'MemoryKernelConfig',
    'HistoryEntry',
    'HistoryManager',
    'HistoryManagerGPU',
    'StateSnapshot',
    'LambdaDensityCalculator',

]
