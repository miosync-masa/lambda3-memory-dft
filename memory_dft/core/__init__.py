"""
Memory-DFT Core Components
==========================

Foundation modules for Memory-DFT simulations.

Memory Kernel Components (v0.4.0):
  1. Field (PowerLaw): Long-range correlations
  2. Phys (StretchedExp): Structural relaxation  
  3. Chem (Step): Irreversible reactions
  4. Exclusion: Distance-direction memory

The same distance r = 0.8 A means DIFFERENT things:
  - Approaching: system is being compressed
  - Departing: system is recovering
DFT cannot distinguish. DSE can!

Modules:
  - memory_kernel: Non-Markovian memory kernels (4 components)
  - repulsive_kernel: Detailed compression history tracking
  - history_manager: State history tracking
  - sparse_engine_unified: Unified sparse Hamiltonian engine (GPU/CPU)
  - lattice: Lattice geometry definitions

v0.5.0: Unified SparseEngine consolidation
  - operators.py, hamiltonian.py, hubbard_engine.py → sparse_engine_unified.py
  - All models (Heisenberg, Ising, XY, Hubbard, Kitaev) in one place
  - GPU/CPU automatic backend selection

Author: Masamichi Iizumi, Tamaki Iizumi
"""

# =============================================================================
# core/environment_operators.py - 環境作用素 B_θ
# =============================================================================
from .environment_operators import (
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

# Memory Kernels (4 components: field, phys, chem, exclusion)
from .memory_kernel import (
    MemoryKernelBase,
    PowerLawKernel,
    StretchedExpKernel,
    StepKernel,
    ExclusionKernel,
    CompositeMemoryKernel,
    CompositeMemoryKernelGPU,
    KernelWeights,
    CatalystMemoryKernel,
    CatalystEvent,
    SimpleMemoryKernel,
    RepulsiveMemoryKernel  # v0.5.0: moved to memory_kernel.py
)

# History Manager
from .history_manager import (
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
    StateSnapshot
)

# =========================================================================
# Unified Sparse Engine (v0.5.0)
# Replaces: operators.py, hamiltonian.py, hubbard_engine.py, sparse_engine.py
# =========================================================================

from .sparse_engine_unified import (
    # Core class
    SparseEngine,
    
    # Data classes
    SystemGeometry,
    ComputeResult,
    
    # Backward compatibility aliases
    SparseHamiltonianEngine,  # = SparseEngine
    HubbardResult,            # = ComputeResult
    SpinOperatorsCompat,      # SpinOperators-like interface
    HubbardEngineCompat,      # HubbardEngine-like interface
    HubbardEngine,            # = HubbardEngineCompat
)

# =========================================================================
# Lattice Geometry (now in sparse_engine_unified)
# =========================================================================

from .sparse_engine_unified import (
    LatticeGeometry2D,
    LatticeGeometry,
    create_chain,
    create_ladder,
    create_square_lattice,
)

# =========================================================================
# Backward Compatibility Aliases
# =========================================================================

# For code that imports SpinOperators
SpinOperators = SpinOperatorsCompat

# HubbardEngine is already imported as HubbardEngineCompat from sparse_engine_unified
# (Do NOT override here!)

# For code that imports HamiltonianBuilder
# Usage: HamiltonianBuilder(lattice, ops) → engine.build_heisenberg(bonds)
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


# Helper functions (backward compatibility)
def pauli_matrices():
    """Return Pauli matrices (σx, σy, σz)."""
    import numpy as np
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)
    return sx, sy, sz


def pauli_matrices_full():
    """Return full Pauli matrices (I, σx, σy, σz, σ+, σ-)."""
    import numpy as np
    I = np.eye(2, dtype=np.complex128)
    sx, sy, sz = pauli_matrices()
    sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    return I, sx, sy, sz, sp, sm


def create_spin_operators(n_sites, use_gpu=False):
    """Create spin operators for n_sites."""
    return SpinOperatorsCompat(n_sites, use_gpu=use_gpu)


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


# =========================================================================
# __all__ for explicit exports
# =========================================================================

__all__ = [
    # Memory Kernels (4 components)
    'MemoryKernelBase',
    'PowerLawKernel',
    'StretchedExpKernel',
    'StepKernel',
    'ExclusionKernel',
    'CompositeMemoryKernel',
    'CompositeMemoryKernelGPU',
    'KernelWeights',
    'CatalystMemoryKernel',
    'CatalystEvent',
    'SimpleMemoryKernel',
    
    # Repulsive Kernel
    'RepulsiveMemoryKernel',
    'CompressionEvent',
    'ExtendedCompositeKernel',
    
    # History Manager
    'HistoryManager',
    'HistoryManagerGPU',
    'LambdaDensityCalculator',
    'StateSnapshot',
    
    # Unified Sparse Engine (v0.5.0)
    'SparseEngine',
    'SystemGeometry',
    'ComputeResult',
    
    # Backward compatibility (Sparse Engine)
    'SparseHamiltonianEngine',
    'HubbardEngine',
    'HubbardEngineCompat',
    'HubbardResult',
    
    # Lattice Geometry
    'LatticeGeometry2D',
    'LatticeGeometry',
    'create_chain',
    'create_ladder',
    'create_square_lattice',
  
    # Physical Constants
    "K_B_EV",
    "K_B_J", 
    "H_EV",
    "HBAR_EV",
    
    # Thermodynamic Utilities
    "T_to_beta",
    "beta_to_T",
    "thermal_energy",
    "boltzmann_weights",
    "partition_function",
    "compute_entropy",
    "compute_free_energy",
    
    # Dislocation
    "Dislocation",
    "compute_peach_koehler_force",
    
    # Environment Operators
    "EnvironmentOperator",
    "TemperatureOperator",
    "StressOperator",
    "EnvironmentBuilder",
    
    # Backward compatibility (Operators/Hamiltonian)
    'SpinOperators',
    'SpinOperatorsCompat',
    'pauli_matrices',
    'pauli_matrices_full',
    'create_spin_operators',
    'compute_total_spin',
    'compute_magnetization',
    'compute_correlation',
    'HamiltonianBuilder',
    'build_hamiltonian',
]
