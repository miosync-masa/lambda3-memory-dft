"""
Memory-DFT Core Components
==========================

Foundation modules for Memory-DFT simulations.

Modules:
  - memory_kernel: Non-Markovian memory kernels
  - repulsive_kernel: Compression/repulsive memory
  - history_manager: State history tracking
  - sparse_engine: Sparse Hamiltonian construction
  - hubbard_engine: Hubbard model implementation
  - lattice: Lattice geometry definitions (NEW)
  - operators: Spin operators (NEW)
  - hamiltonian: Hamiltonian builders (NEW)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

# Memory Kernels
from .memory_kernel import (
    MemoryKernelBase,
    PowerLawKernel,
    StretchedExpKernel,
    StepKernel,
    CompositeMemoryKernel,
    CompositeMemoryKernelGPU,
    KernelWeights,
    CatalystMemoryKernel,
    CatalystEvent,
    SimpleMemoryKernel
)

# Repulsive Kernel
from .repulsive_kernel import (
    RepulsiveMemoryKernel,
    CompressionEvent,
    ExtendedCompositeKernel
)

# History Manager
from .history_manager import (
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
    StateSnapshot
)

# Sparse Engine
from .sparse_engine import (
    SparseHamiltonianEngine,
    # SystemGeometry also exported from lattice.py for compatibility
)

# Hubbard Engine
from .hubbard_engine import (
    HubbardEngine,
    HubbardResult
)

# =========================================================================
# NEW MODULES (Refactored from ladder_dse.py)
# =========================================================================

# Lattice Geometry
from .lattice import (
    # Core classes
    SystemGeometry,          # General geometry (also in sparse_engine)
    LatticeGeometry2D,       # 2D lattice with boundary conditions
    LatticeGeometry,         # Alias for LatticeGeometry2D
    # Factory functions
    create_chain,
    create_ladder,
    create_square_lattice,
)

# Spin Operators
from .operators import (
    # Core class
    SpinOperators,
    # Helper functions
    pauli_matrices,
    pauli_matrices_full,
    create_spin_operators,
    # Utility functions
    compute_total_spin,
    compute_magnetization,
    compute_correlation,
)

# Hamiltonian Builders
from .hamiltonian import (
    # Core class
    HamiltonianBuilder,
    # Factory function
    build_hamiltonian,
)


# =========================================================================
# __all__ for explicit exports
# =========================================================================

__all__ = [
    # Memory Kernels
    'MemoryKernelBase',
    'PowerLawKernel',
    'StretchedExpKernel',
    'StepKernel',
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
    
    # Sparse Engine
    'SparseHamiltonianEngine',
    
    # Hubbard Engine
    'HubbardEngine',
    'HubbardResult',
    
    # Lattice Geometry (NEW)
    'SystemGeometry',
    'LatticeGeometry2D',
    'LatticeGeometry',
    'create_chain',
    'create_ladder',
    'create_square_lattice',
    
    # Spin Operators (NEW)
    'SpinOperators',
    'pauli_matrices',
    'pauli_matrices_full',
    'create_spin_operators',
    'compute_total_spin',
    'compute_magnetization',
    'compute_correlation',
    
    # Hamiltonian Builders (NEW)
    'HamiltonianBuilder',
    'build_hamiltonian',
]
