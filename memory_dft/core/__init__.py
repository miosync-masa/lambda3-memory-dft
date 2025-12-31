"""
Memory-DFT Core Components
==========================

Foundation modules for Memory-DFT simulations.

Memory Kernel Components (v0.4.0):
  1. Field (PowerLaw): Long-range correlations
  2. Phys (StretchedExp): Structural relaxation  
  3. Chem (Step): Irreversible reactions
  4. Exclusion: Distance-direction memory [NEW]

The same distance r = 0.8 A means DIFFERENT things:
  - Approaching: system is being compressed
  - Departing: system is recovering
DFT cannot distinguish. DSE can!

Modules:
  - memory_kernel: Non-Markovian memory kernels (4 components)
  - repulsive_kernel: Detailed compression history tracking
  - history_manager: State history tracking
  - sparse_engine: Sparse Hamiltonian construction
  - hubbard_engine: Hubbard model implementation
  - lattice: Lattice geometry definitions
  - operators: Spin operators
  - hamiltonian: Hamiltonian builders

Author: Masamichi Iizumi, Tamaki Iizumi
"""

# Memory Kernels (4 components: field, phys, chem, exclusion)
from .memory_kernel import (
    MemoryKernelBase,
    PowerLawKernel,
    StretchedExpKernel,
    StepKernel,
    ExclusionKernel,      # NEW in v0.4.0 - distance direction memory
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
    # Memory Kernels (4 components)
    'MemoryKernelBase',
    'PowerLawKernel',
    'StretchedExpKernel',
    'StepKernel',
    'ExclusionKernel',      # NEW - distance direction memory
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
