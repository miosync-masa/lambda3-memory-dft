# =============================================================================
# memory_dft/core/__init__.py
# =============================================================================

"""
Memory-DFT Core Module
======================

Core components for Memory-DFT calculations.
"""

from .sparse_engine_unified import (
    # Main engine
    SparseEngine,
    
    # Geometry
    SystemGeometry,
    LatticeGeometry2D,
    LatticeGeometry,
    
    # Factory functions
    create_chain,
    create_ladder,
    create_square_lattice,
    
    # Result container
    ComputeResult,
)

from .memory_kernel import (
    MemoryKernel,
    MemoryKernelConfig,
    HistoryEntry,
)

from .environment_operators import (
    EnvironmentBuilder,
    EnvironmentOperator,
    TemperatureOperator,
    StressOperator,
    Dislocation,
    # Thermodynamic utilities
    T_to_beta,
    beta_to_T,
    thermal_energy,
    boltzmann_weights,
    partition_function,
    compute_entropy,
    compute_free_energy,
    compute_heat_capacity,
)

__all__ = [
    # Engine
    'SparseEngine',
    
    # Geometry
    'SystemGeometry',
    'LatticeGeometry2D',
    'LatticeGeometry',
    'create_chain',
    'create_ladder',
    'create_square_lattice',
    
    # Result
    'ComputeResult',
    
    # Memory
    'MemoryKernel',
    'MemoryKernelConfig',
    'HistoryEntry',
    
    # Environment
    'EnvironmentBuilder',
    'EnvironmentOperator',
    'TemperatureOperator',
    'StressOperator',
    'Dislocation',
    
    # Thermodynamics
    'T_to_beta',
    'beta_to_T',
    'thermal_energy',
    'boltzmann_weights',
    'partition_function',
    'compute_entropy',
    'compute_free_energy',
    'compute_heat_capacity',
]
