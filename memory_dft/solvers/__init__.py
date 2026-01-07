"""
Memory-DFT Solvers
==================

Numerical solvers for Memory-DFT simulations.

Modules:
  - lanczos_memory: Lanczos-based time evolution with memory
  - time_evolution: General time evolution engine
  - memory_indicators: Memory quantification metrics
  - chemical_reaction: Surface chemistry solver

Note:
  ThermalDSESolver and LadderDSESolver have been refactored.
  - Thermal functions: Use memory_dft.physics.thermodynamics
  - Lattice/Operators/Hamiltonian: Use memory_dft.core.*
  - Example code: See memory_dft.examples.*

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from .dse_solver import (
    # Main solver
    DSESolver,
    
    # Result container
    DSEResult,
    
    # Utility
    lanczos_expm_multiply,
    quick_dse,
)


from .memory_indicators import (
    MemoryIndicator,
    MemoryMetrics,
    HysteresisAnalyzer
)


__all__ = [
    # Memory Indicators
    'MemoryIndicator',
    'MemoryMetrics',
    'HysteresisAnalyzer',

    # Solver
    'DSESolver',
    
    # Result
    'DSEResult',
    
    # Utility
    'lanczos_expm_multiply',
    'quick_dse',
]
