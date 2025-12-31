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

from .lanczos_memory import (
    MemoryLanczosSolver,
    AdaptiveMemorySolver,
    lanczos_expm_multiply
)

from .time_evolution import (
    TimeEvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
    quick_evolve
)

from .memory_indicators import (
    MemoryIndicator,
    MemoryMetrics,
    HysteresisAnalyzer
)

from .chemical_reaction import (
    ChemicalReactionSolver,
    SurfaceHamiltonianEngine,
    LanczosEvolver,
    ReactionEvent,
    ReactionPath,
    PathResult
)

# =========================================================================
# REMOVED (Refactored to core/ and physics/)
# =========================================================================
#
# The following imports have been removed:
#
# from .thermal_dse import (
#     ThermalDSESolver,           # → examples/thermal_path.py
#     thermal_expectation,        # → physics/thermodynamics.py
#     thermal_expectation_zero_T, # → physics/thermodynamics.py
#     compute_entropy,            # → physics/thermodynamics.py
#     T_to_beta,                  # → physics/thermodynamics.py
#     beta_to_T,                  # → physics/thermodynamics.py
# )
#
# from .ladder_dse import (
#     LadderDSESolver,            # → examples/ladder_2d.py
#     LatticeGeometry,            # → core/lattice.py
#     SpinOperators,              # → core/operators.py
#     HamiltonianBuilder,         # → core/hamiltonian.py
# )
#
# Migration guide:
#   # OLD
#   from memory_dft.solvers import T_to_beta, LatticeGeometry
#   
#   # NEW
#   from memory_dft.physics import T_to_beta
#   from memory_dft.core import LatticeGeometry
#   
#   # Or simply
#   from memory_dft import T_to_beta, LatticeGeometry
#
# =========================================================================


__all__ = [
    # Lanczos
    'MemoryLanczosSolver',
    'AdaptiveMemorySolver',
    'lanczos_expm_multiply',
    
    # Time Evolution
    'TimeEvolutionEngine',
    'EvolutionConfig',
    'EvolutionResult',
    'quick_evolve',
    
    # Memory Indicators
    'MemoryIndicator',
    'MemoryMetrics',
    'HysteresisAnalyzer',
    
    # Chemical Reaction
    'ChemicalReactionSolver',
    'SurfaceHamiltonianEngine',
    'LanczosEvolver',
    'ReactionEvent',
    'ReactionPath',
    'PathResult',
]
