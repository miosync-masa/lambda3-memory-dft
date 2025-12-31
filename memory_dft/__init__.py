"""
Direct Schrödinger Evolution (DSE)
==================================

A framework for history-dependent quantum dynamics through
direct solution of the Schrödinger equation.

DFT erases history. DSE remembers.

Key Insight:
  Standard DFT: E[ρ(r)]        - Same structure = Same energy
  DSE:          E[ψ(t)]        - Different history = Different energy

Theoretical Background:
  Correlation decomposition by distance filtering:
    γ_total (r=∞) = 2.604   ← Full correlations
    γ_local (r≤2) = 1.388   ← Markovian sector
    γ_memory      = 1.216   ← Non-Markovian (46.7%)

  This shows that nearly half of quantum correlations
  require history-dependent treatment.

Key Results:
  - gamma_memory = 1.216 (46.7% of correlations are non-Markovian)
  - Path dependence: 1.59 (adsorption order matters)
  - Reaction sequence: 2.18 (A->B->C differs from A->C->B)
  - Thermal path dependence: 2.26 (heating/cooling history matters)
  - Standard DFT cannot distinguish these paths (history-blind)

Structure (Refactored):
  memory_dft/
  ├── core/
  │   ├── memory_kernel.py      # 3-layer kernel (field/phys/chem)
  │   ├── repulsive_kernel.py   # Compression memory
  │   ├── history_manager.py    # History tracking
  │   ├── sparse_engine.py      # Sparse Hamiltonian
  │   ├── hubbard_engine.py     # Hubbard model
  │   ├── lattice.py            # Lattice geometry (NEW)
  │   ├── operators.py          # Spin operators (NEW)
  │   └── hamiltonian.py        # Hamiltonian builders (NEW)
  ├── solvers/
  │   ├── lanczos_memory.py     # Lanczos + memory
  │   ├── time_evolution.py     # Time evolution
  │   ├── memory_indicators.py  # Memory quantification
  │   └── chemical_reaction.py  # Surface chemistry solver
  ├── physics/
  │   ├── lambda3_bridge.py     # Stability diagnostics
  │   ├── vorticity.py          # γ decomposition
  │   └── thermodynamics.py     # Thermal utilities (NEW)
  ├── examples/                  # Example scripts (NEW)
  │   ├── thermal_path.py       # Thermal path demo
  │   └── ladder_2d.py          # 2D lattice demo
  ├── visualization/
  │   └── prl_figures.py        # PRL publication figures
  └── tests/
      └── ...                    # Test suites

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)

DOI: 10.5281/zenodo.18095869

Author: Masamichi Iizumi, Tamaki Iizumi
"""

__version__ = "0.4.0"  # Bumped for refactoring

# =============================================================================
# Core Components
# =============================================================================

# Memory Kernels
from .core.memory_kernel import (
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
from .core.repulsive_kernel import (
    RepulsiveMemoryKernel,
    CompressionEvent,
    ExtendedCompositeKernel
)

# History Manager
from .core.history_manager import (
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
    StateSnapshot
)

# Sparse Engine
from .core.sparse_engine import (
    SparseHamiltonianEngine,
)

# Hubbard Engine
from .core.hubbard_engine import (
    HubbardEngine,
    HubbardResult
)

# Lattice Geometry (NEW - refactored from ladder_dse.py)
from .core.lattice import (
    SystemGeometry,
    LatticeGeometry2D,
    LatticeGeometry,  # Alias
    create_chain,
    create_ladder,
    create_square_lattice,
)

# Spin Operators (NEW - refactored from ladder_dse.py)
from .core.operators import (
    SpinOperators,
    pauli_matrices,
    create_spin_operators,
    compute_total_spin,
    compute_magnetization,
    compute_correlation,
)

# Hamiltonian Builders (NEW - refactored from ladder_dse.py)
from .core.hamiltonian import (
    HamiltonianBuilder,
    build_hamiltonian,
)

# =============================================================================
# Solvers
# =============================================================================

from .solvers.lanczos_memory import (
    MemoryLanczosSolver,
    AdaptiveMemorySolver,
    lanczos_expm_multiply
)

from .solvers.time_evolution import (
    TimeEvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
    quick_evolve
)

from .solvers.memory_indicators import (
    MemoryIndicator,
    MemoryMetrics,
    HysteresisAnalyzer
)

from .solvers.chemical_reaction import (
    ChemicalReactionSolver,
    SurfaceHamiltonianEngine,
    LanczosEvolver,
    ReactionEvent,
    ReactionPath,
    PathResult
)

# =============================================================================
# Physics
# =============================================================================

from .physics.lambda3_bridge import (
    Lambda3Calculator,
    HCSPValidator,
    LambdaState,
    StabilityPhase
)

from .physics.vorticity import (
    VorticityCalculator,
    VorticityResult,
    GammaExtractor,
    MemoryKernelFromGamma
)

# Thermodynamics (NEW - refactored from thermal_dse.py)
from .physics.thermodynamics import (
    K_B_EV,
    T_to_beta,
    beta_to_T,
    thermal_energy,
    boltzmann_weights,
    partition_function,
    thermal_expectation,
    thermal_expectation_zero_T,
    compute_entropy,
    compute_free_energy,
    compute_heat_capacity,
    thermal_density_matrix,
    sample_thermal_state,
)

# Two-Particle Reduced Density Matrix (NEW)
from .physics.rdm import (
    RDM2Result,
    compute_2rdm,
    compute_2rdm_with_ops,
    compute_density_density_correlation,
    compute_connected_correlation,
    compute_correlation_matrix,
    filter_by_distance,
    from_pyscf_rdm2,
    to_pyscf_rdm2,
)

# =============================================================================
# Visualization (optional - requires matplotlib)
# =============================================================================

try:
    from .visualization.prl_figures import (
        fig1_gamma_decomposition,
        fig2_path_evolution,
        fig3_memory_comparison,
        generate_all_prl_figures,
        COLORS as PRL_COLORS
    )
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    
    def fig1_gamma_decomposition(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def fig2_path_evolution(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def fig3_memory_comparison(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def generate_all_prl_figures(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    PRL_COLORS = {}

# =============================================================================
# Backward Compatibility (DEPRECATED)
# =============================================================================

import warnings

def _deprecated_import(name, new_location):
    """Helper for deprecated imports."""
    warnings.warn(
        f"{name} is deprecated. Use {new_location} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# These were previously in solvers/thermal_dse.py and solvers/ladder_dse.py
# Now they're in core/ and physics/
# Kept for backward compatibility but will show deprecation warning

# Note: ThermalDSESolver and LadderDSESolver have been removed.
# Use the new examples/ module for similar functionality:
#   from memory_dft.examples.thermal_path import ThermalPathSolver
#   from memory_dft.examples.ladder_2d import Ladder2DSolver


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Core - Memory Kernels
    'MemoryKernelBase',
    'PowerLawKernel',
    'StretchedExpKernel',
    'StepKernel',
    'CompositeMemoryKernel',
    'CompositeMemoryKernelGPU',
    'KernelWeights',
    'SimpleMemoryKernel',
    'CatalystMemoryKernel',
    'CatalystEvent',
    'RepulsiveMemoryKernel',
    'CompressionEvent',
    'ExtendedCompositeKernel',
    
    # Core - History
    'HistoryManager',
    'HistoryManagerGPU',
    'LambdaDensityCalculator',
    'StateSnapshot',
    
    # Core - Engines
    'SparseHamiltonianEngine',
    'HubbardEngine',
    'HubbardResult',
    
    # Core - Lattice (NEW)
    'SystemGeometry',
    'LatticeGeometry2D',
    'LatticeGeometry',
    'create_chain',
    'create_ladder',
    'create_square_lattice',
    
    # Core - Operators (NEW)
    'SpinOperators',
    'pauli_matrices',
    'create_spin_operators',
    'compute_total_spin',
    'compute_magnetization',
    'compute_correlation',
    
    # Core - Hamiltonian (NEW)
    'HamiltonianBuilder',
    'build_hamiltonian',
    
    # Solvers - Lanczos
    'MemoryLanczosSolver',
    'AdaptiveMemorySolver',
    'lanczos_expm_multiply',
    
    # Solvers - Time Evolution
    'TimeEvolutionEngine',
    'EvolutionConfig',
    'EvolutionResult',
    'quick_evolve',
    
    # Solvers - Memory Indicators
    'MemoryIndicator',
    'MemoryMetrics',
    'HysteresisAnalyzer',
    
    # Solvers - Chemical Reaction
    'ChemicalReactionSolver',
    'SurfaceHamiltonianEngine',
    'LanczosEvolver',
    'ReactionEvent',
    'ReactionPath',
    'PathResult',
    
    # Physics - Stability
    'Lambda3Calculator',
    'LambdaState',
    'StabilityPhase',
    'HCSPValidator',
    
    # Physics - Vorticity
    'VorticityCalculator',
    'VorticityResult',
    'GammaExtractor',
    'MemoryKernelFromGamma',
    
    # Physics - Thermodynamics (NEW)
    'K_B_EV',
    'T_to_beta',
    'beta_to_T',
    'thermal_energy',
    'boltzmann_weights',
    'partition_function',
    'thermal_expectation',
    'thermal_expectation_zero_T',
    'compute_entropy',
    'compute_free_energy',
    'compute_heat_capacity',
    'thermal_density_matrix',
    'sample_thermal_state',
    
    # Physics - 2-RDM (NEW)
    'RDM2Result',
    'compute_2rdm',
    'compute_2rdm_with_ops',
    'compute_density_density_correlation',
    'compute_connected_correlation',
    'compute_correlation_matrix',
    'filter_by_distance',
    'from_pyscf_rdm2',
    'to_pyscf_rdm2',
    
    # Visualization
    'HAS_VISUALIZATION',
    'fig1_gamma_decomposition',
    'fig2_path_evolution',
    'fig3_memory_comparison',
    'generate_all_prl_figures',
    'PRL_COLORS',
]
