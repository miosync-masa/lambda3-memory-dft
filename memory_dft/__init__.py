"""
Memory-DFT: History-Dependent Density Functional Theory
=======================================================

A framework for incorporating memory effects into density
functional theory calculations, capturing path-dependent
phenomena that standard DFT cannot describe.

Key Insight:
  Standard DFT: E[ρ(r)]        - Same structure = Same energy
  Memory-DFT:   E[ρ(r), history] - Different history = Different energy

Theoretical Background:
  Correlation decomposition by distance filtering:
    γ_total (r=∞) = 2.604   ← Full correlations
    γ_local (r≤2) = 1.388   ← Markovian sector
    γ_memory      = 1.216   ← Non-Markovian (46.7%)

  This shows that nearly half of quantum correlations
  require history-dependent treatment.

Key Results:
  - γ_memory = 1.216 (46.7% of correlations are non-Markovian)
  - Path dependence: ΔΛ = 1.59 (adsorption order)
  - Reaction sequence: ΔΛ = 2.18
  - Standard DFT cannot distinguish these paths (ΔΛ ≡ 0)

Structure:
  memory_dft/
  ├── core/
  │   ├── memory_kernel.py      # 3-layer kernel (field/phys/chem)
  │   ├── repulsive_kernel.py   # Compression memory
  │   ├── history_manager.py    # History tracking
  │   ├── sparse_engine.py      # Sparse Hamiltonian
  │   └── hubbard_engine.py     # Hubbard model
  ├── solvers/
  │   ├── lanczos_memory.py     # Lanczos + memory
  │   ├── time_evolution.py     # Time evolution
  │   ├── memory_indicators.py  # Memory quantification (ΔO, M(t), γ)
  │   └── chemical_reaction.py  # Surface chemistry solver
  ├── physics/
  │   ├── lambda3_bridge.py     # Stability diagnostics
  │   └── vorticity.py          # γ decomposition
  ├── visualization/
  │   └── prl_figures.py        # PRL publication figures
  └── tests/
      ├── test_chemical.py      # Chemical tests (A/B/C/D)
      └── test_repulsive.py     # Repulsive tests (E1/E2/E3)

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

__version__ = "0.2.0"

# Core components
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

from .core.history_manager import (
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
    StateSnapshot
)

from .core.sparse_engine import (
    SparseHamiltonianEngine,
    SystemGeometry
)

from .core.hubbard_engine import (
    HubbardEngine,
    HubbardResult
)

from .core.repulsive_kernel import (
    RepulsiveMemoryKernel,
    CompressionEvent,
    ExtendedCompositeKernel
)

# Solvers
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

# Physics
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

# Visualization (optional - requires matplotlib)
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
    # Provide dummy functions
    def fig1_gamma_decomposition(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def fig2_path_evolution(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def fig3_memory_comparison(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def generate_all_prl_figures(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    PRL_COLORS = {}

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
    
    # Core - Catalyst
    'CatalystMemoryKernel',
    'CatalystEvent',
    
    # Core - Repulsive
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
    'SystemGeometry',
    'HubbardEngine',
    'HubbardResult',
    
    # Solvers - Lanczos
    'MemoryLanczosSolver',
    'AdaptiveMemorySolver',
    'lanczos_expm_multiply',
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
    
    # Physics
    'Lambda3Calculator',
    'LambdaState',
    'StabilityPhase',
    'HCSPValidator',
    'VorticityCalculator',
    'VorticityResult',
    'GammaExtractor',
    'MemoryKernelFromGamma',
    
    # Visualization (optional)
    'HAS_VISUALIZATION',
    'fig1_gamma_decomposition',
    'fig2_path_evolution',
    'fig3_memory_comparison',
    'generate_all_prl_figures',
    'PRL_COLORS',
]
