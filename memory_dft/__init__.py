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

v0.5.0 Changes:
  - Unified SparseEngine consolidation
  - operators.py, hamiltonian.py, hubbard_engine.py → sparse_engine_unified.py
  - All models (Heisenberg, Ising, XY, Hubbard, Kitaev) in one place
  - GPU/CPU automatic backend selection

Structure (Refactored):
  memory_dft/
  ├── cli/                      # Command-line interface
  │   ├── __init__.py           # Typer app & command registration
  │   ├── utils.py              # Shared CLI utilities
  │   └── commands/             # Individual command modules
  ├── core/
  │   ├── memory_kernel.py      # 4-layer kernel (field/phys/chem/exclusion)
  │   ├── repulsive_kernel.py   # Compression memory
  │   ├── history_manager.py    # History tracking
  │   ├── sparse_engine_unified.py  # Unified sparse engine (v0.5.0)
  │   └── lattice.py            # Lattice geometry
  ├── solvers/
  │   ├── lanczos_memory.py     # Lanczos + memory
  │   ├── time_evolution.py     # Time evolution
  │   ├── memory_indicators.py  # Memory quantification
  │   └── chemical_reaction.py  # Surface chemistry solver
  ├── engineering/              
  │   ├── base.py               # Common Base
  │   └──thermo_mechanical.py   # ThermoMechanical
  ├── holographic/  
  │   └── dual.py               # AdS/CFT Engine
  ├── physics/
  │   ├── lambda3_bridge.py     # Stability diagnostics
  │   ├── vorticity.py          # γ decomposition
  │   ├── thermodynamics.py     # Thermal utilities
  │   ├── rdm.py                # 2-RDM analysis
  │   ├──  dislocation_dynamics.py # Dislocation Dynamics
  │   └── topology.py           # Topology Engine 
  ├── interfaces/               # External package interfaces
  │   └── pyscf_interface.py    # PySCF DFT vs DSE comparison
  └── visualization/
      └── prl_figures.py        # PRL publication figures

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)

DOI: 10.5281/zenodo.18095869

Author: Masamichi Iizumi, Tamaki Iizumi
"""

__version__ = "0.6.0"

# =============================================================================
# Core Components
# =============================================================================

# Memory Kernels
from .core.memory_kernel import (
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
    SimpleMemoryKernel
)

# Repulsive Kernel (v0.5.0 - now in memory_kernel.py)
from .core import RepulsiveMemoryKernel

# History Manager (optional)
try:
    from .core import (
        HistoryManager,
        HistoryManagerGPU,
        LambdaDensityCalculator,
        StateSnapshot
    )
except ImportError:
    pass  # Use placeholders from core

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
# Solvers
# =============================================================================

from .solvers.lanczos_memory import (
    MemoryLanczosSolver,
    AdaptiveMemorySolver,
    lanczos_expm_multiply
)

# Optional solvers (may not exist yet)
try:
    from .solvers.time_evolution import (
        TimeEvolutionEngine,
        EvolutionConfig,
        EvolutionResult,
        quick_evolve
    )
except ImportError:
    TimeEvolutionEngine = None
    EvolutionConfig = None
    EvolutionResult = None
    quick_evolve = None

try:
    from .solvers.memory_indicators import (
        MemoryIndicator,
        MemoryMetrics,
        HysteresisAnalyzer
    )
except ImportError:
    MemoryIndicator = None
    MemoryMetrics = None
    HysteresisAnalyzer = None

try:
    from .solvers.chemical_reaction import (
        ChemicalReactionSolver,
        SurfaceHamiltonianEngine,
        LanczosEvolver,
        ReactionEvent,
        ReactionPath,
        PathResult
    )
except ImportError:
    ChemicalReactionSolver = None
    SurfaceHamiltonianEngine = None
    LanczosEvolver = None
    ReactionEvent = None
    ReactionPath = None
    PathResult = None

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

# Thermodynamics
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

# Two-Particle Reduced Density Matrix
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

# Topology (NEW!)
from .physics.topology import (
    TopologyResult,
    ReconnectionEvent,
    EnergyTopologyCorrelation,
    MassGapResult,                    # NEW!
    SpinTopologyCalculator,
    BerryPhaseCalculator,
    ZakPhaseCalculator,
    ReconnectionDetector,
    WavefunctionWindingCalculator,
    StateSpaceWindingCalculator,
    EnergyTopologyCorrelator,
    MassGapCalculator,
    TopologyEngine,
    TopologyEngineExtended,
)

# =============================================================================
# Dislocation Dynamics (optional - requires matplotlib for plots)
# =============================================================================
try:
    from .physics.dislocation_dynamics import (
        Dislocation,              # 転位データ構造
        DislocationDynamics,      # 転位動力学エンジン
        plot_pileup_results,      # パイルアップ結果プロット
        plot_hall_petch_dd,       # Hall-Petch プロット
    )
    HAS_DISLOCATION = True
except ImportError:
    HAS_DISLOCATION = False
    
    def DislocationDynamics(*args, **kwargs):
        raise ImportError("dislocation_dynamics not available")
    def plot_pileup_results(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def plot_hall_petch_dd(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    
    Dislocation = None

# =============================================================================
# Holographic (optional - requires matplotlib)
# =============================================================================
try:
    from .holographic.dual import (
        HolographicDual,
        quick_holographic_analysis,
        # Causality analysis
        transfer_entropy,
        crosscorr_at_lags,
        spearman_corr,
        verify_duality,
        plot_duality_analysis,
    )
    from .holographic.measurement import (
        MeasurementRecord,
        HolographicMeasurementResult,
        HolographicMeasurement,
        quick_holographic_measurement,
    )
    HAS_HOLOGRAPHIC = True
except ImportError:
    HAS_HOLOGRAPHIC = False
    
    def HolographicDual(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def quick_holographic_analysis(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def transfer_entropy(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def crosscorr_at_lags(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def spearman_corr(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def verify_duality(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def plot_duality_analysis(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def HolographicMeasurement(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    def quick_holographic_measurement(*args, **kwargs):
        raise ImportError("matplotlib required: pip install matplotlib")
    
    MeasurementRecord = None
    HolographicMeasurementResult = None

# =============================================================================
#  Engineering Solvers
# =============================================================================

from .engineering.base import (
    EngineeringSolver,
    SolverResult,
    MaterialParams,
    ProcessConditions,
)

from .engineering.thermo_mechanical import (
    ThermoMechanicalSolver,
    ThermoMechanicalResult,
    HeatTreatmentType,
)
  
# =============================================================================
# Interfaces (optional - requires PySCF)
# =============================================================================
try:
    from .interfaces import (
        DSECalculator,
        PathResult as DFTPathResult,
        ComparisonResult,
        GeometryStep,
        MemoryKernelDFT,
        create_h2_stretch_path,
        create_h2_compress_path,
        demo_h2_comparison,
        HAS_PYSCF,
    )
except ImportError:
    HAS_PYSCF = False
    
    def DSECalculator(*args, **kwargs):
        raise ImportError("PySCF required: pip install pyscf")
    def create_h2_stretch_path(*args, **kwargs):
        raise ImportError("PySCF required: pip install pyscf")
    def create_h2_compress_path(*args, **kwargs):
        raise ImportError("PySCF required: pip install pyscf")
    def demo_h2_comparison(*args, **kwargs):
        raise ImportError("PySCF required: pip install pyscf")
    
    DFTPathResult = None
    ComparisonResult = None
    GeometryStep = None
    MemoryKernelDFT = None

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
    'ExclusionKernel',
    'CompositeMemoryKernel',
    'CompositeMemoryKernelGPU',
    'KernelWeights',
    'SimpleMemoryKernel',
    'CatalystMemoryKernel',
    'CatalystEvent',
    'RepulsiveMemoryKernel',
    
    # Core - History
    'HistoryManager',
    'HistoryManagerGPU',
    'LambdaDensityCalculator',
    'StateSnapshot',
    
    # Core - Unified Sparse Engine (v0.5.0)
    'SparseEngine',
    'SystemGeometry',
    'ComputeResult',
    'SparseHamiltonianEngine',
    
    # Core - Backward Compatibility
    'HubbardEngine',
    'HubbardEngineCompat',
    'HubbardResult',
    'SpinOperators',
    'SpinOperatorsCompat',
    'pauli_matrices',
    'create_spin_operators',
    'compute_total_spin',
    'compute_magnetization',
    'compute_correlation',
    'HamiltonianBuilder',
    'build_hamiltonian',
    
    # Core - Lattice
    'LatticeGeometry2D',
    'LatticeGeometry',
    'create_chain',
    'create_ladder',
    'create_square_lattice',
    
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
    
    # Physics - Thermodynamics
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
    
    # Physics - 2-RDM
    'RDM2Result',
    'compute_2rdm',
    'compute_2rdm_with_ops',
    'compute_density_density_correlation',
    'compute_connected_correlation',
    'compute_correlation_matrix',
    'filter_by_distance',
    'from_pyscf_rdm2',
    'to_pyscf_rdm2',

    # Topology (NEW!)
    'TopologyResult',
    'ReconnectionEvent',
    'EnergyTopologyCorrelation',
    'SpinTopologyCalculator',
    'BerryPhaseCalculator',
    'ZakPhaseCalculator',
    'ReconnectionDetector',
    'WavefunctionWindingCalculator',
    'StateSpaceWindingCalculator',
    'EnergyTopologyCorrelator',
    'TopologyEngine',
    'TopologyEngineExtended',

    # Dual
    'HolographicDual',
    'quick_holographic_analysis',
    
    # Causality
    'transfer_entropy',
    'crosscorr_at_lags',
    'spearman_corr',
    'verify_duality',
    'plot_duality_analysis',
    
    # Measurement Protocol
    'MeasurementRecord',
    'HolographicMeasurementResult',
    'HolographicMeasurement',
    'quick_holographic_measurement',
    'HAS_HOLOGRAPHIC',

    # EngineeringSolver
    'EngineeringSolver',
    'SolverResult',
    'MaterialParams',
    'ProcessConditions',
    
    # Thermo-Mechanical
    'ThermoMechanicalSolver',
    'ThermoMechanicalResult',
    'HeatTreatmentType',

    # Dislocation Dynamics
    'Dislocation',
    'DislocationDynamics',
    'plot_pileup_results',
    'plot_hall_petch_dd',
    
    # Interfaces - PySCF
    'HAS_PYSCF',
    'DSECalculator',
    'DFTPathResult',
    'ComparisonResult',
    'GeometryStep',
    'MemoryKernelDFT',
    'create_h2_stretch_path',
    'create_h2_compress_path',
    'demo_h2_comparison',
    
    # Visualization
    'HAS_VISUALIZATION',
    'fig1_gamma_decomposition',
    'fig2_path_evolution',
    'fig3_memory_comparison',
    'generate_all_prl_figures',
    'PRL_COLORS',
]
