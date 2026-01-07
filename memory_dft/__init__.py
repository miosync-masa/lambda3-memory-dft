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
  ├── core/
  │   ├── memory_kernel.py      # Memory kernel
  │   ├── history_manager.py    # History tracking
  │   ├── environment_operators.py  
  │   └── sparse_engine_unified.py  # Unified sparse engine (v0.5.0)
  ├── solvers/
  │   ├── dse_solver.py     # Lanczos + memory
  │   ├── memory_indicators.py  # Memory quantification
  │   └── chemical_reaction.py  # Surface chemistry solver
  ├── engineering/              
  │   └──thermo_mechanical.py   # ThermoMechanical
  ├── holographic/ 
  │   ├── measurement.py        # measurement
  │   └── dual.py               # AdS/CFT Engine
  ├── physics/
  │   ├── lambda3_bridge.py     # Stability diagnostics
  │   ├── vorticity.py          # γ decomposition
  │   ├── thermodynamics.py     # Thermal utilities
  │   ├── rdm.py                # 2-RDM analysis
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

__version__ = "1.0.0"

# =============================================================================
# Core Components
# =============================================================================
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

# =============================================================================
# Solvers
# =============================================================================

from .solvers.dse_solver import (
    # Main solver
    DSESolver,
    
    # Result container
    DSEResult,
    
    # Utility
    lanczos_expm_multiply,
    quick_dse,
)

from .solvers.memory_indicators import (
    # Metrics container
    MemoryMetrics,
    
    # Indicator calculator
    MemoryIndicator,
    
    # Hysteresis analysis
    HysteresisAnalyzer,
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
    compute_orbital_distance_matrix,
)

from .physics.rdm import (
    RDMCalculator,
    RDM2Result,
    SystemType,
    HubbardRDM,
    HeisenbergRDM,
    PySCFRDM,
    get_rdm_calculator,
    compute_rdm2,
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
# Interfaces
# =============================================================================
try:
    from .interfaces.pyscf_interface import (
        DSECalculator,
        GeometryStep,
        SinglePointResult,
        PathResult,
        ComparisonResult,
        create_h2_stretch_path,
        create_h2_compress_path,
        create_cyclic_path,
        HAS_PYSCF,
    )
except ImportError:
    HAS_PYSCF = False

# =============================================================================
# __all__
# =============================================================================

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

    # Solver
    'DSESolver',
    
    # Result
    'DSEResult',
    
    # Utility
    'lanczos_expm_multiply',
    'quick_dse',
  
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

    # RDM
    'RDMCalculator',
    'RDM2Result',
    'SystemType',
    'HubbardRDM',
    'HeisenbergRDM',
    'PySCFRDM',
    'get_rdm_calculator',
    'compute_rdm2',
    
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
    
    # Interfaces - PySCF
    'DSECalculator',
    'GeometryStep',
    'SinglePointResult',
    'PathResult',
    'ComparisonResult',
    'create_h2_stretch_path',
    'create_h2_compress_path',
    'create_cyclic_path',
    'HAS_PYSCF',
]
