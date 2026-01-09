"""
Direct Schrödinger Evolution (DSE)
==================================

Author: Masamichi Iizumi, Tamaki Iizumi
"""

__version__ = "1.0.0"

# =============================================================================
# Core Components
# =============================================================================
from .core.sparse_engine_unified import (
    # Main engine
    SparseEngine,
    
    # Geometry
    SystemGeometry,
    LatticeGeometry2D,
    HubbardAndersonGeometry, 
    
    # Factory functions
    create_chain,
    create_ladder,
    create_square_lattice,
    # Hubbard-Anderson
    HubbardAndersonParams,
    LayerEnergies,
    LayerLambda,
    # Result container
    ComputeResult,
)

from .core.memory_kernel import (
    MemoryKernel,
    MemoryKernelConfig,
    HistoryEntry,
)

from .core.history_manager import (
    StateSnapshot,
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
)

from .core.environment_operators import (
    # 正しい有限温度計算
    ThermalEnsemble,
    ThermalObservable,
    
    # Observables
    compute_winding_number,
    compute_phase_entropy,
    compute_vorticity,
    
    # 場の効果
    EnvironmentBuilder,
    StressOperator,
    Dislocation,
    
    # Utilities
    T_to_beta,
    beta_to_T,
    thermal_energy,
    boltzmann_weights,
    partition_function,
    compute_entropy,
    compute_free_energy,
    compute_heat_capacity,
)

from .core.material_failure import (
    ThermalTopologyResult,
    StressTopologyResult,
    FailurePrediction,
    ThermalTopologyAnalyzer,
    StressTopologyAnalyzer,
    CombinedFailureAnalyzer,
    DSETopologyTest,
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

from .solvers.thermal_holographic import (
    # Main class
    ThermalHolographicEvolution,
    
    # Data classes
    ThermalHolographicRecord,   # ← 追加
    ThermalHolographicResult,   # ← 追加
    ThermalPath,
    DualityMetrics,             # ← 追加
    # FailurePrediction,        # material_failure と被るから注意
    
    # Enums
    CoolingMode,
    TopologyState,              # ← 追加
    
    # Constants
    TAU_0,
    C_LIGHT,                    # ← 追加
    V_SOUND,                    # ← 追加
    LAMBDA_LIGHT,
    LAMBDA_PHONON,
    SCALE_RATIO,
    
    # Utility
    info,
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

    # HubbardAnderson
    "HubbardAndersonGeometry",
    "HubbardAndersonParams",
    "LayerEnergies",
    "LayerLambda",

    # Data Classes
    'TopologyResult',
    'ThermalTopologyResult',
    'StressTopologyResult',
    'FailurePrediction',
    
    # Analyzers
    'ThermalTopologyAnalyzer',
    'StressTopologyAnalyzer',
    'CombinedFailureAnalyzer',
    
    # Test Suite
    'DSETopologyTest',
    'LocalThermalEnsemble',
    
    # Utilities
    'get_xp',
    'to_device',
    'to_host',
    'eigsh_wrapper',
    
    # Constants
    'HAS_CUPY',
    'HAS_ENV_OPS',
    'k_B',
    
    # Result
    'ComputeResult',
    
    # Memory
    'MemoryKernel',
    'MemoryKernelConfig',
    'HistoryEntry',

    # HistoryManager
    "StateSnapshot",
    "HistoryManager",
    "HistoryManagerGPU",
    "LambdaDensityCalculator",

    # v2.0 Core
    'ThermalEnsemble',
    'ThermalObservable',
    
    # Observables
    'compute_winding_number',
    'compute_phase_entropy',
    'compute_vorticity',
    
    # Environment (場のみ)
    'EnvironmentBuilder',
    'StressOperator',
    'Dislocation',
    
    # Thermodynamic utilities
    'T_to_beta',
    'beta_to_T',
    'thermal_energy',
    'boltzmann_weights',
    'partition_function',
    'compute_entropy',
    'compute_free_energy',
    'compute_heat_capacity',
    
    # Constants
    'K_B_EV',
    'K_B_J',
    'H_EV',
    'HBAR_EV',
    
    # Deprecated
    'TemperatureOperator',

    # Solver
    'DSESolver',
    
    # Result
    'DSEResult',
    
    # Utility
    'lanczos_expm_multiply',
    'quick_dse',

    # Main class
    "ThermalHolographicEvolution",
    
    # Data classes
    "ThermalHolographicRecord",
    "ThermalHolographicResult",
    "ThermalPath",
    "DualityMetrics",
    
    # Enums
    "CoolingMode",
    "TopologyState",
    
    # Constants
    "TAU_0",
    "C_LIGHT",
    "V_SOUND",
    "LAMBDA_LIGHT",
    "LAMBDA_PHONON",
    "SCALE_RATIO",
    
    # Utility
    "info",
  
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
