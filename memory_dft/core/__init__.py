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
    HubbardAndersonGeometry,  # ← 追加
    
    # Factory functions
    create_chain,
    create_ladder,
    create_square_lattice,
    
    # Result container
    ComputeResult,
    
    # Hubbard-Anderson  # ← 追加
    HubbardAndersonParams,
    LayerEnergies,
    LayerLambda,
)

# Material Failure Analysis
from .material_failure import (
    # Data classes
    TopologyResult,
    ThermalTopologyResult,
    StressTopologyResult,
    FailurePrediction,
    
    # Analyzers
    ThermalTopologyAnalyzer,
    StressTopologyAnalyzer,
    CombinedFailureAnalyzer,
    
    # Test suite
    DSETopologyTest,
    LocalThermalEnsemble,
    
    # Utilities
    get_xp,
    to_device,
    to_host,
    eigsh_wrapper,
)

from .memory_kernel import (
    MemoryKernel,
    MemoryKernelConfig,
    HistoryEntry,
)

from .history_manager import (
    StateSnapshot,
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
)



from .environment_operators import (
    # =========================================================================
    # v2.0 NEW: ThermalEnsemble (正しい有限温度計算)
    # =========================================================================
    ThermalEnsemble,
    ThermalObservable,
    
    # =========================================================================
    # Observable functions (for ThermalEnsemble.register_observable)
    # =========================================================================
    compute_winding_number,
    compute_phase_entropy,
    compute_vorticity,
    
    # =========================================================================
    # Environment Builder (場の効果のみ)
    # =========================================================================
    EnvironmentBuilder,
    StressOperator,
    Dislocation,
    
    # =========================================================================
    # Thermodynamic utilities (これらは正しい)
    # =========================================================================
    T_to_beta,
    beta_to_T,
    thermal_energy,
    boltzmann_weights,
    partition_function,
    compute_entropy,
    compute_free_energy,
    compute_heat_capacity,
    
    # =========================================================================
    # Physical constants
    # =========================================================================
    K_B_EV,
    K_B_J,
    H_EV,
    HBAR_EV,
    # =========================================================================
    # re-export
    # =========================================================================
    SparseEngine,
    HubbardAndersonGeometry,
    HubbardAndersonParams,
    LayerEnergies,
    LayerLambda,
    
    # =========================================================================
    # DEPRECATED (後方互換性のため残す、警告が出る)
    # =========================================================================
    TemperatureOperator,  # ⚠️ DEPRECATED: Use ThermalEnsemble instead
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
    "HubbardAndersonGeometry",
    "HubbardAndersonParams",
    "LayerEnergies",
    "LayerLambda"
    
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
]
