"""
Memory-DFT Physics Components
=============================

Physical analysis and diagnostic tools for Memory-DFT.

Modules:
  - lambda3_bridge: Stability diagnostics and validation
  - vorticity: Correlation decomposition and analysis
  - thermodynamics: Finite-temperature utilities
  - rdm: Two-particle reduced density matrix
  - topology: Topological invariants and reconnection detection

Author: Masamichi Iizumi, Tamaki Iizumi
"""

# Stability Diagnostics
from .lambda3_bridge import (
    Lambda3Calculator,
    LambdaState,
    StabilityPhase,
    HCSPValidator,
    map_kernel_to_environment
)

from .vorticity import (
    VorticityCalculator,
    VorticityResult,
    GammaExtractor,
    compute_orbital_distance_matrix,
)

from .rdm import (
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
from .topology import (
    # Result containers
    TopologyResult,
    ReconnectionEvent,
    EnergyTopologyCorrelation,
    MassGapResult,                    # NEW!
    
    # Spin topology
    SpinTopologyCalculator,
    
    # Berry phase
    BerryPhaseCalculator,
    
    # Zak phase (1D)
    ZakPhaseCalculator,
    
    # Reconnection detection
    ReconnectionDetector,
    
    # Wavefunction phase winding (NEW!)
    WavefunctionWindingCalculator,
    
    # State-space winding (NEW!)
    StateSpaceWindingCalculator,
    
    # Energy-Topology correlator (NEW!)
    EnergyTopologyCorrelator,
    
    # Mass Gap Calculator - E = mc² derivation (NEW!)
    MassGapCalculator,
    
    # Unified engines
    TopologyEngine,
    TopologyEngineExtended,
)

__all__ = [
    # Lambda3 / Stability
    'Lambda3Calculator',
    'LambdaState',
    'StabilityPhase',
    'HCSPValidator',
    'map_kernel_to_environment',
    
    # Vorticity
    'VorticityCalculator',
    'VorticityResult',
    'GammaExtractor',
    'compute_orbital_distance_matrix',
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
    'MassGapResult',
    'SpinTopologyCalculator',
    'BerryPhaseCalculator',
    'ZakPhaseCalculator',
    'ReconnectionDetector',
    'WavefunctionWindingCalculator',
    'StateSpaceWindingCalculator',
    'EnergyTopologyCorrelator',
    'MassGapCalculator',
    'TopologyEngine',
    'TopologyEngineExtended',
]
