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

# Thermodynamics
from .thermodynamics import (
    # Constants
    K_B_EV,
    K_B_J,
    H_EV,
    HBAR_EV,
    
    # Temperature conversion
    T_to_beta,
    beta_to_T,
    thermal_energy,
    
    # Boltzmann statistics
    boltzmann_weights,
    partition_function,
    
    # Thermal expectation values
    thermal_expectation,
    thermal_expectation_zero_T,
    thermal_average_energy,
    thermal_energy_variance,
    
    # Thermodynamic quantities
    compute_entropy,
    compute_free_energy,
    compute_heat_capacity,
    
    # Thermal states
    thermal_density_matrix,
    sample_thermal_state,
    
    # Temperature-dependent Hamiltonian H(T)
    TemperatureDependentHamiltonian,
    ThermalPathEvolver,
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
    
    # Thermodynamics - Constants
    'K_B_EV',
    'K_B_J',
    'H_EV',
    'HBAR_EV',
    
    # Thermodynamics - Temperature
    'T_to_beta',
    'beta_to_T',
    'thermal_energy',
    
    # Thermodynamics - Boltzmann
    'boltzmann_weights',
    'partition_function',
    
    # Thermodynamics - Expectation
    'thermal_expectation',
    'thermal_expectation_zero_T',
    'thermal_average_energy',
    'thermal_energy_variance',
    
    # Thermodynamics - Quantities
    'compute_entropy',
    'compute_free_energy',
    'compute_heat_capacity',
    
    # Thermodynamics - States
    'thermal_density_matrix',
    'sample_thermal_state',
    
    # Thermodynamics - H(T)
    'TemperatureDependentHamiltonian',
    'ThermalPathEvolver',
    
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
