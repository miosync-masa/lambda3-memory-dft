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

# Vorticity and Correlation
from .vorticity import (
    VorticityCalculator,
    VorticityResult,
    GammaExtractor,
    MemoryKernelFromGamma
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

# Two-Particle Reduced Density Matrix
from .rdm import (
    # Result container
    RDM2Result,
    
    # Core computation
    compute_2rdm,
    compute_2rdm_with_ops,
    
    # Correlation analysis
    compute_density_density_correlation,
    compute_connected_correlation,
    compute_correlation_matrix,
    filter_by_distance,
    
    # External interface
    from_pyscf_rdm2,
    to_pyscf_rdm2,
)

# Topology (NEW!)
from .topology import (
    # Result containers
    TopologyResult,
    ReconnectionEvent,
    EnergyTopologyCorrelation,
    
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
    'MemoryKernelFromGamma',
    
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
    
    # 2-RDM
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
]
