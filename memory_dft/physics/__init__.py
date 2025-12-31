"""
Memory-DFT Physics Components
=============================

Physical analysis and diagnostic tools for Memory-DFT.

Modules:
  - lambda3_bridge: Stability diagnostics and validation
  - vorticity: Correlation decomposition and analysis
  - thermodynamics: Finite-temperature utilities
  - rdm: Two-particle reduced density matrix (NEW)

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

# Thermodynamics (NEW - from thermal_dse.py)
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
)

# Two-Particle Reduced Density Matrix (NEW)
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
    
    # 2-RDM (NEW)
    'RDM2Result',
    'compute_2rdm',
    'compute_2rdm_with_ops',
    'compute_density_density_correlation',
    'compute_connected_correlation',
    'compute_correlation_matrix',
    'filter_by_distance',
    'from_pyscf_rdm2',
    'to_pyscf_rdm2',
]
