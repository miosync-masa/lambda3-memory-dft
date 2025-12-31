"""
Memory-DFT Physics Components
=============================

Physical analysis and diagnostic tools for Memory-DFT.

Modules:
  - lambda3_bridge: Λ stability diagnostics and H-CSP validation
  - vorticity: γ decomposition and memory kernel extraction
  - thermodynamics: Finite-temperature utilities (NEW)

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
]
