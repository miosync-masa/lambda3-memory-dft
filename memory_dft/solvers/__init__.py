# =============================================================================
# memory_dft/solvers/__init__.py
# =============================================================================

"""
Memory-DFT Solvers Module
=========================

DSE (Direct Schrödinger Evolution) ソルバーとメモリ指標

【DSE の本質】
  標準量子力学: iℏ ∂ψ/∂t = H ψ（Markovian）
  DSE:         iℏ ∂ψ/∂t = H ψ + ∫ K(t-τ) F[ψ(τ)] dτ（Non-Markovian）

【使用例】
  from memory_dft.solvers import DSESolver, DSEResult, MemoryIndicator
  
  solver = DSESolver(H_K, H_V, gamma_memory=1.2, eta=0.1)
  result1 = solver.run(psi0, t_end=10.0)
  result2 = solver.run(psi0_alt, t_end=10.0)
  
  metrics = MemoryIndicator.from_dse_results(result1, result2)
  print(metrics.summary())

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from .thermal_holographic import (
    ThermalHolographicEvolution,
    CoolingMode,
    ThermalPath,
    info,
    # 物理定数
    TAU_0,
    LAMBDA_LIGHT,
    LAMBDA_PHONON,
    SCALE_RATIO,
)

from .dse_solver import (
    # Main solver
    DSESolver,
    
    # Result container
    DSEResult,
    
    # Utility
    lanczos_expm_multiply,
    quick_dse,
)

from .memory_indicators import (
    # Metrics container
    MemoryMetrics,
    
    # Indicator calculator
    MemoryIndicator,
    
    # Hysteresis analysis
    HysteresisAnalyzer,
)

__all__ = [
    # Main class
    "ThermalHolographicEvolution",
    
    # Data classes
    "ThermalHolographicRecord",
    "ThermalHolographicResult",
    "ThermalPath",
    "DualityMetrics",
    "FailurePrediction",
    
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
  
    # Solver
    'DSESolver',
    'DSEResult',
    'lanczos_expm_multiply',
    'quick_dse',
    
    # Memory indicators
    'MemoryMetrics',
    'MemoryIndicator',
    'HysteresisAnalyzer',
]
