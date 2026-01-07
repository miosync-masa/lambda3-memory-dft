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
