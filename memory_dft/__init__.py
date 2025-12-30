"""
Memory-DFT: Density Functional Theory with Memory
=================================================

H-CSP/Λ³理論に基づく履歴依存密度汎関数理論

理論的背景:
- γ_total = γ_local + γ_memory
- Memory kernel = Σ w_i K_i (H-CSP環境階層)
- 非Markov量子力学の密度汎関数実装

Author: Masamichi Iizumi, Tamaki Iizumi
Based on: Λ³/H-CSP Theory v2.0

🩲→🧪→Λ³
"""

__version__ = "0.1.0"
__author__ = "Masamichi Iizumi, Tamaki Iizumi"

from .core.memory_kernel import (
    PowerLawKernel,
    StretchedExpKernel,
    StepKernel,
    CompositeMemoryKernel,
    CompositeMemoryKernelGPU,
    KernelWeights
)

from .core.history_manager import (
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
    StateSnapshot
)

from .solvers.lanczos_memory import (
    MemoryLanczosSolver,
    AdaptiveMemorySolver,
    lanczos_expm_multiply
)

__all__ = [
    # Kernels
    'PowerLawKernel',
    'StretchedExpKernel', 
    'StepKernel',
    'CompositeMemoryKernel',
    'CompositeMemoryKernelGPU',
    'KernelWeights',
    # History
    'HistoryManager',
    'HistoryManagerGPU',
    'LambdaDensityCalculator',
    'StateSnapshot',
    # Solvers
    'MemoryLanczosSolver',
    'AdaptiveMemorySolver',
    'lanczos_expm_multiply',
]
