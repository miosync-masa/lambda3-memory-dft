"""
Memory-DFT: Density Functional Theory with Memory
=================================================

H-CSP/Λ³理論に基づく履歴依存密度汎関数理論

理論的背景:
- γ_total = γ_local + γ_memory
- Memory kernel = Σ w_i K_i (H-CSP環境階層)
- 非Markov量子力学の密度汎関数実装

Structure:
  memory_dft/
  ├── core/
  │   ├── memory_kernel.py      # 3階層Kernel (field/phys/chem)
  │   ├── history_manager.py    # 履歴保持 + Λ重み付け
  │   └── sparse_engine.py      # CuPy + Sparse 基盤
  ├── solvers/
  │   ├── lanczos_memory.py     # Lanczos + Memory項
  │   └── time_evolution.py     # 時間発展エンジン
  ├── physics/
  │   ├── lambda3_bridge.py     # Λ³理論との接続
  │   └── vorticity.py          # γ計算（PySCF連携）
  └── tests/
      └── test_h2_memory.py     # H2分子での検証

Author: Masamichi Iizumi, Tamaki Iizumi
Based on: Λ³/H-CSP Theory v2.0

🩲→🧪→Λ³
"""

__version__ = "0.1.0"
__author__ = "Masamichi Iizumi, Tamaki Iizumi"

# Core components
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

from .core.sparse_engine import (
    SparseHamiltonianEngine,
    SystemGeometry
)

# Solvers
from .solvers.lanczos_memory import (
    MemoryLanczosSolver,
    AdaptiveMemorySolver,
    lanczos_expm_multiply
)

from .solvers.time_evolution import (
    TimeEvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
    quick_evolve
)

# Physics
from .physics.lambda3_bridge import (
    Lambda3Calculator,
    LambdaState,
    StabilityPhase,
    HCSPValidator
)

from .physics.vorticity import (
    VorticityCalculator,
    VorticityResult,
    GammaExtractor,
    MemoryKernelFromGamma
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
    # Sparse Engine
    'SparseHamiltonianEngine',
    'SystemGeometry',
    # Solvers
    'MemoryLanczosSolver',
    'AdaptiveMemorySolver',
    'lanczos_expm_multiply',
    'TimeEvolutionEngine',
    'EvolutionConfig',
    'EvolutionResult',
    'quick_evolve',
    # Physics
    'Lambda3Calculator',
    'LambdaState',
    'StabilityPhase',
    'HCSPValidator',
    'VorticityCalculator',
    'VorticityResult',
    'GammaExtractor',
    'MemoryKernelFromGamma',
]
