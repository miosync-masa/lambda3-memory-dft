"""
Memory-DFT: Density Functional Theory with Memory
=================================================

H-CSP/Λ³理論に基づく履歴依存密度汎関数理論

理論的背景:
- γ_total = γ_local + γ_memory
- ED距離分解により導出:
    γ_total (r=∞) = 2.604
    γ_local (r≤2) = 1.388  ← Markovian (Lie & Fullwood PRL 2025)
    γ_memory      = 1.216  ← Non-Markovian extension (46.7%)
- Memory kernel = Σ w_i K_i (H-CSP環境階層)
- 非Markov量子力学の密度汎関数実装

Key Results:
- Path dependence: 22.84x amplification
- Catalyst history: Standard QM |ΔΛ|=0, Memory-DFT |ΔΛ|=51.07
- 46.7% of correlations require Memory kernel!

Structure:
  memory_dft/
  ├── core/
  │   ├── memory_kernel.py      # 3階層Kernel (field/phys/chem) + Catalyst
  │   ├── history_manager.py    # 履歴保持 + Λ重み付け
  │   ├── sparse_engine.py      # CuPy + Sparse 基盤
  │   └── hubbard_engine.py     # Hubbard model for chemical tests
  ├── solvers/
  │   ├── lanczos_memory.py     # Lanczos + Memory項
  │   └── time_evolution.py     # 時間発展エンジン
  ├── physics/
  │   ├── lambda3_bridge.py     # Λ³理論との接続
  │   └── vorticity.py          # γ計算（ED距離フィルター）
  └── tests/
      ├── test_h2_memory.py     # H2分子での検証
      └── test_chemical.py      # 化学変化テスト (A/B/C/D)

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)
  "Quantum States Over Time are Uniquely Represented by a CPTP Map"

Author: Masamichi Iizumi, Tamaki Iizumi
Based on: Λ³/H-CSP Theory v2.0

🩲→🧪→Λ³
"""

__version__ = "0.2.0"
__author__ = "Masamichi Iizumi, Tamaki Iizumi"

# Core components
from .core.memory_kernel import (
    PowerLawKernel,
    StretchedExpKernel,
    StepKernel,
    CompositeMemoryKernel,
    CompositeMemoryKernelGPU,
    KernelWeights,
    CatalystMemoryKernel,
    CatalystEvent,
    SimpleMemoryKernel
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

from .core.hubbard_engine import (
    HubbardEngine,
    HubbardResult
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
    'CatalystMemoryKernel',
    'CatalystEvent',
    'SimpleMemoryKernel',
    
    # History
    'HistoryManager',
    'HistoryManagerGPU',
    'LambdaDensityCalculator',
    'StateSnapshot',
    
    # Sparse Engine
    'SparseHamiltonianEngine',
    'SystemGeometry',
    
    # Hubbard Engine
    'HubbardEngine',
    'HubbardResult',
    
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
