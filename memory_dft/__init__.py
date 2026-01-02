"""
Direct Schrödinger Evolution (DSE)
==================================

A framework for history-dependent quantum dynamics through
direct solution of the Schrödinger equation.

DFT erases history. DSE remembers.

Key Insight:
  Standard DFT: E[ρ(r)]        - Same structure = Same energy
  DSE:          E[ψ(t)]        - Different history = Different energy

Theoretical Background:
  Correlation decomposition by distance filtering:
    γ_total (r=∞) = 2.604   ← Full correlations
    γ_local (r≤2) = 1.388   ← Markovian sector
    γ_memory      = 1.216   ← Non-Markovian (46.7%)

  This shows that nearly half of quantum correlations
  require history-dependent treatment.

Key Results:
  - gamma_memory = 1.216 (46.7% of correlations are non-Markovian)
  - Path dependence: 1.59 (adsorption order matters)
  - Reaction sequence: 2.18 (A->B->C differs from A->C->B)
  - Thermal path dependence: 2.26 (heating/cooling history matters)
  - Standard DFT cannot distinguish these paths (history-blind)

v0.5.0 Changes:
  - Unified SparseEngine consolidation
  - operators.py, hamiltonian.py, hubbard_engine.py → sparse_engine_unified.py
  - All models (Heisenberg, Ising, XY, Hubbard, Kitaev) in one place
  - GPU/CPU automatic backend selection

Structure (Refactored):
  memory_dft/
  ├── cli/                      # Command-line interface
  │   ├── __init__.py           # Typer app & command registration
  │   ├── utils.py              # Shared CLI utilities
  │   └── commands/             # Individual command modules
  ├── core/
  │   ├── memory_kernel.py      # 4-layer kernel (field/phys/chem/exclusion)
  │   ├── history_manager.py    # History tracking
  │   └── parse_engine_unified.py  # Unified sparse engine (v0.5.0)
  ├── solvers/
  │   ├── lanczos_memory.py     # Lanczos + memory
  │   ├── time_evolution.py     # Time evolution
  │   ├── memory_indicators.py  # Memory quantification
  │   └── chemical_reaction.py  # Surface chemistry solver
  ├── physics/
  │   ├── lambda3_bridge.py     # Stability diagnostics
  │   ├── vorticity.py          # γ decomposition
  │   ├── thermodynamics.py     # Thermal utilities
  │   └── rdm.py                # 2-RDM analysis
  ├── interfaces/               # External package interfaces
  │   └── pyscf_interface.py    # PySCF DFT vs DSE comparison
  └── visualization/
      └── prl_figures.py        # PRL publication figures

__version__ = "0.5.0"

Foundation modules for Memory-DFT simulations.

Memory Kernel Components (v0.4.0):
  1. Field (PowerLaw): Long-range correlations
  2. Phys (StretchedExp): Structural relaxation  
  3. Chem (Step): Irreversible reactions
  4. Exclusion: Distance-direction memory

The same distance r = 0.8 A means DIFFERENT things:
  - Approaching: system is being compressed
  - Departing: system is recovering
DFT cannot distinguish. DSE can!

Modules:
  - memory_kernel: Non-Markovian memory kernels (4 components)
  - repulsive_kernel: Detailed compression history tracking
  - history_manager: State history tracking
  - sparse_engine_unified: Unified sparse Hamiltonian engine (GPU/CPU)
  - lattice: Lattice geometry definitions

v0.5.0: Unified SparseEngine consolidation
  - operators.py, hamiltonian.py, hubbard_engine.py → sparse_engine_unified.py
  - All models (Heisenberg, Ising, XY, Hubbard, Kitaev) in one place
  - GPU/CPU automatic backend selection

Reference:
  Lie & Fullwood, PRL 135, 230204 (2025)

DOI: 10.5281/zenodo.18095869

Author: Masamichi Iizumi, Tamaki Iizumi
"""

# Memory Kernels (4 components: field, phys, chem, exclusion)
from .memory_kernel import (
    MemoryKernelBase,
    PowerLawKernel,
    StretchedExpKernel,
    StepKernel,
    ExclusionKernel,
    CompositeMemoryKernel,
    CompositeMemoryKernelGPU,
    KernelWeights,
    CatalystMemoryKernel,
    CatalystEvent,
    SimpleMemoryKernel,
    RepulsiveMemoryKernel  # v0.5.0: moved to memory_kernel.py
)

# History Manager
from .history_manager import (
    HistoryManager,
    HistoryManagerGPU,
    LambdaDensityCalculator,
    StateSnapshot
)

# =========================================================================
# Unified Sparse Engine (v0.5.0)
# Replaces: operators.py, hamiltonian.py, hubbard_engine.py, sparse_engine.py
# =========================================================================

from .sparse_engine_unified import (
    # Core class
    SparseEngine,
    
    # Data classes
    SystemGeometry,
    ComputeResult,
    
    # Backward compatibility aliases
    SparseHamiltonianEngine,  # = SparseEngine
    HubbardResult,            # = ComputeResult
    SpinOperatorsCompat,      # SpinOperators-like interface
    HubbardEngineCompat,      # HubbardEngine-like interface
    HubbardEngine,            # = HubbardEngineCompat
)

# =========================================================================
# Lattice Geometry (now in sparse_engine_unified)
# =========================================================================

from .sparse_engine_unified import (
    LatticeGeometry2D,
    LatticeGeometry,
    create_chain,
    create_ladder,
    create_square_lattice,
)

# =========================================================================
# Backward Compatibility Aliases
# =========================================================================

# For code that imports SpinOperators
SpinOperators = SpinOperatorsCompat

# HubbardEngine is already imported as HubbardEngineCompat from sparse_engine_unified
# (Do NOT override here!)

# For code that imports HamiltonianBuilder
# Usage: HamiltonianBuilder(lattice, ops) → engine.build_heisenberg(bonds)
class HamiltonianBuilder:
    """
    Backward compatibility wrapper for HamiltonianBuilder.
    
    New code should use SparseEngine directly:
        engine = SparseEngine(n_sites)
        H = engine.build_heisenberg(bonds, J=1.0)
    """
    def __init__(self, lattice, ops):
        self._engine = ops._engine if hasattr(ops, '_engine') else \
                       SparseEngine(lattice.N_spins, use_gpu=False, verbose=False)
        self._bonds = lattice.bonds_nn if hasattr(lattice, 'bonds_nn') else lattice.bonds
    
    def heisenberg(self, J=1.0, Jz=None):
        return self._engine.build_heisenberg(self._bonds, J=J, Jz=Jz, split_KV=False)
    
    def xy(self, J=1.0):
        return self._engine.build_xy(self._bonds, J=J)
    
    def ising(self, J=1.0, h=0.0):
        return self._engine.build_ising(self._bonds, J=J, h=h, split_KV=False)
    
    def kitaev_rect(self, Kx=1.0, Ky=0.8, Kz_diag=0.5):
        return self._engine.build_kitaev_rect(Kx=Kx, Ky=Ky, Kz_diag=Kz_diag)


# Helper functions (backward compatibility)
def pauli_matrices():
    """Return Pauli matrices (σx, σy, σz)."""
    import numpy as np
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)
    return sx, sy, sz


def pauli_matrices_full():
    """Return full Pauli matrices (I, σx, σy, σz, σ+, σ-)."""
    import numpy as np
    I = np.eye(2, dtype=np.complex128)
    sx, sy, sz = pauli_matrices()
    sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    return I, sx, sy, sz, sp, sm


def create_spin_operators(n_sites, use_gpu=False):
    """Create spin operators for n_sites."""
    return SpinOperatorsCompat(n_sites, use_gpu=use_gpu)


def build_hamiltonian(model, lattice, ops, **kwargs):
    """Build Hamiltonian for given model."""
    builder = HamiltonianBuilder(lattice, ops)
    if model == 'heisenberg':
        return builder.heisenberg(**kwargs)
    elif model == 'xy':
        return builder.xy(**kwargs)
    elif model == 'ising':
        return builder.ising(**kwargs)
    elif model == 'kitaev':
        return builder.kitaev_rect(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")


def compute_total_spin(psi, ops):
    """Compute total spin ⟨S²⟩."""
    if hasattr(ops, '_engine'):
        return ops._engine.compute_total_spin(psi)
    raise ValueError("ops must be SpinOperatorsCompat")


def compute_magnetization(psi, ops):
    """Compute magnetization ⟨Sz⟩/N."""
    if hasattr(ops, '_engine'):
        return ops._engine.compute_magnetization(psi)
    raise ValueError("ops must be SpinOperatorsCompat")


def compute_correlation(psi, ops, i, j, component='Z'):
    """Compute spin-spin correlation ⟨Si·Sj⟩."""
    if hasattr(ops, '_engine'):
        return ops._engine.compute_correlation(psi, i, j, component)
    raise ValueError("ops must be SpinOperatorsCompat")


# =========================================================================
# __all__ for explicit exports
# =========================================================================

__all__ = [
    # Memory Kernels (4 components)
    'MemoryKernelBase',
    'PowerLawKernel',
    'StretchedExpKernel',
    'StepKernel',
    'ExclusionKernel',
    'CompositeMemoryKernel',
    'CompositeMemoryKernelGPU',
    'KernelWeights',
    'CatalystMemoryKernel',
    'CatalystEvent',
    'SimpleMemoryKernel',
    
    # Repulsive Kernel
    'RepulsiveMemoryKernel',
    'CompressionEvent',
    'ExtendedCompositeKernel',
    
    # History Manager
    'HistoryManager',
    'HistoryManagerGPU',
    'LambdaDensityCalculator',
    'StateSnapshot',
    
    # Unified Sparse Engine (v0.5.0)
    'SparseEngine',
    'SystemGeometry',
    'ComputeResult',
    
    # Backward compatibility (Sparse Engine)
    'SparseHamiltonianEngine',
    'HubbardEngine',
    'HubbardEngineCompat',
    'HubbardResult',
    
    # Lattice Geometry
    'LatticeGeometry2D',
    'LatticeGeometry',
    'create_chain',
    'create_ladder',
    'create_square_lattice',
    
    # Backward compatibility (Operators/Hamiltonian)
    'SpinOperators',
    'SpinOperatorsCompat',
    'pauli_matrices',
    'pauli_matrices_full',
    'create_spin_operators',
    'compute_total_spin',
    'compute_magnetization',
    'compute_correlation',
    'HamiltonianBuilder',
    'build_hamiltonian',
]
