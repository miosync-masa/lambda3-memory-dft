"""
Memory-DFT Core Components
==========================

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
    SimpleMemoryKernel
)

# Repulsive Kernel (DEPRECATED in v0.5.0 - will be removed)
# Use ExclusionKernel from memory_kernel.py instead
HAS_REPULSIVE_KERNEL = False

from dataclasses import dataclass

@dataclass
class CompressionEvent:
    """Record of a compression event."""
    time: float
    r_min: float
    pressure: float
    site: int = 0


class RepulsiveMemoryKernel:
    """
    DEPRECATED: Use ExclusionKernel from memory_kernel.py instead.
    
    This is a minimal placeholder for backward compatibility.
    """
    def __init__(self, eta_rep: float = 0.2, tau_rep: float = 3.0,
                 tau_recover: float = 10.0, r_critical: float = 0.8,
                 n_power: float = 12.0):
        self.eta_rep = eta_rep
        self.tau_rep = tau_rep
        self.tau_recover = tau_recover
        self.r_critical = r_critical
        self.n_power = n_power
        self.compression_history = []
        self.state_history = []
    
    def add_state(self, t, r, psi=None):
        self.state_history.append((t, r, psi))
    
    def compute_effective_repulsion(self, r, t):
        return 1.0 / (r ** self.n_power)
    
    def compute_repulsion_enhancement(self, t, r):
        return 0.0
    
    def clear(self):
        self.compression_history = []
        self.state_history = []


class ExtendedCompositeKernel(CompositeMemoryKernel):
    """
    DEPRECATED: Use CompositeMemoryKernel directly.
    """
    def __init__(self, w_field=0.30, w_phys=0.25, w_chem=0.25, w_rep=0.20):
        super().__init__(
            weights=KernelWeights(field=w_field, phys=w_phys, chem=w_chem, exclusion=w_rep),
            include_exclusion=True
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

# For code that imports HubbardEngine
HubbardEngine = SparseEngine

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
