"""
Interfaces Module
=================

External quantum chemistry package interfaces for DSE calculations.

REFACTORED: MemoryKernelDFT is now a wrapper around core.memory_kernel
            (no longer a separate implementation!)

Available Interfaces:
  - pyscf_interface: PySCF integration for DFT vs DSE comparison
"""

from typing import List

__all__: List[str] = []

# PySCF interface (optional)
try:
    from .pyscf_interface import (
        DSECalculator,
        GeometryStep,
        SinglePointResult,
        PathResult,
        ComparisonResult,
        create_h2_stretch_path,
        create_h2_compress_path,
        create_cyclic_path,
        HAS_PYSCF,
    )
    __all__.extend([
        'DSECalculator',
        'GeometryStep',
        'SinglePointResult',
        'PathResult',
        'ComparisonResult',
        'create_h2_stretch_path',
        'create_h2_compress_path',
        'create_cyclic_path',
        'HAS_PYSCF',
    ])
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

__all__.append('HAS_PYSCF')
