"""
Interfaces Module
=================

External quantum chemistry package interfaces for DSE calculations.

Available Interfaces:
  - pyscf_interface: PySCF integration for DFT vs DSE comparison
"""

from typing import List

__all__: List[str] = []

# PySCF interface (optional)
try:
    from .pyscf_interface import (
        DSECalculator,
        PathResult,
        ComparisonResult,
        GeometryStep,
        MemoryKernelDFT,
        create_h2_stretch_path,
        create_h2_compress_path,
        demo_h2_comparison,
    )
    __all__.extend([
        'DSECalculator',
        'PathResult', 
        'ComparisonResult',
        'GeometryStep',
        'MemoryKernelDFT',
        'create_h2_stretch_path',
        'create_h2_compress_path',
        'demo_h2_comparison',
    ])
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

__all__.append('HAS_PYSCF')
