"""
CLI Commands
============

All CLI commands for memory-dft.

Commands:
  - info: Show version and kernel information
  - run: Run DSE time evolution simulation
  - compare: Compare two evolution paths
  - thermal: Thermal path dependence (PySCF)
  - dft-compare: DFT vs DSE comparison (PySCF)
  - lattice: 2D lattice simulation
  - hysteresis: Compression hysteresis analysis
  - gamma: γ decomposition analysis

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from .info import info
from .run import run
from .thermal import thermal
from .compare import compare
from .dft_compare import dft_compare
from .lattice import lattice
from .hysteresis import hysteresis
from .gamma import gamma

__all__ = [
    'info',
    'run',
    'thermal',
    'compare',
    'dft_compare',
    'lattice',
    'hysteresis',
    'gamma',
]
