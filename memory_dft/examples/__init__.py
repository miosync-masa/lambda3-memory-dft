"""
Memory-DFT Examples
===================

Example scripts demonstrating Memory-DFT capabilities.

Examples:
  - thermal_path.py: Thermal path dependence demonstration
  - ladder_2d.py: 2D lattice simulations with various Hamiltonians

These examples show how to use the refactored Memory-DFT
modules for common simulation tasks.

Usage:
    python -m memory_dft.examples.thermal_path
    python -m memory_dft.examples.ladder_2d

Author: Masamichi Iizumi, Tamaki Iizumi
"""

__all__ = [
    'run_thermal_path_demo',
    'run_ladder_2d_demo',
]

# Lazy imports to avoid loading everything
def run_thermal_path_demo():
    """Run thermal path dependence demonstration."""
    from .thermal_path import main
    main()

def run_ladder_2d_demo():
    """Run 2D ladder demonstration."""
    from .ladder_2d import main
    main()
