"""
memory-dft CLI
==============

Command-line interface for Direct Schrödinger Evolution (DSE).

Usage:
    memory-dft info
    memory-dft run -L 4 -T 10.0 --memory
    memory-dft compare --path1 "A,B" --path2 "B,A"
    memory-dft thermal --mol H2 --T-high 300 --T-low 100
    memory-dft dft-compare --mol H2 --basis cc-pvdz
    memory-dft lattice --model heisenberg --Lx 3 --Ly 3
    memory-dft gamma --sizes 4,6,8
    memory-dft hysteresis --r-min 0.6 --r-max 1.2

Architecture:
    cli/
    ├── __init__.py       # This file - app definition
    ├── commands/         # Individual command modules
    │   ├── info.py
    │   ├── run.py
    │   ├── compare.py
    │   ├── thermal.py
    │   ├── dft_compare.py
    │   ├── lattice.py
    │   ├── hysteresis.py
    │   └── gamma.py
    └── utils.py          # Shared utilities

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer

# Create CLI app
app = typer.Typer(
    name="memory-dft",
    help="Direct Schrödinger Evolution - History-dependent quantum dynamics",
    add_completion=False,
)


# =============================================================================
# Register Commands
# =============================================================================

from .commands import (
    info, 
    run, 
    compare,
    thermal, 
    dft_compare,
    lattice,
    hysteresis,
    gamma,
)

# Register all commands
app.command()(info)
app.command()(run)
app.command()(compare)
app.command("thermal")(thermal)
app.command("dft-compare")(dft_compare)
app.command("lattice")(lattice)
app.command()(hysteresis)
app.command()(gamma)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
