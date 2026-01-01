"""
Info Command
============

Show version and kernel information.

Usage:
    memory-dft info

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
from ..utils import print_banner, print_section, print_key_value, check_gpu


def info():
    """Show version and kernel information."""
    print_banner()
    
    print_section("Package Information", "📦")
    print_key_value("Version", "0.5.0")
    print_key_value("Package", "memory-dft")
    print_key_value("License", "MIT")
    
    print_section("Memory Kernel Components (4)", "🧠")
    print_key_value("1. PowerLaw (Field)", "Long-range correlations")
    print_key_value("2. StretchedExp (Phys)", "Structural relaxation")
    print_key_value("3. Step (Chem)", "Irreversible reactions")
    print_key_value("4. Exclusion (Direction)", "Compression history [NEW]")
    
    print_section("Key Insight", "💡")
    typer.echo("  Same distance r = 0.8 Å has DIFFERENT meaning:")
    typer.echo("    • Approaching → Low enhancement")
    typer.echo("    • Departing   → High enhancement (compression memory)")
    typer.echo("  DFT cannot distinguish. DSE can!")
    
    print_section("GPU Status", "🖥️")
    typer.echo(f"  {check_gpu()}")
