"""
CLI Utilities
=============

Common utilities shared across CLI commands.

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
import numpy as np
import json
from pathlib import Path
from typing import Any, Optional

__version__ = "0.5.0"


def print_banner():
    """Print welcome banner."""
    typer.echo("""
╔═══════════════════════════════════════════════════════════════╗
║           Direct Schrödinger Evolution (DSE)                  ║
║                    memory-dft v0.5.0                          ║
║       ~ First-Principles History-Dependent Dynamics ~         ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_section(title: str, emoji: str = "📦"):
    """Print section header."""
    typer.echo(f"\n{emoji} {title}")
    typer.echo("─" * 50)


def print_key_value(key: str, value: Any, indent: int = 2):
    """Print key-value pair."""
    spaces = " " * indent
    typer.echo(f"{spaces}{key}: {value}")


def check_gpu() -> str:
    """Check GPU availability and return status string."""
    try:
        import cupy as cp
        if cp.cuda.is_available():
            device_id = cp.cuda.Device().id
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            device_name = props.get('name', f'GPU {device_id}')
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
            return f"✅ Available ({device_name})"
        else:
            return "❌ Not available (CPU mode)"
    except Exception:
        return "❌ Not available (CPU mode)"


def save_json(data: dict, output: Path, default_serializer=None):
    """Save data to JSON file."""
    if default_serializer is None:
        default_serializer = lambda x: float(x) if isinstance(x, np.floating) else x
    
    output.write_text(json.dumps(data, indent=2, default=default_serializer))
    typer.echo(f"\n💾 Saved to {output}")


def error_exit(message: str, hint: Optional[str] = None):
    """Print error and exit."""
    typer.echo(f"❌ {message}", err=True)
    if hint:
        typer.echo(f"   {hint}", err=True)
    raise typer.Exit(1)


def import_or_exit(module_path: str, error_msg: str, install_hint: Optional[str] = None):
    """Try to import a module, exit with helpful message if fails."""
    try:
        import importlib
        parts = module_path.split('.')
        module = importlib.import_module(parts[0])
        for part in parts[1:]:
            module = getattr(module, part)
        return module
    except ImportError as e:
        error_exit(f"{error_msg}: {e}", install_hint)


class ProgressContext:
    """Context manager for progress bar with timing."""
    
    def __init__(self, iterable, label: str = "Processing"):
        self.iterable = iterable
        self.label = label
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return typer.progressbar(self.iterable, label=self.label)
    
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        typer.echo(f"  ⏱️  Completed in {elapsed:.2f}s")
