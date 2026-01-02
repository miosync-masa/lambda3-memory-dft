"""
Run Command
===========

Run DSE time evolution simulation.

Usage:
    memory-dft run -L 4 -T 10.0 --memory

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
import numpy as np
from typing import Optional
from pathlib import Path

from ..utils import (
    print_banner, print_section, print_key_value, 
    save_json, error_exit
)


def run(
    sites: int = typer.Option(4, "-L", "--sites", help="Number of lattice sites"),
    time: float = typer.Option(10.0, "-T", "--time", help="Total evolution time"),
    dt: float = typer.Option(0.1, "--dt", help="Time step"),
    U: float = typer.Option(2.0, "-U", help="Hubbard U (interaction strength)"),
    t_hop: float = typer.Option(1.0, "-t", help="Hopping parameter"),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Enable memory effects"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Run DSE time evolution simulation.
    
    Example:
        memory-dft run -L 4 -T 10.0 --memory
    """
    print_banner()
    
    typer.echo("🚀 Running DSE simulation")
    typer.echo("─" * 50)
    print_key_value("Sites (L)", sites)
    print_key_value("Time (T)", time)
    print_key_value("Time step", dt)
    print_key_value("U/t", f"{U}/{t_hop}")
    print_key_value("Memory", "ON" if memory else "OFF")
    typer.echo()
    
    # Import core modules (v0.5.0: unified imports)
    try:
        from memory_dft.core import HubbardEngine  # via sparse_engine_unified
        from memory_dft.core.memory_kernel import SimpleMemoryKernel
    except ImportError as e:
        error_exit(
            f"Could not import memory_dft modules: {e}",
            "Make sure memory_dft is installed: pip install -e ."
        )
    
    # Initialize
    engine = HubbardEngine(sites)
    result_init = engine.compute_full(t=t_hop, U=U)
    psi = result_init.psi
    
    if memory:
        mem_kernel = SimpleMemoryKernel(eta=0.2, tau=5.0, gamma=0.5)
    
    # Evolution
    n_steps = int(time / dt)
    lambdas = []
    times = []
    
    with typer.progressbar(range(n_steps), label="Evolving") as progress:
        for step in progress:
            t_current = step * dt
            
            # Simple field oscillation
            h = 0.5 * np.sin(2 * np.pi * t_current / time)
            
            result = engine.compute_full(t=t_hop, U=U, h=h)
            psi = result.psi
            lambda_val = float(result.lambda_val)  # Ensure float
            
            if memory:
                delta_mem = mem_kernel.compute_memory_contribution(t_current, psi)
                # Convert to float if CuPy array
                if hasattr(delta_mem, 'get'):
                    delta_mem = float(delta_mem.get())
                elif hasattr(delta_mem, 'item'):
                    delta_mem = float(delta_mem.item())
                else:
                    delta_mem = float(delta_mem)
                lambda_val += delta_mem
                mem_kernel.add_state(t_current, result.lambda_val, psi)
            
            lambdas.append(lambda_val)
            times.append(t_current)
    
    # Results
    print_section("Results", "📊")
    print_key_value("Initial λ", f"{lambdas[0]:.4f}")
    print_key_value("Final λ", f"{lambdas[-1]:.4f}")
    print_key_value("Max λ", f"{max(lambdas):.4f}")
    print_key_value("Min λ", f"{min(lambdas):.4f}")
    print_key_value("Mean λ", f"{np.mean(lambdas):.4f}")
    
    # Save output
    if output:
        data = {
            'config': {
                'sites': sites,
                'time': time,
                'dt': dt,
                'U': U,
                't_hop': t_hop,
                'memory': memory,
            },
            'results': {
                'times': times,
                'lambdas': lambdas,
            }
        }
        save_json(data, output)
    
    typer.echo("\n✅ Done!")
