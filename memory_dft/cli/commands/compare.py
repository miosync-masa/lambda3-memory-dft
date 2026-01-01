"""
Compare Command
===============

Compare two evolution paths (Hubbard model demo).

This demonstrates that different paths to the same final state
yield different quantum outcomes - something memoryless approaches cannot capture.

Usage:
    memory-dft compare --path1 "A,B" --path2 "B,A"

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..utils import (
    print_banner, print_section, print_key_value, 
    save_json, error_exit
)


# =============================================================================
# Path Comparison Runner
# =============================================================================

class PathComparisonRunner:
    """
    Runs path comparison using Hubbard model.
    
    Demonstrates that different paths to the same final state
    yield different quantum outcomes when memory is included.
    """
    
    def __init__(self, sites: int = 4):
        self.sites = sites
        
        # Import core modules
        try:
            from memory_dft.core.hubbard_engine import HubbardEngine
            from memory_dft.core.memory_kernel import CatalystMemoryKernel, CatalystEvent
            
            self.HubbardEngine = HubbardEngine
            self.CatalystMemoryKernel = CatalystMemoryKernel
            self.CatalystEvent = CatalystEvent
        except ImportError as e:
            raise ImportError(f"Could not import memory_dft modules: {e}")
        
        self.engine = self.HubbardEngine(sites)
    
    def run_path(self, 
                 events: List[str], 
                 time: float, 
                 dt: float,
                 path_name: str) -> Dict[str, Any]:
        """Run evolution along a single path."""
        # Initialize
        result_init = self.engine.compute_full(t=1.0, U=2.0)
        psi_init = result_init.psi
        
        memory = self.CatalystMemoryKernel(eta=0.3, tau_ads=3.0, tau_react=5.0)
        psi = psi_init.copy()
        
        n_steps = int(time / dt)
        lambdas_std = []
        lambdas_mem = []
        
        event_times = [time * 0.3, time * 0.6]
        
        for step in range(n_steps):
            t = step * dt
            
            # Check events
            for i, event_time in enumerate(event_times):
                if i < len(events) and abs(t - event_time) < dt/2:
                    event_type = 'adsorption' if events[i].strip().upper() == 'A' else 'reaction'
                    memory.add_event(self.CatalystEvent(event_type, t, i, 0.5))
            
            result = self.engine.compute_full(t=1.0, U=2.0)
            psi = result.psi
            lambda_std = result.lambda_val
            lambdas_std.append(lambda_std)
            
            delta_mem = memory.compute_memory_contribution(t, psi)
            lambdas_mem.append(lambda_std + delta_mem)
            memory.add_state(t, lambda_std, psi)
        
        return {
            'path_name': path_name,
            'events': events,
            'std': lambdas_std[-1],
            'mem': lambdas_mem[-1],
            'lambdas_std': lambdas_std,
            'lambdas_mem': lambdas_mem,
        }
    
    def compare(self, 
                path1_str: str, 
                path2_str: str, 
                time: float = 10.0,
                dt: float = 0.1) -> Dict[str, Any]:
        """Compare two paths."""
        events1 = path1_str.split(',')
        events2 = path2_str.split(',')
        
        result1 = self.run_path(events1, time, dt, f"Path 1 ({path1_str})")
        result2 = self.run_path(events2, time, dt, f"Path 2 ({path2_str})")
        
        diff_std = abs(result1['std'] - result2['std'])
        diff_mem = abs(result1['mem'] - result2['mem'])
        
        return {
            'path1': result1,
            'path2': result2,
            'diff_std': diff_std,
            'diff_mem': diff_mem,
        }


# =============================================================================
# CLI Command
# =============================================================================

def compare(
    path1: str = typer.Option(..., "--path1", help="First path (e.g., 'A,B' for A→B)"),
    path2: str = typer.Option(..., "--path2", help="Second path (e.g., 'B,A' for B→A)"),
    sites: int = typer.Option(4, "-L", "--sites", help="Number of sites"),
    time: float = typer.Option(10.0, "-T", "--time", help="Total time"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """
    Compare two evolution paths (Hubbard model demo).
    
    This demonstrates that different paths to the same final state
    yield different quantum outcomes - something memoryless approaches cannot capture.
    
    For real DFT comparison, use 'dft-compare' command.
    
    Example:
        memory-dft compare --path1 "A,B" --path2 "B,A"
    """
    print_banner()
    
    typer.echo("🔀 Path Comparison (Hubbard Model Demo)")
    typer.echo("─" * 50)
    print_key_value("Path 1", path1)
    print_key_value("Path 2", path2)
    print_key_value("Sites", sites)
    print_key_value("Time", time)
    typer.echo()
    
    # Run comparison
    try:
        runner = PathComparisonRunner(sites)
    except ImportError as e:
        error_exit(str(e), "pip install -e .")
    
    typer.echo("Running path comparison...")
    
    with typer.progressbar(length=2, label="Computing paths") as progress:
        results = runner.compare(path1, path2, time)
        progress.update(2)
    
    # Display results
    print_section("Results", "📊")
    
    r1 = results['path1']
    r2 = results['path2']
    
    typer.echo(f"\n  {r1['path_name']}:")
    typer.echo(f"    Memoryless:  λ = {r1['std']:.4f}")
    typer.echo(f"    With Memory: λ = {r1['mem']:.4f}")
    
    typer.echo(f"\n  {r2['path_name']}:")
    typer.echo(f"    Memoryless:  λ = {r2['std']:.4f}")
    typer.echo(f"    With Memory: λ = {r2['mem']:.4f}")
    
    typer.echo(f"\n  ═══════════════════════════════════════")
    typer.echo(f"  |Δλ| Memoryless:   {results['diff_std']:.6f}")
    typer.echo(f"  |Δλ| With Memory:  {results['diff_mem']:.4f}")
    
    if results['diff_std'] < 1e-6:
        typer.echo(f"\n  🎯 Memoryless: Cannot distinguish paths! (Δλ ≈ 0)")
        typer.echo(f"  🎯 With Memory: REVEALS difference! (Δλ = {results['diff_mem']:.4f})")
    else:
        ratio = results['diff_mem'] / results['diff_std']
        typer.echo(f"\n  Amplification: {ratio:.1f}x")
    
    # Save
    if output:
        data = {
            'paths': {'path1': path1, 'path2': path2},
            'config': {'sites': sites, 'time': time},
            'results': {
                'diff_std': results['diff_std'],
                'diff_mem': results['diff_mem'],
                'path1_std': r1['std'],
                'path1_mem': r1['mem'],
                'path2_std': r2['std'],
                'path2_mem': r2['mem'],
            }
        }
        save_json(data, output)
    
    typer.echo("\n✅ Done!")
    return results
