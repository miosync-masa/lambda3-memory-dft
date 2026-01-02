"""
Hysteresis Command
==================

Analyze compression hysteresis (Exclusion Kernel demo).

Demonstrates that compression and expansion paths differ
due to history-dependent repulsive memory.

Key insight:
  - Same distance r = 0.8 Å has DIFFERENT meaning:
    • Approaching → Low enhancement
    • Departing   → High enhancement (compression memory)
  - DFT: Area = 0 (no hysteresis)
  - DSE: Area > 0 (compression memory!)

Usage:
    memory-dft hysteresis --r-min 0.6 --r-max 1.2

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
# Hysteresis Analysis Runner
# =============================================================================

class HysteresisRunner:
    """
    Analyzes compression hysteresis using the Exclusion kernel.
    
    The hysteresis demonstrates that the same distance has different
    effective potential depending on whether approaching or departing.
    """
    
    def __init__(self):
        # Import repulsive kernel from core (v0.5.0: deprecated, use ExclusionKernel)
        try:
            from memory_dft.core import RepulsiveMemoryKernel  # compatibility class
            self.RepulsiveMemoryKernel = RepulsiveMemoryKernel
        except ImportError as e:
            raise ImportError(f"Could not import repulsive kernel: {e}")
    
    def run_cycle(self,
                  r_min: float,
                  r_max: float,
                  steps: int,
                  cycles: int = 1) -> Dict[str, Any]:
        """Run compression-expansion cycles."""
        kernel = self.RepulsiveMemoryKernel(
            eta_rep=0.3,
            tau_rep=3.0,
            tau_recover=10.0,
            r_critical=r_min + 0.1
        )
        
        # Create r path for cycles
        r_compress = np.linspace(r_max, r_min, steps)
        r_expand = np.linspace(r_min, r_max, steps)
        
        V_compress_all = []
        V_expand_all = []
        
        t = 0.0
        dt = 1.0
        
        for cycle in range(cycles):
            # Compression phase
            for r in r_compress:
                psi = np.array([1.0, 0.0])  # Dummy state
                kernel.add_state(t, r, psi)
                V_eff = kernel.compute_effective_repulsion(r, t)
                V_compress_all.append(float(V_eff))
                t += dt
            
            # Expansion phase
            for r in r_expand:
                V_eff = kernel.compute_effective_repulsion(r, t)
                V_expand_all.append(float(V_eff))
                t += dt
        
        # Compute hysteresis area (first cycle only)
        V_compress = np.array(V_compress_all[:steps])
        V_expand = np.array(V_expand_all[:steps])
        
        # Area between curves (simple trapezoid)
        r_vals = np.linspace(r_min, r_max, steps)
        area = np.trapz(np.abs(V_expand - V_compress[::-1]), r_vals)
        
        return {
            'r_min': r_min,
            'r_max': r_max,
            'steps': steps,
            'cycles': cycles,
            'V_compress': V_compress_all,
            'V_expand': V_expand_all,
            'hysteresis_area': float(area),
            'r_values': r_vals.tolist(),
        }
    
    def analyze_direction_dependence(self,
                                      r_target: float,
                                      r_approach_start: float,
                                      r_depart_end: float,
                                      steps: int = 20) -> Dict[str, Any]:
        """
        Analyze direction dependence at a specific distance.
        
        Shows that V_eff(r_target, approaching) ≠ V_eff(r_target, departing)
        """
        kernel = self.RepulsiveMemoryKernel(
            eta_rep=0.3,
            tau_rep=3.0,
            tau_recover=10.0,
            r_critical=r_target
        )
        
        # Approaching path
        r_approach = np.linspace(r_approach_start, r_target, steps)
        t = 0.0
        dt = 1.0
        
        for r in r_approach[:-1]:
            psi = np.array([1.0, 0.0])
            kernel.add_state(t, r, psi)
            t += dt
        
        V_approaching = kernel.compute_effective_repulsion(r_target, t)
        
        # Continue past and depart
        r_past = np.linspace(r_target, r_target - 0.2, 5)
        for r in r_past:
            psi = np.array([1.0, 0.0])
            kernel.add_state(t, r, psi)
            t += dt
        
        V_departing = kernel.compute_effective_repulsion(r_target, t)
        
        return {
            'r_target': r_target,
            'V_approaching': float(V_approaching),
            'V_departing': float(V_departing),
            'ratio': float(V_departing / V_approaching) if V_approaching > 0 else float('inf'),
            'insight': "Same r, different history → Different V_eff!"
        }


# =============================================================================
# CLI Command
# =============================================================================

def hysteresis(
    r_min: float = typer.Option(0.6, "--r-min", help="Minimum distance"),
    r_max: float = typer.Option(1.2, "--r-max", help="Maximum distance"),
    steps: int = typer.Option(50, "--steps", "-n", help="Steps per half-cycle"),
    cycles: int = typer.Option(1, "--cycles", help="Number of cycles"),
    analyze_point: Optional[float] = typer.Option(None, "--analyze", "-a",
                                                   help="Analyze direction dependence at this r"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", 
                                          help="Output JSON file"),
):
    """
    Analyze compression hysteresis (Exclusion Kernel demo).
    
    Demonstrates that compression and expansion paths differ
    due to history-dependent repulsive memory.
    
    Key insight:
      Same distance r has DIFFERENT V_eff depending on:
      - Approaching (compressing): lower enhancement
      - Departing (expanding): higher enhancement (memory!)
    
    Examples:
        memory-dft hysteresis --r-min 0.6 --r-max 1.2
        memory-dft hysteresis --analyze 0.8
    """
    print_banner()
    
    typer.echo("🔄 Compression Hysteresis Analysis")
    typer.echo("─" * 50)
    print_key_value("r_min", f"{r_min} Å")
    print_key_value("r_max", f"{r_max} Å")
    print_key_value("Steps", steps)
    print_key_value("Cycles", cycles)
    typer.echo()
    
    # Initialize runner
    try:
        runner = HysteresisRunner()
    except ImportError as e:
        error_exit(str(e), "pip install -e .")
    
    if analyze_point is not None:
        # Direction dependence analysis
        typer.echo(f"Analyzing direction dependence at r = {analyze_point} Å...")
        result = runner.analyze_direction_dependence(
            analyze_point, 
            r_max, 
            r_min
        )
        
        print_section("Direction Dependence", "🎯")
        typer.echo(f"\n  Target distance: r = {result['r_target']} Å")
        typer.echo(f"\n  V_eff (approaching): {result['V_approaching']:.4f}")
        typer.echo(f"  V_eff (departing):   {result['V_departing']:.4f}")
        typer.echo(f"\n  Ratio: {result['ratio']:.2f}x")
        typer.echo(f"\n  💡 {result['insight']}")
        
        results = result
    else:
        # Hysteresis cycle analysis
        typer.echo("Running compression-expansion cycles...")
        result = runner.run_cycle(r_min, r_max, steps, cycles)
        
        print_section("Hysteresis Analysis", "📊")
        
        area = result['hysteresis_area']
        typer.echo(f"\n  Hysteresis area: {area:.4f}")
        
        typer.echo(f"\n  V_eff at r_min (compress): {result['V_compress'][steps-1]:.4f}")
        typer.echo(f"  V_eff at r_min (expand):   {result['V_expand'][0]:.4f}")
        
        typer.echo()
        typer.echo("  💡 Physical interpretation:")
        typer.echo("     DFT: Area = 0 (no hysteresis, history-blind)")
        typer.echo("     DSE: Area > 0 (compression memory!)")
        
        if area > 0.01:
            typer.echo(f"\n  🎯 Significant hysteresis detected!")
            typer.echo(f"     → System 'remembers' compression history")
        
        results = result
    
    # Save output
    if output:
        save_json(results, output)
    
    typer.echo("\n✅ Done!")
    return results
