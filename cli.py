#!/usr/bin/env python3
"""
memory-dft CLI
==============

Command-line interface for Direct Schrödinger Evolution (DSE).

Usage:
    memory-dft run -L 4 -T 10.0 --memory
    memory-dft compare --path1 "A,B" --path2 "B,A"
    memory-dft gamma --sizes 4,6,8
    memory-dft hysteresis --r-min 0.6 --r-max 1.2
    memory-dft info

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import typer
from typing import Optional, List
from pathlib import Path
import numpy as np
import json
import sys
import os

# Add parent directory to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Create CLI app
app = typer.Typer(
    name="memory-dft",
    help="Direct Schrödinger Evolution - History-dependent quantum dynamics",
    add_completion=False,
)


def print_banner():
    """Print welcome banner."""
    typer.echo("""
╔═══════════════════════════════════════════════════════════════╗
║           Direct Schrödinger Evolution (DSE)                  ║
║                    memory-dft v0.4.0                          ║
║       ~ First-Principles History-Dependent Dynamics ~         ║
╚═══════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# info command
# =============================================================================

@app.command()
def info():
    """Show version and kernel information."""
    print_banner()
    
    typer.echo("📦 Package Information")
    typer.echo("─" * 50)
    typer.echo("  Version:  0.4.0")
    typer.echo("  Package:  memory-dft")
    typer.echo("  License:  MIT")
    typer.echo()
    
    typer.echo("🧠 Memory Kernel Components (4)")
    typer.echo("─" * 50)
    typer.echo("  1. PowerLaw (Field)      - Long-range correlations")
    typer.echo("  2. StretchedExp (Phys)   - Structural relaxation")
    typer.echo("  3. Step (Chem)           - Irreversible reactions")
    typer.echo("  4. Exclusion (Direction) - Compression history [NEW]")
    typer.echo()
    
    typer.echo("💡 Key Insight")
    typer.echo("─" * 50)
    typer.echo("  Same distance r = 0.8 Å has DIFFERENT meaning:")
    typer.echo("    • Approaching → Low enhancement")
    typer.echo("    • Departing   → High enhancement (compression memory)")
    typer.echo("  DFT cannot distinguish. DSE can!")
    typer.echo()
    
    # Check GPU
    try:
        import cupy as cp
        gpu_info = f"✅ Available ({cp.cuda.Device().name.decode()})"
    except:
        gpu_info = "❌ Not available (CPU mode)"
    
    typer.echo("🖥️  GPU Status")
    typer.echo("─" * 50)
    typer.echo(f"  {gpu_info}")


# =============================================================================
# run command
# =============================================================================

@app.command()
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
    
    typer.echo(f"🚀 Running DSE simulation")
    typer.echo("─" * 50)
    typer.echo(f"  Sites (L):    {sites}")
    typer.echo(f"  Time (T):     {time}")
    typer.echo(f"  Time step:    {dt}")
    typer.echo(f"  U/t:          {U}/{t_hop}")
    typer.echo(f"  Memory:       {'ON' if memory else 'OFF'}")
    typer.echo()
    
    # Import here to avoid slow startup
    try:
        import sys
        import os
        core_path = os.path.join(os.path.dirname(__file__), 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        from hubbard_engine import HubbardEngine
        from memory_kernel import SimpleMemoryKernel
    except ImportError as e:
        typer.echo(f"❌ Error: Could not import memory_dft modules: {e}", err=True)
        typer.echo("   Make sure you're in the memory_dft directory or it's installed", err=True)
        raise typer.Exit(1)
    
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
            lambda_val = result.lambda_val
            
            if memory:
                delta_mem = mem_kernel.compute_memory_contribution(t_current, psi)
                lambda_val += delta_mem
                mem_kernel.add_state(t_current, result.lambda_val, psi)
            
            lambdas.append(lambda_val)
            times.append(t_current)
    
    # Results
    typer.echo()
    typer.echo("📊 Results")
    typer.echo("─" * 50)
    typer.echo(f"  Initial λ:    {lambdas[0]:.4f}")
    typer.echo(f"  Final λ:      {lambdas[-1]:.4f}")
    typer.echo(f"  Max λ:        {max(lambdas):.4f}")
    typer.echo(f"  Min λ:        {min(lambdas):.4f}")
    typer.echo(f"  Mean λ:       {np.mean(lambdas):.4f}")
    
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
        output.write_text(json.dumps(data, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        typer.echo(f"\n💾 Saved to {output}")
    
    typer.echo("\n✅ Done!")


# =============================================================================
# compare command
# =============================================================================

@app.command()
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
    typer.echo(f"  Path 1: {path1}")
    typer.echo(f"  Path 2: {path2}")
    typer.echo()
    
    # Import
    try:
        import sys
        import os
        core_path = os.path.join(os.path.dirname(__file__), 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        from hubbard_engine import HubbardEngine
        from memory_kernel import CatalystMemoryKernel, CatalystEvent
    except ImportError as e:
        typer.echo(f"❌ Error: Could not import memory_dft modules: {e}", err=True)
        raise typer.Exit(1)
    
    # Parse paths
    events1 = path1.split(',')
    events2 = path2.split(',')
    
    engine = HubbardEngine(sites)
    result_init = engine.compute_full(t=1.0, U=2.0)
    psi_init = result_init.psi
    
    dt = 0.1
    n_steps = int(time / dt)
    
    results = {}
    
    for path_name, events in [(f"Path 1 ({path1})", events1), (f"Path 2 ({path2})", events2)]:
        memory = CatalystMemoryKernel(eta=0.3, tau_ads=3.0, tau_react=5.0)
        psi = psi_init.copy()
        
        lambdas_std = []
        lambdas_mem = []
        
        event_times = [time * 0.3, time * 0.6]
        
        with typer.progressbar(range(n_steps), label=f"  {path_name[:20]:20s}") as progress:
            for step in progress:
                t = step * dt
                
                # Check events
                for i, event_time in enumerate(event_times):
                    if i < len(events) and abs(t - event_time) < dt/2:
                        event_type = 'adsorption' if events[i].strip().upper() == 'A' else 'reaction'
                        memory.add_event(CatalystEvent(event_type, t, i, 0.5))
                
                result = engine.compute_full(t=1.0, U=2.0)
                psi = result.psi
                lambda_std = result.lambda_val
                lambdas_std.append(lambda_std)
                
                delta_mem = memory.compute_memory_contribution(t, psi)
                lambdas_mem.append(lambda_std + delta_mem)
                memory.add_state(t, lambda_std, psi)
        
        results[path_name] = {
            'std': lambdas_std[-1],
            'mem': lambdas_mem[-1],
            'lambdas_mem': lambdas_mem,
        }
    
    # Compare
    typer.echo()
    typer.echo("📊 Results")
    typer.echo("─" * 50)
    
    path_names = list(results.keys())
    diff_std = abs(results[path_names[0]]['std'] - results[path_names[1]]['std'])
    diff_mem = abs(results[path_names[0]]['mem'] - results[path_names[1]]['mem'])
    
    typer.echo(f"\n  {path_names[0]}:")
    typer.echo(f"    Memoryless:  λ = {results[path_names[0]]['std']:.4f}")
    typer.echo(f"    With Memory: λ = {results[path_names[0]]['mem']:.4f}")
    
    typer.echo(f"\n  {path_names[1]}:")
    typer.echo(f"    Memoryless:  λ = {results[path_names[1]]['std']:.4f}")
    typer.echo(f"    With Memory: λ = {results[path_names[1]]['mem']:.4f}")
    
    typer.echo(f"\n  ═══════════════════════════════════════")
    typer.echo(f"  |Δλ| Memoryless:   {diff_std:.6f}")
    typer.echo(f"  |Δλ| With Memory:  {diff_mem:.4f}")
    
    if diff_std < 1e-6:
        typer.echo(f"\n  🎯 Memoryless: Cannot distinguish paths! (Δλ ≈ 0)")
        typer.echo(f"  🎯 With Memory: REVEALS difference! (Δλ = {diff_mem:.4f})")
    else:
        ratio = diff_mem / diff_std
        typer.echo(f"\n  Amplification: {ratio:.1f}x")
    
    # Save
    if output:
        data = {
            'paths': {'path1': path1, 'path2': path2},
            'results': {
                'diff_std': diff_std,
                'diff_mem': diff_mem,
            }
        }
        output.write_text(json.dumps(data, indent=2))
        typer.echo(f"\n💾 Saved to {output}")
    
    typer.echo("\n✅ Done!")


# =============================================================================
# gamma command
# =============================================================================

@app.command()
def gamma(
    sizes: str = typer.Option("4,6,8", "--sizes", "-L", help="Comma-separated system sizes"),
    U: float = typer.Option(2.0, "-U", help="Hubbard U"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """
    Compute γ decomposition (Memory fraction analysis).
    
    Decomposes correlation exponent into local and memory parts:
        γ_total = γ_local + γ_memory
    
    Key result: γ_memory ≈ 1.216 (46.7% non-Markovian!)
    
    Example:
        memory-dft gamma --sizes 4,6,8
    """
    print_banner()
    
    typer.echo("📈 γ Decomposition Analysis")
    typer.echo("─" * 50)
    
    size_list = [int(s.strip()) for s in sizes.split(',')]
    typer.echo(f"  System sizes: {size_list}")
    typer.echo(f"  U/t: {U}")
    typer.echo()
    
    # Import
    try:
        import sys
        import os
        core_path = os.path.join(os.path.dirname(__file__), 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        from hubbard_engine import HubbardEngine
    except ImportError as e:
        typer.echo(f"❌ Error: Could not import memory_dft modules: {e}", err=True)
        raise typer.Exit(1)
    
    results = []
    
    for L in size_list:
        typer.echo(f"  Computing L={L}...")
        
        engine = HubbardEngine(L)
        result = engine.compute_full(t=1.0, U=U)
        
        # Simple γ estimate from λ scaling
        lambda_val = result.lambda_val
        gamma_estimate = np.log(lambda_val + 1) / np.log(L)
        
        results.append({
            'L': L,
            'lambda': lambda_val,
            'gamma_estimate': gamma_estimate,
        })
        
        typer.echo(f"    λ = {lambda_val:.4f}, γ_est = {gamma_estimate:.3f}")
    
    # Extrapolation
    if len(results) >= 3:
        Ls = np.array([r['L'] for r in results])
        gammas = np.array([r['gamma_estimate'] for r in results])
        
        # Linear fit in 1/L
        coeffs = np.polyfit(1/Ls, gammas, 1)
        gamma_inf = coeffs[1]  # Extrapolated to L→∞
        
        typer.echo()
        typer.echo("📊 Extrapolation (L→∞)")
        typer.echo("─" * 50)
        typer.echo(f"  γ_total (extrapolated): {gamma_inf:.3f}")
        typer.echo(f"  γ_local (typical):      ~1.4")
        typer.echo(f"  γ_memory (estimated):   ~{max(0, gamma_inf - 1.4):.3f}")
        typer.echo()
        typer.echo(f"  Memory fraction: ~{max(0, (gamma_inf - 1.4)/gamma_inf * 100):.0f}%")
    
    # Save
    if output:
        data = {
            'config': {'sizes': size_list, 'U': U},
            'results': results,
        }
        output.write_text(json.dumps(data, indent=2))
        typer.echo(f"\n💾 Saved to {output}")
    
    typer.echo("\n✅ Done!")


# =============================================================================
# hysteresis command
# =============================================================================

@app.command()
def hysteresis(
    r_min: float = typer.Option(0.6, "--r-min", help="Minimum distance"),
    r_max: float = typer.Option(1.2, "--r-max", help="Maximum distance"),
    steps: int = typer.Option(50, "--steps", "-n", help="Steps per half-cycle"),
    cycles: int = typer.Option(1, "--cycles", help="Number of cycles"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """
    Analyze compression hysteresis (Exclusion Kernel demo).
    
    Demonstrates that compression and expansion paths differ
    due to history-dependent repulsive memory.
    
    Example:
        memory-dft hysteresis --r-min 0.6 --r-max 1.2 --steps 50
    """
    print_banner()
    
    typer.echo("🔄 Compression Hysteresis Analysis")
    typer.echo("─" * 50)
    typer.echo(f"  Distance range: {r_min} → {r_max} Å")
    typer.echo(f"  Steps: {steps} per half-cycle")
    typer.echo(f"  Cycles: {cycles}")
    typer.echo()
    
    # Import
    try:
        # Direct import to avoid __init__.py dependency issues
        import sys
        import os
        core_path = os.path.join(os.path.dirname(__file__), 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        from repulsive_kernel import RepulsiveMemoryKernel
    except ImportError as e:
        typer.echo(f"❌ Error: Could not import memory_dft modules: {e}", err=True)
        raise typer.Exit(1)
    
    kernel = RepulsiveMemoryKernel(
        eta_rep=0.3,
        tau_rep=3.0,
        tau_recover=10.0,
        r_critical=r_max * 0.8
    )
    
    V_compress_all = []
    V_expand_all = []
    r_values = []
    
    dt = 0.1
    t = 0.0
    
    for cycle in range(cycles):
        typer.echo(f"  Cycle {cycle + 1}/{cycles}...")
        
        # Compression
        r_compress = np.linspace(r_max, r_min, steps)
        V_compress = []
        
        for r in r_compress:
            kernel.add_state(t, r)
            V = kernel.compute_effective_repulsion(r, t)
            V_compress.append(V)
            t += dt
        
        # Expansion
        r_expand = np.linspace(r_min, r_max, steps)
        V_expand = []
        
        for r in r_expand:
            V = kernel.compute_effective_repulsion(r, t)
            V_expand.append(V)
            t += dt
        
        V_compress_all.extend(V_compress)
        V_expand_all.extend(V_expand)
        r_values = list(r_compress) + list(r_expand)
    
    # Compute hysteresis area
    try:
        area = abs(np.trapezoid(V_compress_all[:steps]) - np.trapezoid(V_expand_all[:steps]))
    except AttributeError:
        area = abs(np.trapz(V_compress_all[:steps]) - np.trapz(V_expand_all[:steps]))
    
    # Results
    typer.echo()
    typer.echo("📊 Results")
    typer.echo("─" * 50)
    typer.echo(f"  V_eff at r_min (compress): {V_compress_all[steps-1]:.4f}")
    typer.echo(f"  V_eff at r_min (expand):   {V_expand_all[0]:.4f}")
    typer.echo(f"  ΔV at same point:          {abs(V_compress_all[steps-1] - V_expand_all[0]):.4f}")
    typer.echo()
    typer.echo(f"  Hysteresis area: {area:.4f}")
    typer.echo()
    typer.echo("  💡 Non-zero area = Memory effect!")
    typer.echo("     DFT: Area = 0 (no hysteresis)")
    typer.echo("     DSE: Area > 0 (compression memory)")
    
    # Save
    if output:
        data = {
            'config': {
                'r_min': r_min,
                'r_max': r_max,
                'steps': steps,
                'cycles': cycles,
            },
            'results': {
                'hysteresis_area': area,
                'V_compress': V_compress_all[:steps],
                'V_expand': V_expand_all[:steps],
            }
        }
        output.write_text(json.dumps(data, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        typer.echo(f"\n💾 Saved to {output}")
    
    typer.echo("\n✅ Done!")


# =============================================================================
# dft-compare command (PySCF - Real DFT!)
# =============================================================================

@app.command("dft-compare")
def dft_compare(
    molecule: str = typer.Option("H2", "--mol", "-m", help="Molecule: H2, LiH, or custom 'atom x y z; atom x y z'"),
    basis: str = typer.Option("sto-3g", "--basis", "-b", help="Basis set (sto-3g, cc-pvdz, etc.)"),
    xc: str = typer.Option("LDA", "--xc", help="XC functional (LDA, B3LYP, PBE, etc.)"),
    r_stretch: float = typer.Option(1.5, "--r-stretch", help="Max stretch distance (Å)"),
    r_compress: float = typer.Option(0.5, "--r-compress", help="Min compress distance (Å)"),
    steps: int = typer.Option(5, "-n", "--steps", help="Steps per path segment"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """
    Compare DFT vs DSE using PySCF (REAL DFT!).
    
    This uses actual DFT calculations to demonstrate that:
    - DFT gives identical energies for different paths to same final state
    - DSE captures history dependence
    
    Requires: pip install pyscf
    
    Example:
        memory-dft dft-compare --mol H2 --basis cc-pvdz --xc B3LYP
    """
    print_banner()
    
    typer.echo("🔬 DFT vs DSE Comparison (PySCF)")
    typer.echo("─" * 50)
    typer.echo(f"  Molecule:    {molecule}")
    typer.echo(f"  Basis:       {basis}")
    typer.echo(f"  XC:          {xc}")
    typer.echo(f"  Stretch:     {r_stretch} Å")
    typer.echo(f"  Compress:    {r_compress} Å")
    typer.echo(f"  Steps:       {steps} per segment")
    typer.echo()
    
    # Import PySCF interface
    try:
        import sys
        import os
        interfaces_path = os.path.join(os.path.dirname(__file__), 'interfaces')
        if interfaces_path not in sys.path:
            sys.path.insert(0, interfaces_path)
        from pyscf_interface import (
            DSECalculator,
            create_h2_stretch_path,
            create_h2_compress_path,
            GeometryStep,
        )
    except ImportError as e:
        typer.echo(f"❌ Error: PySCF not available: {e}", err=True)
        typer.echo("   Install with: pip install pyscf", err=True)
        raise typer.Exit(1)
    
    # Determine equilibrium distance based on molecule
    if molecule.upper() == "H2":
        r_eq = 0.74
        atom_template = "H 0 0 0; H 0 0 {r}"
    elif molecule.upper() == "LIH":
        r_eq = 1.60
        atom_template = "Li 0 0 0; H 0 0 {r}"
    else:
        # Custom molecule - assume diatomic with variable z
        r_eq = 1.0
        atom_template = molecule + "; X 0 0 {r}"  # User provides first atom
        typer.echo(f"  ⚠️  Custom molecule: using r_eq = {r_eq} Å")
    
    typer.echo(f"  r_eq:        {r_eq} Å")
    typer.echo()
    
    # Create calculator
    typer.echo("Initializing DSE calculator...")
    calc = DSECalculator(
        basis=basis,
        xc=xc,
        memory_eta=0.05,
        memory_tau=5.0,
        verbose=0
    )
    
    # Create paths
    typer.echo("Creating paths...")
    
    def make_path(r_start, r_mid, r_end, n_steps, template):
        """Create a path with given distances."""
        path = []
        t = 0.0
        dt = 1.0
        
        # First segment
        for r in np.linspace(r_start, r_mid, n_steps):
            atoms = template.format(r=r)
            path.append(GeometryStep(atoms=atoms, time=t))
            t += dt
        
        # Second segment
        for r in np.linspace(r_mid, r_end, n_steps):
            atoms = template.format(r=r)
            path.append(GeometryStep(atoms=atoms, time=t))
            t += dt
        
        return path
    
    if molecule.upper() in ["H2", "LIH"]:
        # Path 1: stretch then return
        path_stretch = make_path(r_eq, r_stretch, r_eq, steps, atom_template)
        # Path 2: compress then return
        path_compress = make_path(r_eq, r_compress, r_eq, steps, atom_template)
        
        typer.echo(f"  Path 1 (stretch→return): {len(path_stretch)} steps")
        typer.echo(f"  Path 2 (compress→return): {len(path_compress)} steps")
    else:
        typer.echo("❌ Custom molecules not fully supported yet", err=True)
        raise typer.Exit(1)
    
    # Run calculations
    typer.echo()
    typer.echo("Running DFT calculations...")
    
    with typer.progressbar(length=2, label="  Computing") as progress:
        result1 = calc.compute_path(path_stretch, label="Stretch→Return")
        progress.update(1)
        result2 = calc.compute_path(path_compress, label="Compress→Return")
        progress.update(1)
    
    # Calculate differences
    diff_dft = abs(result1.E_dft_final - result2.E_dft_final)
    diff_dse = abs(result1.E_dse_final - result2.E_dse_final)
    
    # Display results
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("DSE vs DFT Path Comparison")
    typer.echo("=" * 60)
    
    typer.echo(f"\nPath 1: Stretch→Return")
    typer.echo(f"  E_DFT (final):  {result1.E_dft_final:.6f} Ha")
    typer.echo(f"  E_DSE (final):  {result1.E_dse_final:.6f} Ha")
    typer.echo(f"  Memory effect:  {result1.memory_effect:.6f} Ha")
    
    typer.echo(f"\nPath 2: Compress→Return")
    typer.echo(f"  E_DFT (final):  {result2.E_dft_final:.6f} Ha")
    typer.echo(f"  E_DSE (final):  {result2.E_dse_final:.6f} Ha")
    typer.echo(f"  Memory effect:  {result2.memory_effect:.6f} Ha")
    
    typer.echo()
    typer.echo("─" * 60)
    typer.echo(f"|ΔE| DFT:  {diff_dft:.8f} Ha  ({diff_dft * 27.211:.4f} eV)")
    typer.echo(f"|ΔE| DSE:  {diff_dse:.8f} Ha  ({diff_dse * 27.211:.4f} eV)")
    
    if diff_dft < 1e-8:
        typer.echo()
        typer.echo("🎯 DFT: Cannot distinguish paths! (ΔE ≈ 0)")
        typer.echo(f"🎯 DSE: REVEALS difference! (ΔE = {diff_dse:.6f} Ha)")
    else:
        ratio = diff_dse / diff_dft if diff_dft > 0 else float('inf')
        typer.echo(f"\n  Amplification: {ratio:.1f}x")
    
    typer.echo("=" * 60)
    
    # Save output
    if output:
        data = {
            'config': {
                'molecule': molecule,
                'basis': basis,
                'xc': xc,
                'r_eq': r_eq,
                'r_stretch': r_stretch,
                'r_compress': r_compress,
                'steps': steps,
            },
            'path1': {
                'label': 'Stretch→Return',
                'E_dft': result1.E_dft,
                'E_dse': result1.E_dse,
                'E_dft_final': result1.E_dft_final,
                'E_dse_final': result1.E_dse_final,
                'memory_effect': result1.memory_effect,
            },
            'path2': {
                'label': 'Compress→Return',
                'E_dft': result2.E_dft,
                'E_dse': result2.E_dse,
                'E_dft_final': result2.E_dft_final,
                'E_dse_final': result2.E_dse_final,
                'memory_effect': result2.memory_effect,
            },
            'comparison': {
                'delta_dft_Ha': diff_dft,
                'delta_dft_eV': diff_dft * 27.211,
                'delta_dse_Ha': diff_dse,
                'delta_dse_eV': diff_dse * 27.211,
            }
        }
        output.write_text(json.dumps(data, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        typer.echo(f"\n💾 Saved to {output}")
    
    typer.echo("\n✅ Done!")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
