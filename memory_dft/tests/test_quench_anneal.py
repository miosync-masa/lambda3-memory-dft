"""
Quench vs Anneal Test - Materials Science Core Demonstration
=============================================================

This is THE test for Memory-DFT's practical relevance!

Physical principle:
  - Quench (急冷): Fast cooling traps system in metastable state
  - Anneal (徐冷): Slow cooling allows relaxation to ground state
  
  Same final temperature → DIFFERENT physical properties!

Materials science analogy:
  - Quench → Martensite (hard, brittle, high energy)
  - Anneal → Ferrite/Pearlite (soft, ductile, low energy)

Expected results:
  - Path A (Quench): Higher E, Higher λ (metastable)
  - Path B (Anneal): Lower E, Lower λ (stable)

If Memory-DFT captures this: We prove that
"Same composition + Same final T ≠ Same properties"
when process history differs!

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import time


def test_quench_vs_anneal():
    """
    Test F: Quench vs Anneal - The Ultimate Materials Test
    
    Start: T_high = 1000K (distorted H)
    Path A (Quench): 1000K → 50K in 1 step (instant)
    Path B (Anneal): 1000K → 50K in 100 steps (gradual)
    
    Expected:
      - Quench: Trapped in metastable state (high E, high λ)
      - Anneal: Relaxed to ground state (low E, low λ)
    
    This proves: Same composition + Same T_final ≠ Same properties!
    """
    print("\n" + "=" * 70)
    print("🔥 TEST F: QUENCH vs ANNEAL")
    print("=" * 70)
    print("""
    The ultimate materials science test!
    
    Quench (急冷): Fast cooling → Metastable (Martensite-like)
    Anneal (徐冷): Slow cooling → Stable (Ferrite-like)
    
    Same final T, DIFFERENT physical states!
    """)
    
    # Import modules
    from memory_dft.core.sparse_engine_unified import SparseEngine
    from memory_dft.physics.thermodynamics import ThermalPathEvolver
    
    # Parameters
    n_sites = 4
    T_high = 1000.0  # K - high temperature start
    T_low = 50.0     # K - final temperature
    
    # Strong temperature dependence for clear effect
    alpha = 0.002  # J decreases 0.2% per 100K
    
    print(f"\n  System: {n_sites}-site Hubbard chain")
    print(f"  T_start: {T_high} K")
    print(f"  T_final: {T_low} K")
    print(f"  α (T-dependence): {alpha}")
    
    # Initialize
    engine = SparseEngine(n_sites, use_gpu=True, verbose=False)
    geometry = engine.build_chain(periodic=False)
    bonds = geometry.bonds
    
    evolver = ThermalPathEvolver(
        engine=engine,
        bonds=bonds,
        J0=1.0,
        U0=2.0,
        alpha_J=alpha,
        alpha_U=0.0,
        T_ref=300.0,
        energy_scale=0.1,
        n_eigenstates=14,
        model='hubbard',
        verbose=False
    )
    
    # Show J(T) at key temperatures
    print(f"\n  J(T) values (lattice expansion effect):")
    print(f"    J({T_high}K) = {evolver.H_builder.J_eff(T_high):.4f}")
    print(f"    J(300K)  = {evolver.H_builder.J_eff(300):.4f}")
    print(f"    J({T_low}K)  = {evolver.H_builder.J_eff(T_low):.4f}")
    
    # ========================================
    # Path A: QUENCH (急冷)
    # ========================================
    print("\n" + "-" * 50)
    print("  Path A: QUENCH (急冷)")
    print("  1000K → 50K in 1 step (instant cooling)")
    print("-" * 50)
    
    # Just 2 temperatures: start and end
    path_quench = [T_high, T_low]
    
    t0 = time.time()
    result_quench = evolver.evolve(
        path_quench, 
        dt=0.1, 
        steps_per_T=5  # Few steps per T
    )
    time_quench = time.time() - t0
    
    print(f"  Completed in {time_quench:.2f}s")
    print(f"  λ_final (Quench): {result_quench['lambda_final']:.4f}")
    
    # ========================================
    # Path B: ANNEAL (徐冷)
    # ========================================
    print("\n" + "-" * 50)
    print("  Path B: ANNEAL (徐冷)")
    print("  1000K → 900K → ... → 50K in 100 steps (gradual)")
    print("-" * 50)
    
    # 100 temperature steps
    n_anneal_steps = 100
    path_anneal = list(np.linspace(T_high, T_low, n_anneal_steps))
    
    t0 = time.time()
    result_anneal = evolver.evolve(
        path_anneal, 
        dt=0.1, 
        steps_per_T=5
    )
    time_anneal = time.time() - t0
    
    print(f"  Completed in {time_anneal:.2f}s")
    print(f"  λ_final (Anneal): {result_anneal['lambda_final']:.4f}")
    
    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    
    lambda_quench = result_quench['lambda_final']
    lambda_anneal = result_anneal['lambda_final']
    delta_lambda = abs(lambda_quench - lambda_anneal)
    
    print(f"""
    ┌─────────────────────────────────────────────────┐
    │  Path         │  λ_final    │  Interpretation   │
    ├─────────────────────────────────────────────────┤
    │  Quench (急冷) │  {lambda_quench:8.4f}   │  Metastable       │
    │  Anneal (徐冷) │  {lambda_anneal:8.4f}   │  Stable           │
    ├─────────────────────────────────────────────────┤
    │  |Δλ|         │  {delta_lambda:8.4f}   │                   │
    └─────────────────────────────────────────────────┘
    """)
    
    # Physical interpretation
    print("  Physical Interpretation:")
    if lambda_quench > lambda_anneal:
        print("    ✅ Quench → Higher λ (more kinetic, metastable)")
        print("    ✅ Anneal → Lower λ (more potential, stable)")
        print("\n    → Matches materials science expectation!")
        print("    → Martensite (quench) vs Ferrite (anneal)")
    elif lambda_quench < lambda_anneal:
        print("    ⚠️ Unexpected: Quench < Anneal")
        print("    This might indicate different physics...")
    else:
        print("    ⚠️ No difference detected")
        print("    Try increasing α or T range")
    
    if delta_lambda > 0.01:
        print("\n  " + "🎉" * 20)
        print("  PROOF: Same composition + Same T_final ≠ Same properties!")
        print("  Process history MATTERS in quantum systems!")
        print("  " + "🎉" * 20)
    
    # Memory metrics
    try:
        from memory_dft.solvers.memory_indicators import MemoryIndicator
        metrics = MemoryIndicator.compute_all(
            O_forward=lambda_quench,
            O_backward=lambda_anneal,
            series=np.array(result_anneal['lambdas']),
            dt=0.1
        )
        print(f"\n  Memory Indicators:")
        print(f"    ΔO (path non-commutativity): {metrics.delta_O:.6f}")
        print(f"    M (temporal memory):         {metrics.M_temporal:.6f}")
        print(f"    Non-Markovian? {metrics.is_non_markovian()}")
    except ImportError:
        pass
    
    print("\n" + "=" * 70)
    print("✅ Quench vs Anneal test completed!")
    print("=" * 70)
    
    return {
        'quench': result_quench,
        'anneal': result_anneal,
        'delta_lambda': delta_lambda,
        'quench_higher': lambda_quench > lambda_anneal,
    }


def test_intermediate_rates():
    """
    Test different cooling rates between quench and anneal.
    
    Shows continuous transition from metastable to stable.
    """
    print("\n" + "=" * 70)
    print("📈 TEST: COOLING RATE DEPENDENCE")
    print("=" * 70)
    
    from memory_dft.core.sparse_engine_unified import SparseEngine
    from memory_dft.physics.thermodynamics import ThermalPathEvolver
    
    n_sites = 4
    T_high = 1000.0
    T_low = 50.0
    alpha = 0.002
    
    engine = SparseEngine(n_sites, use_gpu=True, verbose=False)
    geometry = engine.build_chain(periodic=False)
    bonds = geometry.bonds
    
    evolver = ThermalPathEvolver(
        engine=engine,
        bonds=bonds,
        J0=1.0,
        U0=2.0,
        alpha_J=alpha,
        T_ref=300.0,
        energy_scale=0.1,
        n_eigenstates=14,
        model='hubbard',
        verbose=False
    )
    
    # Different cooling rates
    rates = [2, 5, 10, 20, 50, 100]  # Number of steps
    results = []
    
    print(f"\n  Testing {len(rates)} cooling rates...")
    print(f"  {'Steps':>8} │ {'λ_final':>10} │ {'Rate':>15}")
    print(f"  " + "─" * 40)
    
    for n_steps in rates:
        path = list(np.linspace(T_high, T_low, n_steps))
        result = evolver.evolve(path, dt=0.1, steps_per_T=3)
        rate_label = "Fast" if n_steps < 10 else ("Medium" if n_steps < 50 else "Slow")
        
        print(f"  {n_steps:>8} │ {result['lambda_final']:>10.4f} │ {rate_label:>15}")
        results.append((n_steps, result['lambda_final']))
    
    # Check monotonicity
    lambdas = [r[1] for r in results]
    is_decreasing = all(lambdas[i] >= lambdas[i+1] for i in range(len(lambdas)-1))
    
    print(f"\n  λ decreases with slower cooling? {is_decreasing}")
    if is_decreasing:
        print("  ✅ Confirmed: Slower cooling → Lower λ (more stable)")
    
    return results


def test_thermal_cycling():
    """
    Test thermal cycling: multiple heat-cool cycles.
    
    Each cycle should accumulate memory effects.
    """
    print("\n" + "=" * 70)
    print("🔄 TEST: THERMAL CYCLING")
    print("=" * 70)
    
    from memory_dft.core.sparse_engine_unified import SparseEngine
    from memory_dft.physics.thermodynamics import ThermalPathEvolver
    
    n_sites = 4
    T_high = 500.0
    T_low = 50.0
    alpha = 0.002
    
    engine = SparseEngine(n_sites, use_gpu=True, verbose=False)
    geometry = engine.build_chain(periodic=False)
    bonds = geometry.bonds
    
    evolver = ThermalPathEvolver(
        engine=engine,
        bonds=bonds,
        J0=1.0,
        U0=2.0,
        alpha_J=alpha,
        T_ref=300.0,
        energy_scale=0.1,
        n_eigenstates=14,
        model='hubbard',
        verbose=False
    )
    
    # Build thermal cycling path
    n_cycles = 5
    steps_per_leg = 10
    
    path = []
    for cycle in range(n_cycles):
        # Heat
        path.extend(list(np.linspace(T_low, T_high, steps_per_leg)))
        # Cool
        path.extend(list(np.linspace(T_high, T_low, steps_per_leg)))
    
    print(f"\n  {n_cycles} thermal cycles: {T_low}K ↔ {T_high}K")
    print(f"  Total path length: {len(path)} temperatures")
    
    result = evolver.evolve(path, dt=0.1, steps_per_T=3)
    
    # Extract λ at end of each cycle
    lambdas_per_step = result['lambdas']
    steps_per_cycle = steps_per_leg * 2 * 3  # 2 legs * steps_per_T
    
    print(f"\n  λ evolution through cycles:")
    for cycle in range(n_cycles):
        end_idx = min((cycle + 1) * steps_per_cycle - 1, len(lambdas_per_step) - 1)
        lambda_cycle = lambdas_per_step[end_idx]
        print(f"    After cycle {cycle+1}: λ = {lambda_cycle:.4f}")
    
    print(f"\n  Final λ: {result['lambda_final']:.4f}")
    
    return result


def main():
    """Run all quench vs anneal tests."""
    print("\n" + "🔥" * 25)
    print("  MEMORY-DFT: QUENCH vs ANNEAL DEMONSTRATION")
    print("🔥" * 25)
    print("""
    Proving that thermal history determines material properties!
    
    This is THE test for industrial relevance:
      - Steel heat treatment
      - Semiconductor annealing  
      - Glass transition
      - Polymer processing
    """)
    
    # Main test
    result = test_quench_vs_anneal()
    
    # Additional tests
    print("\n")
    test_intermediate_rates()
    
    print("\n")
    test_thermal_cycling()
    
    print("\n" + "=" * 70)
    print("✅ All Quench vs Anneal tests completed!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    main()
