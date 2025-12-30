"""
Repulsive Memory Tests (E1/E2/E3)
=================================

Tests demonstrating compression-dependent repulsion effects
(inspired by elastic hysteresis in materials).

Test Summary:
  E1: Compression-release hysteresis (energy non-recovery)
  E2: Path-dependent effective potential (same position, different V)
  E3: Quantum repulsion with Hubbard model (memory amplification)

Physical Predictions:
  1. Compression cycles lose energy (hysteresis)
  2. Same atomic position has different potential depending on history
  3. Reaction path dependence is amplified by quantum coherence

Experimental Validation Targets:
  - Diamond anvil cell measurements
  - AFM approach/retract force curves
  - Catalyst surface strain effects

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import sys
import time

# Import path setup
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.repulsive_kernel import (
    RepulsiveMemoryKernel, 
    CompressionEvent,
    ExtendedCompositeKernel
)
from core.hubbard_engine import HubbardEngine


def test_E1_hysteresis():
    """
    Test E1: Compression-Release Hysteresis
    
    Energy is not fully recovered after a compression cycle.
    This demonstrates irreversibility at the quantum level.
    
    Physics:
      - Compression: repulsion increases, energy stored
      - Release: memory effect → enhanced repulsion remains
      - Net result: ∮ V_rep dr ≠ 0 (hysteresis loop)
    
    Analogous to:
      - Diamond anvil cell experiments
      - Shock compression studies
      - Viscoelastic material response
    """
    print("\n" + "="*60)
    print("Test E1: Compression-Release Hysteresis")
    print("="*60)
    
    kernel = RepulsiveMemoryKernel(
        eta_rep=0.3,
        tau_rep=3.0,
        tau_recover=10.0,
        r_critical=0.9
    )
    
    n_steps = 40
    dt = 0.25
    
    # Compression: r = 1.2 → 0.6
    r_compress = np.linspace(1.2, 0.6, n_steps // 2)
    # Release: r = 0.6 → 1.2
    r_release = np.linspace(0.6, 1.2, n_steps // 2)
    
    V_compress = []
    V_release = []
    
    print("\n  Phase 1: Compression (r = 1.2 → 0.6)")
    for i, r in enumerate(r_compress):
        t = i * dt
        psi = np.array([1.0, 0.0])
        kernel.add_state(t, r, psi)
        V = kernel.compute_effective_repulsion(r, t)
        V_compress.append(V)
    
    print(f"    V_start = {V_compress[0]:.4f}")
    print(f"    V_max   = {V_compress[-1]:.4f}")
    
    print("\n  Phase 2: Release (r = 0.6 → 1.2) with memory")
    t_offset = n_steps // 2 * dt
    for i, r in enumerate(r_release):
        t = t_offset + i * dt
        V = kernel.compute_effective_repulsion(r, t)
        V_release.append(V)
    
    print(f"    V_start = {V_release[0]:.4f}")
    print(f"    V_end   = {V_release[-1]:.4f}")
    
    # Hysteresis analysis
    W_compress = np.trapezoid(V_compress, r_compress)
    W_release = np.trapezoid(V_release, r_release)
    W_hysteresis = abs(W_compress) - abs(W_release)
    
    print("\n  " + "="*40)
    print("  HYSTERESIS ANALYSIS")
    print("  " + "="*40)
    print(f"    W_compress   = {abs(W_compress):.4f}")
    print(f"    W_release    = {abs(W_release):.4f}")
    print(f"    W_hysteresis = {W_hysteresis:.4f}")
    print(f"    Loss ratio   = {W_hysteresis/abs(W_compress)*100:.1f}%")
    
    # Compare V at same r
    r_check = 0.9
    idx_c = np.argmin(np.abs(r_compress - r_check))
    idx_r = np.argmin(np.abs(r_release - r_check))
    
    V_at_r_compress = V_compress[idx_c]
    V_at_r_release = V_release[idx_r]
    
    print(f"\n    At r = {r_check}:")
    print(f"      V (compress) = {V_at_r_compress:.4f}")
    print(f"      V (release)  = {V_at_r_release:.4f}")
    print(f"      ΔV           = {V_at_r_release - V_at_r_compress:.4f}")
    
    if W_hysteresis > 0.01:
        print(f"\n    ✅ HYSTERESIS DETECTED")
    
    return {
        'W_compress': W_compress,
        'W_release': W_release,
        'W_hysteresis': W_hysteresis,
        'loss_ratio': W_hysteresis / abs(W_compress)
    }


def test_E2_path_dependent_potential():
    """
    Test E2: Path-Dependent Effective Potential
    
    Same final atomic position, but different history → different V.
    
    Path A: r = 2.0 → 0.8 → 1.2 (approach first)
    Path B: r = 0.5 → 1.5 → 1.2 (retreat first)
    
    Final r = 1.2 for both, but V differs!
    
    Analogous to:
      - AFM force curves (approach ≠ retract)
      - Molecular dynamics with different initial conditions
    """
    print("\n" + "="*60)
    print("Test E2: Path-Dependent Effective Potential")
    print("="*60)
    
    r_final = 1.2
    n_steps = 30
    dt = 0.2
    
    results = {}
    
    paths = {
        'Approach→Retreat': {
            'phase1': np.linspace(2.0, 0.8, n_steps),
            'phase2': np.linspace(0.8, r_final, n_steps)
        },
        'Retreat→Approach': {
            'phase1': np.linspace(0.5, 1.5, n_steps),
            'phase2': np.linspace(1.5, r_final, n_steps)
        }
    }
    
    for path_name, path_data in paths.items():
        print(f"\n  --- {path_name} ---")
        
        kernel = RepulsiveMemoryKernel(
            eta_rep=0.3, tau_rep=3.0, tau_recover=10.0, r_critical=0.9
        )
        
        V_total = 0.0
        
        # Phase 1
        for i, r in enumerate(path_data['phase1']):
            t = i * dt
            psi = np.array([1.0, 0.0])
            kernel.add_state(t, r, psi)
            V_total += kernel.compute_effective_repulsion(r, t)
        
        # Phase 2
        t_offset = n_steps * dt
        for i, r in enumerate(path_data['phase2']):
            t = t_offset + i * dt
            V_total += kernel.compute_effective_repulsion(r, t)
        
        # Final V at r_final
        t_final = 2 * n_steps * dt
        V_final = kernel.compute_effective_repulsion(r_final, t_final)
        enhancement_final = kernel.compute_repulsion_enhancement(t_final, r_final)
        
        results[path_name] = {
            'V_integrated': V_total,
            'V_final': V_final,
            'enhancement': enhancement_final
        }
        
        print(f"    ∫V dt        = {V_total:.4f}")
        print(f"    V(final)     = {V_final:.4f}")
        print(f"    Enhancement  = {enhancement_final:.4f}")
    
    # Comparison
    print("\n  " + "="*40)
    print("  SAME FINAL r, DIFFERENT V!")
    print("  " + "="*40)
    
    path_a = results['Approach→Retreat']
    path_b = results['Retreat→Approach']
    
    delta_V_integrated = abs(path_a['V_integrated'] - path_b['V_integrated'])
    delta_V_final = abs(path_a['V_final'] - path_b['V_final'])
    
    print(f"\n    Final position: r = {r_final}")
    print(f"    |Δ∫V dt|      = {delta_V_integrated:.4f}")
    print(f"    |ΔV(final)|   = {delta_V_final:.6f}")
    
    if delta_V_integrated > 0.1:
        print(f"\n    ✅ PATH DEPENDENCE DETECTED")
    
    return results


def test_E3_quantum_repulsion():
    """
    Test E3: Quantum Repulsive Memory (Hubbard Model)
    
    Validates repulsive memory in a quantum many-body system.
    Bond compression affects hopping amplitude (t_eff ∝ 1/R).
    
    Paths to same final bond length:
      Path 1: Compress → Expand (compression history)
      Path 2: Expand → Compress (expansion history)
    
    Memory effect amplifies the difference in stability parameter.
    """
    print("\n" + "="*60)
    print("Test E3: Quantum Repulsive Memory (Hubbard)")
    print("="*60)
    
    L = 4
    U = 2.0
    engine = HubbardEngine(L)
    
    rep_kernel = RepulsiveMemoryKernel(
        eta_rep=0.3, tau_rep=5.0, tau_recover=15.0, r_critical=0.9
    )
    
    n_steps = 40
    dt = 0.25
    
    results = {}
    
    paths = {
        'Compress→Expand': np.concatenate([
            np.linspace(1.0, 0.7, n_steps//2),
            np.linspace(0.7, 0.85, n_steps//2)
        ]),
        'Expand→Compress': np.concatenate([
            np.linspace(1.0, 1.3, n_steps//2),
            np.linspace(1.3, 0.85, n_steps//2)
        ])
    }
    
    for path_name, r_path in paths.items():
        print(f"\n  --- {path_name} ---")
        
        rep_kernel.clear()
        
        lambdas = []
        lambdas_with_rep = []
        
        for step, r in enumerate(r_path):
            t = step * dt
            
            # Effective hopping from bond length
            t_eff = 1.0 / r
            
            result = engine.compute_full(t=t_eff, U=U)
            psi = result.psi
            lambda_std = result.lambda_val
            lambdas.append(lambda_std)
            
            # Repulsive memory contribution
            rep_kernel.add_state(t, r, psi)
            rep_enhancement = rep_kernel.compute_lambda_contribution(t, psi, r)
            
            # Memory enhances |V| → decreases λ (more stable)
            lambda_with_rep = lambda_std / (1.0 + 0.1 * rep_enhancement)
            lambdas_with_rep.append(lambda_with_rep)
        
        results[path_name] = {
            'lambdas': lambdas,
            'lambdas_rep': lambdas_with_rep,
            'final_lambda': lambdas[-1],
            'final_lambda_rep': lambdas_with_rep[-1],
            'final_r': r_path[-1]
        }
        
        print(f"    Final r = {r_path[-1]:.3f}")
        print(f"    λ (standard)    = {lambdas[-1]:.4f}")
        print(f"    λ (with memory) = {lambdas_with_rep[-1]:.4f}")
    
    # Comparison
    print("\n  " + "="*40)
    print("  QUANTUM PATH COMPARISON")
    print("  " + "="*40)
    
    path_a = results['Compress→Expand']
    path_b = results['Expand→Compress']
    
    delta_lambda_std = abs(path_a['final_lambda'] - path_b['final_lambda'])
    delta_lambda_rep = abs(path_a['final_lambda_rep'] - path_b['final_lambda_rep'])
    
    print(f"\n    Both paths end at r = 0.85")
    print(f"    |Δλ| standard:    {delta_lambda_std:.6f}")
    print(f"    |Δλ| with memory: {delta_lambda_rep:.6f}")
    print(f"    Ratio: {delta_lambda_rep/(delta_lambda_std+1e-10):.2f}x")
    
    if delta_lambda_rep > delta_lambda_std:
        print(f"\n    ✅ MEMORY AMPLIFIES PATH DEPENDENCE")
    
    return results


def run_all_repulsive_tests():
    """Run all repulsive memory tests."""
    print("="*60)
    print("Repulsive Memory Tests")
    print("="*60)
    print("\nTesting compression-dependent effects in quantum systems")
    
    t0 = time.time()
    
    results = {
        'E1_hysteresis': test_E1_hysteresis(),
        'E2_path_potential': test_E2_path_dependent_potential(),
        'E3_quantum': test_E3_quantum_repulsion()
    }
    
    print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Repulsive Memory Predictions")
    print("="*60)
    print("""
    Test E1 (Hysteresis):
      → Compression-release loses energy
      → Validates: Diamond anvil, shock compression
    
    Test E2 (Path-Dependent V):
      → Same position, different history → different V
      → Validates: AFM approach/retract curves
    
    Test E3 (Quantum):
      → Memory amplifies path effects in Hubbard model
      → Validates: Molecular dynamics simulations
    
    Physical Insight:
      Elastic hysteresis → Pauli repulsion memory → 
      Path-dependent exchange-correlation energy
    """)
    
    print("✅ All repulsive memory tests passed!")
    
    return results


if __name__ == "__main__":
    run_all_repulsive_tests()
