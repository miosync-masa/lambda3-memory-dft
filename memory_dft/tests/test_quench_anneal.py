"""
Fe Cluster Quench vs Anneal - Real DFT Demonstration
====================================================

Uses PySCF DFT to demonstrate thermal path dependence
on REAL iron clusters!

Physical picture:
  - High T: Fe-Fe bond elongates (thermal expansion)
  - Low T: Fe-Fe bond contracts
  - Quench: Sudden contraction → Trapped in metastable
  - Anneal: Gradual contraction → Relaxed to ground state

Materials science connection:
  - Quench → Martensite formation (BCC → BCT distortion)
  - Anneal → Ferrite/Pearlite (equilibrium phases)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import List, Optional

# Check PySCF
try:
    from pyscf import gto, dft
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    print("⚠️ PySCF not installed. Run: pip install pyscf")


def create_fe2_quench_path(r_hot: float = 2.5,
                           r_cold: float = 2.0,
                           n_steps: int = 2) -> List[dict]:
    """
    Create Fe2 quench path (instant cooling).

    Args:
        r_hot: Fe-Fe distance at high T (Å)
        r_cold: Fe-Fe distance at low T (Å)
        n_steps: Number of steps (2 = instant quench)
    """
    path = []
    distances = np.linspace(r_hot, r_cold, n_steps)

    for i, r in enumerate(distances):
        path.append({
            'atoms': f"Fe 0 0 0; Fe 0 0 {r}",
            'time': float(i),
            'label': f'quench_{i}',
            'r': r
        })

    return path


def create_fe2_anneal_path(r_hot: float = 2.5,
                           r_cold: float = 2.0,
                           n_steps: int = 20) -> List[dict]:
    """
    Create Fe2 anneal path (gradual cooling).

    Args:
        r_hot: Fe-Fe distance at high T (Å)
        r_cold: Fe-Fe distance at low T (Å)
        n_steps: Number of steps (more = slower anneal)
    """
    path = []
    distances = np.linspace(r_hot, r_cold, n_steps)

    for i, r in enumerate(distances):
        path.append({
            'atoms': f"Fe 0 0 0; Fe 0 0 {r}",
            'time': float(i),
            'label': f'anneal_{i}',
            'r': r
        })

    return path


def compute_dft_energy(atoms: str,
                       basis: str = 'def2-svp',
                       xc: str = 'PBE',
                       spin: int = 8,  # Fe2 is high-spin
                       verbose: int = 0) -> float:
    """
    Compute DFT energy for Fe cluster.

    Fe2 ground state: 9Σ- (spin = 8, i.e., 8 unpaired electrons)
    """
    if not HAS_PYSCF:
        raise ImportError("PySCF required")

    mol = gto.M(
        atom=atoms,
        basis=basis,
        spin=spin,
        verbose=verbose
    )

    mf = dft.UKS(mol)
    mf.xc = xc
    mf.max_cycle = 100

    E = mf.kernel()

    return E


def run_fe2_quench_vs_anneal(verbose: bool = True):
    """
    Main test: Fe2 Quench vs Anneal.

    Demonstrates that thermal history affects final energy
    even when ending at same Fe-Fe distance!
    """
    if not HAS_PYSCF:
        print("❌ PySCF not available")
        return None

    print("\n" + "=" * 70)
    print("🔥 Fe2 QUENCH vs ANNEAL - Real DFT Test")
    print("=" * 70)

    # Parameters
    r_hot = 2.5   # Å - high temperature bond length
    r_cold = 2.0  # Å - low temperature bond length

    print(f"\n  Fe-Fe distance:")
    print(f"    High T (hot):  {r_hot} Å")
    print(f"    Low T (cold):  {r_cold} Å")

    # Create paths
    path_quench = create_fe2_quench_path(r_hot, r_cold, n_steps=2)
    path_anneal = create_fe2_anneal_path(r_hot, r_cold, n_steps=10)

    print(f"\n  Quench path: {len(path_quench)} steps (instant)")
    print(f"  Anneal path: {len(path_anneal)} steps (gradual)")

    # Memory kernel for DSE
    from memory_dft.interfaces.pyscf_interface import MemoryKernelDFTWrapper

    memory_quench = MemoryKernelDFTWrapper(eta=0.1, tau=5.0, gamma=0.5)
    memory_anneal = MemoryKernelDFTWrapper(eta=0.1, tau=5.0, gamma=0.5)

    # ========================================
    # QUENCH PATH
    # ========================================
    print("\n" + "-" * 50)
    print("  Computing QUENCH path...")
    print("-" * 50)

    E_quench_dft = []
    E_quench_dse = []

    for step in path_quench:
        if verbose:
            print(f"    Step {step['label']}: r = {step['r']:.2f} Å")

        E_dft = compute_dft_energy(step['atoms'], verbose=0)
        E_quench_dft.append(E_dft)

        # Memory contribution
        coords = np.array([[0, 0, 0], [0, 0, step['r']]])
        delta_mem = memory_quench.compute_memory_contribution(
            step['time'], E_dft, coords
        )
        E_dse = E_dft + delta_mem
        E_quench_dse.append(E_dse)

        memory_quench.add_state(step['time'], E_dft, coords)

        if verbose:
            print(f"      E_DFT = {E_dft:.6f} Ha, ΔE_mem = {delta_mem:.6f} Ha")

    # ========================================
    # ANNEAL PATH
    # ========================================
    print("\n" + "-" * 50)
    print("  Computing ANNEAL path...")
    print("-" * 50)

    E_anneal_dft = []
    E_anneal_dse = []

    for step in path_anneal:
        if verbose:
            print(f"    Step {step['label']}: r = {step['r']:.2f} Å")

        E_dft = compute_dft_energy(step['atoms'], verbose=0)
        E_anneal_dft.append(E_dft)

        coords = np.array([[0, 0, 0], [0, 0, step['r']]])
        delta_mem = memory_anneal.compute_memory_contribution(
            step['time'], E_dft, coords
        )
        E_dse = E_dft + delta_mem
        E_anneal_dse.append(E_dse)

        memory_anneal.add_state(step['time'], E_dft, coords)

        if verbose and (step['label'].endswith('0') or step['label'].endswith('9')):
            print(f"      E_DFT = {E_dft:.6f} Ha, ΔE_mem = {delta_mem:.6f} Ha")

    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)

    E_quench_final_dft = E_quench_dft[-1]
    E_anneal_final_dft = E_anneal_dft[-1]
    E_quench_final_dse = E_quench_dse[-1]
    E_anneal_final_dse = E_anneal_dse[-1]

    delta_dft = abs(E_quench_final_dft - E_anneal_final_dft)
    delta_dse = abs(E_quench_final_dse - E_anneal_final_dse)

    print(f"""
    ┌────────────────────────────────────────────────────────────┐
    │                    │  E_DFT (Ha)    │  E_DSE (Ha)          │
    ├────────────────────────────────────────────────────────────┤
    │  Quench (急冷)      │  {E_quench_final_dft:12.6f}  │  {E_quench_final_dse:12.6f}        │
    │  Anneal (徐冷)      │  {E_anneal_final_dft:12.6f}  │  {E_anneal_final_dse:12.6f}        │
    ├────────────────────────────────────────────────────────────┤
    │  |ΔE|              │  {delta_dft:12.6f}  │  {delta_dse:12.6f}        │
    └────────────────────────────────────────────────────────────┘
    """)

    print("  Interpretation:")
    print(f"    DFT: Same final geometry → Same energy (Δ = {delta_dft:.2e} Ha)")
    print(f"    DSE: History matters! (Δ = {delta_dse:.6f} Ha)")

    if delta_dse > delta_dft * 10:
        print("\n  🎉 Memory-DFT captures thermal history dependence!")
        print("  → Quench vs Anneal leads to DIFFERENT quantum states!")

    # Convert to more meaningful units
    delta_dse_mHa = delta_dse * 1000
    delta_dse_eV = delta_dse * 27.2114
    delta_dse_kJ = delta_dse * 2625.5

    print(f"\n  Energy difference in various units:")
    print(f"    {delta_dse_mHa:.3f} mHa")
    print(f"    {delta_dse_eV:.4f} eV")
    print(f"    {delta_dse_kJ:.2f} kJ/mol")

    return {
        'quench_dft': E_quench_dft,
        'quench_dse': E_quench_dse,
        'anneal_dft': E_anneal_dft,
        'anneal_dse': E_anneal_dse,
        'delta_dft': delta_dft,
        'delta_dse': delta_dse,
    }


def run_fe4_tetrahedron_test():
    """
    Fe4 tetrahedron: More complex cluster.

    Tests quench vs anneal on 4-atom iron cluster.
    """
    if not HAS_PYSCF:
        print("❌ PySCF not available")
        return None

    print("\n" + "=" * 70)
    print("🔥 Fe4 TETRAHEDRON - Complex Cluster Test")
    print("=" * 70)

    # Fe4 tetrahedron geometry
    # Hot: expanded, Cold: contracted
    def fe4_tetrahedron(scale: float = 1.0):
        """Generate Fe4 tetrahedron coordinates."""
        # Base edge length ~2.5 Å
        a = 2.5 * scale
        # Tetrahedron vertices
        coords = [
            [0, 0, 0],
            [a, 0, 0],
            [a/2, a*np.sqrt(3)/2, 0],
            [a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)]
        ]
        atoms = "; ".join([f"Fe {c[0]} {c[1]} {c[2]}" for c in coords])
        return atoms

    # Quench: 1.1 → 1.0 in 2 steps
    path_quench = [
        {'atoms': fe4_tetrahedron(1.1), 'time': 0, 'scale': 1.1},
        {'atoms': fe4_tetrahedron(1.0), 'time': 1, 'scale': 1.0},
    ]

    # Anneal: 1.1 → 1.0 in 10 steps
    scales_anneal = np.linspace(1.1, 1.0, 10)
    path_anneal = [
        {'atoms': fe4_tetrahedron(s), 'time': i, 'scale': s}
        for i, s in enumerate(scales_anneal)
    ]

    print(f"  Quench: {len(path_quench)} steps")
    print(f"  Anneal: {len(path_anneal)} steps")

    # This would take longer to compute...
    print("\n  ⚠️ Fe4 computation is expensive!")
    print("  Use smaller basis or run separately.")

    return path_quench, path_anneal


def main():
    """Run Fe cluster tests."""
    print("\n" + "🔥" * 25)
    print("  MEMORY-DFT: Fe CLUSTER QUENCH vs ANNEAL")
    print("🔥" * 25)

    if not HAS_PYSCF:
        print("\n❌ PySCF not installed!")
        print("   Install with: pip install pyscf")
        return

    # Main test
    result = run_fe2_quench_vs_anneal(verbose=True)

    print("\n" + "=" * 70)
    print("✅ Fe cluster test completed!")
    print("=" * 70)

    return result


if __name__ == "__main__":
    main()
