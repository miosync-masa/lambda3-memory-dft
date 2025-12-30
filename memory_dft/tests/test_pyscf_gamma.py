"""
PySCF Integration Test for Memory-DFT
=====================================

実分子からγ（相関指数）を抽出する

理論:
  γ_total = γ_local + γ_memory
  
- PySCF (FCI/CCSD): γ_total（全相関）
- 差分からγ_memory を推定 → Memory kernelパラメータ

Usage:
  pip install pyscf
  python test_pyscf_gamma.py

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np

# PySCF
try:
    from pyscf import gto, scf, cc, fci
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    print("⚠️ PySCF not found. Install with: pip install pyscf")

# Memory-DFT
try:
    from memory_dft.physics.vorticity import (
        VorticityCalculator,
        GammaExtractor,
        MemoryKernelFromGamma
    )
    HAS_MEMORY_DFT = True
except ImportError:
    import sys
    sys.path.insert(0, '/content/lambda3-memory-dft')
    try:
        from memory_dft.physics.vorticity import (
            VorticityCalculator,
            GammaExtractor,
            MemoryKernelFromGamma
        )
        HAS_MEMORY_DFT = True
    except ImportError:
        HAS_MEMORY_DFT = False
        print("⚠️ Memory-DFT not found")


def compute_h2_properties(bond_length: float = 0.74):
    """
    H2分子のプロパティを計算
    
    Args:
        bond_length: H-H距離 (Å)
    
    Returns:
        dict: 計算結果
    """
    # 分子定義
    mol = gto.Mole()
    mol.atom = f'''
        H  0  0  0
        H  0  0  {bond_length}
    '''
    mol.basis = 'sto-3g'
    mol.build()
    
    n_elec = mol.nelectron
    n_orb = mol.nao
    
    print(f"\n{'='*60}")
    print(f"H2 molecule: R = {bond_length} Å")
    print(f"Electrons: {n_elec}, Orbitals: {n_orb}")
    print('='*60)
    
    # HF計算
    mf = scf.RHF(mol)
    E_hf = mf.kernel()
    print(f"\nHF Energy: {E_hf:.8f} Ha")
    
    # FCI計算（厳密解）
    cisolver = fci.FCI(mf)
    E_fci, ci_vec = cisolver.kernel()
    E_corr = E_fci - E_hf
    print(f"FCI Energy: {E_fci:.8f} Ha")
    print(f"Correlation Energy: {E_corr:.8f} Ha")
    
    # 2-RDM計算
    rdm1, rdm2 = cisolver.make_rdm12(ci_vec, n_orb, n_elec)
    print(f"\n2-RDM shape: {rdm2.shape}")
    
    return {
        'mol': mol,
        'E_hf': E_hf,
        'E_fci': E_fci,
        'E_corr': E_corr,
        'rdm1': rdm1,
        'rdm2': rdm2,
        'n_orb': n_orb,
        'n_elec': n_elec,
        'bond_length': bond_length
    }


def compute_vorticity_from_rdm2(rdm2: np.ndarray, n_orb: int, E_corr: float):
    """
    2-RDMからVorticityとαを計算
    """
    calc = VorticityCalculator(svd_cut=0.95, use_jax=False)
    result = calc.compute_with_energy(rdm2, n_orb, E_corr)
    
    print(f"\nVorticity Analysis:")
    print(f"  V = {result.vorticity:.6f}")
    print(f"  Effective rank k = {result.effective_rank}")
    print(f"  α = |E_xc| / V = {result.alpha:.6f}")
    
    return result


def scan_bond_length_and_extract_gamma():
    """
    結合長をスキャンしてγを抽出
    
    α = |E_xc| / V ∝ N^(-γ)
    """
    extractor = GammaExtractor()
    
    # 結合長スキャン（電子数は固定だが、相関の強さが変化）
    bond_lengths = [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    results = []
    
    print("\n" + "="*60)
    print("Bond Length Scan for γ Extraction")
    print("="*60)
    
    for R in bond_lengths:
        try:
            props = compute_h2_properties(R)
            vort = compute_vorticity_from_rdm2(
                props['rdm2'], 
                props['n_orb'], 
                props['E_corr']
            )
            
            results.append({
                'R': R,
                'E_corr': props['E_corr'],
                'V': vort.vorticity,
                'alpha': vort.alpha
            })
            
            # 擬似的なN（結合長を電子数の代わりに使用）
            # 本来は異なる分子サイズで比較すべき
            extractor.add_data(
                n_electrons=int(R * 10),  # 擬似N
                E_xc=props['E_corr'],
                vorticity=vort.vorticity
            )
            
        except Exception as e:
            print(f"  ⚠️ R={R} Å failed: {e}")
    
    return results, extractor


def multi_molecule_gamma_extraction():
    """
    異なる分子でγを抽出（本格版）
    """
    extractor = GammaExtractor()
    
    molecules = [
        ('H2', 'H 0 0 0; H 0 0 0.74', 2),
        ('LiH', 'Li 0 0 0; H 0 0 1.6', 4),
        ('BeH2', 'Be 0 0 0; H 0 0 1.3; H 0 0 -1.3', 6),
        ('H2O', 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', 10),
    ]
    
    print("\n" + "="*60)
    print("Multi-Molecule γ Extraction")
    print("="*60)
    
    results = []
    
    for name, geom, n_elec_expected in molecules:
        print(f"\n--- {name} ---")
        
        try:
            mol = gto.Mole()
            mol.atom = geom
            mol.basis = 'sto-3g'
            mol.build()
            
            n_orb = mol.nao
            n_elec = mol.nelectron
            
            print(f"  N_elec={n_elec}, N_orb={n_orb}")
            
            # HF
            mf = scf.RHF(mol)
            E_hf = mf.kernel()
            
            # FCI（小さい分子のみ）
            if n_orb <= 6:
                cisolver = fci.FCI(mf)
                E_fci, ci_vec = cisolver.kernel()
                E_corr = E_fci - E_hf
                
                rdm1, rdm2 = cisolver.make_rdm12(ci_vec, n_orb, n_elec)
                
                # Vorticity
                calc = VorticityCalculator(svd_cut=0.95, use_jax=False)
                vort = calc.compute_with_energy(rdm2, n_orb, E_corr)
                
                print(f"  E_corr = {E_corr:.6f} Ha")
                print(f"  V = {vort.vorticity:.6f}")
                print(f"  α = {vort.alpha:.6f}")
                
                results.append({
                    'name': name,
                    'n_elec': n_elec,
                    'E_corr': E_corr,
                    'V': vort.vorticity,
                    'alpha': vort.alpha
                })
                
                extractor.add_data(n_elec, E_corr, vort.vorticity)
                
            else:
                print(f"  ⚠️ Too large for FCI, using CCSD")
                # CCSD
                mycc = cc.CCSD(mf)
                mycc.kernel()
                E_corr = mycc.e_corr
                print(f"  E_corr (CCSD) = {E_corr:.6f} Ha")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    return results, extractor


def main():
    """メイン実行"""
    if not HAS_PYSCF:
        print("❌ PySCF is required. Install with: pip install pyscf")
        return
    
    print("="*60)
    print("🧪 Memory-DFT × PySCF: γ Extraction Test")
    print("="*60)
    
    # Test 1: 単一H2
    print("\n" + "="*60)
    print("TEST 1: Single H2 molecule")
    print("="*60)
    
    props = compute_h2_properties(0.74)
    vort = compute_vorticity_from_rdm2(
        props['rdm2'], 
        props['n_orb'], 
        props['E_corr']
    )
    
    # Test 2: H2結合長スキャン
    print("\n" + "="*60)
    print("TEST 2: H2 Bond Length Scan")
    print("="*60)
    
    results_scan, extractor_scan = scan_bond_length_and_extract_gamma()
    
    # γ抽出（このデータでは意味は限定的）
    gamma_result = extractor_scan.extract_gamma()
    print(f"\nγ extraction from bond scan:")
    for k, v in gamma_result.items():
        print(f"  {k}: {v}")
    
    # Test 3: 複数分子
    print("\n" + "="*60)
    print("TEST 3: Multi-Molecule Analysis")
    print("="*60)
    
    results_multi, extractor_multi = multi_molecule_gamma_extraction()
    
    if len(extractor_multi.data_points) >= 3:
        gamma_result_multi = extractor_multi.extract_gamma()
        print(f"\nγ extraction from multi-molecule:")
        for k, v in gamma_result_multi.items():
            print(f"  {k}: {v}")
        
        # Memory kernelパラメータ推定
        if gamma_result_multi.get('gamma'):
            # γ分解（ED距離フィルターで導出済み: γ_local ≈ 0.53 * γ_total）
            gamma_total = gamma_result_multi['gamma']
            gamma_local_estimate = gamma_total * 0.53  # ED r≤2 から
            
            decomp = extractor_multi.decompose_gamma(
                gamma_total=gamma_total,
                gamma_local=gamma_local_estimate
            )
            
            print(f"\nγ Decomposition (from ED distance filter):")
            for k, v in decomp.items():
                print(f"  {k}: {v}")
            
            # Kernel推定
            kernel_params = MemoryKernelFromGamma.estimate_kernel_params(decomp)
            print(f"\nEstimated Memory Kernel Parameters:")
            for k, v in kernel_params.items():
                print(f"  {k}: {v}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if results_multi:
        print("\nMolecule |  N_elec  |   E_corr   |    V     |    α")
        print("-" * 55)
        for r in results_multi:
            print(f"  {r['name']:5s}  |    {r['n_elec']:2d}    | {r['E_corr']:10.6f} | {r['V']:.4f} | {r['alpha']:.4f}")
    
    print("\n✅ PySCF Integration Test Complete!")
    print("   See test_gamma_distance_scan.py for full γ decomposition")
    print("   γ_memory = 0.916 (45.9% Non-Markovian)")
    print("   \"We implemented one. てへぺろ (・ω<)\"")


if __name__ == "__main__":
    main()
