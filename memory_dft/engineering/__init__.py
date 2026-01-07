"""
Memory-DFT Engineering Solvers
==============================

工学応用向け統合ソルバー

physics/ と solver/ のモジュールを組み合わせて、
実際の材料加工・評価をシミュレート。

Modules:
  - base: 共通基底クラス (EngineeringSolver)
  - thermo_mechanical: 熱間/冷間加工、焼入れ、焼戻し
  - fatigue: 疲労寿命予測 (Basquin則 → Λ³)
  - wear: 摩耗予測 (Archard則 → Λ³)
  - forming: プレス成形限界 (FLC)
  - machining: 切削加工 (白色層形成)

Usage:
    from memory_dft.engineering import ThermoMechanicalSolver
    
    solver = ThermoMechanicalSolver(material='Fe', t=1.0, U=5.0)
    result = solver.solve(
        T_path=[1000, 800, 600, 300],  # K
        sigma_path=[0, 2, 2, 0],        # arb
        dt=0.1
    )
    
    print(f"Final yield stress: {result.sigma_y}")
    print(f"Dislocation density: {result.rho_d}")

Author: Masamichi Iizumi, Tamaki Iizumi
"""




