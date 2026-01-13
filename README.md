# memory_dft パッケージ概要

`memory_dft` は Direct Schrödinger Evolution (DSE) を中心に、履歴依存の量子ダイナミクスと材料記憶効果を扱うための Python パッケージです。メモリカーネルによる非マルコフ的な時間発展、トポロジーやボルテックス解析、熱・応力場の評価、PySCF との連携などを含む構成になっています。

## ディレクトリ構成

- `core/`
  - スパース行列エンジン、履歴管理、メモリカーネル、環境演算子、材料破壊解析などの中核機能を提供します。
- `solvers/`
  - DSE 時間発展ソルバー、記憶指標計算、熱ホログラフィー進化などを実装します。
- `physics/`
  - λ3 安定性評価、渦度解析、RDM (Reduced Density Matrix)、トポロジー指標の計算を担当します。
- `holographic/`
  - ホログラフィックな双対解析や測定プロトコル（`matplotlib` が必要）をまとめます。
- `interfaces/`
  - PySCF を用いた DFT 計算との統合インターフェースを提供します。
- `engineering/`
  - 熱・機械連成など工学系の追加ソルバー群を収録します。
- `tests/`
  - 簡易的なスモークテスト等のテストスイートです。
- `__init__.py`
  - 主要クラス・関数を再エクスポートし、トップレベル API を整理します。
- `__main__.py`
  - `python -m memory_dft` での実行エントリです。

## 主要コンポーネント

### Core

- **SparseEngine (`core/sparse_engine_unified.py`)**
  - スパース Hamiltonian の構築・対角化・GPU/CPU 切替などを扱う統一エンジンです。
- **MemoryKernel (`core/memory_kernel.py`)**
  - 履歴依存効果を表現するメモリカーネルの中核実装です。
- **HistoryManager (`core/history_manager.py`)**
  - 時系列の状態履歴とメモリ評価のための管理ユーティリティを提供します。
- **Environment Operators (`core/environment_operators.py`)**
  - 有限温度・応力・トポロジー解析に必要な演算子や熱力学ユーティリティをまとめます。
- **Material Failure (`core/material_failure.py`)**
  - 熱/応力トポロジーの破壊予測と解析ロジックを提供します。

### Solvers

- **DSESolver (`solvers/dse_solver.py`)**
  - DSE の時間発展計算を行うソルバーです。
- **MemoryIndicator (`solvers/memory_indicators.py`)**
  - ヒステリシスや記憶指標の解析を行います。
- **ThermalHolographicEvolution (`solvers/thermal_holographic.py`)**
  - 熱ホログラフィーの時間発展・双対性指標の評価を担当します。

### Physics

- **Lambda3Calculator (`physics/lambda3_bridge.py`)**
  - λ3 안정性指標の計算や HCSP 検証を行います。
- **VorticityCalculator (`physics/vorticity.py`)**
  - 渦度解析と関連指標の抽出を支援します。
- **RDMCalculator (`physics/rdm.py`)**
  - 2-RDM の計算や Hubbard/Heisenberg 系の RDM 評価を提供します。
- **Topology Engine (`physics/topology.py`)**
  - Berry/Zak 位相や波動関数巻き数などのトポロジー解析を行います。

### Interfaces

- **PySCF Interface (`interfaces/pyscf_interface.py`)**
  - PySCF を利用した DFT 計算に DSE メモリ効果を付加するための統合 API です。

## オプション依存

- **GPU アクセラレーション**: `cupy` がある場合、スパース演算や DSE ソルバーは GPU を利用します。
- **PySCF 連携**: `pyscf` がインストールされている場合、分子系の DFT 計算を経路計算と組み合わせられます。
- **ホログラフィック解析**: `matplotlib` がある場合、ホログラフィック解析の可視化や測定プロトコルが利用可能です。

## 参考: 典型的な利用イメージ

```python
from memory_dft import SparseEngine, DSESolver, MemoryKernel

engine = SparseEngine(n_sites=4)
H_K, H_V = engine.build_heisenberg(J=1.0)
psi0 = engine.compute_ground_state(H_K + H_V)[1]

solver = DSESolver(H_K, H_V, gamma_memory=1.2)
result = solver.run(psi0, t_end=1.0, dt=0.1)
print(result.summary())
```

> 上記は概要例であり、詳細なパラメータや物理モデルは各モジュールの docstring を参照してください。
