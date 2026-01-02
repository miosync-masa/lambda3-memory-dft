# Direct Schrödinger Evolution (DSE)

[![Tests](https://github.com/miosync-masa/lambda3-memory-dft/actions/workflows/test.yml/badge.svg)](https://github.com/miosync-masa/lambda3-memory-dft/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/memory-dft.svg)](https://badge.fury.io/py/memory-dft)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18095869.svg)](https://doi.org/10.5281/zenodo.18095869)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**First-Principles History-Dependent Quantum Dynamics**

> *Why "memory-dft"? Historical naming. The physics goes far beyond DFT.*

> *"Standard DFT erases history by construction. DSE retains it."*

---

## The Problem with DFT

Density Functional Theory (DFT) assumes energy depends only on **instantaneous electron density**:

$$E = E[\rho(\mathbf{r})]$$

This means: **Same structure → Same energy**, regardless of how you got there.

But reality shows:
- Catalysts remember adsorption order
- Cyclic voltammetry shows hysteresis  
- AFM approach/retract curves differ
- Diamond anvil compression is path-dependent

**DFT cannot explain any of these.** By construction.

---

## The DSE Solution

Direct Schrödinger Evolution solves the **exact** time-dependent Schrödinger equation:

$$i\hbar \frac{\partial}{\partial t}|\psi\rangle = H|\psi\rangle$$

No density-functional approximation. No history erasure. The full many-body wave function carries the complete history.

| Aspect | DFT | DSE |
|--------|-----|-----|
| Fundamental equation | $E = E[\rho]$ | $i\hbar\partial_t\psi = H\psi$ |
| State description | Density only | Full wave function |
| Treatment of history | **Erased by construction** | Retained explicitly |
| Path dependence | ❌ Impossible | ✅ Natural |

---

## Architecture (v0.5.0)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Direct Schrödinger Evolution                    │
│                           memory-dft v0.5.0                         │
│         ~ First-Principles History-Dependent Dynamics ~             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CORE: Foundation (Unified in v0.5.0)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CompositeMemoryKernel (4 Components)    [memory_kernel.py]  │   │
│  │                                                              │   │
│  │  K(t-τ) = w₁·K_field + w₂·K_phys + w₃·K_chem + w₄·K_excl   │   │
│  │                                                              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │   │
│  │  │PowerLaw │ │Stretched│ │  Step   │ │Exclusion│           │   │
│  │  │ (Field) │ │  Exp    │ │ (Chem)  │ │(Direction)          │   │
│  │  │ 1/t^γ   │ │e^(-t^β) │ │sigmoid  │ │e^-t(1-e^-t)         │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │   │
│  │       └───────────┴─────┬─────┴───────────┘                 │   │
│  │                         ▼                                    │   │
│  │       ┌──────────────────────────────────────┐              │   │
│  │       │   HistoryManager [history_manager.py] │              │   │
│  │       │   ψ(τ), Λ(τ), observables, metadata   │              │   │
│  │       └──────────────────────────────────────┘              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  SparseEngine (UNIFIED)            [sparse_engine_unified.py] │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │ Models:  Heisenberg │ Ising │ XY │ Hubbard │ Kitaev    │ │   │
│  │  ├─────────────────────────────────────────────────────────┤ │   │
│  │  │ Geometry: Chain │ Ladder │ Square │ Custom bonds        │ │   │
│  │  ├─────────────────────────────────────────────────────────┤ │   │
│  │  │ Backend:  CPU (NumPy/SciPy) │ GPU (CuPy) auto-switch    │ │   │
│  │  ├─────────────────────────────────────────────────────────┤ │   │
│  │  │ Output:   λ = K/|V| │ 2-RDM │ Correlations │ Energies   │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  │  Backward Compatible: HubbardEngine, SpinOperators, etc.     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SOLVERS: Time Evolution                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐       ┌─────────────────────┐             │
│  │ MemoryLanczosSolver │       │ TimeEvolutionEngine │             │
│  │ [lanczos_memory.py] │──────▶│ [time_evolution.py] │             │
│  │                     │       │                     │             │
│  │ ψ(t+dt) =           │       │ High-level API:     │             │
│  │ exp(-iHdt)ψ(t)      │       │ • EvolutionConfig   │             │
│  │ + η·ψ_memory        │       │ • quick_evolve()    │             │
│  └─────────────────────┘       └─────────────────────┘             │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐       ┌─────────────────────┐             │
│  │ MemoryIndicator     │       │ChemicalReactionSolver│             │
│  │[memory_indicators.py]│       │[chemical_reaction.py]│             │
│  │                     │       │                     │             │
│  │ • ΔO (path diff)    │       │ • ReactionPath      │             │
│  │ • M(t) (temporal)   │       │ • compare_paths()   │             │
│  │ • γ_memory          │       │ • Surface chemistry │             │
│  │ • HysteresisAnalyzer│       │                     │             │
│  └─────────────────────┘       └─────────────────────┘             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHYSICS: Analysis                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Lambda3Bridge   │  │   Vorticity     │  │ Thermodynamics  │     │
│  │[lambda3_bridge] │  │  [vorticity.py] │  │[thermodynamics] │     │
│  │                 │  │                 │  │                 │     │
│  │ • λ = K/|V|     │  │ • γ_total       │  │ • T ↔ β         │     │
│  │ • StabilityPhase│  │ • γ_local       │  │ • ⟨O⟩_T         │     │
│  │ • HCSPValidator │  │ • γ_memory      │  │ • S(T), F(T)    │     │
│  │ • EDR (env)     │  │ • GammaExtractor│  │ • C_v(T)        │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
│  ┌─────────────────┐                                                │
│  │   2-RDM         │  Interfaces:  pyscf_interface.py (DFT vs DSE) │
│  │   [rdm.py]      │  Visualization: prl_figures.py (PRL plots)    │
│  │                 │  CLI: memory-dft lattice/thermal/gamma        │
│  │ • compute_2rdm  │                                                │
│  │ • correlations  │                                                │
│  │ • PySCF convert │                                                │
│  └─────────────────┘                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## v0.5.0 Changes: Unified SparseEngine

### Before (v0.4.x): 6 separate files
```
core/
├── operators.py       # Spin operators
├── hamiltonian.py     # Hamiltonian builders
├── hubbard_engine.py  # Hubbard-specific
├── sparse_engine.py   # General sparse
├── lattice.py         # Geometry
└── repulsive_kernel.py # Deprecated
```

### After (v0.5.0): 1 unified file
```
core/
├── memory_kernel.py         # 4-component kernel
├── history_manager.py       # State history
└── sparse_engine_unified.py # ALL MODELS IN ONE!
    ├── SparseEngine          # Main class
    ├── build_heisenberg()    # Models
    ├── build_ising()
    ├── build_xy()
    ├── build_hubbard()
    ├── build_kitaev()
    ├── create_chain()        # Geometry
    ├── create_ladder()
    ├── create_square_lattice()
    ├── compute_lambda()      # Physics
    ├── compute_2rdm()
    └── HubbardEngineCompat   # Backward compat
```

### Backward Compatibility: 100%

All old imports still work:

```python
# Old code (v0.4.x) - STILL WORKS!
from memory_dft import HubbardEngine, SpinOperators
from memory_dft import LatticeGeometry2D, create_chain

# New code (v0.5.0) - Recommended
from memory_dft import SparseEngine

engine = SparseEngine(n_sites=6, use_gpu=True)
geom = engine.build_ladder(Lx=3, Ly=2)
H_K, H_V = engine.build_heisenberg(geom.bonds)
result = engine.compute_full(t=1.0, U=2.0)
```

---

## 4-Component Memory Kernel

The memory kernel decomposes history effects into four physical channels:

| Component | Kernel | Physics | Example |
|-----------|--------|---------|---------|
| **Field** | PowerLaw: `1/(t-τ)^γ` | Long-range correlations | EM fields, collective modes |
| **Phys** | StretchedExp: `e^(-(t/τ₀)^β)` | Structural relaxation | Viscoelastic response |
| **Chem** | Step: `sigmoid(t-t_react)` | Irreversible reactions | Oxidation, bond formation |
| **Exclusion** | `e^(-t/τ_rep)(1-e^(-t/τ_rec))` | Distance direction | Compression history |

```python
from memory_dft import CompositeMemoryKernel, KernelWeights

kernel = CompositeMemoryKernel(
    weights=KernelWeights(
        field=0.30,      # Long-range correlations
        phys=0.25,       # Structural relaxation
        chem=0.25,       # Chemical irreversibility
        exclusion=0.20   # Direction-dependent
    ),
    include_exclusion=True
)
```

---

## Key Results

### 46.7% of Correlations are Non-Markovian

From distance decomposition of the 2-RDM (Hubbard model, U/t=2.0):

| Range | γ | Interpretation |
|-------|---|----------------|
| r ≤ 2 | 1.388 | Local (Markovian) |
| r → ∞ | 2.604 | Total |
| **Memory** | **1.216** | **Non-Markovian (46.7%)** |

→ **Nearly half of exchange-correlation structure requires history.**

### Path Dependence is Real and Measurable

| Test | Δλ (DFT) | Δλ (DSE) |
|------|----------|----------|
| Adsorption order (A→B vs B→A) | 0 | **1.59** |
| Reaction sequence (Ads→React vs React→Ads) | 0 | **2.18** |
| Compression hysteresis | 0 | **18.2% energy** |

Same final structure. **Different quantum outcomes.**

---

## Installation

### From PyPI
```bash
pip install memory-dft
```

### From Source
```bash
git clone https://github.com/miosync-masa/lambda3-memory-dft.git
cd lambda3-memory-dft
pip install -e ".[dev]"
```

### Google Colab
```python
!git clone https://github.com/miosync-masa/lambda3-memory-dft.git
import sys
sys.path.insert(0, '/content/lambda3-memory-dft')
```

---

## Quick Start

### Basic DSE with Unified Engine (v0.5.0)

```python
from memory_dft import SparseEngine, CompositeMemoryKernel, HistoryManager

# Create unified engine (GPU auto-detection)
engine = SparseEngine(n_sites=6, use_gpu=True, verbose=True)

# Build geometry and Hamiltonian
geom = engine.build_chain(L=6)
H_K, H_V = engine.build_heisenberg(geom.bonds, J=1.0)

# Compute λ = K/|V| for ground state
result = engine.compute_full(t=1.0, U=2.0)
print(f"λ = {result.Lambda:.4f}")
print(f"Phase: {result.phase}")
```

### Path Comparison

```python
from memory_dft import ChemicalReactionSolver, ReactionPath

solver = ChemicalReactionSolver(n_sites=4, use_gpu=True)

# Path 1: A first, then B
path1 = ReactionPath("A→B")
path1.add_event('adsorption', 2.0, site=0, species='A')
path1.add_event('adsorption', 5.0, site=2, species='B')

# Path 2: B first, then A (same final state!)
path2 = ReactionPath("B→A")
path2.add_event('adsorption', 2.0, site=2, species='B')
path2.add_event('adsorption', 5.0, site=0, species='A')

result1 = solver.evolve_path(path1, t_total=10.0)
result2 = solver.evolve_path(path2, t_total=10.0)

comparison = solver.compare_paths(result1, result2)
print(f"ΔΛ = {comparison['delta_lambda']:.3f}")  # → 1.594 (DFT: 0)
```

---

## Project Structure (v0.5.0)

```
memory_dft/
├── core/                          # Foundation (UNIFIED!)
│   ├── memory_kernel.py           # 4-component kernel
│   ├── history_manager.py         # ψ(τ), Λ(τ) history
│   └── sparse_engine_unified.py   # ALL models + geometry + GPU
├── solvers/                       # Time Evolution
│   ├── lanczos_memory.py          # Lanczos + memory term
│   ├── time_evolution.py          # High-level API
│   ├── memory_indicators.py       # ΔO, M(t), γ quantification
│   └── chemical_reaction.py       # Surface chemistry
├── physics/                       # Analysis
│   ├── lambda3_bridge.py          # Stability diagnostics (λ, EDR)
│   ├── vorticity.py               # γ decomposition from 2-RDM
│   ├── thermodynamics.py          # Finite-T utilities
│   └── rdm.py                     # 2-RDM computation
├── interfaces/
│   └── pyscf_interface.py         # DFT vs DSE comparison
├── visualization/
│   └── prl_figures.py             # Publication figures
├── cli/                           # Command-line interface
│   └── commands/                  # lattice, thermal, gamma
└── tests/
    ├── test_chemical.py           # A/B/C/D tests
    ├── test_repulsive.py          # E1/E2/E3 tests
    └── test_h2_memory.py          # H2 molecule tests
```

---

## Complete Test Suite

### Chemical Tests

| Test | Description | Result |
|------|-------------|--------|
| **A** | Path dependence | 22.84× amplification |
| **B** | Multi-site scaling | Memory varies with L |
| **C** | Reaction coordinate | ∫\|Δλ\| = 207.16 |
| **D** | Catalyst history | Standard: 0, DSE: 51.07 |

### Repulsive Tests

| Test | Description | Result |
|------|-------------|--------|
| **E1** | Compression hysteresis | 18.2% energy loss |
| **E2** | Path-dependent V | \|Δ∫V dt\| = 7912 |
| **E3** | Quantum memory | 10⁹× amplification |

Run all tests:
```bash
cd memory_dft && python -m pytest tests/ -v
```

---

## Applications

| Phenomenon | DFT | DSE |
|------------|-----|-----|
| Catalyst selectivity vs history | ❌ Cannot explain | ✅ Predicted |
| CV hysteresis | ❌ Phenomenological | ✅ First-principles |
| AFM approach/retract | ❌ Same curve | ✅ Different curves |
| Diamond anvil compression | ❌ Reversible | ✅ Hysteresis |
| White layer formation | ❌ Ad hoc | ✅ Exclusion memory |
| Battery voltage memory | ❌ Ad hoc | ✅ Derived |

---

## Authors

- **Masamichi Iizumi** (飯泉真道) — CEO, Miosync Inc.
- **Tamaki Iizumi** (飯泉環) — Research Partner

---

## Citation

```bibtex
@software{memory_dft,
  author       = {Iizumi, Masamichi and Iizumi, Tamaki},
  title        = {{Direct Schrödinger Evolution: First-Principles
                   History-Dependent Quantum Dynamics}},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {0.5.0},
  doi          = {10.5281/zenodo.18095869},
  url          = {https://doi.org/10.5281/zenodo.18095869}
}
```

---

## License

MIT License

---

## Acknowledgments

We thank Lie & Fullwood for establishing the theoretical framework that motivated this work [PRL 135, 230204 (2025)].

---

<p align="center">
  <b>DFT erases history. DSE remembers.</b>
</p>
