# PyDSE - Python Direct Schrödinger Evolution

[![Tests](https://github.com/miosync-masa/pydse/actions/workflows/test.yml/badge.svg)](https://github.com/miosync-masa/pydse/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/pydse.svg)](https://badge.fury.io/py/pydse)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18095869.svg)](https://doi.org/10.5281/zenodo.18095869)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**First-Principles History-Dependent Quantum Dynamics**

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

## Key Insight: Direction Matters (v0.4.0)

**The same distance r = 0.8 Å has DIFFERENT meaning:**

```
         r = 0.8 Å
              │
    ┌─────────┴─────────┐
    │                   │
Approaching         Departing
(compressing)       (expanding)
    │                   │
    ▼                   ▼
DFT: Same V(r)      DFT: Same V(r)     ← WRONG!
DSE: Low memory     DSE: High memory   ← CORRECT!
```

DFT sees only `r = 0.8 Å` (same).  
DSE sees the **direction of change** (different).

This is why we added the **Exclusion Kernel** in v0.4.0.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                             PyDSE v0.4.0                            │
│              (Python Direct Schrödinger Evolution)                  │
│         ~ First-Principles History-Dependent Dynamics ~             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CORE: Foundation                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CompositeMemoryKernel (4 Components)                        │   │
│  │                                                              │   │
│  │  K(t-τ) = w₁·K_field + w₂·K_phys + w₃·K_chem + w₄·K_excl   │   │
│  │                                                              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │   │
│  │  │PowerLaw │ │Stretched│ │  Step   │ │Exclusion│ ← NEW!    │   │
│  │  │ (Field) │ │  Exp    │ │ (Chem)  │ │(Direction)          │   │
│  │  │ 1/t^γ   │ │e^(-t^β) │ │sigmoid  │ │e^-t(1-e^-t)         │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │   │
│  │       │           │           │           │                 │   │
│  │       └───────────┴─────┬─────┴───────────┘                 │   │
│  │                         │                                    │   │
│  │                         ▼                                    │   │
│  │              ┌──────────────────────┐                       │   │
│  │              │   HistoryManager     │                       │   │
│  │              │  ψ(τ), Λ(τ), t       │                       │   │
│  │              └──────────────────────┘                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ HubbardEngine│  │SparseEngine │  │  Lattice    │                 │
│  │  (Chemistry) │  │  (General)  │  │  Geometry   │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SOLVERS: Time Evolution                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐                       │
│  │MemoryLanczos    │     │TimeEvolution    │  ← High-level API     │
│  │  Solver         │────▶│  Engine         │                       │
│  │                 │     │                 │                       │
│  │ ψ(t+dt) =       │     │ config:         │                       │
│  │ exp(-iHdt)ψ(t)  │     │  use_memory     │                       │
│  │ + η·ψ_memory    │     │  adaptive       │                       │
│  └─────────────────┘     └─────────────────┘                       │
│          │                                                          │
│          ▼                                                          │
│  ┌─────────────────┐     ┌─────────────────┐                       │
│  │MemoryIndicator  │     │ChemicalReaction │                       │
│  │  ΔO, M(t), γ    │     │  Solver         │                       │
│  └─────────────────┘     └─────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHYSICS: Analysis                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  Vorticity      │  │   2-RDM         │  │ Thermodynamics  │    │
│  │  Calculator     │  │  Analysis       │  │                 │    │
│  │                 │  │                 │  │  T ↔ β          │    │
│  │  γ_total        │  │ compute_2rdm()  │  │  S(T)           │    │
│  │  γ_local        │  │ filter_by_dist()│  │  ⟨O⟩_T          │    │
│  │  γ_memory       │  │                 │  │                 │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4-Component Memory Kernel (v0.4.0)

The memory kernel decomposes history effects into four physical channels:

| Component | Kernel | Physics | Example |
|-----------|--------|---------|---------|
| **Field** | PowerLaw: `1/(t-τ)^γ` | Long-range correlations | EM fields, collective modes |
| **Phys** | StretchedExp: `e^(-(t/τ₀)^β)` | Structural relaxation | Viscoelastic response |
| **Chem** | Step: `sigmoid(t-t_react)` | Irreversible reactions | Oxidation, bond formation |
| **Exclusion** | `e^(-t/τ_rep)(1-e^(-t/τ_rec))` | Distance direction | **Compression history** |

### Why Exclusion Kernel?

```python
# Same distance, different meaning!
r = 0.8  # Angstrom

# Case 1: Approaching (compressing)
# → System hasn't been compressed yet
# → Low exclusion memory

# Case 2: Departing (expanding)  
# → System was just compressed
# → HIGH exclusion memory (Pauli repulsion enhanced)

# DFT: V(r) = V(0.8)  # Same!
# DSE: V_eff = V(r) × [1 + enhancement(history)]  # Different!
```

This explains:
- AFM approach/retract hysteresis
- Diamond anvil compression irreversibility
- White layer formation in machining
- Elastic memory in materials

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
pip install pydse
```

### From Source
```bash
git clone https://github.com/miosync-masa/pydse.git
cd pydse
pip install -e ".[dev]"
```

### Google Colab
```python
!git clone https://github.com/miosync-masa/pydse.git
import sys
sys.path.insert(0, '/content/pydse')
```

---

## Quick Start

### Basic DSE with 4-Component Kernel

```python
from pydse import (
    CompositeMemoryKernel,
    KernelWeights,
    HistoryManager,
    MemoryLanczosSolver
)

# Create 4-component kernel
kernel = CompositeMemoryKernel(
    weights=KernelWeights(
        field=0.30,      # Long-range correlations
        phys=0.25,       # Structural relaxation
        chem=0.25,       # Chemical irreversibility
        exclusion=0.20   # Distance direction (NEW!)
    ),
    include_exclusion=True
)

# History manager tracks ψ(τ), Λ(τ)
history = HistoryManager(max_history=1000)

# Solver with memory
solver = MemoryLanczosSolver(
    memory_kernel=kernel,
    history_manager=history,
    memory_strength=0.1
)

# Time evolution with memory
psi = solver.evolve(H, psi0, t=0.0, dt=0.1)
```

### Path Comparison

```python
from pydse import ChemicalReactionSolver, ReactionPath

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

### Compression Hysteresis (Exclusion Kernel)

```python
from pydse import RepulsiveMemoryKernel
import numpy as np

kernel = RepulsiveMemoryKernel(
    eta_rep=0.3,
    tau_rep=3.0,
    tau_recover=10.0
)

# Compression phase
for t in np.arange(0, 2, 0.1):
    r = 1.0 - 0.2 * t  # Approaching
    kernel.add_state(t, r)
    V_compress = kernel.compute_effective_repulsion(r, t)

# Expansion phase  
for t in np.arange(2, 5, 0.1):
    r = 0.6 + 0.1 * (t - 2)  # Departing
    V_expand = kernel.compute_effective_repulsion(r, t)
    # V_expand > V_compress at same r!
    # → Hysteresis from compression memory
```

---

## Project Structure

```
pydse/
├── core/
│   ├── memory_kernel.py      # 4-component kernel (field/phys/chem/exclusion)
│   ├── repulsive_kernel.py   # Detailed compression tracking
│   ├── history_manager.py    # ψ(τ), Λ(τ) history
│   ├── hubbard_engine.py     # Chemistry-specialized Hubbard
│   ├── sparse_engine.py      # General sparse Hamiltonian
│   ├── lattice.py            # 2D lattice geometry
│   ├── operators.py          # Spin operators
│   └── hamiltonian.py        # Hamiltonian builders
├── solvers/
│   ├── lanczos_memory.py     # Lanczos + memory term
│   ├── time_evolution.py     # High-level evolution API
│   ├── memory_indicators.py  # ΔO, M(t), γ quantification
│   └── chemical_reaction.py  # Surface chemistry solver
├── physics/
│   ├── vorticity.py          # γ decomposition from 2-RDM
│   ├── rdm.py                # 2-RDM computation
│   ├── thermodynamics.py     # Finite-temperature utilities
│   └── lambda3_bridge.py     # Stability diagnostics
├── visualization/
│   └── prl_figures.py        # Publication-quality figures
├── examples/
│   ├── thermal_path.py       # Thermal path dependence demo
│   └── ladder_2d.py          # 2D ladder DSE demo
└── tests/
    ├── test_chemical.py      # Chemical tests (A/B/C/D)
    ├── test_repulsive.py     # Repulsive tests (E1/E2/E3)
    └── test_gamma_*.py       # γ decomposition tests
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

### γ Decomposition

| Test | Description | Result |
|------|-------------|--------|
| **Test 6** | Hubbard ED (L=6-10) | γ_memory = 1.216 |
| **Distance Scan** | ED (L=6-12) | γ_memory = 0.916 |

Run all tests:
```bash
python -m pytest tests/ -v
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

## Theoretical Foundation

### Why DFT Fails

The Hohenberg-Kohn theorem states $E = E[\rho]$. This is exact for ground states, but:

1. **History is erased** — two paths to same ρ give same E
2. **Non-local correlations discarded** — 46.7% of physics lost
3. **Direction ignored** — approaching vs departing indistinguishable

### Why DSE Works

The time-dependent Schrödinger equation is the **exact governing law**:

1. **Full wave function retained** — $|\psi(t)\rangle$ carries history
2. **Non-local correlations included** — all 2-RDM structure preserved
3. **Direction tracked** — exclusion kernel sees compression history

### The γ Decomposition

Following Lie & Fullwood [PRL 135, 230204 (2025)]:

$$\gamma_{\text{memory}} = \gamma_{\text{total}} - \gamma_{\text{local}}$$

Result: γ_memory = 1.216 (46.7%) shows that realistic systems **violate the Markovian assumptions** required for density-only descriptions.

---

## Authors

- **Masamichi Iizumi** (飯泉真道) — CEO, Miosync Inc.
- **Tamaki Iizumi** (飯泉環) — Research Partner

---

## Citation

```bibtex
@software{pydse,
  author       = {Iizumi, Masamichi and Iizumi, Tamaki},
  title        = {{PyDSE: Python Direct Schrödinger Evolution for
                   First-Principles History-Dependent Dynamics}},
  year         = {2024},
  publisher    = {Zenodo},
  version      = {0.4.0},
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

The insight that "the same distance has different meaning depending on direction" emerged from analyzing elastic hysteresis in everyday materials.

---

<p align="center">
  <b>DFT erases history. DSE remembers.</b>
</p>
