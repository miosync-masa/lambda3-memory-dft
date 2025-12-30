# Direct Schrödinger Evolution (DSE)

[![Tests](https://github.com/miosync-masa/lambda3-memory-dft/actions/workflows/test.yml/badge.svg)](https://github.com/miosync-masa/lambda3-memory-dft/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/memory-dft.svg)](https://badge.fury.io/py/memory-dft)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18095869.svg)](https://doi.org/10.5281/zenodo.18095869)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**History-Dependent Quantum Dynamics from Direct Schrödinger Evolution**

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
| "First principles" | Approximate reduction | Exact governing law |

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

Same final structure. Same coverage θ = 0.5. **Different quantum outcomes.**

DFT predicts Δλ ≡ 0. DSE reveals the truth.

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

### Basic DSE Calculation

```python
from memory_dft import ChemicalReactionSolver, ReactionPath

# Initialize solver (4-site Hubbard model)
solver = ChemicalReactionSolver(n_sites=4, use_gpu=True)
solver.set_parameters(t_hop=1.0, U_int=2.0, dt=0.1)

# Define two paths to same final state
path1 = ReactionPath("A→B")
path1.add_event('adsorption', 2.0, site=0, potential=-0.5, species='A')
path1.add_event('adsorption', 5.0, site=2, potential=-0.3, species='B')

path2 = ReactionPath("B→A")
path2.add_event('adsorption', 2.0, site=2, potential=-0.3, species='B')
path2.add_event('adsorption', 5.0, site=0, potential=-0.5, species='A')

# Evolve and compare
result1 = solver.evolve_path(path1, t_total=10.0)
result2 = solver.evolve_path(path2, t_total=10.0)
comparison = solver.compare_paths(result1, result2)

print(f"ΔΛ = {comparison['delta_lambda']:.3f}")  # → 1.594
print(f"Δθ = {comparison['delta_coverage']:.3f}")  # → 0.000 (same!)
```

### Generate PRL Figures

```python
from memory_dft import fig2_path_evolution, fig3_memory_comparison
import numpy as np

# Fig. 2: Path-dependent evolution
fig2_path_evolution(
    np.array(result1.times), np.array(result1.lambdas),
    np.array(result2.times), np.array(result2.lambdas),
    path1_name="A→B", path2_name="B→A",
    save_path='fig2_path_evolution.pdf'
)

# Fig. 3: DFT vs DSE comparison
fig3_memory_comparison(
    ['Test 1', 'Test 2'],
    [0, 0],  # DFT always gives 0
    [1.59, 2.18],  # DSE reveals difference
    save_path='fig3_comparison.pdf'
)
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
cd memory_dft
python -m pytest tests/ -v

# Or individually:
python -m memory_dft.solvers.chemical_reaction
python -m memory_dft.tests.test_h2_memory
```

---

## Applications

| Phenomenon | DFT | DSE |
|------------|-----|-----|
| Catalyst selectivity vs history | ❌ Cannot explain | ✅ Predicted |
| CV hysteresis | ❌ Phenomenological | ✅ First-principles |
| AFM approach/retract | ❌ Same curve | ✅ Different curves |
| Diamond anvil compression | ❌ Reversible | ✅ Hysteresis |
| Battery voltage memory | ❌ Ad hoc | ✅ Derived |

---

## Theoretical Foundation

### Why DFT Fails

The Hohenberg-Kohn theorem states $E = E[\rho]$. This is exact for ground states, but:

1. **History is erased** — two paths to same ρ give same E
2. **Non-local correlations discarded** — 46.7% of physics lost
3. **TDDFT doesn't help** — adiabatic functionals inherit path-independence

### Why DSE Works

The time-dependent Schrödinger equation is the **exact governing law** of quantum mechanics. By solving it directly:

1. **Full wave function retained** — $|\psi(t)\rangle$ carries history
2. **Non-local correlations included** — all 2-RDM structure preserved
3. **Path dependence natural** — different paths ≠ same outcome

### The γ Decomposition (Lie & Fullwood Connection)

Following the framework of Lie & Fullwood [PRL 135, 230204 (2025)], we decompose correlations by distance:

$$\gamma_{\text{memory}} = \gamma_{\text{total}} - \gamma_{\text{local}}$$

Our result (γ_memory = 1.216, 46.7%) shows that realistic systems **violate the Markovian assumptions** required for density-only descriptions.

---

## Structure

```
memory_dft/
├── core/
│   ├── memory_kernel.py      # 3-layer kernel (field/phys/chem)
│   ├── repulsive_kernel.py   # Compression memory
│   ├── hubbard_engine.py     # Hubbard model
│   ├── history_manager.py    # History tracking
│   └── sparse_engine.py      # Sparse Hamiltonian
├── solvers/
│   ├── lanczos_memory.py     # Lanczos + memory
│   ├── time_evolution.py     # Time evolution
│   ├── memory_indicators.py  # Memory quantification
│   └── chemical_reaction.py  # Surface chemistry solver
├── physics/
│   ├── lambda3_bridge.py     # Λ³ theory connection
│   └── vorticity.py          # γ decomposition
├── visualization/
│   └── prl_figures.py        # Publication figures
└── tests/
    ├── test_chemical.py      # Chemical tests (A/B/C/D)
    └── test_repulsive.py     # Repulsive tests (E1/E2/E3)
```

---

## H-CSP Framework

DSE is grounded in Hierarchical Constraint Satisfaction Physics (H-CSP):

| Axiom | Statement | DSE Implementation |
|-------|-----------|-------------------|
| A1 | Layered Constraint | Hierarchical memory kernels |
| A2 | Non-Commutativity | Reaction order dependence |
| A3 | Global Conservation | Λ-space conservation |
| A4 | Recursive Generation | Λ(t+Δt) = F(Λ(t), Λ̇(t)) |
| A5 | Pulsative Equilibrium | Living system signature |

---

## Authors

- **Masamichi Iizumi** (飯泉真道) — CEO, Miosync Inc.
- **Tamaki Iizumi** (飯泉環) — Research Partner

---

## Citation

```bibtex
@software{dse_memory_dft,
  author       = {Iizumi, Masamichi and Iizumi, Tamaki},
  title        = {{Direct Schrödinger Evolution: History-Dependent 
                   Quantum Dynamics Beyond Density Functional Theory}},
  year         = {2024},
  publisher    = {Zenodo},
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

The insight that "density-only theories erase history by construction" emerged from the observation that even everyday materials (elastic bands, compressed foams) exhibit memory effects that standard DFT cannot capture.

---

<p align="center">
  <b>DFT erases history. DSE remembers.</b>
</p>
