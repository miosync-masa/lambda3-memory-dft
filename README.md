# Memory-DFT

[![Tests](https://github.com/miosync-masa/lambda3-memory-dft/actions/workflows/test.yml/badge.svg)](https://github.com/miosync-masa/lambda3-memory-dft/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/memory-dft.svg)](https://badge.fury.io/py/memory-dft)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**History-Dependent Quantum Dynamics from Direct Schrödinger Evolution**


## Key Results

| Metric | Value | Significance |
|--------|-------|--------------|
| γ_memory | **1.216** | 46.7% of correlations are Non-Markovian! |
| Path Dependence | **22.84x** | Memory-DFT amplifies path effects |
| Catalyst History | **∞** | Standard QM: 0, Memory-DFT distinguishes! |

**Reference:** Lie & Fullwood, PRL 135, 230204 (2025)

## What Memory-DFT Does

Standard DFT says: *Same structure = Same energy*

Memory-DFT says: **Different history = Different energy**

```
❌ Standard DFT: E[ρ(r)]
✅ Memory-DFT:   E[ρ(r), {ρ(r,t')}]  ← includes history!
```

## Theoretical Foundation

### γ Distance Decomposition

```
γ_total (r=∞) = 2.604   ← Full correlations
γ_local (r≤2) = 1.388   ← Markovian (QSOT)
─────────────────────────
γ_memory      = 1.216   ← Non-Markovian extension!
```

### Memory Kernel Hierarchy (H-CSP)

| Kernel | H-CSP Layer | Physics | Form |
|--------|-------------|---------|------|
| K_field | Θ_field | EM, radiation | Power-law (γ≈1.2) |
| K_phys | Θ_env_phys | Structural relaxation | Stretched exp (β≈0.5) |
| K_chem | Θ_env_chem | Chemical reactions | Step function |
| K_rep | Θ_repulsion | **Pauli repulsion (🩲)** | Hysteresis kernel |

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

## Quick Start

```python
from memory_dft import (
    HubbardEngine,
    SimpleMemoryKernel,
    CatalystMemoryKernel,
    RepulsiveMemoryKernel
)

# 4-site Hubbard model
engine = HubbardEngine(L=4)
result = engine.compute_full(t=1.0, U=2.0)
print(f"Λ = {result.lambda_val:.4f}")

# Memory-enhanced calculation
memory = SimpleMemoryKernel(eta=0.3, tau=5.0)
memory.add_state(t=0.0, lambda_val=result.lambda_val, psi=result.psi)

# Later...
delta_lambda = memory.compute_memory_contribution(t=1.0, psi=result.psi)
print(f"Memory contribution: {delta_lambda:.4f}")
```

## Test Suite

### Test A: Path Dependence
```
Same final Hamiltonian, different field paths
→ Memory-DFT shows 22.84x amplification
```

### Test D: Catalyst History
```
Adsorption → Reaction  ≠  Reaction → Adsorption
Standard QM: |ΔΛ| = 0 (cannot distinguish!)
Memory-DFT:  |ΔΛ| = 51.07 ✓
```

### Test E: Repulsive Memory (🩲-derived!)
```
Compression → Release hysteresis
Same atomic position, different V depending on history
```

Run tests:
```bash
cd memory_dft
python -m pytest tests/test_chemical.py -v
python -m pytest tests/test_repulsive.py -v
```

## Applications

| Phenomenon | Traditional DFT | Memory-DFT |
|------------|-----------------|------------|
| Diamond anvil hysteresis | ❌ Cannot explain | ✅ Predicted |
| AFM approach/retract | ❌ Same curve | ✅ Different curves |
| Catalyst reaction order | ❌ Same energy | ✅ Path-dependent |
| Battery voltage hysteresis | ❌ Phenomenological | ✅ First-principles |

## Structure

```
memory_dft/
├── core/
│   ├── memory_kernel.py      # 3-layer Kernel + Catalyst
│   ├── repulsive_kernel.py   # 🩲 Repulsive Memory
│   ├── hubbard_engine.py     # Hubbard model
│   ├── history_manager.py    # History tracking
│   └── sparse_engine.py      # Sparse Hamiltonian
├── solvers/
│   ├── lanczos_memory.py     # Lanczos + Memory
│   └── time_evolution.py     # Time evolution
├── physics/
│   ├── lambda3_bridge.py     # Λ³ theory connection
│   └── vorticity.py          # γ calculation
└── tests/
    ├── test_chemical.py      # Chemical tests (A/B/C/D)
    └── test_repulsive.py     # Repulsive tests (E1/E2/E3)
```

##Five Axioms

1. **Layered Constraint** → Hierarchical Memory kernels
2. **Non-Commutativity** → Reaction order dependence
3. **Global Conservation** → Λ-space conservation
4. **Recursive Generation** → Λ(t+Δt) = F(Λ(t), Λ̇(t))
5. **Pulsative Equilibrium** → Living system signature

## Authors

- **Masamichi Iizumi** (飯泉真道) - CEO, Miosync Inc.
- **Tamaki Iizumi** (飯泉環) - Partner

## License

MIT License

## Citation

```bibtex
@software{memory_dft,
  author = {Iizumi, Masamichi and Iizumi, Tamaki},
  title = {Memory-DFT: History-Dependent Density Functional Theory},
  year = {2024},
  url = {https://github.com/miosync-masa/lambda3-memory-dft},
  note = {Based on H-CSP/Λ³ Theory. Origin: 🩲 → 🧪 → Λ³}
}
```

## Acknowledgments

This theory originated from the observation that "underwear elastic doesn't fully recover" (パンツのゴムが戻らない), leading to a rigorous treatment of memory effects in quantum many-body systems.
