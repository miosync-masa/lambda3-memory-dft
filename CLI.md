# memory-dft CLI Documentation

Command-line interface for **Direct Schrödinger Evolution (DSE)**.

## Installation

```bash
cd lambda3-memory-dft
pip install -e .
```

After installation, the `memory-dft` command is available globally.
Alternatively, run directly with `python cli.py`.

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `info` | Show version and kernel information |
| `run` | Run DSE time evolution simulation |
| `compare` | Compare two evolution paths (Hubbard model demo) |
| `dft-compare` | Compare DFT vs DSE using PySCF ⭐ **REAL DFT!** |
| `gamma` | Compute γ decomposition (memory fraction) |
| `hysteresis` | Analyze compression hysteresis |

---

## `info` - Package Information

Show version, kernel components, and GPU status.

```bash
memory-dft info
```

**Output:**
```
╔═══════════════════════════════════════════════════════════════╗
║           Direct Schrödinger Evolution (DSE)                  ║
║                    memory-dft v0.4.0                          ║
╚═══════════════════════════════════════════════════════════════╝

🧠 Memory Kernel Components (4)
──────────────────────────────────────────────────
  1. PowerLaw (Field)      - Long-range correlations
  2. StretchedExp (Phys)   - Structural relaxation
  3. Step (Chem)           - Irreversible reactions
  4. Exclusion (Direction) - Compression history [NEW]

💡 Key Insight
──────────────────────────────────────────────────
  Same distance r = 0.8 Å has DIFFERENT meaning:
    • Approaching → Low enhancement
    • Departing   → High enhancement (compression memory)
  DFT cannot distinguish. DSE can!
```

---

## `run` - Time Evolution Simulation

Run a DSE time evolution simulation with memory effects.

```bash
memory-dft run [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-L, --sites` | 4 | Number of lattice sites |
| `-T, --time` | 10.0 | Total evolution time |
| `--dt` | 0.1 | Time step |
| `-U` | 2.0 | Hubbard U (interaction strength) |
| `-t` | 1.0 | Hopping parameter |
| `--memory/--no-memory` | ON | Enable/disable memory effects |
| `-o, --output` | None | Output JSON file |
| `-v, --verbose` | False | Verbose output |

**Examples:**

```bash
# Basic run
memory-dft run -L 4 -T 5.0

# Without memory (memoryless mode)
memory-dft run -L 4 -T 5.0 --no-memory

# Save results
memory-dft run -L 4 -T 10.0 -o results.json
```

**Output:**
```
🚀 Running DSE simulation
──────────────────────────────────────────────────
  Sites (L):    4
  Time (T):     5.0
  Memory:       ON

Evolving  [####################################]  100%

📊 Results
──────────────────────────────────────────────────
  Initial λ:    9.1966
  Final λ:      108.2888
  Max λ:        161.4574
  Min λ:        6.7173
  Mean λ:       37.1282
```

---

## `compare` - Path Comparison (Hubbard Model)

Compare two evolution paths using Hubbard model to demonstrate path dependence.

```bash
memory-dft compare --path1 "A,B" --path2 "B,A" [OPTIONS]
```

**Path notation:**
- `A` = Adsorption event
- `B` = Reaction event (or any other event type)

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--path1` | Required | First path (e.g., "A,B") |
| `--path2` | Required | Second path (e.g., "B,A") |
| `-L, --sites` | 4 | Number of sites |
| `-T, --time` | 10.0 | Total time |
| `-o, --output` | None | Output JSON file |

**Example:**

```bash
memory-dft compare --path1 "A,B" --path2 "B,A"
```

**Output:**
```
🔀 Path Comparison (Hubbard Model Demo)
──────────────────────────────────────────────────
  Path 1: A,B
  Path 2: B,A

📊 Results
──────────────────────────────────────────────────

  Path 1 (A,B):
    Memoryless: λ = 9.1966
    With Memory: λ = 21.7928

  Path 2 (B,A):
    Memoryless: λ = 9.1966
    With Memory: λ = 21.7837

  ═══════════════════════════════════════
  |Δλ| Memoryless:    0.000000
  |Δλ| With Memory:   0.0091

  🎯 Memoryless: Cannot distinguish paths! (Δλ ≈ 0)
  🎯 With Memory: REVEALS difference! (Δλ = 0.0091)
```

**Note:** This uses the Hubbard model for educational demonstration.
For real DFT comparison, use `dft-compare`.

---

## `dft-compare` - DFT vs DSE (PySCF) ⭐

**This is the publication-ready feature!**

Compare actual DFT calculations with DSE to demonstrate that:
- DFT gives identical energies for different paths to the same final state
- DSE captures history dependence

**Requires:** `pip install pyscf`

```bash
memory-dft dft-compare [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--mol, -m` | "H2" | Molecule (H2, LiH) |
| `--basis, -b` | "sto-3g" | Basis set (sto-3g, cc-pvdz, etc.) |
| `--xc` | "LDA" | XC functional (LDA, B3LYP, PBE) |
| `--r-stretch` | 1.5 | Max stretch distance (Å) |
| `--r-compress` | 0.5 | Min compress distance (Å) |
| `-n, --steps` | 5 | Steps per path segment |
| `-o, --output` | None | Output JSON file |

**Example:**

```bash
# Quick test
memory-dft dft-compare --mol H2 --steps 3

# Publication quality
memory-dft dft-compare --mol H2 --basis cc-pvdz --xc B3LYP --steps 10
```

**Output:**
```
🔬 DFT vs DSE Comparison (PySCF)
──────────────────────────────────────────────────
  Molecule:    H2
  Basis:       sto-3g
  XC:          LDA
  r_eq:        0.74 Å

============================================================
DSE vs DFT Path Comparison
============================================================

Path 1: Stretch→Return
  E_DFT (final):  -1.025008 Ha
  E_DSE (final):  -1.024160 Ha
  Memory effect:  0.000848 Ha

Path 2: Compress→Return
  E_DFT (final):  -1.025008 Ha
  E_DSE (final):  -1.023480 Ha
  Memory effect:  0.001528 Ha

────────────────────────────────────────────────────────────
|ΔE| DFT:  0.00000000 Ha  (0.0000 eV)
|ΔE| DSE:  0.00068018 Ha  (0.0185 eV)

🎯 DFT: Cannot distinguish paths! (ΔE ≈ 0)
🎯 DSE: REVEALS difference! (ΔE = 0.000680 Ha)
============================================================
```

**Key insight:** The same final geometry (r = 0.74 Å) gives:
- **DFT:** Identical energies (history-blind)
- **DSE:** Different energies depending on path (history-aware)

This is direct evidence that standard DFT cannot capture path-dependent physics!

---

## `gamma` - γ Decomposition

Compute the correlation exponent decomposition to analyze memory fraction.

```bash
memory-dft gamma [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-L, --sizes` | "4,6,8" | Comma-separated system sizes |
| `-U` | 2.0 | Hubbard U |
| `-o, --output` | None | Output JSON file |

**Example:**

```bash
memory-dft gamma --sizes 3,4,5,6
```

**Output:**
```
📈 γ Decomposition Analysis
──────────────────────────────────────────────────
  System sizes: [3, 4, 5, 6]
  U/t: 2.0

  Computing L=3...
    λ = 0.0000, γ_est = 0.000
  Computing L=4...
    λ = 9.1966, γ_est = 1.675
  Computing L=5...
    λ = 20.0241, γ_est = 1.892

📊 Extrapolation (L→∞)
──────────────────────────────────────────────────
  γ_total (extrapolated): 5.052
  γ_local (typical):      ~1.4
  γ_memory (estimated):   ~3.652

  Memory fraction: ~72%
```

**Interpretation:** A high memory fraction indicates strong non-Markovian
effects that standard DFT cannot capture.

---

## `hysteresis` - Compression Hysteresis

Analyze compression hysteresis using the Exclusion Kernel.

```bash
memory-dft hysteresis [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--r-min` | 0.6 | Minimum distance (Å) |
| `--r-max` | 1.2 | Maximum distance (Å) |
| `-n, --steps` | 50 | Steps per half-cycle |
| `--cycles` | 1 | Number of cycles |
| `-o, --output` | None | Output JSON file |

**Example:**

```bash
memory-dft hysteresis --r-min 0.6 --r-max 1.2 --steps 50
```

**Output:**
```
🔄 Compression Hysteresis Analysis
──────────────────────────────────────────────────
  Distance range: 0.6 → 1.2 Å
  Steps: 50 per half-cycle
  Cycles: 1

📊 Results
──────────────────────────────────────────────────
  V_eff at r_min (compress): 1133.1551
  V_eff at r_min (expand):   1175.9304
  ΔV at same point:          42.7753

  Hysteresis area: 1282.8106

  💡 Non-zero area = Memory effect!
     DFT: Area = 0 (no hysteresis)
     DSE: Area > 0 (compression memory)
```

**Key insight:** At the same distance (r = 0.6 Å), the effective potential
differs depending on whether we're compressing or expanding. This is the
**direction memory** captured by the Exclusion Kernel!

---

## Output Files

All commands support `-o, --output` to save results as JSON:

```bash
memory-dft run -L 4 -T 5.0 -o run_results.json
memory-dft compare --path1 "A,B" --path2 "B,A" -o compare_results.json
memory-dft hysteresis -o hysteresis_results.json
```

---

## Help

```bash
# General help
memory-dft --help

# Command-specific help
memory-dft run --help
memory-dft compare --help
```

---

## Quick Start

```bash
# 1. Check installation
memory-dft info

# 2. Run hysteresis demo (fast)
memory-dft hysteresis --r-min 0.6 --r-max 1.2

# 3. Compare paths with Hubbard model (educational)
memory-dft compare --path1 "A,B" --path2 "B,A"

# 4. Compare DFT vs DSE with PySCF (publication-ready!) ⭐
pip install pyscf  # if not installed
memory-dft dft-compare --mol H2 --basis cc-pvdz --xc B3LYP

# 5. Full simulation
memory-dft run -L 4 -T 10.0 --memory
```

---

## Authors

- Masamichi Iizumi
- Tamaki Iizumi

## License

MIT
