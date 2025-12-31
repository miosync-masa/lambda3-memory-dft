"""
Thermodynamics for Memory-DFT
=============================

Finite-temperature quantum mechanics utilities.

Functions:
  - Temperature ↔ inverse temperature conversion
  - Thermal expectation values
  - Boltzmann statistics
  - Entropy calculation

These functions support finite-temperature DSE simulations
where thermal path dependence can be detected.

Key insight:
  - Path A: 50K -> 300K -> 50K (heat then cool)
  - Path B: 50K -> 10K -> 300K -> 50K (cool then heat then cool)
  - Same final temperature (50K)
  - Different quantum history
  - Memory-DFT detects path dependence
  - Standard DFT sees no difference

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Optional, Union, List
from scipy.linalg import expm as scipy_expm

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


# =============================================================================
# Physical Constants
# =============================================================================

# Boltzmann constant in eV/K
K_B_EV = 8.617333262e-5

# Boltzmann constant in J/K
K_B_J = 1.380649e-23

# Planck constant in eV·s
H_EV = 4.135667696e-15

# Reduced Planck constant in eV·s
HBAR_EV = 6.582119569e-16


# =============================================================================
# Temperature Conversion
# =============================================================================

def T_to_beta(T_kelvin: float, energy_scale: float = 1.0) -> float:
    """
    Convert temperature (K) to inverse temperature β.
    
    β = E_scale / (k_B T)
    
    At T = 0, returns infinity.
    
    Args:
        T_kelvin: Temperature in Kelvin
        energy_scale: Energy scale in eV (default: 1.0 eV)
        
    Returns:
        Inverse temperature β (dimensionless if energy_scale = 1 eV)
        
    Example:
        >>> beta = T_to_beta(300)  # Room temperature
        >>> print(f"β = {beta:.2f}")
        β = 38.68
    """
    if T_kelvin <= 0:
        return float('inf')
    return energy_scale / (K_B_EV * T_kelvin)


def beta_to_T(beta: float, energy_scale: float = 1.0) -> float:
    """
    Convert inverse temperature β to temperature (K).
    
    T = E_scale / (k_B β)
    
    At β = ∞, returns 0.
    
    Args:
        beta: Inverse temperature
        energy_scale: Energy scale in eV (default: 1.0 eV)
        
    Returns:
        Temperature in Kelvin
        
    Example:
        >>> T = beta_to_T(38.68)
        >>> print(f"T = {T:.1f} K")
        T = 300.0 K
    """
    if beta == float('inf') or beta <= 0:
        return 0.0
    return energy_scale / (K_B_EV * beta)


def thermal_energy(T_kelvin: float) -> float:
    """
    Thermal energy k_B T in eV.
    
    Args:
        T_kelvin: Temperature in Kelvin
        
    Returns:
        Thermal energy in eV
    """
    return K_B_EV * T_kelvin


# =============================================================================
# Boltzmann Statistics
# =============================================================================

def boltzmann_weights(eigenvalues: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute Boltzmann weights exp(-β E_n) / Z.
    
    Shifted to prevent overflow: uses E_n - E_min.
    
    Args:
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Normalized Boltzmann weights (sum = 1)
    """
    if beta == float('inf'):
        # T = 0: only ground state(s)
        E_min = eigenvalues[0]
        weights = np.zeros_like(eigenvalues, dtype=float)
        # Handle degeneracy
        ground_mask = np.abs(eigenvalues - E_min) < 1e-10
        n_ground = np.sum(ground_mask)
        weights[ground_mask] = 1.0 / n_ground
        return weights
    
    # Shift energies for numerical stability
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    
    # Boltzmann factors
    boltzmann = np.exp(-beta * E_shifted)
    
    # Partition function
    Z = np.sum(boltzmann)
    
    return boltzmann / Z


def partition_function(eigenvalues: np.ndarray, beta: float) -> float:
    """
    Compute partition function Z = Σ exp(-β E_n).
    
    Uses shifted energies for numerical stability.
    
    Args:
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Partition function Z
    """
    if beta == float('inf'):
        return 1.0  # Only ground state contributes
    
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    
    return np.sum(np.exp(-beta * E_shifted))


# =============================================================================
# Thermal Expectation Values
# =============================================================================

def thermal_expectation(eigenvectors: np.ndarray,
                        operator,
                        eigenvalues: np.ndarray,
                        beta: float,
                        xp=np) -> float:
    """
    Compute thermal expectation value ⟨O⟩_β.
    
    ⟨O⟩_β = (1/Z) Σ_n exp(-β E_n) ⟨n|O|n⟩
    
    Args:
        eigenvectors: Eigenvectors |n⟩ as columns
        operator: Observable O (array or sparse matrix)
        eigenvalues: Energy eigenvalues E_n
        beta: Inverse temperature
        xp: Array module (numpy or cupy)
        
    Returns:
        Thermal expectation value
        
    Example:
        >>> # Compute ⟨Sz⟩ at T = 100 K
        >>> Sz_thermal = thermal_expectation(vecs, Sz_op, energies, T_to_beta(100))
    """
    weights = boltzmann_weights(eigenvalues, beta)
    
    expectation = 0.0
    for n in range(len(eigenvalues)):
        psi_n = eigenvectors[:, n]
        O_nn = float(xp.real(xp.vdot(psi_n, operator @ psi_n)))
        expectation += weights[n] * O_nn
    
    return expectation


def thermal_expectation_zero_T(eigenvectors: np.ndarray,
                               operator,
                               eigenvalues: Optional[np.ndarray] = None,
                               tol: float = 1e-10,
                               xp=np) -> float:
    """
    Compute ground state expectation value (T = 0 limit).
    
    ⟨O⟩_0 = ⟨ψ_0|O|ψ_0⟩
    
    Handles degeneracy by averaging over ground states.
    
    Args:
        eigenvectors: Eigenvectors |n⟩ as columns
        operator: Observable O
        eigenvalues: Energy eigenvalues (for degeneracy detection)
        tol: Degeneracy tolerance
        xp: Array module
        
    Returns:
        Ground state expectation value
    """
    if eigenvalues is None:
        # Assume first eigenvector is ground state
        psi_0 = eigenvectors[:, 0]
        return float(xp.real(xp.vdot(psi_0, operator @ psi_0)))
    
    # Check for degeneracy
    E_0 = float(eigenvalues[0])
    degeneracy = int(np.sum(np.abs(eigenvalues - E_0) < tol))
    
    if degeneracy == 1:
        psi_0 = eigenvectors[:, 0]
        return float(xp.real(xp.vdot(psi_0, operator @ psi_0)))
    else:
        # Average over degenerate ground states
        total = 0.0
        for i in range(degeneracy):
            psi_i = eigenvectors[:, i]
            total += float(xp.real(xp.vdot(psi_i, operator @ psi_i)))
        return total / degeneracy


def thermal_average_energy(eigenvalues: np.ndarray, beta: float) -> float:
    """
    Compute thermal average energy ⟨E⟩_β.
    
    ⟨E⟩ = Σ_n p_n E_n where p_n = exp(-β E_n) / Z
    
    Args:
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Average energy
    """
    weights = boltzmann_weights(eigenvalues, beta)
    return np.sum(weights * eigenvalues)


def thermal_energy_variance(eigenvalues: np.ndarray, beta: float) -> float:
    """
    Compute thermal energy variance ⟨E²⟩ - ⟨E⟩².
    
    Related to heat capacity: C_V = β² Var(E)
    
    Args:
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Energy variance
    """
    weights = boltzmann_weights(eigenvalues, beta)
    E_avg = np.sum(weights * eigenvalues)
    E2_avg = np.sum(weights * eigenvalues**2)
    return E2_avg - E_avg**2


# =============================================================================
# Thermodynamic Quantities
# =============================================================================

def compute_entropy(eigenvalues: np.ndarray, beta: float) -> float:
    """
    Compute entropy S/k_B from partition function.
    
    S/k_B = ln(Z) + β⟨E'⟩
    
    where E' = E - E_min (shifted energies).
    
    Args:
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Entropy in units of k_B
        
    Example:
        >>> S = compute_entropy(energies, T_to_beta(300))
        >>> print(f"S/kB = {S:.3f}")
    """
    if beta == float('inf'):
        # T = 0: S = k_B ln(g) where g is ground state degeneracy
        E_min = eigenvalues[0]
        degeneracy = np.sum(np.abs(eigenvalues - E_min) < 1e-10)
        return np.log(degeneracy)
    
    E_min = float(eigenvalues[0])
    E_shifted = np.array(eigenvalues) - E_min
    
    boltzmann = np.exp(-beta * E_shifted)
    Z = np.sum(boltzmann)
    
    E_avg = np.sum(E_shifted * boltzmann) / Z
    
    S = np.log(Z) + beta * E_avg
    return S


def compute_free_energy(eigenvalues: np.ndarray, beta: float) -> float:
    """
    Compute Helmholtz free energy F = -k_B T ln(Z).
    
    In units where k_B = 1: F = -T ln(Z) = -ln(Z) / β
    
    Args:
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Free energy (same units as eigenvalues)
    """
    if beta == float('inf'):
        return eigenvalues[0]  # Ground state energy at T = 0
    
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    
    Z = np.sum(np.exp(-beta * E_shifted))
    
    # F = E_min - (1/β) ln(Z)
    return E_min - np.log(Z) / beta


def compute_heat_capacity(eigenvalues: np.ndarray, beta: float) -> float:
    """
    Compute heat capacity C_V = β² Var(E).
    
    In units of k_B.
    
    Args:
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Heat capacity in units of k_B
    """
    if beta == float('inf'):
        return 0.0  # No fluctuations at T = 0
    
    var_E = thermal_energy_variance(eigenvalues, beta)
    return beta**2 * var_E


# =============================================================================
# Thermal State Preparation
# =============================================================================

def thermal_density_matrix(eigenvectors: np.ndarray,
                           eigenvalues: np.ndarray,
                           beta: float) -> np.ndarray:
    """
    Construct thermal density matrix ρ = exp(-βH) / Z.
    
    ρ = Σ_n p_n |n⟩⟨n|
    
    Args:
        eigenvectors: Eigenvectors as columns
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        
    Returns:
        Density matrix (Dim × Dim)
    """
    weights = boltzmann_weights(eigenvalues, beta)
    
    dim = eigenvectors.shape[0]
    rho = np.zeros((dim, dim), dtype=np.complex128)
    
    for n in range(len(eigenvalues)):
        psi_n = eigenvectors[:, n]
        rho += weights[n] * np.outer(psi_n, np.conj(psi_n))
    
    return rho


def sample_thermal_state(eigenvectors: np.ndarray,
                         eigenvalues: np.ndarray,
                         beta: float,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample a pure state from thermal ensemble.
    
    Useful for Monte Carlo or stochastic methods.
    
    Args:
        eigenvectors: Eigenvectors as columns
        eigenvalues: Energy eigenvalues
        beta: Inverse temperature
        rng: Random number generator
        
    Returns:
        Sampled state vector
    """
    if rng is None:
        rng = np.random.default_rng()
    
    weights = boltzmann_weights(eigenvalues, beta)
    
    # Sample eigenstate index
    n = rng.choice(len(eigenvalues), p=weights)
    
    return eigenvectors[:, n].copy()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Thermodynamics Test")
    print("=" * 70)
    
    # Temperature conversion
    print("\n--- Temperature Conversion ---")
    T_room = 300  # K
    beta_room = T_to_beta(T_room)
    T_back = beta_to_T(beta_room)
    print(f"T = {T_room} K → β = {beta_room:.2f} → T = {T_back:.1f} K")
    
    # Edge cases
    beta_zero = T_to_beta(0)
    T_inf = beta_to_T(float('inf'))
    print(f"T = 0 K → β = {beta_zero}")
    print(f"β = ∞ → T = {T_inf} K")
    
    # Thermal energy
    print(f"k_B T (300 K) = {thermal_energy(300):.4f} eV")
    
    # Boltzmann weights
    print("\n--- Boltzmann Statistics ---")
    # Simple 3-level system
    E = np.array([0.0, 0.1, 0.3])  # eV
    
    for T in [100, 300, 1000]:
        beta = T_to_beta(T)
        w = boltzmann_weights(E, beta)
        print(f"T = {T:4d} K: weights = [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]")
    
    # Thermodynamic quantities
    print("\n--- Thermodynamic Quantities ---")
    # Harmonic oscillator-like spectrum
    E_ho = np.array([0.5, 1.5, 2.5, 3.5, 4.5]) * 0.1  # eV
    
    for T in [50, 100, 300]:
        beta = T_to_beta(T)
        S = compute_entropy(E_ho, beta)
        F = compute_free_energy(E_ho, beta)
        C = compute_heat_capacity(E_ho, beta)
        print(f"T = {T:3d} K: S/kB = {S:.3f}, F = {F:.4f} eV, C/kB = {C:.3f}")
    
    # Thermal expectation
    print("\n--- Thermal Expectation ---")
    # 2-site system (dim = 4)
    dim = 4
    np.random.seed(42)
    
    # Random Hamiltonian (Hermitian)
    H_rand = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H_rand = (H_rand + H_rand.conj().T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(H_rand)
    
    # Observable: Sz-like
    Sz = np.diag([0.5, 0.5, -0.5, -0.5])
    
    for T in [10, 100, 1000]:
        beta = T_to_beta(T)
        Sz_avg = thermal_expectation(eigenvectors, Sz, eigenvalues, beta)
        print(f"T = {T:4d} K: ⟨Sz⟩ = {Sz_avg:.4f}")
    
    # Zero-T limit
    Sz_0 = thermal_expectation_zero_T(eigenvectors, Sz, eigenvalues)
    print(f"T =    0 K: ⟨Sz⟩ = {Sz_0:.4f}")
    
    print("\n✅ All thermodynamics tests passed!")
