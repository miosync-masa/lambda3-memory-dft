"""
Thermodynamics for Memory-DFT
=============================

Finite-temperature quantum mechanics utilities.

Functions:
  - Temperature ↔ inverse temperature conversion
  - Thermal expectation values
  - Boltzmann statistics
  - Entropy calculation

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Optional, Union, List

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

K_B_EV = 8.617333262e-5   # Boltzmann constant in eV/K
K_B_J = 1.380649e-23      # Boltzmann constant in J/K
H_EV = 4.135667696e-15    # Planck constant in eV·s
HBAR_EV = 6.582119569e-16 # Reduced Planck constant in eV·s


# =============================================================================
# Temperature Conversion
# =============================================================================

def T_to_beta(T_kelvin: float, energy_scale: float = 1.0) -> float:
    """Convert temperature (K) to inverse temperature β."""
    if T_kelvin <= 0:
        return float('inf')
    return energy_scale / (K_B_EV * T_kelvin)


def beta_to_T(beta: float, energy_scale: float = 1.0) -> float:
    """Convert inverse temperature β to temperature (K)."""
    if beta == float('inf') or beta <= 0:
        return 0.0
    return energy_scale / (K_B_EV * beta)


def thermal_energy(T_kelvin: float) -> float:
    """Thermal energy k_B T in eV."""
    return K_B_EV * T_kelvin


# =============================================================================
# Boltzmann Statistics
# =============================================================================

def boltzmann_weights(eigenvalues: np.ndarray, beta: float) -> np.ndarray:
    """Compute Boltzmann weights exp(-β E_n) / Z."""
    if beta == float('inf'):
        E_min = eigenvalues[0]
        weights = np.zeros_like(eigenvalues, dtype=float)
        ground_mask = np.abs(eigenvalues - E_min) < 1e-10
        n_ground = np.sum(ground_mask)
        weights[ground_mask] = 1.0 / n_ground
        return weights
    
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    boltzmann = np.exp(-beta * E_shifted)
    Z = np.sum(boltzmann)
    return boltzmann / Z


def partition_function(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute partition function Z = Σ exp(-β E_n)."""
    if beta == float('inf'):
        return 1.0
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
    """Compute thermal expectation value ⟨O⟩_β."""
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
    """Compute ground state expectation value (T = 0 limit)."""
    if eigenvalues is None:
        psi_0 = eigenvectors[:, 0]
        return float(xp.real(xp.vdot(psi_0, operator @ psi_0)))
    
    E_0 = float(eigenvalues[0])
    degeneracy = int(np.sum(np.abs(eigenvalues - E_0) < tol))
    
    if degeneracy == 1:
        psi_0 = eigenvectors[:, 0]
        return float(xp.real(xp.vdot(psi_0, operator @ psi_0)))
    else:
        total = 0.0
        for i in range(degeneracy):
            psi_i = eigenvectors[:, i]
            total += float(xp.real(xp.vdot(psi_i, operator @ psi_i)))
        return total / degeneracy


def thermal_average_energy(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute thermal average energy ⟨E⟩_β."""
    weights = boltzmann_weights(eigenvalues, beta)
    return np.sum(weights * eigenvalues)


def thermal_energy_variance(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute thermal energy variance ⟨E²⟩ - ⟨E⟩²."""
    weights = boltzmann_weights(eigenvalues, beta)
    E_avg = np.sum(weights * eigenvalues)
    E2_avg = np.sum(weights * eigenvalues**2)
    return E2_avg - E_avg**2


# =============================================================================
# Thermodynamic Quantities
# =============================================================================

def compute_entropy(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute entropy S/k_B from partition function."""
    if beta == float('inf'):
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
    """Compute Helmholtz free energy F = -k_B T ln(Z)."""
    if beta == float('inf'):
        return eigenvalues[0]
    E_min = np.min(eigenvalues)
    E_shifted = eigenvalues - E_min
    Z = np.sum(np.exp(-beta * E_shifted))
    return E_min - np.log(Z) / beta


def compute_heat_capacity(eigenvalues: np.ndarray, beta: float) -> float:
    """Compute heat capacity C_V = β² Var(E) in units of k_B."""
    if beta == float('inf'):
        return 0.0
    var_E = thermal_energy_variance(eigenvalues, beta)
    return beta**2 * var_E


# =============================================================================
# Thermal State Preparation
# =============================================================================

def thermal_density_matrix(eigenvectors: np.ndarray,
                           eigenvalues: np.ndarray,
                           beta: float) -> np.ndarray:
    """Construct thermal density matrix ρ = exp(-βH) / Z."""
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
    """Sample a pure state from thermal ensemble."""
    if rng is None:
        rng = np.random.default_rng()
    weights = boltzmann_weights(eigenvalues, beta)
    n = rng.choice(len(eigenvalues), p=weights)
    return eigenvectors[:, n].copy()


# =============================================================================
# Temperature-Dependent Hamiltonian H(T)
# =============================================================================

class TemperatureDependentHamiltonian:
    """
    Temperature-dependent Hamiltonian H(T).
    
    Physical models:
      - Lattice expansion: J(T) = J₀(1 - α(T - T_ref)/T_ref)
      - Anharmonic effects: U(T) can also vary
      - Debye model: phonon-mediated changes
    
    This is KEY for true thermal path dependence!
    Without H(T), different temperature paths converge to same equilibrium.
    
    Usage:
        h_builder = TemperatureDependentHamiltonian(
            engine, bonds, J0=1.0, alpha=0.001, T_ref=300
        )
        H_300K = h_builder.build(T=300)
        H_500K = h_builder.build(T=500)  # Different H!
    """
    
    def __init__(self, engine, bonds, 
                 J0: float = 1.0, U0: float = 2.0,
                 alpha_J: float = 0.001, alpha_U: float = 0.0,
                 T_ref: float = 300.0,
                 model: str = 'hubbard'):
        """
        Initialize temperature-dependent Hamiltonian builder.
        
        Args:
            engine: SparseEngine instance
            bonds: List of bond pairs
            J0: Base coupling at T_ref
            U0: Base interaction at T_ref  
            alpha_J: Temperature coefficient for J (α > 0: J decreases with T)
            alpha_U: Temperature coefficient for U
            T_ref: Reference temperature (K)
            model: 'hubbard' or 'heisenberg'
        """
        self.engine = engine
        self.bonds = bonds
        self.J0 = J0
        self.U0 = U0
        self.alpha_J = alpha_J
        self.alpha_U = alpha_U
        self.T_ref = T_ref
        self.model = model
        self.xp = engine.xp
    
    def J_eff(self, T: float) -> float:
        """
        Effective coupling J(T).
        
        J(T) = J₀ * (1 - α_J * (T - T_ref) / T_ref)
        
        Physical interpretation:
          - α_J > 0: Lattice expansion weakens bonds at high T
          - α_J < 0: Bonds strengthen at high T (rare)
        """
        return self.J0 * (1 - self.alpha_J * (T - self.T_ref) / self.T_ref)
    
    def U_eff(self, T: float) -> float:
        """
        Effective interaction U(T).
        
        U(T) = U₀ * (1 - α_U * (T - T_ref) / T_ref)
        """
        return self.U0 * (1 - self.alpha_U * (T - self.T_ref) / self.T_ref)
    
    def build(self, T: float):
        """
        Build Hamiltonian at temperature T.
        
        Returns:
            H, H_K, H_V (sparse matrices)
        """
        J_T = self.J_eff(T)
        U_T = self.U_eff(T)
        
        if self.model == 'hubbard':
            H_K, H_V = self.engine.build_hubbard(
                self.bonds, t=J_T, U=U_T, split_KV=True
            )
        elif self.model == 'heisenberg':
            H = self.engine.build_heisenberg(self.bonds, J=J_T)
            # For Heisenberg, split into exchange (K) and field (V) if needed
            H_K = H
            H_V = self.engine.sparse_module.csr_matrix(H.shape, dtype=H.dtype)
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        H = H_K + H_V
        return H, H_K, H_V
    
    def __repr__(self):
        return (f"TemperatureDependentHamiltonian("
                f"model={self.model}, J₀={self.J0}, α_J={self.alpha_J}, "
                f"U₀={self.U0}, α_U={self.alpha_U}, T_ref={self.T_ref}K)")


# =============================================================================
# Thermal Path Evolver with H(T)
# =============================================================================

class ThermalPathEvolver:
    """
    Thermal path evolution with temperature-dependent Hamiltonian.
    
    This is the correct implementation for thermal path dependence!
    
    Key difference from naive approach:
      - Naive: H fixed, only Boltzmann weights change → no path dependence
      - Correct: H(T) changes → true path dependence
    
    Physical picture:
      - As T changes, lattice expands/contracts
      - Bond lengths change → J(T), U(T) change
      - Different paths through T space → different quantum evolution
    
    Usage:
        evolver = ThermalPathEvolver(engine, bonds, alpha_J=0.001)
        
        # Path 1: heat then cool
        result1 = evolver.evolve([50, 150, 300, 150, 50])
        
        # Path 2: cool then heat  
        result2 = evolver.evolve([50, 25, 10, 150, 300, 150, 50])
        
        # Same final T, different λ!
        print(f"Δλ = {abs(result1['lambda_final'] - result2['lambda_final'])}")
    """
    
    def __init__(self, engine, bonds,
                 J0: float = 1.0, U0: float = 2.0,
                 alpha_J: float = 0.001, alpha_U: float = 0.0,
                 T_ref: float = 300.0,
                 energy_scale: float = 0.1,
                 n_eigenstates: int = 14,
                 model: str = 'hubbard',
                 verbose: bool = True):
        """
        Initialize thermal path evolver.
        
        Args:
            engine: SparseEngine instance
            bonds: List of bond pairs
            J0, U0: Base parameters
            alpha_J, alpha_U: Temperature coefficients
            T_ref: Reference temperature
            energy_scale: For β conversion (eV)
            n_eigenstates: Number of eigenstates to track
            model: 'hubbard' or 'heisenberg'
            verbose: Print progress
        """
        self.engine = engine
        self.bonds = bonds
        self.energy_scale = energy_scale
        self.n_eigenstates = n_eigenstates
        self.verbose = verbose
        self.xp = engine.xp
        
        # Build H(T) helper
        self.H_builder = TemperatureDependentHamiltonian(
            engine, bonds, J0, U0, alpha_J, alpha_U, T_ref, model
        )
        
        if verbose:
            print(f"  ThermalPathEvolver: {self.H_builder}")
            print(f"  Energy scale: {energy_scale} eV")
    
    def _diagonalize(self, H, n_states: int):
        """Diagonalize H and return eigenvalues/vectors on CPU."""
        dim = H.shape[0]
        
        # CuPy eigsh requires k < n - 1 with margin
        max_k = max(1, dim - 3)  # Safety margin for CuPy
        n_states = min(n_states, max_k)
        
        # For small matrices, use dense solver
        if dim <= 32 or n_states >= dim - 2:
            if self.engine.use_gpu:
                import cupy as cp
                H_dense = H.toarray()
                eigenvalues, eigenvectors = cp.linalg.eigh(H_dense)
                eigenvalues = eigenvalues[:n_states].get()
                eigenvectors = eigenvectors[:, :n_states]
            else:
                H_dense = H.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
                eigenvalues = eigenvalues[:n_states]
                eigenvectors = eigenvectors[:, :n_states]
            return eigenvalues, eigenvectors
        
        # Sparse eigensolver for larger matrices
        eigenvalues, eigenvectors = self.engine.eigsh(H, k=n_states, which='SA')
        
        # Eigenvalues always to CPU for Boltzmann
        if self.engine.use_gpu:
            eigenvalues_cpu = eigenvalues.get()
        else:
            eigenvalues_cpu = eigenvalues
        
        return eigenvalues_cpu, eigenvectors
    
    def evolve(self, temperatures: List[float],
               dt: float = 0.1,
               steps_per_T: int = 10) -> dict:
        """
        Evolve system along temperature path with H(T).
        
        At each temperature:
          1. Build H(T) with temperature-dependent parameters
          2. Diagonalize to get eigenstates
          3. Time-evolve under H(T)
          4. Compute λ = K/|V|
        
        Args:
            temperatures: Temperature path [T1, T2, ...]
            dt: Time step
            steps_per_T: Evolution steps per temperature
            
        Returns:
            Dictionary with evolution results
        """
        xp = self.xp
        
        if self.verbose:
            print(f"\n  Evolving through {len(temperatures)} temperatures...")
            print(f"  T_start={temperatures[0]}K → T_end={temperatures[-1]}K")
        
        # Initialize at first temperature
        T0 = temperatures[0]
        H, H_K, H_V = self.H_builder.build(T0)
        
        n_states = min(self.n_eigenstates, H.shape[0] - 2)
        eigenvalues, eigenvectors = self._diagonalize(H, n_states)
        
        # Boltzmann weights at T0
        beta0 = T_to_beta(T0, self.energy_scale)
        weights = boltzmann_weights(eigenvalues, beta0)
        
        # Active states
        active_mask = weights > 1e-10
        active_indices = np.where(active_mask)[0]
        
        # Initialize evolved states
        evolved_psis = [eigenvectors[:, i].copy() for i in active_indices]
        evolved_weights = [weights[i] for i in active_indices]
        
        # Track results
        times = []
        lambdas = []
        J_values = []
        t = 0.0
        
        for T_idx, T in enumerate(temperatures):
            # Rebuild H(T) at new temperature - THIS IS KEY!
            H, H_K, H_V = self.H_builder.build(T)
            J_T = self.H_builder.J_eff(T)
            
            if self.verbose and T_idx % max(1, len(temperatures)//5) == 0:
                print(f"    T={T:.1f}K, J={J_T:.4f}")
            
            for step in range(steps_per_T):
                # Compute λ = K/|V|
                K_total = 0.0
                V_total = 0.0
                
                for i, psi in enumerate(evolved_psis):
                    w = evolved_weights[i]
                    K = float(xp.real(xp.vdot(psi, H_K @ psi)))
                    V = float(xp.real(xp.vdot(psi, H_V @ psi)))
                    K_total += w * K
                    V_total += w * V
                
                lam = abs(K_total) / (abs(V_total) + 1e-10)
                
                times.append(t)
                lambdas.append(lam)
                J_values.append(J_T)
                
                # Time evolution under H(T)
                for i in range(len(evolved_psis)):
                    evolved_psis[i] = self._time_evolve(evolved_psis[i], H, dt)
                    evolved_psis[i] = evolved_psis[i] / xp.linalg.norm(evolved_psis[i])
                
                t += dt
        
        return {
            'times': times,
            'lambdas': lambdas,
            'J_values': J_values,
            'lambda_final': lambdas[-1] if lambdas else 0.0,
            'temperatures': temperatures,
            'n_active': len(active_indices),
        }
    
    def _time_evolve(self, psi, H, dt: float):
        """Time evolution: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩."""
        xp = self.xp
        
        try:
            from memory_dft.solvers import lanczos_expm_multiply
            return lanczos_expm_multiply(H, psi, dt, krylov_dim=20)
        except ImportError:
            from scipy.linalg import expm
            if self.engine.use_gpu:
                H_dense = H.toarray().get()
                psi_np = psi.get() if hasattr(psi, 'get') else psi
                U = expm(-1j * H_dense * dt)
                result = U @ psi_np
                return xp.asarray(result)
            else:
                H_dense = H.toarray()
                U = expm(-1j * H_dense * dt)
                return U @ psi
    
    def compare_paths(self, path1_temps: List[float], path2_temps: List[float],
                      dt: float = 0.1, steps_per_T: int = 10) -> dict:
        """
        Compare two temperature paths.
        
        Returns:
            Dictionary with both results and difference
        """
        if self.verbose:
            print(f"\n  Path 1: {path1_temps[0]}K → ... → {path1_temps[-1]}K")
            print(f"  Path 2: {path2_temps[0]}K → ... → {path2_temps[-1]}K")
        
        result1 = self.evolve(path1_temps, dt, steps_per_T)
        result2 = self.evolve(path2_temps, dt, steps_per_T)
        
        delta_lambda = abs(result1['lambda_final'] - result2['lambda_final'])
        
        return {
            'path1': result1,
            'path2': result2,
            'delta_lambda': delta_lambda,
            'is_path_dependent': delta_lambda > 0.01,
        }
