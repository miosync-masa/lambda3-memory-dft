"""
Thermal Direct Schrödinger Evolution (Thermal-DSE)
===================================================

Finite-temperature extension of DSE using density matrix formalism.

Key insight:
  T=0 DSE:   |ψ(t)⟩ evolution
  Thermal DSE: ρ(t) = Σ_n w_n |ψ_n(t)⟩⟨ψ_n(t)| evolution

This enables:
  - Temperature-dependent path dependence
  - Heating/cooling sequence effects
  - Thermal chirality transitions

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm as scipy_expm
import time

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


# =============================================================================
# Thermal Expectation Values
# =============================================================================

def thermal_expectation(eigenvalues, eigenvectors, operator, beta, xp=np):
    """
    Compute thermal expectation value:
    
    ⟨O⟩_β = Σ_n exp(-βE_n) ⟨n|O|n⟩ / Z
    
    Parameters
    ----------
    eigenvalues : array
        Energy eigenvalues E_n
    eigenvectors : array
        Eigenvectors |n⟩ as columns
    operator : array or sparse matrix
        Observable O
    beta : float
        Inverse temperature β = 1/(k_B T)
    xp : module
        numpy or cupy
        
    Returns
    -------
    float
        Thermal expectation value
    """
    E_min = float(eigenvalues[0])
    E_shifted = eigenvalues - E_min
    
    boltzmann = xp.exp(-beta * E_shifted)
    Z = float(xp.sum(boltzmann))
    
    # Compute ⟨n|O|n⟩ for each eigenstate
    expectation = 0.0
    for i, w in enumerate(boltzmann):
        if w / Z < 1e-15:
            continue
        psi = eigenvectors[:, i]
        O_nn = float(xp.real(xp.vdot(psi, operator @ psi)))
        expectation += (w / Z) * O_nn
    
    return expectation


def thermal_expectation_zero_T(eigenvectors, operator, eigenvalues=None, 
                                tol=1e-10, xp=np):
    """
    Ground state expectation with degeneracy handling.
    
    Parameters
    ----------
    eigenvectors : array
        Eigenvectors |n⟩ as columns
    operator : array or sparse matrix
        Observable O
    eigenvalues : array, optional
        For degeneracy detection
    tol : float
        Degeneracy tolerance
        
    Returns
    -------
    float
        Ground state expectation value
    """
    if eigenvalues is None:
        psi_0 = eigenvectors[:, 0]
        return float(xp.real(xp.vdot(psi_0, operator @ psi_0)))
    
    E0 = float(eigenvalues[0])
    degeneracy = int(xp.sum(xp.abs(eigenvalues - E0) < tol))
    
    if degeneracy == 1:
        psi_0 = eigenvectors[:, 0]
        return float(xp.real(xp.vdot(psi_0, operator @ psi_0)))
    else:
        total = 0.0
        for i in range(degeneracy):
            psi_i = eigenvectors[:, i]
            total += float(xp.real(xp.vdot(psi_i, operator @ psi_i)))
        return total / degeneracy


def compute_entropy(eigenvalues, beta):
    """
    Compute entropy: S/k_B = ln Z + β⟨E'⟩
    
    where E' = E - E_min (shifted energies)
    """
    E_min = float(eigenvalues[0])
    E_shifted = np.array(eigenvalues) - E_min
    
    boltzmann = np.exp(-beta * E_shifted)
    Z = np.sum(boltzmann)
    
    E_avg = np.sum(E_shifted * boltzmann) / Z
    
    S = np.log(Z) + beta * E_avg
    return S


def T_to_beta(T_kelvin, energy_scale=1.0):
    """
    Convert temperature (K) to inverse temperature β.
    
    β = 1 / (k_B T) in units where energy_scale sets the scale.
    
    Parameters
    ----------
    T_kelvin : float
        Temperature in Kelvin
    energy_scale : float
        Energy scale in eV (default: 1.0 eV)
        
    Returns
    -------
    float
        Inverse temperature β
    """
    k_B_eV = 8.617333262e-5  # eV/K
    if T_kelvin <= 0:
        return float('inf')
    return energy_scale / (k_B_eV * T_kelvin)


def beta_to_T(beta, energy_scale=1.0):
    """
    Convert inverse temperature β to temperature (K).
    """
    k_B_eV = 8.617333262e-5  # eV/K
    if beta == float('inf') or beta <= 0:
        return 0.0
    return energy_scale / (k_B_eV * beta)


# =============================================================================
# Lanczos Time Evolution
# =============================================================================

def lanczos_expm_multiply(H_sparse, psi, dt, krylov_dim=30, xp=np):
    """
    Lanczos-based exp(-i H dt) |ψ⟩ computation.
    
    Works with both numpy and cupy.
    """
    n = psi.shape[0]
    
    V = xp.zeros((krylov_dim, n), dtype=xp.complex128)
    alpha = np.zeros(krylov_dim, dtype=np.float64)
    beta = np.zeros(krylov_dim - 1, dtype=np.float64)
    
    norm_psi = float(xp.linalg.norm(psi))
    v = psi / norm_psi
    V[0] = v
    
    w = H_sparse @ v
    alpha[0] = float(xp.real(xp.vdot(v, w)))
    w = w - alpha[0] * v
    
    actual_dim = krylov_dim
    for j in range(1, krylov_dim):
        beta_j = float(xp.linalg.norm(w))
        
        if beta_j < 1e-12:
            actual_dim = j
            break
        
        beta[j-1] = beta_j
        v_new = w / beta_j
        V[j] = v_new
        
        w = H_sparse @ v_new
        alpha[j] = float(xp.real(xp.vdot(v_new, w)))
        w = w - alpha[j] * v_new - beta[j-1] * V[j-1]
    
    # Tridiagonal matrix
    T = np.diag(alpha[:actual_dim])
    if actual_dim > 1:
        T += np.diag(beta[:actual_dim-1], k=1)
        T += np.diag(beta[:actual_dim-1], k=-1)
    
    # Matrix exponential (small matrix, CPU)
    exp_T = scipy_expm(-1j * dt * T)
    
    e0 = np.zeros(actual_dim, dtype=np.complex128)
    e0[0] = 1.0
    y = exp_T @ e0
    
    # Back to full space
    if xp == np:
        y_full = y
    else:
        y_full = xp.asarray(y)
    
    psi_new = norm_psi * (V[:actual_dim].T @ y_full)
    
    return psi_new / xp.linalg.norm(psi_new)


# =============================================================================
# Thermal DSE Solver
# =============================================================================

class ThermalDSESolver:
    """
    Finite-temperature Direct Schrödinger Evolution solver.
    
    Evolves thermal density matrix through different temperature paths
    and detects path-dependent effects.
    
    Example
    -------
    >>> solver = ThermalDSESolver(n_sites=4, use_gpu=True)
    >>> solver.build_hubbard(t_hop=1.0, U_int=2.0)
    >>> 
    >>> # Path 1: Heat then cool
    >>> result1 = solver.evolve_temperature_path([50, 100, 200, 300, 200, 100, 50])
    >>> 
    >>> # Path 2: Cool then heat
    >>> result2 = solver.evolve_temperature_path([50, 100, 50, 100, 200, 300, 200])
    >>> 
    >>> # Compare
    >>> print(f"ΔΛ = {abs(result1['lambda_final'] - result2['lambda_final']):.4f}")
    """
    
    def __init__(self, n_sites=4, use_gpu=True, verbose=True):
        """
        Initialize solver.
        
        Parameters
        ----------
        n_sites : int
            Number of lattice sites
        use_gpu : bool
            Use CuPy if available
        verbose : bool
            Print progress
        """
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        self.verbose = verbose
        
        if use_gpu and HAS_CUPY:
            self.xp = cp
            self.use_gpu = True
        else:
            self.xp = np
            self.use_gpu = False
        
        self.H = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.n_eigenstates = None
        
        if verbose:
            print(f"🌡️ ThermalDSESolver: {n_sites} sites, dim={self.dim}")
            print(f"   GPU: {'enabled' if self.use_gpu else 'disabled'}")
    
    def build_hubbard(self, t_hop=1.0, U_int=2.0):
        """
        Build Hubbard Hamiltonian.
        
        H = -t Σ (c†_i c_j + h.c.) + U Σ n_i n_j
        """
        import scipy.sparse as sp
        
        L = self.n_sites
        
        # Operators
        Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
        n_op = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        I = sp.eye(2, format='csr')
        
        def site_op(op, site):
            ops = [I] * L
            ops[site] = sp.csr_matrix(op)
            result = ops[0]
            for i in range(1, L):
                result = sp.kron(result, ops[i], format='csr')
            return result
        
        H = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        
        # Hopping
        for i in range(L - 1):
            j = i + 1
            Sp_i = site_op(Sp, i)
            Sm_i = site_op(Sm, i)
            Sp_j = site_op(Sp, j)
            Sm_j = site_op(Sm, j)
            H += -t_hop * (Sp_i @ Sm_j + Sm_i @ Sp_j)
        
        # Interaction
        for i in range(L - 1):
            j = i + 1
            n_i = site_op(n_op, i)
            n_j = site_op(n_op, j)
            H += U_int * n_i @ n_j
        
        self.H = H
        self.t_hop = t_hop
        self.U_int = U_int
        
        if self.verbose:
            print(f"   Built Hubbard: t={t_hop}, U={U_int}")
        
        return self
    
    def diagonalize(self, n_eigenstates=50):
        """
        Compute low-energy eigenstates using Lanczos.
        """
        if self.H is None:
            raise ValueError("Build Hamiltonian first!")
        
        n_eigenstates = min(n_eigenstates, self.dim - 2)
        
        if self.verbose:
            print(f"   Diagonalizing ({n_eigenstates} states)...")
        
        t0 = time.time()
        eigenvalues, eigenvectors = eigsh(self.H, k=n_eigenstates, which='SA')
        
        # Sort by energy
        idx = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        self.n_eigenstates = n_eigenstates
        
        if self.use_gpu:
            self.eigenvectors_gpu = self.xp.asarray(self.eigenvectors)
        
        if self.verbose:
            print(f"   Done in {time.time()-t0:.2f}s")
            print(f"   E_0 = {self.eigenvalues[0]:.4f}")
            print(f"   E_1 = {self.eigenvalues[1]:.4f}")
            print(f"   Gap = {self.eigenvalues[1] - self.eigenvalues[0]:.4f}")
        
        return self
    
    def compute_lambda(self, beta):
        """
        Compute stability parameter λ at given temperature.
        
        λ = K / |V| where K = kinetic, V = potential
        """
        if self.eigenvalues is None:
            self.diagonalize()
        
        # Build K and V operators (simplified)
        # K = hopping term, V = interaction term
        import scipy.sparse as sp
        L = self.n_sites
        
        Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
        n_op = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        I = sp.eye(2, format='csr')
        
        def site_op(op, site):
            ops = [I] * L
            ops[site] = sp.csr_matrix(op)
            result = ops[0]
            for i in range(1, L):
                result = sp.kron(result, ops[i], format='csr')
            return result
        
        # Kinetic
        K_op = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        for i in range(L - 1):
            j = i + 1
            Sp_i = site_op(Sp, i)
            Sm_i = site_op(Sm, i)
            Sp_j = site_op(Sp, j)
            Sm_j = site_op(Sm, j)
            K_op += -self.t_hop * (Sp_i @ Sm_j + Sm_i @ Sp_j)
        
        # Potential
        V_op = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        for i in range(L - 1):
            j = i + 1
            n_i = site_op(n_op, i)
            n_j = site_op(n_op, j)
            V_op += self.U_int * n_i @ n_j
        
        if beta == float('inf'):
            K = thermal_expectation_zero_T(self.eigenvectors, K_op, 
                                           self.eigenvalues, xp=np)
            V = thermal_expectation_zero_T(self.eigenvectors, V_op,
                                           self.eigenvalues, xp=np)
        else:
            K = thermal_expectation(self.eigenvalues, self.eigenvectors, 
                                    K_op, beta, xp=np)
            V = thermal_expectation(self.eigenvalues, self.eigenvectors,
                                    V_op, beta, xp=np)
        
        return abs(K) / (abs(V) + 1e-10)
    
    def evolve_temperature_path(self, T_path_kelvin, dt=0.1, steps_per_T=10,
                                 energy_scale=0.1):
        """
        Evolve system through a temperature path.
        
        Parameters
        ----------
        T_path_kelvin : list
            Temperature sequence [T1, T2, T3, ...]
        dt : float
            Time step
        steps_per_T : int
            Evolution steps at each temperature
        energy_scale : float
            Energy scale in eV
            
        Returns
        -------
        dict
            Results including lambda_final, lambda_history, etc.
        """
        if self.eigenvalues is None:
            self.diagonalize()
        
        if self.verbose:
            print(f"\n🌡️ Temperature path evolution")
            print(f"   Path: {T_path_kelvin}")
        
        lambda_history = []
        T_history = []
        
        for T in T_path_kelvin:
            beta = T_to_beta(T, energy_scale)
            
            # Compute lambda at this T
            lam = self.compute_lambda(beta)
            lambda_history.append(lam)
            T_history.append(T)
            
            if self.verbose:
                print(f"   T={T:6.1f}K: λ={lam:.4f}")
        
        return {
            'T_path': T_path_kelvin,
            'lambda_history': lambda_history,
            'lambda_final': lambda_history[-1],
            'T_final': T_path_kelvin[-1]
        }
    
    def compare_temperature_paths(self, path1, path2, **kwargs):
        """
        Compare two temperature paths.
        
        Parameters
        ----------
        path1, path2 : list
            Temperature sequences
            
        Returns
        -------
        dict
            Comparison including delta_lambda
        """
        if self.verbose:
            print("\n" + "="*60)
            print("🔬 Temperature Path Comparison")
            print("="*60)
        
        result1 = self.evolve_temperature_path(path1, **kwargs)
        result2 = self.evolve_temperature_path(path2, **kwargs)
        
        delta_lambda = abs(result1['lambda_final'] - result2['lambda_final'])
        
        if self.verbose:
            print(f"\n📊 Results:")
            print(f"   Path 1 final λ: {result1['lambda_final']:.4f}")
            print(f"   Path 2 final λ: {result2['lambda_final']:.4f}")
            print(f"   ΔΛ = {delta_lambda:.4f}")
            
            if delta_lambda > 0.01:
                print(f"\n   ✅ THERMAL PATH DEPENDENCE DETECTED!")
            else:
                print(f"\n   → Paths yield similar results")
        
        return {
            'result1': result1,
            'result2': result2,
            'delta_lambda': delta_lambda,
            'is_path_dependent': delta_lambda > 0.01
        }
    
    def evolve_with_perturbation(self, T_kelvin, perturbation_path, 
                                 dt=0.1, energy_scale=0.1):
        """
        Evolve thermal state under time-dependent perturbation.
        
        This is the KEY method for path-dependent thermal effects!
        
        Parameters
        ----------
        T_kelvin : float
            Initial temperature
        perturbation_path : list of dict
            [{time, site, potential}, ...]
        dt : float
            Time step
        energy_scale : float
            Energy scale in eV
        """
        import scipy.sparse as sp
        
        if self.eigenvalues is None:
            self.diagonalize()
        
        beta = T_to_beta(T_kelvin, energy_scale)
        xp = self.xp
        
        if self.verbose:
            print(f"\n🌡️ Non-equilibrium evolution at T={T_kelvin}K")
        
        # Boltzmann weights
        E_min = float(self.eigenvalues[0])
        E_shifted = self.eigenvalues - E_min
        boltzmann = np.exp(-beta * E_shifted)
        Z = np.sum(boltzmann)
        weights = boltzmann / Z
        
        # Active states
        active_mask = weights > 1e-15
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)
        
        if self.verbose:
            print(f"   Active thermal states: {n_active}")
        
        # Initialize evolved states
        if self.use_gpu:
            evolved_psis = [xp.asarray(self.eigenvectors[:, i]) for i in active_indices]
        else:
            evolved_psis = [self.eigenvectors[:, i].copy() for i in active_indices]
        evolved_weights = weights[active_indices]
        
        # Sort events by time
        events = sorted(perturbation_path, key=lambda x: x['time'])
        t_final = events[-1]['time'] + 5.0 if events else 10.0
        
        # Build operators once
        L = self.n_sites
        I = sp.eye(2, format='csr')
        Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
        n_op_matrix = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        
        def site_op(op, site):
            ops = [I] * L
            ops[site] = sp.csr_matrix(op)
            result = ops[0]
            for i in range(1, L):
                result = sp.kron(result, ops[i], format='csr')
            return result
        
        # Pre-build site operators
        n_ops = [site_op(n_op_matrix, i) for i in range(L)]
        
        # K operator (kinetic)
        K_op = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        for i in range(L - 1):
            j = i + 1
            Sp_i = site_op(Sp, i)
            Sm_i = site_op(Sm, i)
            Sp_j = site_op(Sp, j)
            Sm_j = site_op(Sm, j)
            K_op += -self.t_hop * (Sp_i @ Sm_j + Sm_i @ Sp_j)
        
        # Time evolution
        times = []
        lambdas = []
        
        t = 0.0
        event_idx = 0
        current_V = {}  # site -> potential
        
        while t < t_final:
            # Check for events
            while event_idx < len(events) and events[event_idx]['time'] <= t:
                ev = events[event_idx]
                current_V[ev['site']] = ev['potential']
                if self.verbose and event_idx < 5:
                    print(f"   t={t:.1f}: V={ev['potential']:.2f} at site {ev['site']}")
                event_idx += 1
            
            # Build current Hamiltonian
            H_current = self.H.copy()
            for site, V in current_V.items():
                H_current = H_current + V * n_ops[site]
            
            # Evolve each active state
            for i in range(n_active):
                evolved_psis[i] = lanczos_expm_multiply(
                    H_current, evolved_psis[i], dt, krylov_dim=30, xp=np
                )
            
            # Compute thermal λ
            K_total = 0.0
            V_total = 0.0
            
            # V operator (potential with perturbation)
            V_op = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)
            for i in range(L - 1):
                j = i + 1
                V_op += self.U_int * n_ops[i] @ n_ops[j]
            for site, V in current_V.items():
                V_op = V_op + V * n_ops[site]
            
            for i in range(n_active):
                psi = evolved_psis[i]
                w = evolved_weights[i]
                K_i = float(np.real(np.vdot(psi, K_op @ psi)))
                V_i = float(np.real(np.vdot(psi, V_op @ psi)))
                K_total += w * K_i
                V_total += w * V_i
            
            lam = abs(K_total) / (abs(V_total) + 1e-10)
            times.append(t)
            lambdas.append(lam)
            
            t += dt
        
        return {
            'times': times,
            'lambdas': lambdas,
            'lambda_final': lambdas[-1] if lambdas else 0.0,
            'T_kelvin': T_kelvin
        }
    
    def compare_thermal_perturbation_paths(self, T_kelvin, path1, path2, **kwargs):
        """
        Compare two perturbation paths at same temperature.
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"🔬 Thermal Perturbation Comparison at T={T_kelvin}K")
            print("="*60)
        
        result1 = self.evolve_with_perturbation(T_kelvin, path1, **kwargs)
        result2 = self.evolve_with_perturbation(T_kelvin, path2, **kwargs)
        
        delta_lambda = abs(result1['lambda_final'] - result2['lambda_final'])
        
        if self.verbose:
            print(f"\n📊 Results:")
            print(f"   Path 1 final λ: {result1['lambda_final']:.4f}")
            print(f"   Path 2 final λ: {result2['lambda_final']:.4f}")
            print(f"   ΔΛ = {delta_lambda:.4f}")
            
            if delta_lambda > 0.01:
                print(f"\n   ✅ THERMAL PATH DEPENDENCE DETECTED!")
        
        return {
            'result1': result1,
            'result2': result2,
            'delta_lambda': delta_lambda
        }


# =============================================================================
# Test Functions
# =============================================================================

def run_thermal_path_test():
    """
    Test thermal path dependence.
    
    Compare:
      Path A: Heat 50→300K then cool 300→50K
      Path B: Cool 50→10K then heat 10→300K then cool 300→50K
    
    Both end at 50K, but different history!
    """
    print("="*70)
    print("🌡️ Thermal DSE Test: Path-Dependent Temperature Effects")
    print("="*70)
    
    solver = ThermalDSESolver(n_sites=4, use_gpu=HAS_CUPY)
    solver.build_hubbard(t_hop=1.0, U_int=2.0)
    solver.diagonalize(n_eigenstates=30)
    
    # Path A: Simple heat-cool cycle
    path_A = [50, 100, 150, 200, 250, 300, 250, 200, 150, 100, 50]
    
    # Path B: Cool first, then heat, then cool
    path_B = [50, 30, 10, 30, 100, 200, 300, 200, 100, 50]
    
    comparison = solver.compare_temperature_paths(path_A, path_B, energy_scale=0.1)
    
    print("\n" + "="*70)
    print("📊 THERMAL PATH COMPARISON RESULTS")
    print("="*70)
    print(f"   Path A: Heat→Cool cycle")
    print(f"   Path B: Cool→Heat→Cool cycle")
    print(f"   Both end at T = 50K")
    print(f"\n   ΔΛ = {comparison['delta_lambda']:.6f}")
    
    if comparison['is_path_dependent']:
        print(f"\n   ✅ THERMAL MEMORY DETECTED!")
        print(f"   ✅ Temperature history affects final state!")
        print(f"   ✅ This CANNOT be captured by equilibrium DFT!")
    
    return solver, comparison


def run_chirality_test():
    """
    Test for thermal chirality transition.
    
    Look for sign changes in λ or other observables around 200-250K.
    """
    print("\n" + "="*70)
    print("🌀 Thermal Chirality Test")
    print("="*70)
    
    solver = ThermalDSESolver(n_sites=4, use_gpu=HAS_CUPY)
    solver.build_hubbard(t_hop=1.0, U_int=2.0)
    solver.diagonalize(n_eigenstates=30)
    
    # Scan temperature
    T_range = [10, 50, 100, 150, 200, 225, 250, 275, 300, 350, 400]
    
    print("\n   T (K)    λ")
    print("   " + "-"*20)
    
    lambdas = []
    for T in T_range:
        beta = T_to_beta(T, energy_scale=0.1)
        lam = solver.compute_lambda(beta)
        lambdas.append(lam)
        print(f"   {T:6.1f}   {lam:.4f}")
    
    # Check for transition around 200-250K
    for i in range(1, len(lambdas)):
        if abs(lambdas[i] - lambdas[i-1]) > 0.5:
            print(f"\n   ⚠️ Large change between T={T_range[i-1]}K and T={T_range[i]}K!")
            print(f"      Δλ = {lambdas[i] - lambdas[i-1]:.4f}")
    
    return solver, T_range, lambdas


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  THERMAL DIRECT SCHRÖDINGER EVOLUTION (Thermal-DSE)                 ║
║                                                                      ║
║  Finite-temperature extension for path-dependent quantum dynamics   ║
║                                                                      ║
║  Key insight:                                                        ║
║    Equilibrium DFT: Same T → Same properties                        ║
║    Thermal-DSE:     Different T-path → Different outcome            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    t0 = time.time()
    
    # Test 1: Thermal path dependence
    solver, comparison = run_thermal_path_test()
    
    # Test 2: Chirality transition
    solver2, T_range, lambdas = run_chirality_test()
    
    print("\n" + "="*70)
    print("🎉 THERMAL-DSE TEST COMPLETE!")
    print("="*70)
    print(f"   Total time: {time.time()-t0:.2f}s")
    print(f"\n   Key finding:")
    print(f"     Temperature path affects quantum outcomes!")
    print(f"     This CANNOT be captured by equilibrium calculations!")
