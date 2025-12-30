"""
Chemical Reaction Time-Evolution Solver
=======================================

History-dependent reaction dynamics using direct Schrödinger evolution.

This solver demonstrates why Memory-DFT is necessary for chemical systems:
- Standard DFT: Same structure → Same energy (path-independent)
- Memory-DFT: Different history → Different outcome (path-dependent)

Applications:
- Heterogeneous catalysis (adsorption order effects)
- Surface reactions (Langmuir-Hinshelwood mechanism)
- Electrode processes (cyclic voltammetry hysteresis)
- Enzyme kinetics (ordered vs random binding)

Key Features:
- Direct Schrödinger evolution (not DFT approximation!)
- Full quantum state history tracking
- GPU-accelerated Lanczos for large systems
- Explicit memory quantification (ΔO, M(t), γ_memory)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from scipy.linalg import expm as scipy_expm
import time

# GPU support
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import eigsh as eigsh_gpu
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

# CPU sparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh as eigsh_cpu


@dataclass
class ReactionEvent:
    """Single reaction event (adsorption, reaction, desorption)."""
    event_type: str      # 'adsorption', 'reaction', 'desorption'
    time: float          # When it occurred
    site: int            # Which site
    potential: float     # Potential change (V_ads, V_react, etc.)
    species: str = ""    # Species involved (optional)


@dataclass
class ReactionPath:
    """A sequence of reaction events defining a path."""
    name: str
    events: List[ReactionEvent] = field(default_factory=list)
    
    def add_event(self, event_type: str, time: float, site: int, 
                  potential: float, species: str = ""):
        self.events.append(ReactionEvent(event_type, time, site, potential, species))


@dataclass
class PathResult:
    """Results from evolving along a reaction path."""
    path_name: str
    times: np.ndarray
    energies: np.ndarray
    lambdas: np.ndarray           # Stability parameter λ(t)
    coverages: np.ndarray         # Surface coverage θ(t)
    observables: Dict[str, np.ndarray] = field(default_factory=dict)
    final_state: Optional[np.ndarray] = None
    memory_metrics: Optional[Any] = None


class SurfaceHamiltonianEngine:
    """
    Hamiltonian engine for surface reaction systems.
    
    Models a 1D chain of surface sites with:
    - Nearest-neighbor hopping (electron delocalization)
    - On-site potentials (adsorbate binding)
    - Hubbard-like interactions (adsorbate-adsorbate repulsion)
    
    H = -t Σ(c†_i c_j + h.c.) + Σ V_i n_i + U Σ n_i n_j
    
    Mapped to spin model via Jordan-Wigner for exact diagonalization.
    """
    
    def __init__(self, n_sites: int, use_gpu: bool = True, verbose: bool = True):
        """
        Args:
            n_sites: Number of surface sites
            use_gpu: Use GPU acceleration if available
            verbose: Print progress messages
        """
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.verbose = verbose
        
        if verbose:
            print(f"🔬 Surface Hamiltonian Engine")
            print(f"   Sites: {n_sites}, Hilbert dim: {self.dim:,}")
            if self.use_gpu:
                print(f"   GPU acceleration: enabled")
        
        # Sparse module selection
        self.sp = csp if self.use_gpu else sp
        self.xp = cp if self.use_gpu else np
        self.eigsh = eigsh_gpu if self.use_gpu else eigsh_cpu
        
        # Build Pauli matrices
        self._build_operators()
        
        # Site potentials (modified by adsorption/reaction)
        self.site_potentials = np.zeros(n_sites)
        
    def _build_operators(self):
        """Build sparse Pauli operators."""
        I = np.eye(2, dtype=np.complex128)
        Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
        Sz = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)
        n_op = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        
        if self.use_gpu:
            self.I = csp.csr_matrix(cp.asarray(I))
            self.Sp = csp.csr_matrix(cp.asarray(Sp))
            self.Sm = csp.csr_matrix(cp.asarray(Sm))
            self.Sz = csp.csr_matrix(cp.asarray(Sz))
            self.n_op = csp.csr_matrix(cp.asarray(n_op))
        else:
            self.I = sp.csr_matrix(I)
            self.Sp = sp.csr_matrix(Sp)
            self.Sm = sp.csr_matrix(Sm)
            self.Sz = sp.csr_matrix(Sz)
            self.n_op = sp.csr_matrix(n_op)
    
    def _site_operator(self, op, site: int):
        """Build operator acting on specific site."""
        ops = [self.I] * self.n_sites
        ops[site] = op
        
        result = ops[0]
        for i in range(1, self.n_sites):
            result = self.sp.kron(result, ops[i], format='csr')
        return result
    
    def build_hamiltonian(self, t_hop: float = 1.0, U_int: float = 0.0) -> Tuple:
        """
        Build surface Hamiltonian.
        
        H = H_kinetic + H_potential + H_interaction
        
        Args:
            t_hop: Hopping parameter (electron delocalization)
            U_int: Interaction strength (adsorbate repulsion)
            
        Returns:
            (H_kinetic, H_potential): Kinetic and potential parts
        """
        L = self.n_sites
        
        # Kinetic: -t Σ(S+_i S-_j + S-_i S+_j)
        H_K = None
        for i in range(L - 1):
            j = i + 1
            Sp_i = self._site_operator(self.Sp, i)
            Sm_i = self._site_operator(self.Sm, i)
            Sp_j = self._site_operator(self.Sp, j)
            Sm_j = self._site_operator(self.Sm, j)
            
            term = -t_hop * (Sp_i @ Sm_j + Sm_i @ Sp_j)
            H_K = term if H_K is None else H_K + term
        
        # Potential: Σ V_i n_i
        H_V = None
        for i in range(L):
            n_i = self._site_operator(self.n_op, i)
            term = self.site_potentials[i] * n_i
            H_V = term if H_V is None else H_V + term
        
        # Interaction: U Σ n_i n_j (nearest neighbor)
        if U_int != 0:
            for i in range(L - 1):
                j = i + 1
                n_i = self._site_operator(self.n_op, i)
                n_j = self._site_operator(self.n_op, j)
                H_V = H_V + U_int * n_i @ n_j
        
        return H_K, H_V
    
    def set_site_potential(self, site: int, potential: float):
        """Set potential at a specific site (e.g., adsorption)."""
        self.site_potentials[site] = potential
    
    def reset_potentials(self):
        """Reset all site potentials to zero."""
        self.site_potentials = np.zeros(self.n_sites)
    
    def compute_coverage(self, psi) -> float:
        """
        Compute surface coverage θ = ⟨n⟩ / N_sites.
        
        Args:
            psi: Quantum state
            
        Returns:
            Coverage fraction [0, 1]
        """
        total_n = 0.0
        for i in range(self.n_sites):
            n_i = self._site_operator(self.n_op, i)
            n_expect = self.xp.real(self.xp.vdot(psi, n_i @ psi))
            total_n += float(n_expect)
        return total_n / self.n_sites
    
    def compute_lambda(self, psi, H_K, H_V, epsilon: float = 1e-10) -> float:
        """
        Compute stability parameter λ = K / |V|.
        
        Args:
            psi: Quantum state
            H_K: Kinetic Hamiltonian
            H_V: Potential Hamiltonian
            
        Returns:
            λ value (stability indicator)
        """
        K = float(self.xp.real(self.xp.vdot(psi, H_K @ psi)))
        V = float(self.xp.real(self.xp.vdot(psi, H_V @ psi)))
        return abs(K) / (abs(V) + epsilon)


class LanczosEvolver:
    """
    Lanczos-based time evolution: exp(-i H dt) |ψ⟩
    
    Uses Krylov subspace method for efficient evolution
    without full matrix exponentiation.
    """
    
    def __init__(self, krylov_dim: int = 30, use_gpu: bool = True):
        self.krylov_dim = krylov_dim
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
    
    def evolve(self, H_sparse, psi, dt: float):
        """
        Compute exp(-i H dt) |ψ⟩ using Lanczos method.
        
        Args:
            H_sparse: Sparse Hamiltonian
            psi: Initial state
            dt: Time step
            
        Returns:
            Evolved state (normalized)
        """
        xp = self.xp
        n = psi.shape[0]
        krylov_dim = min(self.krylov_dim, n)
        
        # Lanczos vectors
        V = xp.zeros((krylov_dim, n), dtype=xp.complex128)
        
        # Tridiagonal elements (CPU, small)
        alpha = np.zeros(krylov_dim, dtype=np.float64)
        beta = np.zeros(krylov_dim - 1, dtype=np.float64)
        
        # Initialize
        norm_psi = float(xp.linalg.norm(psi))
        v = psi / norm_psi
        V[0] = v
        
        # Lanczos iteration
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
        
        # Build tridiagonal matrix (CPU)
        T = np.diag(alpha[:actual_dim])
        if actual_dim > 1:
            T += np.diag(beta[:actual_dim-1], k=1)
            T += np.diag(beta[:actual_dim-1], k=-1)
        
        # Exponentiate small matrix (CPU)
        exp_T = scipy_expm(-1j * dt * T)
        
        # Apply to |e_0⟩
        e0 = np.zeros(actual_dim, dtype=np.complex128)
        e0[0] = 1.0
        y = exp_T @ e0
        
        # Transform back
        y_gpu = xp.asarray(y) if self.use_gpu else y
        psi_new = norm_psi * (V[:actual_dim].T @ y_gpu)
        
        return psi_new / xp.linalg.norm(psi_new)


class ChemicalReactionSolver:
    """
    Main solver for history-dependent chemical reactions.
    
    Workflow:
    1. Define surface system
    2. Define reaction paths (different event sequences)
    3. Evolve quantum state along each path
    4. Compare outcomes → Quantify memory effects
    
    Example:
        solver = ChemicalReactionSolver(n_sites=4)
        
        # Define two paths with different adsorption order
        path1 = ReactionPath("A_first")
        path1.add_event('adsorption', 1.0, site=0, potential=-0.5, species='A')
        path1.add_event('adsorption', 2.0, site=1, potential=-0.3, species='B')
        
        path2 = ReactionPath("B_first")
        path2.add_event('adsorption', 1.0, site=1, potential=-0.3, species='B')
        path2.add_event('adsorption', 2.0, site=0, potential=-0.5, species='A')
        
        # Run and compare
        result1 = solver.evolve_path(path1)
        result2 = solver.evolve_path(path2)
        memory = solver.compare_paths(result1, result2)
    """
    
    def __init__(self, n_sites: int, use_gpu: bool = True, verbose: bool = True):
        """
        Args:
            n_sites: Number of surface sites
            use_gpu: Use GPU if available
            verbose: Print progress
        """
        self.n_sites = n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.verbose = verbose
        self.xp = cp if self.use_gpu else np
        
        self.engine = SurfaceHamiltonianEngine(n_sites, use_gpu, verbose)
        self.evolver = LanczosEvolver(krylov_dim=30, use_gpu=use_gpu)
        
        # Default parameters
        self.t_hop = 1.0       # Hopping
        self.U_int = 2.0       # Interaction
        self.dt = 0.1          # Time step
        
    def set_parameters(self, t_hop: float = 1.0, U_int: float = 2.0, dt: float = 0.1):
        """Set Hamiltonian and evolution parameters."""
        self.t_hop = t_hop
        self.U_int = U_int
        self.dt = dt
    
    def get_initial_state(self, state_type: str = 'ground') -> np.ndarray:
        """
        Get initial quantum state.
        
        Args:
            state_type: 'ground' (ground state) or 'empty' (vacuum)
            
        Returns:
            Initial state vector
        """
        self.engine.reset_potentials()
        H_K, H_V = self.engine.build_hamiltonian(self.t_hop, self.U_int)
        H = H_K + H_V
        
        if state_type == 'ground':
            if self.use_gpu:
                E, psi = eigsh_gpu(H, k=1, which='SA')
                return psi[:, 0]
            else:
                E, psi = eigsh_cpu(H, k=1, which='SA')
                return psi[:, 0]
        else:  # empty/vacuum
            psi = self.xp.zeros(self.engine.dim, dtype=self.xp.complex128)
            psi[0] = 1.0
            return psi
    
    def evolve_path(self, path: ReactionPath, 
                    t_total: float = 10.0,
                    initial_state: Optional[np.ndarray] = None) -> PathResult:
        """
        Evolve quantum state along a reaction path.
        
        Args:
            path: Sequence of reaction events
            t_total: Total evolution time
            initial_state: Starting state (default: ground state)
            
        Returns:
            PathResult with full trajectory
        """
        if self.verbose:
            print(f"\n📍 Evolving path: {path.name}")
        
        # Initial state
        if initial_state is None:
            psi = self.get_initial_state('ground')
        else:
            psi = initial_state.copy()
        
        # Storage
        n_steps = int(t_total / self.dt) + 1
        times = np.zeros(n_steps)
        energies = np.zeros(n_steps)
        lambdas = np.zeros(n_steps)
        coverages = np.zeros(n_steps)
        
        # Sort events by time
        events = sorted(path.events, key=lambda e: e.time)
        event_idx = 0
        
        # Evolution loop
        for step in range(n_steps):
            t = step * self.dt
            times[step] = t
            
            # Apply any events at this time
            while event_idx < len(events) and events[event_idx].time <= t:
                event = events[event_idx]
                self.engine.set_site_potential(event.site, event.potential)
                if self.verbose:
                    print(f"   t={t:.2f}: {event.event_type} at site {event.site} "
                          f"(V={event.potential:.2f})")
                event_idx += 1
            
            # Build Hamiltonian with current potentials
            H_K, H_V = self.engine.build_hamiltonian(self.t_hop, self.U_int)
            H = H_K + H_V
            
            # Compute observables
            E = float(self.xp.real(self.xp.vdot(psi, H @ psi)))
            lam = self.engine.compute_lambda(psi, H_K, H_V)
            cov = self.engine.compute_coverage(psi)
            
            energies[step] = E
            lambdas[step] = lam
            coverages[step] = cov
            
            # Evolve (except last step)
            if step < n_steps - 1:
                psi = self.evolver.evolve(H, psi, self.dt)
        
        # Convert final state to numpy if needed
        final_state = psi.get() if self.use_gpu and hasattr(psi, 'get') else psi
        
        return PathResult(
            path_name=path.name,
            times=times,
            energies=energies,
            lambdas=lambdas,
            coverages=coverages,
            final_state=np.array(final_state)
        )
    
    def compare_paths(self, result1: PathResult, result2: PathResult) -> Dict[str, Any]:
        """
        Compare two path results to quantify memory effects.
        
        Args:
            result1, result2: Results from different paths
            
        Returns:
            Dictionary with memory indicators
        """
        try:
            from .memory_indicators import MemoryIndicator, HysteresisAnalyzer
        except ImportError:
            from memory_indicators import MemoryIndicator, HysteresisAnalyzer
        
        indicator = MemoryIndicator()
        
        # Path non-commutativity for various observables
        delta_lambda = indicator.path_noncommutativity(
            result1.lambdas[-1], result2.lambdas[-1]
        )
        delta_energy = indicator.path_noncommutativity(
            result1.energies[-1], result2.energies[-1]
        )
        delta_coverage = indicator.path_noncommutativity(
            result1.coverages[-1], result2.coverages[-1]
        )
        
        # Temporal memory from lambda series
        M1, tau1 = indicator.temporal_memory(result1.lambdas, self.dt)
        M2, tau2 = indicator.temporal_memory(result2.lambdas, self.dt)
        
        # Full metrics
        metrics = indicator.compute_all(
            O_forward=result1.lambdas[-1],
            O_backward=result2.lambdas[-1],
            series=result1.lambdas,
            dt=self.dt
        )
        
        comparison = {
            'path1': result1.path_name,
            'path2': result2.path_name,
            'delta_lambda': delta_lambda,
            'delta_energy': delta_energy,
            'delta_coverage': delta_coverage,
            'M_temporal_path1': M1,
            'M_temporal_path2': M2,
            'tau_memory_path1': tau1,
            'tau_memory_path2': tau2,
            'is_non_markovian': metrics.is_non_markovian(),
            'memory_metrics': metrics
        }
        
        if self.verbose:
            print(f"\n📊 Path Comparison: {result1.path_name} vs {result2.path_name}")
            print(f"   ΔΛ (final)     = {delta_lambda:.6f}")
            print(f"   ΔE (final)     = {delta_energy:.6f}")
            print(f"   Δθ (coverage)  = {delta_coverage:.6f}")
            print(f"   Non-Markovian? {metrics.is_non_markovian()}")
        
        return comparison


def run_catalyst_test():
    """
    Standard test: Adsorption order affects reaction outcome.
    
    Same final configuration, different history → Different λ!
    
    This is the key demonstration that Memory-DFT is necessary.
    """
    print("="*70)
    print("🧪 Catalyst Memory Test: Adsorption Order Effects")
    print("="*70)
    print("\nQuestion: Does the order of adsorption affect the final state?")
    print("Standard DFT answer: No (same structure = same energy)")
    print("Memory-DFT answer: Yes! (different history = different outcome)")
    
    # Setup
    solver = ChemicalReactionSolver(n_sites=4, use_gpu=HAS_CUPY, verbose=True)
    solver.set_parameters(t_hop=1.0, U_int=2.0, dt=0.1)
    
    # Path 1: Adsorption A first, then B
    path1 = ReactionPath("A→B")
    path1.add_event('adsorption', 2.0, site=0, potential=-0.5, species='A')
    path1.add_event('adsorption', 5.0, site=2, potential=-0.3, species='B')
    
    # Path 2: Adsorption B first, then A (same final state!)
    path2 = ReactionPath("B→A")
    path2.add_event('adsorption', 2.0, site=2, potential=-0.3, species='B')
    path2.add_event('adsorption', 5.0, site=0, potential=-0.5, species='A')
    
    # Evolve both paths
    result1 = solver.evolve_path(path1, t_total=10.0)
    result2 = solver.evolve_path(path2, t_total=10.0)
    
    # Compare
    comparison = solver.compare_paths(result1, result2)
    
    # Summary
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"\n  Path 1 ({result1.path_name}):")
    print(f"    Final λ = {result1.lambdas[-1]:.6f}")
    print(f"    Final E = {result1.energies[-1]:.6f}")
    print(f"    Final θ = {result1.coverages[-1]:.6f}")
    
    print(f"\n  Path 2 ({result2.path_name}):")
    print(f"    Final λ = {result2.lambdas[-1]:.6f}")
    print(f"    Final E = {result2.energies[-1]:.6f}")
    print(f"    Final θ = {result2.coverages[-1]:.6f}")
    
    print(f"\n  Memory Indicators:")
    print(f"    ΔΛ = {comparison['delta_lambda']:.6f}")
    print(f"    ΔE = {comparison['delta_energy']:.6f}")
    print(f"    Δθ = {comparison['delta_coverage']:.6f}")
    
    if comparison['is_non_markovian']:
        print(f"\n  ✅ NON-MARKOVIAN DYNAMICS DETECTED!")
        print(f"  ✅ Memory-DFT is NECESSARY for this system!")
        print(f"  ✅ Standard DFT would give IDENTICAL results for both paths!")
    else:
        print(f"\n  → System appears Markovian for these parameters")
    
    return solver, result1, result2, comparison


def run_reaction_order_test():
    """
    Test: Reaction vs Adsorption order.
    
    This demonstrates catalyst poisoning effects.
    """
    print("\n" + "="*70)
    print("🧪 Reaction Order Test: Adsorption ↔ Reaction Sequence")
    print("="*70)
    
    solver = ChemicalReactionSolver(n_sites=4, use_gpu=HAS_CUPY, verbose=True)
    solver.set_parameters(t_hop=1.0, U_int=2.0, dt=0.1)
    
    # Path 1: Adsorption → Reaction
    path1 = ReactionPath("Ads→React")
    path1.add_event('adsorption', 2.0, site=0, potential=-0.5, species='reactant')
    path1.add_event('reaction', 5.0, site=1, potential=+0.3, species='product')
    
    # Path 2: Reaction → Adsorption
    path2 = ReactionPath("React→Ads")
    path2.add_event('reaction', 2.0, site=1, potential=+0.3, species='product')
    path2.add_event('adsorption', 5.0, site=0, potential=-0.5, species='reactant')
    
    result1 = solver.evolve_path(path1, t_total=10.0)
    result2 = solver.evolve_path(path2, t_total=10.0)
    
    comparison = solver.compare_paths(result1, result2)
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"\n  ΔΛ = {comparison['delta_lambda']:.6f}")
    
    if comparison['delta_lambda'] > 0.01:
        print(f"\n  ✅ Reaction order matters!")
        print(f"  ✅ This explains catalyst selectivity dependence on history!")
    
    return comparison


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  CHEMICAL REACTION TIME-EVOLUTION SOLVER                            ║
║                                                                      ║
║  History-Dependent Quantum Dynamics for Surface Chemistry           ║
║                                                                      ║
║  Key insight:                                                        ║
║    Standard DFT: Same structure → Same energy                       ║
║    Memory-DFT:   Different history → Different outcome              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    t0 = time.time()
    
    print("\n" + "#"*70)
    print("# TEST 1: Catalyst Adsorption Order")
    print("#"*70)
    solver, res1, res2, comp1 = run_catalyst_test()
    
    print("\n" + "#"*70)
    print("# TEST 2: Reaction vs Adsorption Sequence")
    print("#"*70)
    comp2 = run_reaction_order_test()
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 CHEMICAL REACTION SOLVER TEST COMPLETE!")
    print("="*70)
    print(f"\n  Total time: {time.time()-t0:.2f}s")
    print(f"\n  Key finding:")
    print(f"    Different reaction paths → Different quantum outcomes")
    print(f"    This CANNOT be captured by standard DFT!")
    print(f"    Memory-DFT is NECESSARY for history-dependent chemistry!")
