"""
Dislocation Dynamics Module (CuPy Unified)
==========================================

転位の生成・移動・パイルアップを Λ³ ベースでシミュレート
GPU加速対応（CuPy）

Author: Masamichi Iizumi, Tamaki Iizumi
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

# CuPy support
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_sparse = None
    cp_eigsh = None
    HAS_CUPY = False

# SciPy (CPU fallback)
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

# Type hints
try:
    from memory_dft.core.sparse_engine_unified import SparseEngine, SystemGeometry
except ImportError:
    SparseEngine = Any
    SystemGeometry = Any


# =============================================================================
# Dislocation Data Structure
# =============================================================================

@dataclass
class Dislocation:
    """Single dislocation representation."""
    site: int
    burgers: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    slip_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    pinned: bool = False
    history: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.site not in self.history:
            self.history.append(self.site)
    
    @property
    def burgers_magnitude(self) -> float:
        return float(np.sqrt(sum(b**2 for b in self.burgers)))
    
    def move_to(self, new_site: int):
        if not self.pinned:
            self.history.append(new_site)
            self.site = new_site


# =============================================================================
# Dislocation Dynamics Engine (CuPy Unified)
# =============================================================================

class DislocationDynamics:
    """
    Λ³-based dislocation dynamics simulator.
    GPU acceleration via CuPy.
    """
    
    def __init__(self,
                 engine: SparseEngine,
                 geometry: SystemGeometry,
                 t: float = 1.0,
                 U: float = 5.0,
                 lambda_critical: float = 0.5,
                 use_gpu: bool = False,
                 verbose: bool = True):
        """
        Initialize dislocation dynamics.
        
        Args:
            engine: SparseEngine instance
            geometry: Lattice geometry
            t: Hopping parameter
            U: On-site interaction
            lambda_critical: Threshold for motion
            use_gpu: Use GPU acceleration
            verbose: Print progress
        """
        self.engine = engine
        self.geometry = geometry
        self.t = t
        self.U = U
        self.lambda_critical = lambda_critical
        self.verbose = verbose
        
        # Backend selection
        self.use_gpu = use_gpu and HAS_CUPY
        if self.use_gpu:
            self.xp = cp
            self.sp_module = cp_sparse
        else:
            self.xp = np
            self.sp_module = sp
        
        # Dislocation list
        self.dislocations: List[Dislocation] = []
        
        # Build neighbor map
        self._build_neighbor_map()
        
        # State
        self.psi = None
        self.H_K = None
        self.H_V = None
        self.lambda_local = None
        
        if verbose:
            print("=" * 60)
            print("Dislocation Dynamics Engine")
            print("=" * 60)
            print(f"  Sites: {geometry.n_sites}")
            print(f"  t = {t}, U = {U}, U/t = {U/t:.1f}")
            print(f"  λ_critical = {lambda_critical}")
            print(f"  Backend: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'}")
            print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Array Conversion
    # -------------------------------------------------------------------------
    
    def _to_device(self, arr):
        """Convert to device"""
        if self.use_gpu and not isinstance(arr, cp.ndarray):
            return cp.asarray(arr)
        return arr
    
    def _to_host(self, arr):
        """Convert to host"""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def _build_neighbor_map(self):
        """Build neighbor map for dislocation motion"""
        self.neighbors = {i: [] for i in range(self.geometry.n_sites)}
        for (i, j) in self.geometry.bonds:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
    
    # -------------------------------------------------------------------------
    # Dislocation Management
    # -------------------------------------------------------------------------
    
    def add_dislocation(self,
                        site: int,
                        burgers: Tuple[float, float, float] = (1, 0, 0),
                        slip_direction: Tuple[float, float, float] = (1, 0, 0)):
        """Add a dislocation at specified site"""
        disl = Dislocation(
            site=site,
            burgers=burgers,
            slip_direction=slip_direction
        )
        self.dislocations.append(disl)
        
        if self.verbose:
            print(f"  Added dislocation at site {site}, b={burgers}")
        
        return disl
    
    def add_frank_read_source(self, site: int, n_emit: int = 3):
        """Add Frank-Read source (dislocation generator)."""
        source = Dislocation(
            site=site,
            burgers=(1, 0, 0),
            pinned=True
        )
        source.is_source = True
        source.n_emit = n_emit
        source.emitted = 0
        self.dislocations.append(source)
        
        if self.verbose:
            print(f"  Added Frank-Read source at site {site}")
    
    def remove_dislocation(self, site: int):
        """Remove dislocation at site"""
        self.dislocations = [d for d in self.dislocations if d.site != site]
    
    def get_dislocation_sites(self) -> List[int]:
        """Get list of sites with dislocations"""
        return [d.site for d in self.dislocations]
    
    # -------------------------------------------------------------------------
    # Hamiltonian and λ Computation
    # -------------------------------------------------------------------------
    
    def build_hamiltonian(self, stress: float = 0.0):
        """Build Hamiltonian with current dislocation configuration."""
        xp = self.xp
        
        # Mark weak bonds near dislocations
        weak_bonds = list(self.geometry.weak_bonds or [])
        
        for disl in self.dislocations:
            site = disl.site
            for neighbor in self.neighbors.get(site, []):
                bond = (min(site, neighbor), max(site, neighbor))
                if bond not in weak_bonds:
                    weak_bonds.append(bond)
        
        self.geometry.weak_bonds = weak_bonds
        
        # Build Hubbard with defects
        self.H_K, self.H_V = self.engine.build_hubbard_with_defects(
            self.geometry,
            t=self.t,
            U=self.U,
            t_weak=0.3 * self.t
        )
        
        # Add stress gradient
        if stress > 0:
            H_stress = self._build_stress_hamiltonian(stress)
            self.H_V = self.H_V + H_stress
        
        return self.H_K, self.H_V
    
    def _build_stress_hamiltonian(self, stress: float):
        """Build stress gradient Hamiltonian"""
        xp = self.xp
        n = self.geometry.n_sites
        Lx = getattr(self.geometry, 'Lx', int(np.sqrt(n)))
        
        dim = 2**n if n < 10 else n
        diag = xp.zeros(dim, dtype=xp.float64)
        
        if n < 10:
            for state in range(dim):
                for site in range(n):
                    if (state >> site) & 1:
                        x = site % Lx
                        diag[state] += stress * (x - Lx/2) / Lx
        
        if self.use_gpu:
            return cp_sparse.diags(diag, format='csr', dtype=cp.complex128)
        else:
            return sp.diags(self._to_host(diag), format='csr', dtype=np.complex128)
    
    def compute_ground_state(self):
        """Compute ground state"""
        xp = self.xp
        H = self.H_K + self.H_V
        
        if self.use_gpu and cp_eigsh is not None:
            try:
                E0, psi0 = cp_eigsh(H, k=1, which='SA')
                self.psi = psi0[:, 0]
                self.psi = self.psi / xp.linalg.norm(self.psi)
                return float(E0[0]), self.psi
            except Exception:
                pass
        
        # CPU fallback
        H_cpu = self._to_host(H.toarray()) if hasattr(H, 'toarray') else self._to_host(H)
        H_sp = sp.csr_matrix(H_cpu)
        
        try:
            E0, psi0 = eigsh(H_sp, k=1, which='SA')
            psi = psi0[:, 0]
            psi = psi / np.linalg.norm(psi)
            self.psi = self._to_device(psi)
            return float(E0[0]), self.psi
        except Exception as e:
            if self.verbose:
                print(f"  Warning: eigsh failed: {e}")
            dim = H_sp.shape[0]
            psi = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi = psi / np.linalg.norm(psi)
            self.psi = self._to_device(psi)
            return 0.0, self.psi
    
    def compute_local_lambda(self) -> np.ndarray:
        """Compute local λ at each site."""
        if self.psi is None:
            self.compute_ground_state()
        
        psi_host = self._to_host(self.psi)
        
        self.lambda_local = self.engine.compute_local_lambda(
            psi_host, self.H_K, self.H_V, self.geometry
        )
        
        return self.lambda_local
    
    # -------------------------------------------------------------------------
    # Peach-Koehler Force
    # -------------------------------------------------------------------------
    
    def compute_peach_koehler_force(self, disl: Dislocation, stress: float) -> float:
        """Compute Peach-Koehler force on dislocation."""
        b = disl.burgers_magnitude
        return stress * b
    
    # -------------------------------------------------------------------------
    # Dislocation Motion
    # -------------------------------------------------------------------------
    
    def attempt_move(self, disl: Dislocation, stress: float) -> bool:
        """Attempt to move dislocation under stress."""
        if disl.pinned:
            if hasattr(disl, 'is_source') and disl.is_source:
                return self._emit_from_source(disl, stress)
            return False
        
        site = disl.site
        F = self.compute_peach_koehler_force(disl, stress)
        
        candidates = self.neighbors.get(site, [])
        if not candidates:
            return False
        
        best_site = None
        best_score = -float('inf')
        
        for candidate in candidates:
            if self.lambda_local is None or candidate >= len(self.lambda_local):
                continue
            lam = self.lambda_local[candidate]
            score = F * lam
            
            if score > best_score and lam > self.lambda_critical * 0.5:
                best_score = score
                best_site = candidate
        
        if best_site is not None and best_score > self.lambda_critical:
            disl.move_to(best_site)
            return True
        
        return False
    
    def _emit_from_source(self, source: Dislocation, stress: float) -> bool:
        """Emit new dislocation from Frank-Read source"""
        if source.emitted >= source.n_emit:
            return False
        
        if stress > 1.0:
            neighbors = self.neighbors.get(source.site, [])
            if neighbors:
                new_site = neighbors[source.emitted % len(neighbors)]
                self.add_dislocation(site=new_site, burgers=source.burgers)
                source.emitted += 1
                return True
        
        return False
    
    # -------------------------------------------------------------------------
    # Pileup Simulation
    # -------------------------------------------------------------------------
    
    def simulate_under_stress(self,
                               stress_range: np.ndarray,
                               grain_boundary_sites: List[int],
                               n_steps_per_stress: int = 10,
                               dt: float = 0.1) -> Dict[str, Any]:
        """Simulate dislocation motion under increasing stress."""
        if self.verbose:
            print(f"\n[Pileup Simulation] σ: {stress_range[0]:.2f} → {stress_range[-1]:.2f}")
            print(f"  GB sites: {grain_boundary_sites}")
            print(f"  Initial dislocations: {len(self.dislocations)}")
        
        results = {
            'stress': [],
            'pileup_count': [],
            'lambda_max': [],
            'lambda_at_gb': [],
            'energy': [],
            'yielded': False,
            'sigma_y': None,
        }
        
        gb_set = set(grain_boundary_sites)
        
        for sigma in stress_range:
            self.build_hamiltonian(stress=sigma)
            E0, _ = self.compute_ground_state()
            self.compute_local_lambda()
            
            for disl in self.dislocations:
                if disl.site in gb_set:
                    disl.pinned = True
                else:
                    self.attempt_move(disl, sigma)
            
            pileup = sum(1 for d in self.dislocations if d.site in gb_set)
            
            gb_lambdas = [self.lambda_local[s] for s in grain_boundary_sites 
                         if s < len(self.lambda_local)]
            lambda_gb = float(np.mean(gb_lambdas)) if gb_lambdas else 0.5
            
            results['stress'].append(sigma)
            results['pileup_count'].append(pileup)
            results['lambda_max'].append(float(np.max(self.lambda_local)))
            results['lambda_at_gb'].append(lambda_gb)
            results['energy'].append(E0)
            
            if not results['yielded'] and lambda_gb > self.lambda_critical:
                results['yielded'] = True
                results['sigma_y'] = sigma
                if self.verbose:
                    print(f"  → Yield at σ = {sigma:.3f}, pileup = {pileup}")
            
            if self.verbose and len(results['stress']) % 10 == 0:
                print(f"  σ = {sigma:.2f}, pileup = {pileup}, λ_gb = {lambda_gb:.4f}")
        
        for key in ['stress', 'pileup_count', 'lambda_max', 'lambda_at_gb', 'energy']:
            results[key] = np.array(results[key])
        
        return results
    
    # -------------------------------------------------------------------------
    # Hall-Petch Simulation
    # -------------------------------------------------------------------------
    
    def simulate_hall_petch(self,
                            grain_sizes: List[int],
                            n_dislocations: int = 3,
                            stress_max: float = 5.0,
                            n_stress_points: int = 25) -> Dict[str, Any]:
        """Simulate Hall-Petch relation: σ_y vs 1/√d"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔬 HALL-PETCH SIMULATION (Dislocation Dynamics)")
            print("=" * 60)
        
        results = {
            'd': [],
            'inv_sqrt_d': [],
            'sigma_y': [],
            'pileup_final': [],
            'lambda_gb_at_yield': [],
        }
        
        stress_range = np.linspace(0, stress_max, n_stress_points)
        
        for gs in grain_sizes:
            if self.verbose:
                print(f"\n--- Grain size: {gs} sites ---")
            
            Lx = gs * 2
            Ly = max(4, gs // 2)
            
            gb_x = gs
            gb_sites = [y * Lx + gb_x for y in range(Ly)]
            
            self.dislocations = []
            
            for i in range(n_dislocations):
                site = (i % Ly) * Lx + (gs // 2)
                self.add_dislocation(site, burgers=(1, 0, 0))
            
            sim_result = self.simulate_under_stress(
                stress_range=stress_range,
                grain_boundary_sites=gb_sites
            )
            
            d_eff = gs * 0.248
            sigma_y = sim_result['sigma_y'] or stress_max
            
            results['d'].append(d_eff)
            results['inv_sqrt_d'].append(1.0 / np.sqrt(d_eff))
            results['sigma_y'].append(sigma_y)
            results['pileup_final'].append(sim_result['pileup_count'][-1])
            
            yield_idx = np.argmax(sim_result['stress'] >= sigma_y) if sigma_y < stress_max else -1
            results['lambda_gb_at_yield'].append(sim_result['lambda_at_gb'][yield_idx])
            
            if self.verbose:
                print(f"  d = {d_eff:.2f} nm, σ_y = {sigma_y:.3f}")
        
        for key in results:
            results[key] = np.array(results[key])
        
        # Fit Hall-Petch
        coeffs = np.polyfit(results['inv_sqrt_d'], results['sigma_y'], 1)
        k_HP = coeffs[0]
        sigma_0 = coeffs[1]
        
        results['k_HP'] = k_HP
        results['sigma_0'] = sigma_0
        results['fit_coeffs'] = coeffs
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("📊 HALL-PETCH FIT RESULTS")
            print("=" * 60)
            print(f"  σ_y = σ_0 + k/√d")
            print(f"  σ_0 = {sigma_0:.4f}")
            print(f"  k   = {k_HP:.4f}")
            print("=" * 60)
        
        return results


# =============================================================================
# Plotting (matplotlib required)
# =============================================================================

def plot_pileup_results(results: Dict[str, Any], save: bool = True):
    """Plot pileup simulation results"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    ax = axes[0, 0]
    ax.plot(results['stress'], results['pileup_count'], 'b-o', markersize=3)
    ax.set_xlabel('Stress σ')
    ax.set_ylabel('Pileup count')
    ax.set_title('Dislocation Pileup')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(results['stress'], results['lambda_at_gb'], 'r-s', markersize=3)
    ax.axhline(y=0.5, color='k', linestyle='--', label='λ_critical')
    if results['sigma_y']:
        ax.axvline(x=results['sigma_y'], color='g', linestyle='--', 
                   label=f"σ_y={results['sigma_y']:.2f}")
    ax.set_xlabel('Stress σ')
    ax.set_ylabel('λ at grain boundary')
    ax.set_title('Stability at GB')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(results['stress'], results['lambda_max'], 'g-^', markersize=3)
    ax.set_xlabel('Stress σ')
    ax.set_ylabel('λ_max')
    ax.set_title('Maximum λ')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(results['stress'], results['energy'], 'm-d', markersize=3)
    ax.set_xlabel('Stress σ')
    ax.set_ylabel('Energy')
    ax.set_title('System Energy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        fname = 'dislocation_pileup_results.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"\n[Saved] {fname}")
    
    plt.close()
    return fig


def plot_hall_petch_dd(results: Dict[str, Any], save: bool = True):
    """Plot Hall-Petch results"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.scatter(results['inv_sqrt_d'], results['sigma_y'], s=100, c='blue', label='Data')
    
    x_fit = np.linspace(0, max(results['inv_sqrt_d']) * 1.1, 100)
    y_fit = results['sigma_0'] + results['k_HP'] * x_fit
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f"Fit: k={results['k_HP']:.3f}")
    
    ax.set_xlabel('1/√d (nm⁻¹/²)', fontsize=12)
    ax.set_ylabel('σ_y', fontsize=12)
    ax.set_title('Hall-Petch Relation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.bar(range(len(results['d'])), results['pileup_final'], color='orange')
    ax.set_xticks(range(len(results['d'])))
    ax.set_xticklabels([f"{d:.1f}" for d in results['d']])
    ax.set_xlabel('Grain size d (nm)', fontsize=12)
    ax.set_ylabel('Final pileup count', fontsize=12)
    ax.set_title('Dislocation Pileup vs Grain Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        fname = 'hall_petch_dislocation_dynamics.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"\n[Saved] {fname}")
    
    plt.close()
    return fig


# =============================================================================
# Main (Test)
# =============================================================================

def main():
    """Test dislocation dynamics"""
    print("\n" + "🔧" * 25)
    print("  DISLOCATION DYNAMICS TEST")
    print(f"  CuPy available: {HAS_CUPY}")
    print("🔧" * 25)
    
    try:
        from memory_dft.core.sparse_engine_unified import SparseEngine
    except ImportError:
        print("⚠️ SparseEngine not available.")
        return
    
    n_sites = 16
    use_gpu = HAS_CUPY
    
    engine = SparseEngine(n_sites=n_sites, use_gpu=use_gpu, verbose=False)
    geom = engine.build_edge_dislocation(Lx=4, Ly=4, dislocation_y=2)
    
    dd = DislocationDynamics(
        engine=engine,
        geometry=geom,
        t=1.0,
        U=5.0,
        lambda_critical=0.5,
        use_gpu=use_gpu,
        verbose=True
    )
    
    dd.add_dislocation(site=1, burgers=(1, 0, 0))
    dd.add_dislocation(site=5, burgers=(1, 0, 0))
    
    gb_sites = [3, 7, 11, 15]
    stress_range = np.linspace(0, 3.0, 20)
    
    results = dd.simulate_under_stress(
        stress_range=stress_range,
        grain_boundary_sites=gb_sites
    )
    
    print(f"\n  Final pileup: {results['pileup_count'][-1]}")
    print(f"  σ_y = {results['sigma_y']}")
    
    print("\n" + "=" * 60)
    print("✅ Dislocation Dynamics Test Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
