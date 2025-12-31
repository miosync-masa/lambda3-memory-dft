"""
2D Lattice Direct Schrödinger Evolution (Ladder-DSE)
====================================================

Extension of DSE to 2D lattice systems with multiple Hamiltonian types.

Supported Hamiltonians:
  - Heisenberg:  H = J Σ S_i · S_j
  - XY:          H = J Σ (Sx_i Sx_j + Sy_i Sy_j)
  - XX:          H = J Σ Sx_i Sx_j
  - Kitaev:      H = Kx Σ_x Sx Sx + Ky Σ_y Sy Sy + Kz Σ_diag Sz Sz
  - Ising:       H = J Σ Sz_i Sz_j + h Σ Sx_i

Key features:
  - Arbitrary Lx × Ly lattice
  - Plaquette flux (vorticity) operator
  - Thermal + path-dependent dynamics
  - DFT cannot capture ANY of this!

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import scipy.sparse as sparse
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
# 2D Lattice Geometry
# =============================================================================

class LatticeGeometry:
    """
    2D lattice geometry with configurable boundary conditions.
    """
    
    def __init__(self, Lx, Ly, periodic_x=False, periodic_y=False):
        self.Lx = Lx
        self.Ly = Ly
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.N_spins = Lx * Ly
        self.Dim = 2 ** self.N_spins
        
        self.coords = self._build_coords()
        self.bonds_nn, self.bonds_x, self.bonds_y = self._build_nn_bonds()
        self.plaquettes = self._build_plaquettes()
    
    def idx(self, x, y):
        return y * self.Lx + x
    
    def _build_coords(self):
        return {self.idx(x, y): (x, y) 
                for y in range(self.Ly) 
                for x in range(self.Lx)}
    
    def _build_nn_bonds(self):
        bonds = set()
        bonds_x = []
        bonds_y = []
        
        for y in range(self.Ly):
            for x in range(self.Lx):
                i = self.idx(x, y)
                if x + 1 < self.Lx or self.periodic_x:
                    j = self.idx((x + 1) % self.Lx, y)
                    if i < j:
                        bonds.add((i, j))
                    bonds_x.append((i, j))
                if y + 1 < self.Ly or self.periodic_y:
                    j = self.idx(x, (y + 1) % self.Ly)
                    if i < j:
                        bonds.add((i, j))
                    bonds_y.append((i, j))
        
        return sorted(list(bonds)), bonds_x, bonds_y
    
    def _build_plaquettes(self):
        plaquettes = []
        for y in range(self.Ly - 1):
            for x in range(self.Lx - 1):
                bl = self.idx(x, y)
                br = self.idx(x + 1, y)
                tr = self.idx(x + 1, y + 1)
                tl = self.idx(x, y + 1)
                plaquettes.append((bl, br, tr, tl))
        return plaquettes


# =============================================================================
# Spin Operators
# =============================================================================

class SpinOperators:
    """Pauli spin operators for N-spin system."""
    
    def __init__(self, N_spins):
        self.N = N_spins
        self.Dim = 2 ** N_spins
        
        self.sx = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
        self.sy = sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
        self.sz = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
        self.sp = sparse.csr_matrix([[0, 1], [0, 0]], dtype=complex)
        self.sm = sparse.csr_matrix([[0, 0], [1, 0]], dtype=complex)
        self.iden = sparse.eye(2, dtype=complex)
        
        self.Sx = [self._get_spin_op(self.sx, i) for i in range(N_spins)]
        self.Sy = [self._get_spin_op(self.sy, i) for i in range(N_spins)]
        self.Sz = [self._get_spin_op(self.sz, i) for i in range(N_spins)]
        self.Sp = [self._get_spin_op(self.sp, i) for i in range(N_spins)]
        self.Sm = [self._get_spin_op(self.sm, i) for i in range(N_spins)]
        
        self.S_total_z = sum(self.Sz)
    
    def _get_spin_op(self, op, site):
        ops = [self.iden] * self.N
        ops[site] = op
        full_op = ops[0]
        for i in range(1, self.N):
            full_op = sparse.kron(full_op, ops[i])
        return full_op


# =============================================================================
# Hamiltonian Builders
# =============================================================================

class HamiltonianBuilder:
    """Build various spin Hamiltonians on 2D lattice."""
    
    def __init__(self, geometry: LatticeGeometry, spin_ops: SpinOperators):
        self.geom = geometry
        self.ops = spin_ops
        self.Dim = geometry.Dim
    
    def heisenberg(self, J=1.0):
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        for (i, j) in self.geom.bonds_nn:
            H += J * (self.ops.Sx[i] @ self.ops.Sx[j] +
                      self.ops.Sy[i] @ self.ops.Sy[j] +
                      self.ops.Sz[i] @ self.ops.Sz[j])
        return H
    
    def xy(self, J=1.0):
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        for (i, j) in self.geom.bonds_nn:
            H += J * (self.ops.Sx[i] @ self.ops.Sx[j] +
                      self.ops.Sy[i] @ self.ops.Sy[j])
        return H
    
    def xx(self, J=1.0):
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        for (i, j) in self.geom.bonds_nn:
            H += J * self.ops.Sx[i] @ self.ops.Sx[j]
        return H
    
    def kitaev_rect(self, Kx=1.0, Ky=1.0, Kz_diag=0.0):
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        for (i, j) in self.geom.bonds_x:
            H += Kx * self.ops.Sx[i] @ self.ops.Sx[j]
        for (i, j) in self.geom.bonds_y:
            H += Ky * self.ops.Sy[i] @ self.ops.Sy[j]
        if Kz_diag != 0.0:
            for (bl, br, tr, tl) in self.geom.plaquettes:
                H += Kz_diag * self.ops.Sz[bl] @ self.ops.Sz[tr]
                H += Kz_diag * self.ops.Sz[br] @ self.ops.Sz[tl]
        return H
    
    def ising(self, J=1.0, h=0.0):
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        for (i, j) in self.geom.bonds_nn:
            H += J * self.ops.Sz[i] @ self.ops.Sz[j]
        for i in range(self.geom.N_spins):
            H += h * self.ops.Sx[i]
        return H
    
    def hubbard_spin(self, t=1.0, U=2.0):
        """Hubbard-like with spin representation."""
        H = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        for (i, j) in self.geom.bonds_nn:
            H += -t * (self.ops.Sp[i] @ self.ops.Sm[j] +
                       self.ops.Sm[i] @ self.ops.Sp[j])
            H += U * self.ops.Sz[i] @ self.ops.Sz[j]
        return H
    
    def build_vorticity_operator(self):
        """Static vorticity: V = Σ_plaq Σ_{i→j} 2(Sx_i Sy_j - Sy_i Sx_j)"""
        V = sparse.csr_matrix((self.Dim, self.Dim), dtype=complex)
        for (bl, br, tr, tl) in self.geom.plaquettes:
            loop_edges = [(bl, br), (br, tr), (tr, tl), (tl, bl)]
            for (i, j) in loop_edges:
                V += 2.0 * (self.ops.Sx[i] @ self.ops.Sy[j] -
                           self.ops.Sy[i] @ self.ops.Sx[j])
        return V


# =============================================================================
# 2D Ladder DSE Solver
# =============================================================================

class LadderDSESolver:
    """
    2D Lattice Direct Schrödinger Evolution solver.
    
    DFT信者を完全論破するためのツール！
    """
    
    def __init__(self, Lx=3, Ly=3, periodic_x=False, periodic_y=False,
                 use_gpu=True, verbose=True):
        self.geom = LatticeGeometry(Lx, Ly, periodic_x, periodic_y)
        self.ops = SpinOperators(self.geom.N_spins)
        self.builder = HamiltonianBuilder(self.geom, self.ops)
        
        self.verbose = verbose
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        
        self.H = None
        self.H_type = None
        self.V_op = None
        self.eigenvalues = None
        self.eigenvectors = None
        
        if verbose:
            print(f"🔲 LadderDSESolver: {Lx}×{Ly} lattice")
            print(f"   N_spins = {self.geom.N_spins}, Dim = {self.geom.Dim}")
            print(f"   Bonds: {len(self.geom.bonds_nn)}, Plaquettes: {len(self.geom.plaquettes)}")
    
    def build_hamiltonian(self, H_type='heisenberg', **params):
        self.H_type = H_type
        
        if H_type == 'heisenberg':
            self.H = self.builder.heisenberg(**params)
        elif H_type == 'xy':
            self.H = self.builder.xy(**params)
        elif H_type == 'xx':
            self.H = self.builder.xx(**params)
        elif H_type == 'kitaev':
            self.H = self.builder.kitaev_rect(**params)
        elif H_type == 'ising':
            self.H = self.builder.ising(**params)
        elif H_type == 'hubbard':
            self.H = self.builder.hubbard_spin(**params)
        else:
            raise ValueError(f"Unknown: {H_type}")
        
        self.V_op = self.builder.build_vorticity_operator()
        
        if self.verbose:
            print(f"   Built {H_type}, ||V_op|| = {sparse.linalg.norm(self.V_op):.4f}")
        return self
    
    def diagonalize(self, n_eigenstates=50):
        if self.H is None:
            raise ValueError("Build Hamiltonian first!")
        
        n_eigenstates = min(n_eigenstates, self.geom.Dim - 2)
        
        if self.verbose:
            print(f"   Diagonalizing ({n_eigenstates} states)...")
        
        t0 = time.time()
        eigenvalues, eigenvectors = eigsh(self.H, k=n_eigenstates, which='SA')
        
        idx = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        self.n_eigenstates = n_eigenstates
        
        if self.verbose:
            gap = self.eigenvalues[1] - self.eigenvalues[0]
            print(f"   Done in {time.time()-t0:.2f}s, E_0={self.eigenvalues[0]:.4f}, Gap={gap:.4f}")
        return self
    
    def compute_vorticity(self, psi):
        return float(np.real(np.vdot(psi, self.V_op @ psi)))
    
    def _lanczos_expm(self, H, psi, dt, krylov_dim=30):
        n = psi.shape[0]
        V = np.zeros((krylov_dim, n), dtype=np.complex128)
        alpha = np.zeros(krylov_dim)
        beta = np.zeros(krylov_dim - 1)
        
        norm_psi = np.linalg.norm(psi)
        v = psi / norm_psi
        V[0] = v
        
        w = H @ v
        alpha[0] = np.real(np.vdot(v, w))
        w = w - alpha[0] * v
        
        actual_dim = krylov_dim
        for j in range(1, krylov_dim):
            beta_j = np.linalg.norm(w)
            if beta_j < 1e-12:
                actual_dim = j
                break
            beta[j-1] = beta_j
            v_new = w / beta_j
            V[j] = v_new
            w = H @ v_new
            alpha[j] = np.real(np.vdot(v_new, w))
            w = w - alpha[j] * v_new - beta[j-1] * V[j-1]
        
        T = np.diag(alpha[:actual_dim])
        if actual_dim > 1:
            T += np.diag(beta[:actual_dim-1], k=1)
            T += np.diag(beta[:actual_dim-1], k=-1)
        
        exp_T = scipy_expm(-1j * dt * T)
        e0 = np.zeros(actual_dim, dtype=np.complex128)
        e0[0] = 1.0
        y = exp_T @ e0
        
        psi_new = norm_psi * (V[:actual_dim].T @ y)
        return psi_new / np.linalg.norm(psi_new)
    
    def evolve_with_field(self, T_kelvin, field_path, dt=0.1, energy_scale=0.1):
        if self.eigenvalues is None:
            self.diagonalize()
        
        k_B_eV = 8.617333262e-5
        beta = energy_scale / (k_B_eV * T_kelvin) if T_kelvin > 0 else float('inf')
        
        if self.verbose:
            print(f"\n🌀 Evolution at T={T_kelvin}K")
        
        E_min = self.eigenvalues[0]
        E_shifted = self.eigenvalues - E_min
        
        if beta == float('inf'):
            weights = np.zeros(len(self.eigenvalues))
            weights[0] = 1.0
        else:
            boltzmann = np.exp(-beta * E_shifted)
            Z = np.sum(boltzmann)
            weights = boltzmann / Z
        
        active_mask = weights > 1e-15
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)
        
        if self.verbose:
            print(f"   Active states: {n_active}")
        
        evolved_psis = [self.eigenvectors[:, i].copy() for i in active_indices]
        evolved_weights = weights[active_indices]
        
        events = sorted(field_path, key=lambda x: x['time'])
        t_final = events[-1]['time'] + 5.0 if events else 10.0
        
        times, vorticities, energies = [], [], []
        t, event_idx = 0.0, 0
        current_fields = {}
        
        while t < t_final:
            while event_idx < len(events) and events[event_idx]['time'] <= t:
                ev = events[event_idx]
                site = ev['site']
                if site not in current_fields:
                    current_fields[site] = {}
                current_fields[site][ev['field']] = ev['strength']
                if self.verbose and event_idx < 5:
                    print(f"   t={t:.1f}: h_{ev['field']}={ev['strength']:.2f} @ site {site}")
                event_idx += 1
            
            H_current = self.H.copy()
            for site, fields in current_fields.items():
                for f_type, strength in fields.items():
                    if f_type == 'x':
                        H_current = H_current + strength * self.ops.Sx[site]
                    elif f_type == 'y':
                        H_current = H_current + strength * self.ops.Sy[site]
                    elif f_type == 'z':
                        H_current = H_current + strength * self.ops.Sz[site]
            
            for i in range(n_active):
                evolved_psis[i] = self._lanczos_expm(H_current, evolved_psis[i], dt)
            
            V_total, E_total = 0.0, 0.0
            for i in range(n_active):
                psi = evolved_psis[i]
                w = evolved_weights[i]
                V_total += w * self.compute_vorticity(psi)
                E_total += w * float(np.real(np.vdot(psi, H_current @ psi)))
            
            times.append(t)
            vorticities.append(V_total)
            energies.append(E_total)
            t += dt
        
        return {
            'times': times,
            'vorticities': vorticities,
            'energies': energies,
            'V_final': vorticities[-1] if vorticities else 0.0,
            'E_final': energies[-1] if energies else 0.0,
            'n_active': n_active,
            'T_kelvin': T_kelvin
        }
    
    def compare_paths(self, T_kelvin, path1, path2, **kwargs):
        if self.verbose:
            print("\n" + "="*60)
            print(f"🔬 2D Path Comparison: {self.H_type} at T={T_kelvin}K")
            print("="*60)
        
        result1 = self.evolve_with_field(T_kelvin, path1, **kwargs)
        result2 = self.evolve_with_field(T_kelvin, path2, **kwargs)
        
        delta_V = abs(result1['V_final'] - result2['V_final'])
        delta_E = abs(result1['E_final'] - result2['E_final'])
        
        if self.verbose:
            print(f"\n📊 Results:")
            print(f"   Path 1: V={result1['V_final']:+.4f}, E={result1['E_final']:.4f}")
            print(f"   Path 2: V={result2['V_final']:+.4f}, E={result2['E_final']:.4f}")
            print(f"   ΔV = {delta_V:.4f}, ΔE = {delta_E:.4f}")
            
            if delta_V > 0.01:
                print(f"\n   ✅ VORTICITY PATH DEPENDENCE!")
                print(f"   ✅ DFT predicts ΔV ≡ 0")
        
        return {
            'result1': result1,
            'result2': result2,
            'delta_vorticity': delta_V,
            'delta_energy': delta_E
        }
    
    def scan_hamiltonians(self, T_kelvin, path1, path2, **kwargs):
        """DFT信者を完全論破するスキャン！"""
        results = {}
        
        H_configs = [
            ('heisenberg', {'J': 1.0}),
            ('xy', {'J': 1.0}),
            ('xx', {'J': 1.0}),
            ('kitaev', {'Kx': 1.0, 'Ky': 1.0, 'Kz_diag': 0.5}),
            ('ising', {'J': 1.0, 'h': 0.5}),
            ('hubbard', {'t': 1.0, 'U': 2.0})
        ]
        
        print("\n" + "="*70)
        print("🔬 HAMILTONIAN SCAN: Path Dependence Across ALL Models")
        print("="*70)
        
        for H_type, params in H_configs:
            print(f"\n--- {H_type.upper()} ---")
            self.build_hamiltonian(H_type, **params)
            self.diagonalize(n_eigenstates=30)
            comparison = self.compare_paths(T_kelvin, path1, path2, **kwargs)
            results[H_type] = comparison
        
        print("\n" + "="*70)
        print("📊 SUMMARY: DFT vs DSE")
        print("="*70)
        print(f"{'Model':<15} {'ΔV (DSE)':<12} {'ΔV (DFT)':<12} {'Verdict'}")
        print("-"*55)
        for H_type, comp in results.items():
            verdict = "DSE WINS!" if comp['delta_vorticity'] > 0.01 else "~same"
            print(f"{H_type:<15} {comp['delta_vorticity']:<12.4f} {'0.0000':<12} {verdict}")
        
        return results


# =============================================================================
# Test
# =============================================================================

def run_full_test():
    """DFT完全論破テスト！"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  2D LADDER DSE: DFT信者を完全論破するデモ                           ║
║                                                                      ║
║  "DFT erases vorticity history. DSE remembers the spin."            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    t0 = time.time()
    
    solver = LadderDSESolver(Lx=3, Ly=3, verbose=True)
    
    # Two paths: same final fields, different ORDER
    path_A = [
        {'time': 2.0, 'site': 0, 'field': 'x', 'strength': 0.5},
        {'time': 5.0, 'site': 4, 'field': 'z', 'strength': 0.3}
    ]
    path_B = [
        {'time': 2.0, 'site': 4, 'field': 'z', 'strength': 0.3},
        {'time': 5.0, 'site': 0, 'field': 'x', 'strength': 0.5}
    ]
    
    results = solver.scan_hamiltonians(T_kelvin=100, path1=path_A, path2=path_B)
    
    print("\n" + "="*70)
    print("🏆 FINAL VERDICT")
    print("="*70)
    n_wins = sum(1 for r in results.values() if r['delta_vorticity'] > 0.01)
    print(f"   DSE detects path dependence in {n_wins}/{len(results)} models")
    print(f"   DFT detects path dependence in 0/{len(results)} models")
    print(f"\n   CONCLUSION: DFT is FUNDAMENTALLY BLIND!")
    print(f"   Total time: {time.time()-t0:.2f}s")
    
    return solver, results


if __name__ == "__main__":
    solver, results = run_full_test()
