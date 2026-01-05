#!/usr/bin/env python3
"""
Topology Module for Memory-DFT
==============================

Unified topological invariants for quantum systems.
Backend: CuPy (GPU) / NumPy (CPU) - NO JAX!

Key Insight (from Λ³ theory):
  Energy = Topological Tension = Berry Connection
  E = ⟨Ψ|H|Ψ⟩ = iℏ⟨Ψ|∂_t|Ψ⟩ = ℏ × A_t
  
  Reconnection = Topological charge change = Integer jump

Implemented invariants:
  1. Q_Lambda (Spin Topological Charge)
     - Winding number on plaquettes
     - Physical space topology
     
  2. Berry Phase
     - Winding in parameter space
     - γ = ∮ i⟨Ψ|∇_R|Ψ⟩ dR = n × 2π
     
  3. Zak Phase (1D systems)
     - Band topology indicator
     - 0 or π (Z₂ classification)

Author: Tamaki & Masamichi Iizumi
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass, field
import scipy.sparse as sp

# CuPy support (consistent with memory-dft framework)
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    HAS_CUPY = True
except ImportError:
    cp = np
    csp = sp
    HAS_CUPY = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TopologyResult:
    """Container for topological invariants."""
    Q_Lambda: float = 0.0           # Spin topological charge
    berry_phase: float = 0.0        # Berry phase (mod 2π)
    winding_number: int = 0         # Integer winding number
    zak_phase: float = 0.0          # Zak phase (0 or π)
    
    # Per-site/per-plaquette data
    site_phases: Optional[np.ndarray] = None
    plaquette_windings: Optional[np.ndarray] = None
    
    def is_topological(self, threshold: float = 0.1) -> bool:
        """Check if system has non-trivial topology."""
        return (abs(self.winding_number) >= 1 or 
                abs(self.Q_Lambda) > threshold or
                abs(self.zak_phase) > threshold)
    
    def __repr__(self):
        return (f"TopologyResult(Q_Λ={self.Q_Lambda:.4f}, "
                f"γ_Berry={self.berry_phase:.4f}, "
                f"n={self.winding_number})")


@dataclass
class ReconnectionEvent:
    """A topological reconnection event."""
    time: float
    parameter_value: float
    Q_before: float
    Q_after: float
    delta_Q: float
    berry_phase_jump: float
    
    @property
    def is_integer_jump(self) -> bool:
        """Check if this is a true topological transition."""
        return abs(round(self.delta_Q) - self.delta_Q) < 0.1


# =============================================================================
# Spin Topological Charge (Q_Lambda)
# =============================================================================

class SpinTopologyCalculator:
    """
    Calculate spin topological charge Q_Λ.
    
    Q_Λ measures the winding number of the spin configuration
    around plaquettes in the lattice.
    
    For a plaquette with sites (i, j, k, l):
      Q = (1/2π) Σ Δθ_{ij}
    
    where Δθ_{ij} = θ_j - θ_i (wrapped to [-π, π])
    and θ_i = arctan2(⟨S_y^i⟩, ⟨S_x^i⟩)
    """
    
    def __init__(self, n_sites: int, use_gpu: bool = True):
        self.n_sites = n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
    
    def compute_site_phases(self, psi: np.ndarray, 
                            Sx: List, Sy: List) -> np.ndarray:
        """
        Compute spin phase θ_i = arctan2(⟨S_y^i⟩, ⟨S_x^i⟩) for each site.
        
        Args:
            psi: Wavefunction
            Sx: List of S_x operators for each site
            Sy: List of S_y operators for each site
            
        Returns:
            Array of phases [θ_0, θ_1, ..., θ_{N-1}]
        """
        xp = self.xp
        
        # Ensure psi is on correct device
        if self.use_gpu and not hasattr(psi, 'device'):
            psi = xp.asarray(psi)
        
        phases = xp.zeros(self.n_sites, dtype=xp.float64)
        
        for site in range(self.n_sites):
            # ⟨S_x⟩ and ⟨S_y⟩
            sx_exp = xp.real(xp.vdot(psi, Sx[site] @ psi))
            sy_exp = xp.real(xp.vdot(psi, Sy[site] @ psi))
            
            # Magnitude
            r = xp.sqrt(sx_exp**2 + sy_exp**2)
            
            # Phase (handle r ≈ 0)
            if float(r) > 1e-10:
                phases[site] = xp.arctan2(sy_exp, sx_exp)
            else:
                phases[site] = 0.0
        
        return phases
    
    def compute_Q_Lambda(self, psi: np.ndarray,
                         Sx: List, Sy: List,
                         plaquettes: List[Tuple[int, ...]]) -> TopologyResult:
        """
        Compute total topological charge Q_Λ.
        
        Args:
            psi: Wavefunction
            Sx: List of S_x operators
            Sy: List of S_y operators
            plaquettes: List of plaquette tuples, e.g. [(0,1,3,2), ...]
            
        Returns:
            TopologyResult with Q_Lambda and diagnostics
        """
        xp = self.xp
        
        # Get site phases
        phases = self.compute_site_phases(psi, Sx, Sy)
        
        # Compute winding for each plaquette
        Q_total = 0.0
        plaquette_windings = []
        
        for plaq in plaquettes:
            # Close the loop: [i, j, k, l, i]
            sites = list(plaq) + [plaq[0]]
            
            winding = 0.0
            for k in range(len(plaq)):
                i, j = sites[k], sites[k + 1]
                
                # Phase difference
                dtheta = float(phases[j] - phases[i])
                
                # Wrap to [-π, π]
                dtheta = ((dtheta + np.pi) % (2 * np.pi)) - np.pi
                
                winding += dtheta
            
            # Normalize by 2π
            Q_plaq = winding / (2 * np.pi)
            plaquette_windings.append(Q_plaq)
            Q_total += Q_plaq
        
        # Convert phases to numpy for storage
        if self.use_gpu:
            phases_np = phases.get()
        else:
            phases_np = phases
        
        return TopologyResult(
            Q_Lambda=float(Q_total),
            winding_number=int(round(Q_total)),
            site_phases=phases_np,
            plaquette_windings=np.array(plaquette_windings)
        )


# =============================================================================
# Berry Phase Calculator
# =============================================================================

class BerryPhaseCalculator:
    """
    Calculate Berry phase along a parameter path.
    
    Berry phase is the geometric phase acquired when a quantum state
    is adiabatically transported around a closed loop in parameter space.
    
    γ = i ∮ ⟨ψ(R)|∇_R|ψ(R)⟩ · dR
    
    Discretized version:
    γ = Im[ Σ_i log⟨ψ_i|ψ_{i+1}⟩ ]
    
    For a closed loop: γ = n × 2π (integer winding number)
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
    
    def compute_berry_phase(self, psi_list: List[np.ndarray],
                            closed_loop: bool = True) -> TopologyResult:
        """
        Compute Berry phase from a sequence of states.
        
        Args:
            psi_list: List of wavefunctions along parameter path
            closed_loop: If True, connect last state back to first
            
        Returns:
            TopologyResult with berry_phase and winding_number
        """
        xp = self.xp
        n_states = len(psi_list)
        
        if n_states < 2:
            return TopologyResult(berry_phase=0.0, winding_number=0)
        
        # Compute cumulative phase
        total_phase = 0.0
        
        for i in range(n_states - 1):
            psi_i = psi_list[i]
            psi_j = psi_list[i + 1]
            
            # Ensure on correct device
            if self.use_gpu:
                if not hasattr(psi_i, 'device'):
                    psi_i = xp.asarray(psi_i)
                if not hasattr(psi_j, 'device'):
                    psi_j = xp.asarray(psi_j)
            
            # Overlap
            overlap = xp.vdot(psi_i, psi_j)
            
            # Phase increment
            phase = float(xp.angle(overlap))
            total_phase += phase
        
        # Close the loop if requested
        if closed_loop and n_states > 2:
            psi_first = psi_list[0]
            psi_last = psi_list[-1]
            
            if self.use_gpu:
                if not hasattr(psi_first, 'device'):
                    psi_first = xp.asarray(psi_first)
                if not hasattr(psi_last, 'device'):
                    psi_last = xp.asarray(psi_last)
            
            overlap = xp.vdot(psi_last, psi_first)
            total_phase += float(xp.angle(overlap))
        
        # Winding number
        winding = int(round(total_phase / (2 * np.pi)))
        
        return TopologyResult(
            berry_phase=total_phase,
            winding_number=winding
        )
    
    def compute_berry_connection(self, psi_list: List[np.ndarray],
                                 dR: float = 1.0) -> np.ndarray:
        """
        Compute Berry connection A_i = i⟨ψ_i|∂_R|ψ_i⟩ ≈ i⟨ψ_i|ψ_{i+1} - ψ_i⟩/dR
        
        Returns:
            Array of Berry connection values
        """
        xp = self.xp
        n_states = len(psi_list)
        
        A = np.zeros(n_states - 1)
        
        for i in range(n_states - 1):
            psi_i = psi_list[i]
            psi_j = psi_list[i + 1]
            
            if self.use_gpu:
                if not hasattr(psi_i, 'device'):
                    psi_i = xp.asarray(psi_i)
                if not hasattr(psi_j, 'device'):
                    psi_j = xp.asarray(psi_j)
            
            # A = i⟨ψ|(|ψ'⟩ - |ψ⟩)/dR = i(⟨ψ|ψ'⟩ - 1)/dR
            overlap = xp.vdot(psi_i, psi_j)
            A[i] = float(xp.imag(overlap - 1.0) / dR)
        
        return A


# =============================================================================
# Zak Phase (1D systems)
# =============================================================================

class ZakPhaseCalculator:
    """
    Calculate Zak phase for 1D systems.
    
    The Zak phase is the Berry phase across the Brillouin zone:
    γ_Zak = ∫_0^{2π/a} A(k) dk
    
    For inversion-symmetric systems: γ_Zak = 0 or π (Z₂ classification)
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
    
    def compute_zak_phase(self, 
                          psi_k_list: List[np.ndarray],
                          k_points: np.ndarray) -> TopologyResult:
        """
        Compute Zak phase from Bloch states across BZ.
        
        Args:
            psi_k_list: List of Bloch states |ψ(k)⟩
            k_points: k-point values (should span 0 to 2π/a)
            
        Returns:
            TopologyResult with zak_phase
        """
        # Use Berry phase calculator
        berry_calc = BerryPhaseCalculator(use_gpu=self.use_gpu)
        result = berry_calc.compute_berry_phase(psi_k_list, closed_loop=True)
        
        # Zak phase should be 0 or π (mod 2π) for inversion-symmetric systems
        zak = result.berry_phase % (2 * np.pi)
        
        # Wrap to [0, 2π)
        if zak > np.pi:
            zak = zak - 2 * np.pi
        
        return TopologyResult(
            zak_phase=zak,
            berry_phase=result.berry_phase,
            winding_number=result.winding_number
        )


# =============================================================================
# Reconnection Detector
# =============================================================================

class ReconnectionDetector:
    """
    Detect topological reconnection events.
    
    A reconnection occurs when a topological invariant
    (Q_Λ, Berry phase, winding number) changes discontinuously.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Minimum change to count as reconnection
        """
        self.threshold = threshold
        self.history: List[TopologyResult] = []
        self.events: List[ReconnectionEvent] = []
    
    def update(self, result: TopologyResult, 
               time: float = 0.0,
               parameter: float = 0.0) -> Optional[ReconnectionEvent]:
        """
        Update with new topology result and check for reconnection.
        
        Returns:
            ReconnectionEvent if detected, None otherwise
        """
        event = None
        
        if self.history:
            prev = self.history[-1]
            delta_Q = result.Q_Lambda - prev.Q_Lambda
            delta_berry = result.berry_phase - prev.berry_phase
            
            # Check for significant change
            if abs(delta_Q) > self.threshold:
                event = ReconnectionEvent(
                    time=time,
                    parameter_value=parameter,
                    Q_before=prev.Q_Lambda,
                    Q_after=result.Q_Lambda,
                    delta_Q=delta_Q,
                    berry_phase_jump=delta_berry
                )
                self.events.append(event)
        
        self.history.append(result)
        return event
    
    def get_reconnection_count(self) -> int:
        """Total number of reconnection events detected."""
        return len(self.events)
    
    def get_total_Q_change(self) -> float:
        """Total change in Q_Λ across all events."""
        return sum(e.delta_Q for e in self.events)


# =============================================================================
# Unified Topology Engine
# =============================================================================

class TopologyEngine:
    """
    Unified engine for all topological calculations.
    
    Combines:
      - Spin topology (Q_Λ)
      - Berry phase
      - Zak phase
      - Reconnection detection
    
    Example:
        >>> engine = TopologyEngine(n_sites=4, use_gpu=False)
        >>> result = engine.compute_all(psi, Sx, Sy, plaquettes)
        >>> print(f"Q_Λ = {result.Q_Lambda}, γ = {result.berry_phase}")
    """
    
    def __init__(self, n_sites: int, use_gpu: bool = True):
        self.n_sites = n_sites
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        
        # Sub-calculators
        self.spin_calc = SpinTopologyCalculator(n_sites, use_gpu)
        self.berry_calc = BerryPhaseCalculator(use_gpu)
        self.zak_calc = ZakPhaseCalculator(use_gpu)
        self.reconnection_detector = ReconnectionDetector()
    
    def compute_Q_Lambda(self, psi: np.ndarray,
                         Sx: List, Sy: List,
                         plaquettes: List[Tuple[int, ...]]) -> TopologyResult:
        """Compute spin topological charge."""
        return self.spin_calc.compute_Q_Lambda(psi, Sx, Sy, plaquettes)
    
    def compute_berry_phase(self, psi_list: List[np.ndarray],
                            closed_loop: bool = True) -> TopologyResult:
        """Compute Berry phase along parameter path."""
        return self.berry_calc.compute_berry_phase(psi_list, closed_loop)
    
    def compute_berry_phase_cycle(self, 
                                  hamiltonian_builder: Callable[[float], Any],
                                  param_values: np.ndarray,
                                  initial_psi: Optional[np.ndarray] = None) -> TopologyResult:
        """
        Compute Berry phase by cycling a parameter.
        
        Args:
            hamiltonian_builder: Function R -> H(R)
            param_values: Parameter values forming a closed loop
            initial_psi: Initial state (if None, uses ground state)
            
        Returns:
            TopologyResult with Berry phase
        """
        from scipy.sparse.linalg import eigsh
        
        psi_list = []
        
        for R in param_values:
            H = hamiltonian_builder(R)
            
            # Convert to scipy sparse if needed
            if hasattr(H, 'get'):
                H = sp.csr_matrix(H.get())
            
            # Get ground state
            E, psi = eigsh(H, k=1, which='SA')
            psi = psi[:, 0]
            
            # Fix gauge (make first nonzero element real and positive)
            idx = np.argmax(np.abs(psi))
            phase = np.angle(psi[idx])
            psi = psi * np.exp(-1j * phase)
            
            psi_list.append(psi)
        
        return self.berry_calc.compute_berry_phase(psi_list, closed_loop=True)
    
    def track_reconnection(self, result: TopologyResult,
                           time: float = 0.0,
                           parameter: float = 0.0) -> Optional[ReconnectionEvent]:
        """Track topology and detect reconnection."""
        return self.reconnection_detector.update(result, time, parameter)


# =============================================================================
# Test / Demo
# =============================================================================

def test_berry_phase_simple():
    """Test Berry phase with a simple two-level system."""
    print("=" * 60)
    print("TEST: Berry Phase (Two-Level System)")
    print("=" * 60)
    
    # Two-level system: H(θ) = cos(θ)σ_z + sin(θ)σ_x
    # Berry phase should be π for a full cycle
    
    n_points = 50
    theta_values = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    psi_list = []
    for theta in theta_values:
        # Ground state of H(θ)
        # |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩  (for θ in [0, π])
        # More generally, need to solve eigenvalue problem
        
        H = np.array([
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ])
        
        E, V = np.linalg.eigh(H)
        psi = V[:, 0]  # Ground state
        
        # Fix gauge
        if psi[0] != 0:
            psi = psi * np.exp(-1j * np.angle(psi[0]))
        
        psi_list.append(psi)
    
    # Compute Berry phase
    calc = BerryPhaseCalculator(use_gpu=False)
    result = calc.compute_berry_phase(psi_list, closed_loop=True)
    
    print(f"  θ range: 0 → 2π ({n_points} points)")
    print(f"  Berry phase: γ = {result.berry_phase:.4f}")
    print(f"  Expected: γ = π = {np.pi:.4f}")
    print(f"  Winding number: n = {result.winding_number}")
    print(f"  Expected: n = 0 or ±1 (mod gauge)")
    
    # Check
    # Note: The Berry phase for this system should be ±π
    gamma_mod = result.berry_phase % (2 * np.pi)
    if gamma_mod > np.pi:
        gamma_mod -= 2 * np.pi
    
    print(f"  γ (mod 2π): {gamma_mod:.4f}")
    
    if abs(abs(gamma_mod) - np.pi) < 0.3:
        print("  ✅ Berry phase test PASSED!")
    else:
        print("  ⚠️ Berry phase test needs investigation")
    
    return result


def test_Q_Lambda_simple():
    """Test Q_Lambda with a simple 2x2 plaquette."""
    print("\n" + "=" * 60)
    print("TEST: Q_Lambda (2x2 Plaquette)")
    print("=" * 60)
    
    # 4-site system with one plaquette
    n_sites = 4
    
    # Build spin operators (S = 1/2)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    I2 = np.eye(2, dtype=complex)
    
    def kron_list(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    Sx = []
    Sy = []
    for site in range(n_sites):
        ops_x = [I2] * n_sites
        ops_x[site] = sigma_x
        Sx.append(kron_list(ops_x))
        
        ops_y = [I2] * n_sites
        ops_y[site] = sigma_y
        Sy.append(kron_list(ops_y))
    
    # Test state: |↑↓↑↓⟩ (Néel state) - should have Q ≈ 0
    psi_neel = np.zeros(16, dtype=complex)
    psi_neel[0b0101] = 1.0  # |↑↓↑↓⟩
    
    # Plaquette: sites 0-1-3-2 (square)
    #  0 -- 1
    #  |    |
    #  2 -- 3
    plaquettes = [(0, 1, 3, 2)]
    
    calc = SpinTopologyCalculator(n_sites, use_gpu=False)
    result = calc.compute_Q_Lambda(psi_neel, Sx, Sy, plaquettes)
    
    print(f"  State: |↑↓↑↓⟩ (Néel)")
    print(f"  Plaquette: {plaquettes[0]}")
    print(f"  Site phases: {result.site_phases}")
    print(f"  Q_Lambda: {result.Q_Lambda:.4f}")
    print(f"  Winding: {result.winding_number}")
    
    # Test state: superposition (should have non-trivial Q)
    psi_super = np.ones(16, dtype=complex) / 4
    result2 = calc.compute_Q_Lambda(psi_super, Sx, Sy, plaquettes)
    
    print(f"\n  State: equal superposition")
    print(f"  Q_Lambda: {result2.Q_Lambda:.4f}")
    
    print("  ✅ Q_Lambda test completed!")
    
    return result, result2


if __name__ == "__main__":
    print("\n" + "🔬" * 20)
    print("TOPOLOGY MODULE TEST")
    print("🔬" * 20)
    
    test_berry_phase_simple()
    test_Q_Lambda_simple()
    
    print("\n" + "=" * 60)
    print("✅ All topology tests completed!")
    print("=" * 60)
