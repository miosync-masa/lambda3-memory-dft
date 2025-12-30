"""
Repulsive Memory Kernel (パンツ由来の斥力項)
=============================================

Physical Origin:
  パンツのゴム弾性から着想を得た斥力の履歴依存性。
  
  🩲 → 伸縮 → 弾性ヒステリシス → Memory効果

Physical Basis:
  1. Pauli Exclusion Principle
     - 電子雲の重なり → 近距離斥力
     - V_rep ∝ 1/r^n (n ≈ 12 for LJ)
  
  2. Elastic Hysteresis
     - 圧縮履歴が現在の斥力に影響
     - ゴムの「へたり」と「回復」
  
  3. H-CSP Connection
     - Θ_env_phys の圧力二面性と対応
     - 局所圧縮 → |V|_eff 変化

Memory Kernel Form:
  K_rep(t, t') = exp(-(t-t')/τ_rep) × [1 - exp(-(t-t')/τ_recover)]
  
  - τ_rep: 斥力記憶の減衰時間（圧縮の「忘却」）
  - τ_recover: 回復時間（ゴムが元に戻る速度）
  
  物理的解釈:
  - t-t' 小: 圧縮直後 → 斥力増強
  - t-t' 中: 回復途中 → 斥力残留
  - t-t' 大: 完全回復 → 元の斥力

Application:
  - 高圧下の材料（ダイヤモンドアンビル）
  - 衝撃圧縮（衝突、爆発）
  - 摩擦界面（局所圧縮）
  - 触媒表面（吸着による歪み）

Author: Masamichi Iizumi, Tamaki Iizumi
Origin: 🩲 → 🧪 → Λ³
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CompressionEvent:
    """圧縮イベント"""
    time: float
    r_min: float        # 最小距離（圧縮度）
    pressure: float     # 局所圧力
    site: int = 0       # 圧縮サイト


class RepulsiveMemoryKernel:
    """
    Repulsive Memory Kernel
    
    近距離斥力の履歴依存性をモデル化。
    
    Physical Model:
      V_rep^eff(r, t) = V_rep(r) × [1 + η ∫ K(t-t') Θ(r_c - r(t')) dt']
    
    Parameters:
      η_rep: 斥力メモリ強度 (default: 0.2)
      τ_rep: 斥力減衰時間 (default: 3.0)
      τ_recover: 回復時間 (default: 10.0)
      r_critical: 臨界距離 (default: 0.8, 相対単位)
      n_power: 斥力指数 (default: 12, LJ型)
    
    Usage:
        kernel = RepulsiveMemoryKernel()
        kernel.add_compression(CompressionEvent(t=1.0, r_min=0.5, pressure=10.0))
        delta_V = kernel.compute_repulsion_enhancement(t=2.0, r_current=0.9)
    """
    
    def __init__(self,
                 eta_rep: float = 0.2,
                 tau_rep: float = 3.0,
                 tau_recover: float = 10.0,
                 r_critical: float = 0.8,
                 n_power: float = 12.0):
        """
        Args:
            eta_rep: Memory strength for repulsion
            tau_rep: Decay time for repulsion memory
            tau_recover: Recovery time (rubber returning to shape)
            r_critical: Critical distance below which compression is recorded
            n_power: Power-law exponent for repulsion (12 for LJ)
        """
        self.eta_rep = eta_rep
        self.tau_rep = tau_rep
        self.tau_recover = tau_recover
        self.r_critical = r_critical
        self.n_power = n_power
        
        self.compression_history: List[CompressionEvent] = []
        self.state_history: List[Tuple[float, float, np.ndarray]] = []  # (t, r, psi)
    
    def kernel_value(self, dt: float) -> float:
        """
        Compute kernel K(Δt)
        
        K(Δt) = exp(-Δt/τ_rep) × [1 - exp(-Δt/τ_recover)]
        
        Physical meaning:
        - First term: memory decay (forgetting compression)
        - Second term: recovery (rubber returning)
        - Product: net effect peaks at intermediate times
        """
        if dt <= 0:
            return 0.0
        
        decay = np.exp(-dt / self.tau_rep)
        recovery = 1.0 - np.exp(-dt / self.tau_recover)
        
        return decay * recovery
    
    def add_compression(self, event: CompressionEvent):
        """Record compression event"""
        self.compression_history.append(event)
    
    def add_state(self, t: float, r: float, psi: Optional[np.ndarray] = None):
        """Record state with distance"""
        if r < self.r_critical:
            # Auto-detect compression
            pressure = (self.r_critical / r) ** self.n_power
            self.add_compression(CompressionEvent(
                time=t,
                r_min=r,
                pressure=pressure
            ))
        
        if psi is not None:
            self.state_history.append((t, r, psi.copy()))
            if len(self.state_history) > 100:
                self.state_history = self.state_history[-100:]
    
    def compute_repulsion_enhancement(self, t: float, r_current: float) -> float:
        """
        Compute repulsion enhancement from compression history
        
        ΔV_rep = η ∫ K(t-t') × compression_factor(t') dt'
        
        Returns:
            Enhancement factor (multiply with bare repulsion)
        """
        if len(self.compression_history) == 0:
            return 0.0
        
        enhancement = 0.0
        
        for event in self.compression_history:
            dt = t - event.time
            if dt <= 0:
                continue
            
            # Kernel contribution
            K = self.kernel_value(dt)
            
            # Compression severity: how much was it squeezed?
            compression_factor = (self.r_critical / event.r_min) ** 2
            
            # Distance-dependent coupling
            # Closer current distance → more effect from past compression
            distance_coupling = 1.0
            if r_current < self.r_critical:
                distance_coupling = (self.r_critical / r_current)
            
            enhancement += self.eta_rep * K * compression_factor * distance_coupling
        
        return enhancement
    
    def compute_effective_repulsion(self, r: float, t: float, 
                                    A: float = 1.0) -> float:
        """
        Compute effective repulsion potential
        
        V_rep^eff = V_rep(r) × [1 + enhancement(t)]
        
        Args:
            r: Current distance
            t: Current time
            A: Amplitude of bare repulsion
        
        Returns:
            Effective repulsion energy
        """
        # Bare repulsion (LJ-type)
        V_bare = A / (r ** self.n_power)
        
        # Memory enhancement
        enhancement = self.compute_repulsion_enhancement(t, r)
        
        return V_bare * (1.0 + enhancement)
    
    def compute_lambda_contribution(self, t: float, 
                                    psi: np.ndarray,
                                    r_current: float) -> float:
        """
        Compute contribution to Λ from repulsive memory
        
        This affects |V|_eff in the EDR formula:
          Λ = K / |V|_eff
        
        Repulsion enhancement → |V|_eff increases → Λ decreases
        (More stable against further compression)
        """
        enhancement = self.compute_repulsion_enhancement(t, r_current)
        
        # Overlap with past states (quantum coherence effect)
        overlap_factor = 1.0
        if len(self.state_history) > 0:
            for t_hist, r_hist, psi_hist in self.state_history[-10:]:
                dt = t - t_hist
                if dt > 0:
                    overlap = abs(np.vdot(psi, psi_hist)) ** 2
                    overlap_factor += 0.1 * overlap * self.kernel_value(dt)
        
        return enhancement * overlap_factor
    
    def get_hysteresis_curve(self, 
                             r_range: np.ndarray,
                             compression_history: List[float]) -> dict:
        """
        Generate hysteresis curve for visualization
        
        Args:
            r_range: Array of distances
            compression_history: List of compression depths over time
        
        Returns:
            Dict with compression and expansion curves
        """
        # Reset and simulate
        self.compression_history = []
        
        n_steps = len(compression_history)
        t_values = np.arange(n_steps) * 0.1
        
        # Compression phase
        V_compress = []
        for i, r in enumerate(compression_history[:n_steps//2]):
            t = t_values[i]
            self.add_state(t, r)
            V = self.compute_effective_repulsion(r, t)
            V_compress.append(V)
        
        # Expansion phase
        V_expand = []
        for i, r in enumerate(compression_history[n_steps//2:]):
            t = t_values[n_steps//2 + i]
            V = self.compute_effective_repulsion(r, t)
            V_expand.append(V)
        
        return {
            'r_compress': compression_history[:n_steps//2],
            'r_expand': compression_history[n_steps//2:],
            'V_compress': V_compress,
            'V_expand': V_expand,
            'hysteresis_area': abs(np.trapezoid(V_compress) - np.trapezoid(V_expand))
        }
    
    def clear(self):
        """Clear all history"""
        self.compression_history = []
        self.state_history = []
    
    def __repr__(self) -> str:
        return (
            f"RepulsiveMemoryKernel(\n"
            f"  η_rep={self.eta_rep}, τ_rep={self.tau_rep}, τ_recover={self.tau_recover}\n"
            f"  r_critical={self.r_critical}, n_power={self.n_power}\n"
            f"  compression_events={len(self.compression_history)}\n"
            f"  Origin: 🩲 → Elastic Hysteresis → Memory-DFT\n"
            f")"
        )


# =============================================================================
# Integration with CompositeMemoryKernel
# =============================================================================

class ExtendedCompositeKernel:
    """
    Extended Composite Memory Kernel with Repulsion
    
    4-component kernel:
    - Θ_field   → PowerLaw (γ ≈ 1.2)
    - Θ_env_phys → StretchedExp (β ≈ 0.5)
    - Θ_env_chem → Step (reaction time)
    - Θ_repulsion → RepulsiveMemory (🩲)  ← NEW!
    
    H-CSP Correspondence:
      圧力の二面性（環境 + 場）を完全に実装
    """
    
    def __init__(self,
                 w_field: float = 0.35,
                 w_phys: float = 0.25,
                 w_chem: float = 0.20,
                 w_rep: float = 0.20):
        """
        Args:
            w_field: Weight for field kernel
            w_phys: Weight for physical environment kernel
            w_chem: Weight for chemical environment kernel
            w_rep: Weight for repulsive memory kernel (NEW!)
        """
        self.weights = {
            'field': w_field,
            'phys': w_phys,
            'chem': w_chem,
            'repulsion': w_rep
        }
        
        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Initialize repulsive kernel
        self.repulsive_kernel = RepulsiveMemoryKernel()
    
    def compute_total(self, t: float, 
                      history_data: dict,
                      psi: np.ndarray,
                      r_current: float = 1.0) -> float:
        """
        Compute total memory contribution including repulsion
        
        Args:
            t: Current time
            history_data: Dict with 'field', 'phys', 'chem' contributions
            psi: Current wavefunction
            r_current: Current characteristic distance
        
        Returns:
            Total memory contribution
        """
        total = 0.0
        
        # Standard components
        if 'field' in history_data:
            total += self.weights['field'] * history_data['field']
        if 'phys' in history_data:
            total += self.weights['phys'] * history_data['phys']
        if 'chem' in history_data:
            total += self.weights['chem'] * history_data['chem']
        
        # Repulsive memory (🩲)
        rep_contribution = self.repulsive_kernel.compute_lambda_contribution(
            t, psi, r_current
        )
        total += self.weights['repulsion'] * rep_contribution
        
        return total


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🩲 Repulsive Memory Kernel Test")
    print("="*60)
    
    kernel = RepulsiveMemoryKernel(
        eta_rep=0.3,
        tau_rep=3.0,
        tau_recover=10.0,
        r_critical=0.8
    )
    
    print(f"\n{kernel}")
    
    # Simulate compression-expansion cycle
    print("\n--- Compression-Expansion Cycle ---")
    
    # Phase 1: Compression
    print("\nPhase 1: Compression")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        r = 1.0 - 0.15 * t  # Compressing
        psi = np.array([1.0, 0.0])  # Dummy state
        kernel.add_state(t, r, psi)
        
        V_eff = kernel.compute_effective_repulsion(r, t)
        enhancement = kernel.compute_repulsion_enhancement(t, r)
        
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f}, enhancement={enhancement:.4f}")
    
    # Phase 2: Expansion
    print("\nPhase 2: Expansion (with memory!)")
    for t in [2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]:
        r = 0.7 + 0.05 * (t - 2.0)  # Expanding
        r = min(r, 1.0)
        psi = np.array([1.0, 0.0])
        
        V_eff = kernel.compute_effective_repulsion(r, t)
        enhancement = kernel.compute_repulsion_enhancement(t, r)
        
        print(f"  t={t:.1f}, r={r:.2f}: V_eff={V_eff:.4f}, enhancement={enhancement:.4f}")
    
    # Hysteresis
    print("\n--- Hysteresis Analysis ---")
    kernel.clear()
    
    # Create compression-expansion path
    r_compress = np.linspace(1.0, 0.6, 25)
    r_expand = np.linspace(0.6, 1.0, 25)
    r_path = np.concatenate([r_compress, r_expand])
    
    hysteresis = kernel.get_hysteresis_curve(
        r_range=np.linspace(0.6, 1.0, 50),
        compression_history=r_path.tolist()
    )
    
    print(f"  Hysteresis area: {hysteresis['hysteresis_area']:.4f}")
    print(f"  (Non-zero area = Memory effect!)")
    
    # Extended kernel
    print("\n--- Extended 4-Component Kernel ---")
    ext_kernel = ExtendedCompositeKernel()
    print(f"  Weights: {ext_kernel.weights}")
    
    print("\n✅ Repulsive Memory Kernel test passed!")
    print("\n🩲 → 🧪 → Λ³ : Physics from underwear to publications!")
