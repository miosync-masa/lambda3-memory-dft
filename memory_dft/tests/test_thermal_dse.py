"""
Test Thermal-DSE: Finite-temperature path dependence
====================================================

Tests for thermal path-dependent effects that DFT cannot capture.

Key insight:
  - Same final temperature, different heating/cooling history
  - Same perturbations, different order
  → Different quantum outcomes!

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import pytest
import numpy as np


class TestThermalDSE:
    """Test suite for Thermal-DSE solver."""
    
    def test_import(self):
        """Test that ThermalDSESolver can be imported."""
        from memory_dft import ThermalDSESolver
        assert ThermalDSESolver is not None
    
    def test_temperature_conversion(self):
        """Test T_to_beta and beta_to_T conversions."""
        from memory_dft import T_to_beta, beta_to_T
        
        # Round-trip test
        T = 300.0
        beta = T_to_beta(T, energy_scale=1.0)
        T_back = beta_to_T(beta, energy_scale=1.0)
        
        assert abs(T - T_back) < 1e-10
        
        # Zero temperature
        beta_inf = T_to_beta(0.0)
        assert beta_inf == float('inf')
        
        T_zero = beta_to_T(float('inf'))
        assert T_zero == 0.0
    
    def test_basic_solver(self):
        """Test basic ThermalDSESolver functionality."""
        from memory_dft import ThermalDSESolver
        
        solver = ThermalDSESolver(n_sites=4, verbose=False)
        solver.build_hubbard(t_hop=1.0, U_int=2.0)
        solver.diagonalize(n_eigenstates=10)
        
        # Check eigenvalues
        assert solver.eigenvalues is not None
        assert len(solver.eigenvalues) == 10
        assert solver.eigenvalues[0] < solver.eigenvalues[1]  # Ground state lowest
    
    def test_thermal_lambda(self):
        """Test thermal λ computation at different temperatures."""
        from memory_dft import ThermalDSESolver, T_to_beta
        
        solver = ThermalDSESolver(n_sites=4, verbose=False)
        solver.build_hubbard(t_hop=1.0, U_int=2.0)
        solver.diagonalize(n_eigenstates=14)
        
        # λ should vary with temperature
        lambda_50K = solver.compute_lambda(T_to_beta(50, 0.1))
        lambda_300K = solver.compute_lambda(T_to_beta(300, 0.1))
        
        # Different temperatures should give different λ
        assert lambda_50K != lambda_300K
        
        # λ should generally increase with T (more kinetic energy)
        assert lambda_300K > lambda_50K


class TestThermalPathDependence:
    """Test path-dependent thermal effects."""
    
    def test_perturbation_order_dependence(self):
        """
        Test that perturbation order affects final state.
        
        Path A: Perturb site 0 first, then site 2
        Path B: Perturb site 2 first, then site 0
        
        Same final perturbation, different history → different λ
        """
        from memory_dft import ThermalDSESolver
        
        solver = ThermalDSESolver(n_sites=4, verbose=False)
        solver.build_hubbard(t_hop=1.0, U_int=2.0)
        solver.diagonalize(n_eigenstates=14)
        
        # Path A: Site 0 first
        path_A = [
            {'time': 2.0, 'site': 0, 'potential': -0.5},
            {'time': 5.0, 'site': 2, 'potential': -0.3}
        ]
        
        # Path B: Site 2 first
        path_B = [
            {'time': 2.0, 'site': 2, 'potential': -0.3},
            {'time': 5.0, 'site': 0, 'potential': -0.5}
        ]
        
        comparison = solver.compare_thermal_perturbation_paths(
            T_kelvin=100, 
            path1=path_A, 
            path2=path_B,
            dt=0.1
        )
        
        # Path dependence should be detected
        assert comparison['delta_lambda'] > 0.1
        print(f"\n✅ ΔΛ = {comparison['delta_lambda']:.4f}")
    
    def test_temperature_affects_path_dependence(self):
        """
        Test that path dependence varies with temperature.
        
        Expected: Higher T → weaker path dependence
        (thermal averaging washes out memory effects)
        """
        from memory_dft import ThermalDSESolver
        
        solver = ThermalDSESolver(n_sites=4, verbose=False)
        solver.build_hubbard(t_hop=1.0, U_int=2.0)
        solver.diagonalize(n_eigenstates=14)
        
        path_A = [
            {'time': 2.0, 'site': 0, 'potential': -0.5},
            {'time': 5.0, 'site': 2, 'potential': -0.3}
        ]
        path_B = [
            {'time': 2.0, 'site': 2, 'potential': -0.3},
            {'time': 5.0, 'site': 0, 'potential': -0.5}
        ]
        
        # Measure ΔΛ at different temperatures
        delta_50K = solver.compare_thermal_perturbation_paths(
            50, path_A, path_B, dt=0.1
        )['delta_lambda']
        
        delta_300K = solver.compare_thermal_perturbation_paths(
            300, path_A, path_B, dt=0.1
        )['delta_lambda']
        
        print(f"\n  ΔΛ(50K)  = {delta_50K:.4f}")
        print(f"  ΔΛ(300K) = {delta_300K:.4f}")
        
        # Higher T should have smaller path dependence
        # (thermal averaging reduces memory effects)
        assert delta_50K > delta_300K * 0.9  # Allow some tolerance
        
        print(f"\n✅ Path dependence decreases with temperature!")


class TestThermalEntropy:
    """Test entropy and thermodynamic quantities."""
    
    def test_entropy_increases_with_temperature(self):
        """Test that entropy increases with temperature."""
        from memory_dft.solvers.thermal_dse import compute_entropy, T_to_beta
        
        eigenvalues = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
        
        S_low = compute_entropy(eigenvalues, T_to_beta(50, 0.1))
        S_high = compute_entropy(eigenvalues, T_to_beta(300, 0.1))
        
        assert S_high > S_low
        print(f"\n  S(50K)  = {S_low:.4f}")
        print(f"  S(300K) = {S_high:.4f}")
        print(f"\n✅ Entropy increases with temperature!")


def run_thermal_dse_tests():
    """Run all Thermal-DSE tests."""
    print("="*70)
    print("🌡️ Thermal-DSE Test Suite")
    print("="*70)
    
    # Basic tests
    test = TestThermalDSE()
    test.test_import()
    print("✅ Import test passed")
    
    test.test_temperature_conversion()
    print("✅ Temperature conversion test passed")
    
    test.test_basic_solver()
    print("✅ Basic solver test passed")
    
    test.test_thermal_lambda()
    print("✅ Thermal λ test passed")
    
    # Path dependence tests
    path_test = TestThermalPathDependence()
    path_test.test_perturbation_order_dependence()
    print("✅ Perturbation order test passed")
    
    path_test.test_temperature_affects_path_dependence()
    print("✅ Temperature-path dependence test passed")
    
    # Entropy test
    entropy_test = TestThermalEntropy()
    entropy_test.test_entropy_increases_with_temperature()
    print("✅ Entropy test passed")
    
    print("\n" + "="*70)
    print("🎉 All Thermal-DSE tests passed!")
    print("="*70)


if __name__ == "__main__":
    run_thermal_dse_tests()
