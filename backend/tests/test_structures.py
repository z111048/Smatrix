"""
Comprehensive test suite for Smatrix structural analysis engine.
Tests cover single-span, multi-span, and various load configurations.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.structure import Structure, SupportType
from app.beam_element import BeamElement


class TestBeamElement:
    """Tests for BeamElement class"""
    
    def test_stiffness_matrix_symmetry(self):
        """Stiffness matrix must be symmetric"""
        beam = BeamElement(E=200e9, I=1e-4, L=4.0)
        K = beam.stiffness_matrix_bending()
        assert np.allclose(K, K.T), "Stiffness matrix is not symmetric"
    
    def test_stiffness_matrix_values(self):
        """Verify specific stiffness matrix coefficients"""
        E, I, L = 200e9, 1e-4, 4.0
        beam = BeamElement(E=E, I=I, L=L)
        K = beam.stiffness_matrix_bending()
        
        # K[0,0] = 12EI/L³
        assert abs(K[0, 0] - 12 * E * I / L**3) < 1, "K[0,0] incorrect"
        # K[1,1] = 4EI/L
        assert abs(K[1, 1] - 4 * E * I / L) < 1, "K[1,1] incorrect"
        # K[0,1] = 6EI/L²
        assert abs(K[0, 1] - 6 * E * I / L**2) < 1, "K[0,1] incorrect"
    
    def test_fixed_end_forces_udl(self):
        """Verify fixed-end forces for UDL"""
        beam = BeamElement(E=200e9, I=1e-4, L=6.0)
        w = -10000  # 10 kN/m downward
        fem = beam.fixed_end_forces_udl(w)
        
        # V = wL/2
        expected_V = w * 6 / 2
        assert abs(fem[0] - expected_V) < 1, "FEM V_i incorrect"
        assert abs(fem[2] - expected_V) < 1, "FEM V_j incorrect"
        
        # M = wL²/12
        expected_M = w * 36 / 12
        assert abs(fem[1] - expected_M) < 1, "FEM M_i incorrect"


class TestSingleSpan:
    """Tests for single-span beam configurations"""
    
    E = 200e9  # Pa
    I = 1e-4   # m^4
    
    def test_T1_1_simply_supported_center_load(self):
        """T1-1: Simply supported beam with center point load"""
        L = 10.0
        P = 100000  # 100 kN
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.PIN)
        struct.add_node(2, x=L/2)
        struct.add_node(3, x=L, support=SupportType.ROLLER)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_point_load(2, Fy=-P)
        
        struct.solve()
        v_center = struct.get_node_displacement(2)[0]
        
        # δ = PL³ / 48EI
        delta_theory = P * L**3 / (48 * self.E * self.I)
        error = abs(-v_center - delta_theory) / delta_theory
        
        assert error < 0.01, f"Displacement error {error*100:.2f}% > 1%"
    
    def test_T1_2_simply_supported_udl(self):
        """T1-2: Simply supported beam with uniformly distributed load"""
        L = 8.0
        w = 10000  # 10 kN/m
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.PIN)
        struct.add_node(2, x=L/2)
        struct.add_node(3, x=L, support=SupportType.ROLLER)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_udl(1, w=-w)
        struct.add_udl(2, w=-w)
        
        result = struct.solve()
        v_center = struct.get_node_displacement(2)[0]
        
        # δ = 5wL⁴ / 384EI
        delta_theory = 5 * w * L**4 / (384 * self.E * self.I)
        error = abs(-v_center - delta_theory) / delta_theory
        
        assert error < 0.05, f"Displacement error {error*100:.2f}% > 5%"
        
        # Check reactions: R = wL/2
        R_expected = w * L / 2
        R_A = result["reactions"][1][0]
        R_C = result["reactions"][3][0]
        
        assert abs(R_A - R_expected) / R_expected < 0.01, "R_A incorrect"
        assert abs(R_C - R_expected) / R_expected < 0.01, "R_C incorrect"
    
    def test_T1_3_cantilever_tip_load(self):
        """T1-3: Cantilever beam with tip point load"""
        L = 4.0
        P = 50000  # 50 kN
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.FIXED)
        struct.add_node(2, x=L)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_point_load(2, Fy=-P)
        
        result = struct.solve()
        v_tip = struct.get_node_displacement(2)[0]
        
        # δ = PL³ / 3EI
        delta_theory = P * L**3 / (3 * self.E * self.I)
        error = abs(-v_tip - delta_theory) / delta_theory
        
        assert error < 0.01, f"Tip displacement error {error*100:.2f}% > 1%"
        
        # Fixed moment = PL
        M_fixed = result["reactions"][1][1]
        M_expected = P * L
        assert abs(abs(M_fixed) - M_expected) / M_expected < 0.01, "Fixed moment incorrect"
    
    def test_T1_4_cantilever_udl(self):
        """T1-4: Cantilever beam with UDL"""
        L = 4.0
        w = 20000  # 20 kN/m
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.FIXED)
        struct.add_node(2, x=L)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_udl(1, w=-w)
        
        result = struct.solve()
        v_tip = struct.get_node_displacement(2)[0]
        
        # δ = wL⁴ / 8EI
        delta_theory = w * L**4 / (8 * self.E * self.I)
        error = abs(-v_tip - delta_theory) / delta_theory
        
        assert error < 0.02, f"Tip displacement error {error*100:.2f}% > 2%"
    
    def test_T1_5_fixed_fixed_center_load(self):
        """T1-5: Fixed-fixed beam with center point load"""
        L = 8.0
        P = 100000  # 100 kN
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.FIXED)
        struct.add_node(2, x=L/2)
        struct.add_node(3, x=L, support=SupportType.FIXED)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_point_load(2, Fy=-P)
        
        result = struct.solve()
        
        # Fixed-end moment = PL/8
        M_expected = P * L / 8
        M_A = abs(result["reactions"][1][1])
        M_C = abs(result["reactions"][3][1])
        
        assert abs(M_A - M_expected) / M_expected < 0.05, "M_A incorrect"
        assert abs(M_C - M_expected) / M_expected < 0.05, "M_C incorrect"
    
    def test_T1_6_fixed_fixed_udl(self):
        """T1-6: Fixed-fixed beam with UDL"""
        L = 6.0
        w = 15000  # 15 kN/m
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.FIXED)
        struct.add_node(2, x=L, support=SupportType.FIXED)  # Both ends fixed
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_udl(1, w=-w)
        
        result = struct.solve()
        
        # For fixed-fixed beam with UDL, each reaction = wL/2
        R_A = result["reactions"][1][0]
        R_B = result["reactions"][2][0]
        R_total = R_A + R_B
        
        total_load = w * L  # 90 kN
        
        # Equilibrium check
        assert abs(R_total - total_load) / total_load < 0.01, \
            f"R_total = {R_total}, expected {total_load}"


class TestThreeMembers:
    """Tests for three-member structures"""
    
    E = 200e9
    I = 1e-4
    
    def test_T3_1_two_span_continuous_full_udl(self):
        """T3-1: Two-span continuous beam with UDL on both spans"""
        L = 6.0
        w = 20000  # 20 kN/m
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.PIN)
        struct.add_node(2, x=L, support=SupportType.PIN)
        struct.add_node(3, x=2*L, support=SupportType.ROLLER)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_udl(1, w=-w)
        struct.add_udl(2, w=-w)
        
        result = struct.solve()
        
        R_A = result["reactions"][1][0]
        R_B = result["reactions"][2][0]
        R_C = result["reactions"][3][0]
        
        # Middle support should have higher reaction
        assert R_B > R_A, "R_B should be greater than R_A"
        assert R_B > R_C, "R_B should be greater than R_C"
        
        # Total reaction should equal total load
        total_load = w * 2 * L
        total_reaction = R_A + R_B + R_C
        assert abs(total_reaction - total_load) / total_load < 0.01, "Equilibrium check failed"
    
    def test_T3_2_two_span_continuous_single_udl(self):
        """T3-2: Two-span continuous beam with UDL on one span only"""
        L = 6.0
        w = 20000  # 20 kN/m
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.PIN)
        struct.add_node(2, x=L, support=SupportType.PIN)
        struct.add_node(3, x=2*L, support=SupportType.ROLLER)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_udl(1, w=-w)
        
        result = struct.solve()
        
        R_A = result["reactions"][1][0]
        R_B = result["reactions"][2][0]
        R_C = result["reactions"][3][0]
        
        # Reactions should be asymmetric
        assert abs(R_A - R_C) > 1000, "Reactions should be asymmetric"
        assert R_A > R_C, "R_A should be greater than R_C"


class TestMultiMember:
    """Tests for multi-member systems"""
    
    E = 200e9
    I = 1e-4
    
    def test_TM_1_three_span_continuous(self):
        """TM-1: Three-span continuous beam with UDL"""
        L = 5.0
        w = 15000  # 15 kN/m
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.PIN)
        struct.add_node(2, x=L, support=SupportType.PIN)
        struct.add_node(3, x=2*L, support=SupportType.PIN)
        struct.add_node(4, x=3*L, support=SupportType.ROLLER)
        
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_element(3, 3, 4, E=self.E, I=self.I)
        
        struct.add_udl(1, w=-w)
        struct.add_udl(2, w=-w)
        struct.add_udl(3, w=-w)
        
        result = struct.solve()
        
        # Check equilibrium
        total_load = w * 3 * L
        total_reaction = sum(r[0] for r in result["reactions"].values())
        assert abs(total_reaction - total_load) / total_load < 0.01, "Equilibrium check failed"


class TestInternalForces:
    """Tests for internal force calculations"""
    
    E = 200e9
    I = 1e-4
    
    def test_shear_simply_supported_center_load(self):
        """Shear force diagram for simply supported beam with center load"""
        L = 10.0
        P = 100000  # 100 kN
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.PIN)
        struct.add_node(2, x=L/2)
        struct.add_node(3, x=L, support=SupportType.ROLLER)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_point_load(2, Fy=-P)
        
        struct.solve()
        forces = struct.compute_internal_forces()
        
        # Shear should be approximately P/2
        V_left = forces[1]["V"]
        assert abs(abs(V_left[0]) - P/2) < 1000, "V_left incorrect"
    
    def test_moment_simply_supported_center_load(self):
        """Bending moment diagram for simply supported beam with center load"""
        L = 10.0
        P = 100000  # 100 kN
        
        struct = Structure()
        struct.add_node(1, x=0, support=SupportType.PIN)
        struct.add_node(2, x=L/2)
        struct.add_node(3, x=L, support=SupportType.ROLLER)
        struct.add_element(1, 1, 2, E=self.E, I=self.I)
        struct.add_element(2, 2, 3, E=self.E, I=self.I)
        struct.add_point_load(2, Fy=-P)
        
        struct.solve()
        forces = struct.compute_internal_forces()
        
        # Max moment at center = PL/4
        M_max_expected = P * L / 4
        M_left_end = abs(forces[1]["M"][-1])
        
        assert abs(M_left_end - M_max_expected) / M_max_expected < 0.05, \
            f"M_max = {M_left_end}, expected {M_max_expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
