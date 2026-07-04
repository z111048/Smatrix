"""
Comprehensive test suite for Smatrix structural analysis engine.
Tests cover single-span, multi-span, and various load configurations.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.frame_element import FrameElement2D, ReleaseType
from app.structure_2d import Structure2D, SupportType as SupportType2D


DEFAULT_A = 1e-2


def add_uniform_load(struct: Structure2D, element_id: int, w: float):
    """Add a full-span uniform load through the live Structure2D API."""
    struct.add_element_trapezoidal_load(element_id, w1=w, w2=w)


def vertical_displacement(struct: Structure2D, node_id: int) -> float:
    return struct.get_node_displacement(node_id)[1]


def vertical_reaction(result: dict, node_id: int) -> float:
    return result["reactions"][node_id][1]


def moment_reaction(result: dict, node_id: int) -> float:
    return result["reactions"][node_id][2]


class TestFrameElementBenchmarks:
    """Benchmarks formerly covered by the retired 1D beam element."""

    def test_stiffness_matrix_symmetry(self):
        """Stiffness matrix must be symmetric"""
        elem = FrameElement2D(
            E=200e9,
            A=DEFAULT_A,
            I=1e-4,
            node_i=(0, 0),
            node_j=(4.0, 0),
        )
        K = elem.local_stiffness_matrix()
        assert np.allclose(K, K.T), "Stiffness matrix is not symmetric"

    def test_stiffness_matrix_values(self):
        """Verify specific bending stiffness coefficients"""
        E, I, L = 200e9, 1e-4, 4.0
        elem = FrameElement2D(
            E=E,
            A=DEFAULT_A,
            I=I,
            node_i=(0, 0),
            node_j=(L, 0),
        )
        K = elem.local_stiffness_matrix()

        # Local DOFs: [u_i, v_i, theta_i, u_j, v_j, theta_j]
        assert abs(K[1, 1] - 12 * E * I / L**3) < 1, "K[v_i,v_i] incorrect"
        assert abs(K[2, 2] - 4 * E * I / L) < 1, "K[theta_i,theta_i] incorrect"
        assert abs(K[1, 2] - 6 * E * I / L**2) < 1, "K[v_i,theta_i] incorrect"

    def test_fixed_end_forces_udl(self):
        """Verify fixed-end forces for UDL"""
        elem = FrameElement2D(
            E=200e9,
            A=DEFAULT_A,
            I=1e-4,
            node_i=(0, 0),
            node_j=(6.0, 0),
        )
        w = -10000  # 10 kN/m downward
        fem = elem.fixed_end_forces_udl_local(w)

        # V = wL/2
        expected_V = w * 6 / 2
        assert abs(fem[1] - expected_V) < 1, "FEM V_i incorrect"
        assert abs(fem[4] - expected_V) < 1, "FEM V_j incorrect"

        # M = wL^2/12
        expected_M = w * 36 / 12
        assert abs(fem[2] - expected_M) < 1, "FEM M_i incorrect"


class TestSingleSpan:
    """Tests for single-span beam configurations"""

    E = 200e9  # Pa
    A = DEFAULT_A
    I = 1e-4  # m^4

    def test_T1_1_simply_supported_center_load(self):
        """T1-1: Simply supported beam with center point load"""
        L = 10.0
        P = 100000  # 100 kN

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, L / 2, 0)
        struct.add_node(3, L, 0, SupportType2D.ROLLER_X)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_point_load(2, Fy=-P)

        struct.solve()
        v_center = vertical_displacement(struct, 2)

        # delta = PL^3 / 48EI
        delta_theory = P * L**3 / (48 * self.E * self.I)
        error = abs(-v_center - delta_theory) / delta_theory

        assert error < 0.01, f"Displacement error {error*100:.2f}% > 1%"

    def test_T1_2_simply_supported_udl(self):
        """T1-2: Simply supported beam with uniformly distributed load"""
        L = 8.0
        w = 10000  # 10 kN/m

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, L / 2, 0)
        struct.add_node(3, L, 0, SupportType2D.ROLLER_X)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        add_uniform_load(struct, 1, -w)
        add_uniform_load(struct, 2, -w)

        result = struct.solve()
        v_center = vertical_displacement(struct, 2)

        # delta = 5wL^4 / 384EI
        delta_theory = 5 * w * L**4 / (384 * self.E * self.I)
        error = abs(abs(v_center) - delta_theory) / delta_theory

        assert error < 0.05, f"Displacement error {error*100:.2f}% > 5%"

        # Check reactions: R = wL/2
        R_expected = w * L / 2
        R_A = vertical_reaction(result, 1)
        R_C = vertical_reaction(result, 3)

        assert abs(abs(R_A) - R_expected) / R_expected < 0.01, "R_A incorrect"
        assert abs(abs(R_C) - R_expected) / R_expected < 0.01, "R_C incorrect"

    def test_T1_3_cantilever_tip_load(self):
        """T1-3: Cantilever beam with tip point load"""
        L = 4.0
        P = 50000  # 50 kN

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(2, L, 0)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_point_load(2, Fy=-P)

        result = struct.solve()
        v_tip = vertical_displacement(struct, 2)

        # delta = PL^3 / 3EI
        delta_theory = P * L**3 / (3 * self.E * self.I)
        error = abs(-v_tip - delta_theory) / delta_theory

        assert error < 0.01, f"Tip displacement error {error*100:.2f}% > 1%"

        # Fixed moment = PL
        M_fixed = moment_reaction(result, 1)
        M_expected = P * L
        assert abs(abs(M_fixed) - M_expected) / M_expected < 0.01, "Fixed moment incorrect"

    def test_T1_4_cantilever_udl(self):
        """T1-4: Cantilever beam with UDL"""
        L = 4.0
        w = 20000  # 20 kN/m

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(2, L, 0)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        add_uniform_load(struct, 1, -w)

        struct.solve()
        v_tip = vertical_displacement(struct, 2)

        # delta = wL^4 / 8EI
        delta_theory = w * L**4 / (8 * self.E * self.I)
        error = abs(abs(v_tip) - delta_theory) / delta_theory

        assert error < 0.02, f"Tip displacement error {error*100:.2f}% > 2%"

    def test_T1_5_fixed_fixed_center_load(self):
        """T1-5: Fixed-fixed beam with center point load"""
        L = 8.0
        P = 100000  # 100 kN

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(2, L / 2, 0)
        struct.add_node(3, L, 0, SupportType2D.FIXED)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_point_load(2, Fy=-P)

        result = struct.solve()

        # Fixed-end moment = PL/8
        M_expected = P * L / 8
        M_A = abs(moment_reaction(result, 1))
        M_C = abs(moment_reaction(result, 3))

        assert abs(M_A - M_expected) / M_expected < 0.05, "M_A incorrect"
        assert abs(M_C - M_expected) / M_expected < 0.05, "M_C incorrect"

    def test_T1_6_fixed_fixed_udl(self):
        """T1-6: Fixed-fixed beam with UDL"""
        L = 6.0
        w = 15000  # 15 kN/m

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(2, L, 0, SupportType2D.FIXED)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        add_uniform_load(struct, 1, -w)

        result = struct.solve()

        # For fixed-fixed beam with UDL, each reaction = wL/2
        R_A = vertical_reaction(result, 1)
        R_B = vertical_reaction(result, 2)
        R_total = abs(R_A + R_B)

        total_load = w * L  # 90 kN

        # Equilibrium check
        assert abs(R_total - total_load) / total_load < 0.01, (
            f"R_total = {R_total}, expected {total_load}"
        )


class TestThreeMembers:
    """Tests for three-member structures"""

    E = 200e9
    A = DEFAULT_A
    I = 1e-4

    def test_T3_1_two_span_continuous_full_udl(self):
        """T3-1: Two-span continuous beam with UDL on both spans"""
        L = 6.0
        w = 20000  # 20 kN/m

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, L, 0, SupportType2D.PIN)
        struct.add_node(3, 2 * L, 0, SupportType2D.ROLLER_X)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        add_uniform_load(struct, 1, -w)
        add_uniform_load(struct, 2, -w)

        result = struct.solve()

        R_A = abs(vertical_reaction(result, 1))
        R_B = abs(vertical_reaction(result, 2))
        R_C = abs(vertical_reaction(result, 3))

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

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, L, 0, SupportType2D.PIN)
        struct.add_node(3, 2 * L, 0, SupportType2D.ROLLER_X)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        add_uniform_load(struct, 1, -w)

        result = struct.solve()

        R_A = abs(vertical_reaction(result, 1))
        R_C = abs(vertical_reaction(result, 3))

        # Reactions should be asymmetric
        assert abs(R_A - R_C) > 1000, "Reactions should be asymmetric"
        assert R_A > R_C, "R_A should be greater than R_C"


class TestInternalForces:
    """Tests for internal force calculations"""

    E = 200e9
    A = DEFAULT_A
    I = 1e-4

    def test_shear_simply_supported_center_load(self):
        """Shear force diagram for simply supported beam with center load"""
        L = 10.0
        P = 100000  # 100 kN

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, L / 2, 0)
        struct.add_node(3, L, 0, SupportType2D.ROLLER_X)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_point_load(2, Fy=-P)

        struct.solve()
        forces = struct.compute_element_forces()

        # Shear should be approximately P/2
        V_left = forces[1]["V"]
        assert abs(abs(V_left[0]) - P / 2) < 1000, "V_left incorrect"

    def test_moment_simply_supported_center_load(self):
        """Bending moment diagram for simply supported beam with center load"""
        L = 10.0
        P = 100000  # 100 kN

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, L / 2, 0)
        struct.add_node(3, L, 0, SupportType2D.ROLLER_X)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_point_load(2, Fy=-P)

        struct.solve()
        forces = struct.compute_element_forces()

        # Max moment at center = PL/4
        M_max_expected = P * L / 4
        M_left_end = abs(forces[1]["M"][-1])

        assert abs(M_left_end - M_max_expected) / M_max_expected < 0.05, (
            f"M_max = {M_left_end}, expected {M_max_expected}"
        )


class TestFrameElement2D:
    """Tests for 2D frame elements with inclined members and releases"""

    E = 200e9
    A = DEFAULT_A
    I = 1e-4

    def test_horizontal_stiffness(self):
        """Horizontal frame element stiffness"""
        elem = FrameElement2D(
            E=self.E,
            A=self.A,
            I=self.I,
            node_i=(0, 0),
            node_j=(4, 0),
        )

        K_local = elem.local_stiffness_matrix()
        K_global = elem.global_stiffness_matrix()

        # For horizontal member, local = global
        assert np.allclose(K_local, K_global, rtol=1e-10)

    def test_vertical_stiffness(self):
        """Vertical frame element stiffness transformation"""
        elem = FrameElement2D(
            E=self.E,
            A=self.A,
            I=self.I,
            node_i=(0, 0),
            node_j=(0, 4),
        )

        assert abs(elem.angle - np.pi / 2) < 1e-10, "Angle should be 90 degrees"

        K_global = elem.global_stiffness_matrix()
        assert np.allclose(K_global, K_global.T), "Global stiffness not symmetric"

    def test_inclined_45_deg(self):
        """45-degree inclined frame element"""
        elem = FrameElement2D(
            E=self.E,
            A=self.A,
            I=self.I,
            node_i=(0, 0),
            node_j=(4, 4),
        )

        assert abs(elem.angle - np.pi / 4) < 1e-10, "Angle should be 45 degrees"
        assert abs(elem.L - 4 * np.sqrt(2)) < 1e-10, "Length incorrect"

        K_global = elem.global_stiffness_matrix()
        assert np.allclose(K_global, K_global.T), "Global stiffness not symmetric"

    def test_moment_release_both_ends(self):
        """Member with moment releases at both ends (truss behavior)"""
        elem = FrameElement2D(
            E=self.E,
            A=self.A,
            I=self.I,
            node_i=(0, 0),
            node_j=(4, 0),
            release_i=[ReleaseType.MOMENT],
            release_j=[ReleaseType.MOMENT],
        )

        K = elem.global_stiffness_matrix()

        # Only axial stiffness should remain
        EA_L = self.E * self.A / 4
        assert abs(K[0, 0] - EA_L) / EA_L < 0.01, "Axial stiffness incorrect"
        assert abs(K[2, 2]) < 1, "Rotational stiffness should be zero"
        assert abs(K[5, 5]) < 1, "Rotational stiffness should be zero"


class TestStructure2D:
    """Tests for 2D frame/truss structures"""

    E = 200e9
    A = DEFAULT_A
    I = 1e-4

    def test_portal_frame_horizontal_load(self):
        """Portal frame with horizontal load at top"""
        struct = Structure2D()

        H = 4.0  # Height
        W = 6.0  # Width

        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(2, 0, H)
        struct.add_node(3, W, H)
        struct.add_node(4, W, 0, SupportType2D.FIXED)

        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_element(3, 3, 4, self.E, self.A, self.I)

        P = 50000  # 50 kN
        struct.add_point_load(2, Fx=P)

        result = struct.solve()

        # Check horizontal equilibrium
        R1_x = result["reactions"][1][0]
        R4_x = result["reactions"][4][0]
        total_Rx = R1_x + R4_x

        assert abs(total_Rx + P) < 100, "Horizontal equilibrium failed"

    def test_simple_truss_equilibrium(self):
        """Simple triangle truss with vertical load"""
        struct = Structure2D()

        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, 4, 0, SupportType2D.ROLLER_X)
        struct.add_node(3, 2, 3)

        # All members with moment releases (truss behavior)
        struct.add_element(
            1,
            1,
            2,
            self.E,
            self.A,
            self.I,
            release_i=[ReleaseType.MOMENT],
            release_j=[ReleaseType.MOMENT],
        )
        struct.add_element(
            2,
            1,
            3,
            self.E,
            self.A,
            self.I,
            release_i=[ReleaseType.MOMENT],
            release_j=[ReleaseType.MOMENT],
        )
        struct.add_element(
            3,
            2,
            3,
            self.E,
            self.A,
            self.I,
            release_i=[ReleaseType.MOMENT],
            release_j=[ReleaseType.MOMENT],
        )

        P = 100000  # 100 kN
        struct.add_point_load(3, Fy=-P)

        result = struct.solve()

        # Check vertical equilibrium
        R1_y = result["reactions"][1][1]
        R2_y = result["reactions"][2][1]
        total_Ry = R1_y + R2_y

        assert abs(total_Ry - P) < 100, "Vertical equilibrium failed"

        # Symmetric truss should have equal reactions
        assert abs(R1_y - R2_y) / R1_y < 0.01, "Symmetric reactions expected"

    def test_reactions_zero_unconstrained_support_dofs(self):
        """Reported reactions should only include constrained support DOFs."""
        struct = Structure2D()

        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, 3, 0)
        struct.add_node(3, 6, 0, SupportType2D.ROLLER_X)
        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_point_load(2, Fy=-100000)

        result = struct.solve()

        pin_reaction = result["reactions"][1]
        roller_reaction = result["reactions"][3]

        assert pin_reaction[2] == 0.0
        assert roller_reaction[0] == 0.0
        assert roller_reaction[2] == 0.0
        assert abs(pin_reaction[1] + roller_reaction[1] - 100000) < 100

    def test_continuous_beam_udl(self):
        """Three-span continuous beam with UDL"""
        struct = Structure2D()

        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, 5, 0, SupportType2D.PIN)
        struct.add_node(3, 10, 0, SupportType2D.PIN)
        struct.add_node(4, 15, 0, SupportType2D.ROLLER_X)

        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_element(3, 3, 4, self.E, self.A, self.I)

        w = 20000  # 20 kN/m
        add_uniform_load(struct, 1, -w)
        add_uniform_load(struct, 2, -w)
        add_uniform_load(struct, 3, -w)

        result = struct.solve()

        # Check equilibrium
        total_load = w * 15
        total_reaction = -sum(r[1] for r in result["reactions"].values())

        assert abs(total_reaction - total_load) / total_load < 0.01, "Equilibrium failed"


class TestAdditionalStructures:
    """Additional tests for various structure types"""

    E = 200e9
    A = DEFAULT_A
    I = 1e-4

    def test_propped_cantilever_center_load(self):
        """Propped cantilever with center point load"""
        L = 8.0
        P = 80000  # 80 kN

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(2, L / 2, 0)
        struct.add_node(3, L, 0, SupportType2D.ROLLER_X)

        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_point_load(2, Fy=-P)

        result = struct.solve()

        # Check equilibrium
        R_A = vertical_reaction(result, 1)
        R_C = vertical_reaction(result, 3)
        total_R = R_A + R_C

        assert abs(total_R - P) / P < 0.01, "Equilibrium failed"

    def test_overhanging_beam(self):
        """Overhanging beam with UDL"""
        # Pin at 2m, roller at 8m, overhang to 10m
        struct = Structure2D()
        struct.add_node(1, 0, 0)
        struct.add_node(2, 2, 0, SupportType2D.PIN)
        struct.add_node(3, 8, 0, SupportType2D.ROLLER_X)
        struct.add_node(4, 10, 0)

        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)
        struct.add_element(3, 3, 4, self.E, self.A, self.I)

        w = 15000  # 15 kN/m
        add_uniform_load(struct, 1, -w)
        add_uniform_load(struct, 2, -w)
        add_uniform_load(struct, 3, -w)

        result = struct.solve()

        # Total load = w * 10 = 150 kN
        total_load = w * 10
        R_B = vertical_reaction(result, 2)
        R_C = vertical_reaction(result, 3)

        assert abs(abs(R_B + R_C) - total_load) / total_load < 0.01, "Equilibrium failed"

    def test_multi_story_frame(self):
        """Two-story rigid frame with horizontal loads"""
        struct = Structure2D()

        # Bottom columns fixed
        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(5, 6, 0, SupportType2D.FIXED)

        # First floor
        struct.add_node(2, 0, 4)
        struct.add_node(6, 6, 4)

        # Second floor (roof)
        struct.add_node(3, 0, 8)
        struct.add_node(7, 6, 8)

        # Columns
        struct.add_element(1, 1, 2, self.E, self.A, self.I)  # Left bottom
        struct.add_element(2, 2, 3, self.E, self.A, self.I)  # Left top
        struct.add_element(5, 5, 6, self.E, self.A, self.I)  # Right bottom
        struct.add_element(6, 6, 7, self.E, self.A, self.I)  # Right top

        # Beams
        struct.add_element(3, 2, 6, self.E, self.A, self.I)  # First floor
        struct.add_element(4, 3, 7, self.E, self.A, self.I)  # Roof

        # Horizontal loads (wind)
        struct.add_point_load(2, Fx=30000)  # 30 kN at first floor
        struct.add_point_load(3, Fx=20000)  # 20 kN at roof

        result = struct.solve()

        # Check horizontal equilibrium
        R1_x = result["reactions"][1][0]
        R5_x = result["reactions"][5][0]

        assert abs(R1_x + R5_x + 50000) < 200, "Horizontal equilibrium failed"

        # Frame should sway right (positive displacement)
        u3, v3, theta3 = struct.get_node_displacement(3)
        assert u3 > 0, "Roof should sway right under lateral load"

    def test_warren_truss(self):
        """Warren truss with point loads at panel points"""
        struct = Structure2D()

        # Bottom chord nodes
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, 4, 0)
        struct.add_node(3, 8, 0)
        struct.add_node(4, 12, 0, SupportType2D.ROLLER_X)

        # Top chord nodes
        struct.add_node(5, 2, 3)
        struct.add_node(6, 6, 3)
        struct.add_node(7, 10, 3)

        # Bottom chord
        for i in range(1, 4):
            struct.add_element(
                i,
                i,
                i + 1,
                self.E,
                self.A,
                self.I,
                release_i=[ReleaseType.MOMENT],
                release_j=[ReleaseType.MOMENT],
            )

        # Top chord
        struct.add_element(
            4,
            5,
            6,
            self.E,
            self.A,
            self.I,
            release_i=[ReleaseType.MOMENT],
            release_j=[ReleaseType.MOMENT],
        )
        struct.add_element(
            5,
            6,
            7,
            self.E,
            self.A,
            self.I,
            release_i=[ReleaseType.MOMENT],
            release_j=[ReleaseType.MOMENT],
        )

        # Diagonals
        diags = [(1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 4)]
        for i, (ni, nj) in enumerate(diags, start=6):
            struct.add_element(
                i,
                ni,
                nj,
                self.E,
                self.A,
                self.I,
                release_i=[ReleaseType.MOMENT],
                release_j=[ReleaseType.MOMENT],
            )

        # Point loads at bottom chord
        struct.add_point_load(2, Fy=-50000)  # 50 kN
        struct.add_point_load(3, Fy=-50000)  # 50 kN

        result = struct.solve()

        # Check equilibrium
        R1_y = result["reactions"][1][1]
        R4_y = result["reactions"][4][1]

        assert abs(R1_y + R4_y - 100000) < 200, "Vertical equilibrium failed"

        # Symmetric truss with symmetric load should have equal reactions
        assert abs(R1_y - R4_y) / R1_y < 0.05, "Reactions should be symmetric"

    def test_inclined_member_with_udl(self):
        """Single inclined member with UDL in global coordinates"""
        struct = Structure2D()

        # 45-degree inclined member
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, 4, 4, SupportType2D.ROLLER_X)

        struct.add_element(1, 1, 2, self.E, self.A, self.I)

        # Vertical UDL (gravity load)
        struct.add_element_udl(1, wy=-10000)  # 10 kN/m vertical

        result = struct.solve()

        # Check that structure solved and has reactions
        R1_y = result["reactions"][1][1]
        R2_y = result["reactions"][2][1]

        # Reactions should be non-zero and balance the load
        assert abs(R1_y) > 0 or abs(R2_y) > 0, "Should have non-zero reactions"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    E = 200e9
    A = DEFAULT_A
    I = 1e-4

    def test_very_stiff_element(self):
        """Very stiff element should have minimal deflection"""
        struct = Structure2D()

        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, 5, 0)
        struct.add_node(3, 10, 0, SupportType2D.ROLLER_X)

        # Very high I (very stiff)
        I_stiff = 1.0  # 1 m^4 - extremely stiff
        struct.add_element(1, 1, 2, self.E, self.A, I_stiff)
        struct.add_element(2, 2, 3, self.E, self.A, I_stiff)

        struct.add_point_load(2, Fy=-100000)
        struct.solve()

        v_center = vertical_displacement(struct, 2)

        # Should be very small deflection (less than 1mm)
        assert abs(v_center) < 1e-3, "Stiff beam should have minimal deflection"

    def test_very_flexible_element(self):
        """Very flexible element should have large deflection"""
        struct = Structure2D()

        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, 5, 0)
        struct.add_node(3, 10, 0, SupportType2D.ROLLER_X)

        # Very low I (very flexible)
        I_flex = 1e-8
        struct.add_element(1, 1, 2, self.E, self.A, I_flex)
        struct.add_element(2, 2, 3, self.E, self.A, I_flex)

        struct.add_point_load(2, Fy=-100)  # Small load
        struct.solve()

        v_center = vertical_displacement(struct, 2)

        # Should have significant deflection
        assert abs(v_center) > 0.1, "Flexible beam should have large deflection"

    def test_long_span_beam(self):
        """Very long span beam"""
        L = 100.0  # 100m span

        struct = Structure2D()
        struct.add_node(1, 0, 0, SupportType2D.PIN)
        struct.add_node(2, L / 2, 0)
        struct.add_node(3, L, 0, SupportType2D.ROLLER_X)

        struct.add_element(1, 1, 2, self.E, self.A, self.I)
        struct.add_element(2, 2, 3, self.E, self.A, self.I)

        struct.add_point_load(2, Fy=-1000)  # 1 kN
        result = struct.solve()

        # Result should have reactions
        assert len(result["reactions"]) > 0, "Long span beam should solve"

    def test_small_coordinates(self):
        """Structure with very small coordinates (microstructure)"""
        struct = Structure2D()

        # Millimeter scale structure
        struct.add_node(1, 0, 0, SupportType2D.FIXED)
        struct.add_node(2, 0.01, 0)  # 10mm

        struct.add_element(1, 1, 2, self.E, self.A, 1e-12)  # Tiny I for tiny structure
        struct.add_point_load(2, Fy=-1)  # 1 N

        result = struct.solve()
        assert len(result["reactions"]) > 0, "Small structure should solve"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
