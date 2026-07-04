"""
Tests for linearly varying distributed loads on 2D frame elements.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models import UDLInput
from app.structure_2d import Structure2D, SupportType


E = 200e9
A = 1e-2
I = 1e-4


def build_single_element_structure(
    length: float,
    support_i: SupportType,
    support_j: SupportType = SupportType.NONE,
) -> Structure2D:
    struct = Structure2D()
    struct.add_node(1, 0.0, 0.0, support_i)
    struct.add_node(2, length, 0.0, support_j)
    struct.add_element(1, 1, 2, E, A, I)
    return struct


class TestTrapezoidalDistributedLoads:
    def test_udl_input_accepts_legacy_w(self):
        """Legacy API payloads with w should expand to equal endpoint loads."""
        load = UDLInput(element_id=1, w=-10000.0)

        assert load.w1 == -10000.0
        assert load.w2 == -10000.0

    def test_simply_supported_full_triangular_load(self):
        """Triangular load w1=0, w2=w has closed-form reactions and Mmax."""
        length = 6.0
        w = 12000.0

        struct = build_single_element_structure(
            length,
            SupportType.PIN,
            SupportType.ROLLER_X,
        )
        struct.add_element_trapezoidal_load(1, w1=0.0, w2=-w)

        result = struct.solve()

        reaction_i = result["reactions"][1][1]
        reaction_j = result["reactions"][2][1]
        assert abs(abs(reaction_i) - w * length / 6) < 1e-6
        assert abs(abs(reaction_j) - w * length / 3) < 1e-6

        forces = struct.compute_element_forces(n_points=2001)[1]
        max_index = int(np.argmax(forces["M"]))
        max_x = forces["stations"][max_index] * length
        max_moment = forces["M"][max_index]

        assert abs(max_x - length / math.sqrt(3)) < length / 2000
        expected_max_moment = w * length**2 / (9 * math.sqrt(3))
        assert abs(max_moment - expected_max_moment) / expected_max_moment < 1e-4

    def test_cantilever_full_triangular_load(self):
        """Cantilever triangular load balances to total force and fixed-end moment."""
        length = 5.0
        w = 15000.0

        struct = build_single_element_structure(length, SupportType.FIXED)
        struct.add_element_trapezoidal_load(1, w1=0.0, w2=-w)

        result = struct.solve()
        fixed_reaction = result["reactions"][1]

        assert abs(abs(fixed_reaction[1]) - w * length / 2) < 1e-6
        assert abs(abs(fixed_reaction[2]) - w * length**2 / 3) < 1e-6

        forces = struct.compute_element_forces(n_points=101)[1]
        mid_x = length / 2
        mid_index = 50
        expected_mid_shear = w * length / 2 - w * mid_x**2 / (2 * length)

        assert abs(forces["V"][-1]) < 1e-6
        assert abs(forces["V"][mid_index] - expected_mid_shear) < 1e-6

    def test_trapezoid_equals_udl_plus_triangle_superposition(self):
        """A trapezoid should equal a UDL plus the remaining triangular load."""
        length = 8.0

        trapezoid = build_single_element_structure(
            length,
            SupportType.PIN,
            SupportType.ROLLER_X,
        )
        trapezoid.add_element_trapezoidal_load(1, w1=-10000.0, w2=-30000.0)
        trapezoid_result = trapezoid.solve()
        trapezoid_forces = trapezoid.compute_element_forces(n_points=101)[1]

        superposed = build_single_element_structure(
            length,
            SupportType.PIN,
            SupportType.ROLLER_X,
        )
        superposed.add_element_trapezoidal_load(1, w1=-10000.0, w2=-10000.0)
        superposed.add_element_trapezoidal_load(1, w1=0.0, w2=-20000.0)
        superposed_result = superposed.solve()
        superposed_forces = superposed.compute_element_forces(n_points=101)[1]

        np.testing.assert_allclose(
            trapezoid_result["displacements"][1],
            superposed_result["displacements"][1],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            trapezoid_result["displacements"][2],
            superposed_result["displacements"][2],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            trapezoid_result["reactions"][1],
            superposed_result["reactions"][1],
            rtol=1e-10,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            trapezoid_result["reactions"][2],
            superposed_result["reactions"][2],
            rtol=1e-10,
            atol=1e-8,
        )
        np.testing.assert_allclose(trapezoid_forces["V"], superposed_forces["V"], rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(trapezoid_forces["M"], superposed_forces["M"], rtol=1e-10, atol=1e-8)
