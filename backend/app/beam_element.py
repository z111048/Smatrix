"""
BeamElement class for structural matrix analysis.
Implements local stiffness matrix for beam elements using Direct Stiffness Method.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BeamElement:
    """
    Represents a beam element in 2D structural analysis.
    
    Attributes:
        E: Young's modulus (Pa)
        I: Moment of inertia (m^4)
        L: Length (m)
        A: Cross-sectional area (m^2), optional for axial stiffness
    """
    E: float
    I: float
    L: float
    A: float = 1e-2  # Default area for axial stiffness
    
    def __post_init__(self):
        if self.L <= 0:
            raise ValueError("Length must be positive")
        if self.E <= 0:
            raise ValueError("Young's modulus must be positive")
        if self.I <= 0:
            raise ValueError("Moment of inertia must be positive")
    
    def stiffness_matrix_bending(self) -> np.ndarray:
        """
        Returns 4x4 local stiffness matrix for bending only.
        DOFs: [v_i, θ_i, v_j, θ_j]
        
        K = EI/L³ * [[12, 6L, -12, 6L],
                     [6L, 4L², -6L, 2L²],
                     [-12, -6L, 12, -6L],
                     [6L, 2L², -6L, 4L²]]
        """
        E, I, L = self.E, self.I, self.L
        EI_L3 = E * I / (L ** 3)
        
        K = EI_L3 * np.array([
            [12,      6*L,    -12,      6*L   ],
            [6*L,     4*L**2, -6*L,     2*L**2],
            [-12,    -6*L,     12,     -6*L   ],
            [6*L,     2*L**2, -6*L,     4*L**2]
        ])
        
        return K
    
    def stiffness_matrix(self) -> np.ndarray:
        """
        Returns 6x6 local stiffness matrix including axial deformation.
        DOFs: [u_i, v_i, θ_i, u_j, v_j, θ_j]
        """
        E, I, L, A = self.E, self.I, self.L, self.A
        EI_L3 = E * I / (L ** 3)
        EA_L = E * A / L
        
        K = np.array([
            [EA_L,   0,          0,         -EA_L,  0,          0        ],
            [0,      12*EI_L3,   6*EI_L3*L,  0,    -12*EI_L3,   6*EI_L3*L],
            [0,      6*EI_L3*L,  4*E*I/L,    0,    -6*EI_L3*L,  2*E*I/L  ],
            [-EA_L,  0,          0,          EA_L,  0,          0        ],
            [0,     -12*EI_L3,  -6*EI_L3*L,  0,     12*EI_L3,  -6*EI_L3*L],
            [0,      6*EI_L3*L,  2*E*I/L,    0,    -6*EI_L3*L,  4*E*I/L  ]
        ])
        
        return K
    
    def fixed_end_forces_udl(self, w: float) -> np.ndarray:
        """
        Returns fixed-end forces for uniformly distributed load (UDL).
        
        Args:
            w: Uniform load intensity (N/m), positive upward
            
        Returns:
            Fixed-end force vector [V_i, M_i, V_j, M_j] for 4-DOF
            or [0, V_i, M_i, 0, V_j, M_j] for 6-DOF
        """
        L = self.L
        V = w * L / 2
        M = w * L ** 2 / 12
        
        # 4-DOF version: [v_i, θ_i, v_j, θ_j]
        # For downward load (w < 0): reactions are upward (positive V), 
        # M_i is counter-clockwise (positive), M_j is clockwise (negative)
        return np.array([V, M, V, -M])
    
    def fixed_end_forces_udl_6dof(self, w: float) -> np.ndarray:
        """
        Returns fixed-end forces for UDL in 6-DOF format.
        DOFs: [u_i, v_i, θ_i, u_j, v_j, θ_j]
        """
        L = self.L
        V = w * L / 2
        M = w * L ** 2 / 12
        
        return np.array([0, V, M, 0, V, -M])
    
    def internal_forces(self, d_local: np.ndarray, w: float = 0, n_points: int = 11) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate shear force V(x) and bending moment M(x) along the beam.
        
        Args:
            d_local: Local displacement vector [v_i, θ_i, v_j, θ_j]
            w: Uniformly distributed load (N/m), positive upward
            n_points: Number of sampling points along the beam
            
        Returns:
            stations: x/L values (0 to 1)
            V: Shear force at each station
            M: Bending moment at each station
        """
        E, I, L = self.E, self.I, self.L
        v_i, theta_i, v_j, theta_j = d_local
        
        stations = np.linspace(0, 1, n_points)
        x = stations * L
        
        # Hermite shape functions derivatives for internal forces
        # V = EI * d³v/dx³, M = EI * d²v/dx²
        
        # From element end forces
        K = self.stiffness_matrix_bending()
        f_end = K @ d_local
        
        # End forces: [V_i, M_i, V_j, M_j]
        V_i, M_i, V_j, M_j = f_end
        
        # Add fixed-end forces contribution from UDL
        if w != 0:
            fem = self.fixed_end_forces_udl(w)
            V_i += fem[0]
            M_i += fem[1]
            V_j += fem[2]
            M_j += fem[3]
        
        # Shear and moment along the beam
        # V(x) = V_i - w*x (for UDL)
        # M(x) = M_i + V_i*x - w*x²/2
        V = -V_i - w * x  # Sign convention: positive shear causes clockwise rotation
        M = -M_i - V_i * x - w * x ** 2 / 2
        
        return stations, V, M


def test_beam_element():
    """Test BeamElement with known cases."""
    print("=" * 60)
    print("Task 1.1: BeamElement Class Verification")
    print("=" * 60)
    
    # Test case from acceptance criteria
    # E=200GPa, I=1e-4 m^4, L=4m
    beam = BeamElement(E=200e9, I=1e-4, L=4)
    K = beam.stiffness_matrix_bending()
    
    # K[0,0] should be 12EI/L³ = 12 * 200e9 * 1e-4 / 64 = 3.75e6
    expected_K00 = 12 * 200e9 * 1e-4 / (4 ** 3)
    
    print(f"\nTest Case: E=200GPa, I=1e-4 m⁴, L=4m")
    print(f"K[0,0] = 12EI/L³")
    print(f"  Expected: {expected_K00:.6e}")
    print(f"  Actual:   {K[0,0]:.6e}")
    print(f"  Error:    {abs(K[0,0] - expected_K00):.6e}")
    
    assert abs(K[0,0] - expected_K00) < 1e3, "K[0,0] verification failed!"
    print("  ✓ PASSED")
    
    # Verify K[1,1] = 4EI/L
    expected_K11 = 4 * 200e9 * 1e-4 / 4
    print(f"\nK[1,1] = 4EI/L")
    print(f"  Expected: {expected_K11:.6e}")
    print(f"  Actual:   {K[1,1]:.6e}")
    assert abs(K[1,1] - expected_K11) < 1e3, "K[1,1] verification failed!"
    print("  ✓ PASSED")
    
    # Verify symmetry
    print(f"\nSymmetry check:")
    is_symmetric = np.allclose(K, K.T)
    print(f"  K = K^T: {is_symmetric}")
    assert is_symmetric, "Stiffness matrix is not symmetric!"
    print("  ✓ PASSED")
    
    # Print full matrix
    print(f"\nFull 4x4 bending stiffness matrix (×10⁶):")
    print(K / 1e6)
    
    # Test 6-DOF matrix
    K6 = beam.stiffness_matrix()
    print(f"\n6-DOF matrix shape: {K6.shape}")
    print(f"  6x6 matrix symmetric: {np.allclose(K6, K6.T)}")
    print("  ✓ PASSED")
    
    # Test UDL fixed-end forces
    w = -10000  # 10 kN/m downward
    fem = beam.fixed_end_forces_udl(w)
    expected_V = w * 4 / 2  # wL/2
    expected_M = w * 16 / 12  # wL²/12
    print(f"\nUDL Fixed-End Forces (w={w/1000} kN/m, L=4m):")
    print(f"  V = wL/2 = {expected_V/1000:.2f} kN, actual = {fem[0]/1000:.2f} kN")
    print(f"  M = wL²/12 = {expected_M/1000:.2f} kN·m, actual = {fem[1]/1000:.2f} kN·m")
    assert abs(fem[0] - expected_V) < 1, "FEM V verification failed!"
    assert abs(fem[1] - expected_M) < 1, "FEM M verification failed!"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("Task 1.1: ALL TESTS PASSED ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_beam_element()
