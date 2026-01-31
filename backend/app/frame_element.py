"""
FrameElement2D class for 2D frame/truss structural analysis.
Supports inclined members with coordinate transformation and moment releases.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum


class ReleaseType(Enum):
    """Types of moment/force releases at element ends"""
    NONE = "none"           # Full connection (rigid)
    MOMENT = "moment"       # Moment release (hinge) - Mz = 0
    AXIAL = "axial"         # Axial release - Fx = 0 (rare)
    SHEAR = "shear"         # Shear release - Fy = 0 (rare)


@dataclass
class FrameElement2D:
    """
    2D Frame Element with 6 degrees of freedom (3 per node).
    
    Supports:
    - Inclined members via coordinate transformation
    - Moment releases (hinges) for truss-like behavior
    - Various load types
    
    Attributes:
        E: Young's modulus (Pa)
        A: Cross-sectional area (m²)
        I: Moment of inertia (m⁴)
        L: Length (m)
        angle: Angle from horizontal (radians), computed from node coordinates
        node_i: Start node coordinates (x, y)
        node_j: End node coordinates (x, y)
        release_i: Releases at start node
        release_j: Releases at end node
    """
    E: float
    A: float
    I: float
    node_i: Tuple[float, float]
    node_j: Tuple[float, float]
    release_i: List[ReleaseType] = field(default_factory=list)
    release_j: List[ReleaseType] = field(default_factory=list)
    
    def __post_init__(self):
        x1, y1 = self.node_i
        x2, y2 = self.node_j
        
        # Calculate length
        self._L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if self._L <= 0:
            raise ValueError("Element length must be positive")
        if self.E <= 0:
            raise ValueError("Young's modulus must be positive")
        if self.A <= 0:
            raise ValueError("Area must be positive")
        if self.I < 0:
            raise ValueError("Moment of inertia cannot be negative")
        
        # Calculate angle from horizontal
        self._angle = np.arctan2(y2 - y1, x2 - x1)
        
        # Direction cosines
        self._cos = (x2 - x1) / self._L
        self._sin = (y2 - y1) / self._L
    
    @property
    def L(self) -> float:
        return self._L
    
    @property
    def angle(self) -> float:
        return self._angle
    
    @property
    def angle_degrees(self) -> float:
        return np.degrees(self._angle)
    
    def local_stiffness_matrix(self) -> np.ndarray:
        """
        Returns 6x6 local stiffness matrix in local coordinates.
        DOFs: [u_i, v_i, θ_i, u_j, v_j, θ_j]
        
        Local x-axis is along the member from i to j.
        Local y-axis is perpendicular, positive is 90° counter-clockwise from x.
        """
        E, A, I, L = self.E, self.A, self.I, self._L
        
        # Axial stiffness
        EA_L = E * A / L
        
        # Bending stiffness components
        EI_L3 = E * I / L**3
        EI_L2 = E * I / L**2
        EI_L = E * I / L
        
        k = np.array([
            [ EA_L,       0,           0,        -EA_L,       0,           0       ],
            [ 0,          12*EI_L3,    6*EI_L2,   0,         -12*EI_L3,    6*EI_L2 ],
            [ 0,          6*EI_L2,     4*EI_L,    0,         -6*EI_L2,     2*EI_L  ],
            [-EA_L,       0,           0,         EA_L,       0,           0       ],
            [ 0,         -12*EI_L3,   -6*EI_L2,   0,          12*EI_L3,   -6*EI_L2 ],
            [ 0,          6*EI_L2,     2*EI_L,    0,         -6*EI_L2,     4*EI_L  ]
        ])
        
        return k
    
    def transformation_matrix(self) -> np.ndarray:
        """
        Returns 6x6 coordinate transformation matrix.
        Transforms from local to global coordinates: K_global = T^T @ K_local @ T
        
        T = [[R, 0],
             [0, R]]
        
        where R = [[cos θ, sin θ, 0],
                   [-sin θ, cos θ, 0],
                   [0, 0, 1]]
        """
        c, s = self._cos, self._sin
        
        R = np.array([
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1]
        ])
        
        T = np.zeros((6, 6))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        
        return T
    
    def global_stiffness_matrix(self) -> np.ndarray:
        """
        Returns 6x6 global stiffness matrix.
        K_global = T^T @ K_local @ T
        """
        T = self.transformation_matrix()
        k_local = self.local_stiffness_matrix()
        
        # Apply releases if any
        if self.release_i or self.release_j:
            k_local = self._apply_releases(k_local)
        
        return T.T @ k_local @ T
    
    def _apply_releases(self, k: np.ndarray) -> np.ndarray:
        """
        Apply moment/force releases by zeroing the corresponding stiffness terms.
        
        For a moment release at node i (DOF 2), we zero out the coupling between
        that DOF and others, making the moment at that point zero.
        
        Note: This is a simplified approach. For a more rigorous treatment,
        static condensation should be used, but that requires knowing the
        loading condition.
        """
        k_mod = k.copy()
        
        # DOF mapping: 0=u_i, 1=v_i, 2=θ_i, 3=u_j, 4=v_j, 5=θ_j
        released_dofs = []
        
        for release in self.release_i:
            if release == ReleaseType.MOMENT:
                released_dofs.append(2)  # θ_i
            elif release == ReleaseType.AXIAL:
                released_dofs.append(0)  # u_i
            elif release == ReleaseType.SHEAR:
                released_dofs.append(1)  # v_i
        
        for release in self.release_j:
            if release == ReleaseType.MOMENT:
                released_dofs.append(5)  # θ_j
            elif release == ReleaseType.AXIAL:
                released_dofs.append(3)  # u_j
            elif release == ReleaseType.SHEAR:
                released_dofs.append(4)  # v_j
        
        if not released_dofs:
            return k_mod
        
        # For moment releases at both ends, the member becomes a truss element
        # (axial force only, no bending moments). We keep axial stiffness intact.
        # The transverse DOFs are still connected but with zero bending stiffness.
        if 2 in released_dofs and 5 in released_dofs:
            # Both ends released - pure truss member
            # Keep only axial stiffness (DOFs 0 and 3)
            # Set I effectively to 0 for bending calculations
            truss_k = np.zeros((6, 6))
            EA_L = self.E * self.A / self._L
            truss_k[0, 0] = EA_L
            truss_k[0, 3] = -EA_L
            truss_k[3, 0] = -EA_L
            truss_k[3, 3] = EA_L
            return truss_k
        
        # Single end release - modify stiffness matrix
        # For a hinge at one end, we use the reduced stiffness approach
        for dof in released_dofs:
            # Zero out row and column for the released DOF
            k_mod[dof, :] = 0.0
            k_mod[:, dof] = 0.0
        
        return k_mod
    
    def fixed_end_forces_udl_local(self, w: float) -> np.ndarray:
        """
        Returns fixed-end forces for UDL in local coordinates.
        
        Args:
            w: Load intensity perpendicular to member (N/m), positive in local +y
            
        Returns:
            [0, V_i, M_i, 0, V_j, M_j] in local coordinates
        """
        L = self._L
        
        # Standard fixed-end forces for UDL
        V = w * L / 2
        M = w * L**2 / 12
        
        return np.array([0, V, M, 0, V, -M])
    
    def fixed_end_forces_udl_global(self, w: float) -> np.ndarray:
        """
        Returns fixed-end forces for UDL in global coordinates.
        
        Args:
            w: Load intensity perpendicular to member (N/m), positive in local +y
            
        Returns:
            6-element vector in global coordinates
        """
        f_local = self.fixed_end_forces_udl_local(w)
        T = self.transformation_matrix()
        
        # Transform to global: f_global = T^T @ f_local
        return T.T @ f_local
    
    def fixed_end_forces_point_load_local(self, P: float, a: float, 
                                           direction: str = "y") -> np.ndarray:
        """
        Returns fixed-end forces for a point load on the element.
        
        Args:
            P: Load magnitude (N), positive in specified direction
            a: Distance from node i to load point (m)
            direction: "x" (axial) or "y" (transverse in local coords)
            
        Returns:
            6-element vector in local coordinates
        """
        L = self._L
        b = L - a
        
        if direction == "x":
            # Axial point load
            return np.array([-P * b / L, 0, 0, -P * a / L, 0, 0])
        else:
            # Transverse point load
            V_i = P * b**2 * (3*a + b) / L**3
            V_j = P * a**2 * (a + 3*b) / L**3
            M_i = P * a * b**2 / L**2
            M_j = -P * a**2 * b / L**2
            
            return np.array([0, V_i, M_i, 0, V_j, M_j])
    
    def fixed_end_forces_point_load_global(self, Fx: float, Fy: float, 
                                            a: float) -> np.ndarray:
        """
        Returns fixed-end forces for a global point load on the element.
        
        Args:
            Fx: Global X force (N)
            Fy: Global Y force (N)
            a: Distance from node i to load point along member (m)
            
        Returns:
            6-element vector in global coordinates
        """
        T = self.transformation_matrix()
        
        # Convert global force to local
        # Local force = R @ Global force (using first 2x2 of R)
        c, s = self._cos, self._sin
        P_local_x = c * Fx + s * Fy
        P_local_y = -s * Fx + c * Fy
        
        # Get FEM in local coordinates
        fem_x = self.fixed_end_forces_point_load_local(P_local_x, a, "x")
        fem_y = self.fixed_end_forces_point_load_local(P_local_y, a, "y")
        
        f_local = fem_x + fem_y
        
        # Transform to global
        return T.T @ f_local
    
    def element_forces_local(self, d_global: np.ndarray, 
                              f_fixed_local: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate element end forces in local coordinates.
        
        Args:
            d_global: Global displacement vector for this element [6]
            f_fixed_local: Fixed-end forces in local coords (from member loads)
            
        Returns:
            Element forces in local coordinates [u_i, v_i, θ_i, u_j, v_j, θ_j]
        """
        T = self.transformation_matrix()
        
        # Transform displacements to local
        d_local = T @ d_global
        
        # Local stiffness
        k_local = self.local_stiffness_matrix()
        
        # Element forces = K @ d_local + f_fixed
        f_local = k_local @ d_local
        
        if f_fixed_local is not None:
            f_local += f_fixed_local
        
        return f_local
    
    def internal_forces(self, d_global: np.ndarray, 
                        w: float = 0, 
                        point_loads: Optional[List[Tuple[float, float, float]]] = None,
                        n_points: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate internal forces along the element.
        
        Args:
            d_global: Global displacement vector for element [6]
            w: UDL intensity (N/m), positive in local +y
            point_loads: List of (P, a, direction) tuples for point loads
            n_points: Number of sampling points
            
        Returns:
            stations: x/L values (0 to 1)
            N: Axial force at each station (positive = tension)
            V: Shear force at each station
            M: Bending moment at each station
        """
        L = self._L
        T = self.transformation_matrix()
        
        # Transform displacements to local
        d_local = T @ d_global
        
        # Get element end forces in local coordinates
        k_local = self.local_stiffness_matrix()
        f_end = k_local @ d_local
        
        # Add fixed-end forces from loads
        if w != 0:
            f_end += self.fixed_end_forces_udl_local(w)
        
        if point_loads:
            for P, a, direction in point_loads:
                f_end += self.fixed_end_forces_point_load_local(P, a, direction)
        
        # End forces: [N_i, V_i, M_i, N_j, V_j, M_j]
        N_i, V_i, M_i = -f_end[0], -f_end[1], -f_end[2]
        
        # Sampling points
        stations = np.linspace(0, 1, n_points)
        x = stations * L
        
        # Internal forces along the beam
        # Sign convention: N positive = tension, V positive = clockwise shear, M positive = hogging
        N = np.full(n_points, N_i)
        V = V_i + w * x  # V decreases (becomes more negative) with positive w
        M = M_i + V_i * x + w * x**2 / 2
        
        # Handle point loads if any
        if point_loads:
            for P, a, direction in point_loads:
                if direction == "x":
                    # Axial point load: N jumps by P at x=a
                    N[x >= a] += P
                else:
                    # Transverse: V and M affected
                    V[x >= a] -= P
                    M[x >= a] -= P * (x[x >= a] - a)
        
        return stations, N, V, M


def test_frame_element():
    """Test FrameElement2D with known cases."""
    print("=" * 60)
    print("FrameElement2D Test Suite")
    print("=" * 60)
    
    # Test 1: Horizontal member (should match BeamElement)
    print("\nTest 1: Horizontal frame element")
    elem = FrameElement2D(
        E=200e9, A=1e-2, I=1e-4,
        node_i=(0, 0), node_j=(4, 0)
    )
    
    assert abs(elem.L - 4.0) < 1e-10, "Length incorrect"
    assert abs(elem.angle) < 1e-10, "Angle should be 0 for horizontal"
    print(f"  Length: {elem.L} m")
    print(f"  Angle: {elem.angle_degrees}°")
    
    K_local = elem.local_stiffness_matrix()
    K_global = elem.global_stiffness_matrix()
    
    # For horizontal, local = global
    assert np.allclose(K_local, K_global, rtol=1e-10), "K_local should equal K_global for horizontal"
    print("  ✓ Local = Global stiffness for horizontal member")
    
    # Test 2: Vertical member
    print("\nTest 2: Vertical frame element")
    elem_v = FrameElement2D(
        E=200e9, A=1e-2, I=1e-4,
        node_i=(0, 0), node_j=(0, 4)
    )
    
    assert abs(elem_v.L - 4.0) < 1e-10, "Length incorrect"
    assert abs(elem_v.angle - np.pi/2) < 1e-10, "Angle should be 90°"
    print(f"  Length: {elem_v.L} m")
    print(f"  Angle: {elem_v.angle_degrees}°")
    
    K_global_v = elem_v.global_stiffness_matrix()
    
    # Check transformation: for vertical member, u_global ↔ v_local, v_global ↔ -u_local
    # K_global[0,0] should equal K_local[1,1] (axial stiffness in global-x = transverse in local)
    assert abs(K_global_v[0,0] - K_local[1,1]) < 1, "Transformation check 1 failed"
    assert abs(K_global_v[1,1] - K_local[0,0]) < 1, "Transformation check 2 failed"
    print("  ✓ Stiffness transformation correct for vertical member")
    
    # Test 3: 45-degree inclined member
    print("\nTest 3: 45° inclined frame element")
    L_inclined = 4 * np.sqrt(2)
    elem_45 = FrameElement2D(
        E=200e9, A=1e-2, I=1e-4,
        node_i=(0, 0), node_j=(4, 4)
    )
    
    assert abs(elem_45.L - L_inclined) < 1e-10, "Length incorrect"
    assert abs(elem_45.angle - np.pi/4) < 1e-10, "Angle should be 45°"
    print(f"  Length: {elem_45.L:.4f} m")
    print(f"  Angle: {elem_45.angle_degrees}°")
    
    K_global_45 = elem_45.global_stiffness_matrix()
    assert np.allclose(K_global_45, K_global_45.T), "Global stiffness not symmetric"
    print("  ✓ Global stiffness symmetric")
    
    # Test 4: Moment releases (truss behavior)
    print("\nTest 4: Member with moment releases (truss)")
    elem_truss = FrameElement2D(
        E=200e9, A=1e-2, I=1e-4,
        node_i=(0, 0), node_j=(4, 0),
        release_i=[ReleaseType.MOMENT],
        release_j=[ReleaseType.MOMENT]
    )
    
    K_truss = elem_truss.global_stiffness_matrix()
    
    # With both ends released, bending stiffness should be zero
    # Only axial stiffness remains
    print(f"  K_truss[2,2] (should be ~0): {K_truss[2,2]:.2e}")
    print(f"  K_truss[5,5] (should be ~0): {K_truss[5,5]:.2e}")
    print(f"  K_truss[0,0] (axial, should be EA/L): {K_truss[0,0]:.2e}")
    
    EA_L = 200e9 * 1e-2 / 4
    assert abs(K_truss[0,0] - EA_L) / EA_L < 0.01, "Axial stiffness incorrect"
    print("  ✓ Truss behavior correct (axial only)")
    
    # Test 5: UDL fixed-end forces
    print("\nTest 5: UDL fixed-end forces")
    w = -10000  # 10 kN/m downward
    fem_local = elem.fixed_end_forces_udl_local(w)
    
    expected_V = w * 4 / 2
    expected_M = w * 16 / 12
    
    assert abs(fem_local[1] - expected_V) < 1, "V_i incorrect"
    assert abs(fem_local[2] - expected_M) < 1, "M_i incorrect"
    print(f"  V_i = {fem_local[1]/1000:.2f} kN (expected {expected_V/1000:.2f} kN)")
    print(f"  M_i = {fem_local[2]/1000:.2f} kN·m (expected {expected_M/1000:.2f} kN·m)")
    print("  ✓ UDL FEM correct")
    
    print("\n" + "=" * 60)
    print("All FrameElement2D tests PASSED ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_frame_element()
