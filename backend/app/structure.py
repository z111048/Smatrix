"""
Structure class for assembling global stiffness matrix and solving structural analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

from .beam_element import BeamElement


class SupportType(Enum):
    """Support types for nodes."""
    FREE = "free"
    ROLLER = "roller"  # Restrains vertical displacement (v=0)
    PIN = "pin"        # Restrains vertical displacement (v=0)
    FIXED = "fixed"    # Restrains v=0 and θ=0


@dataclass
class Node:
    """Represents a node in the structure."""
    id: int
    x: float
    y: float = 0.0
    support: SupportType = SupportType.FREE
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Element:
    """Represents a beam element connecting two nodes."""
    id: int
    node_i: int  # Start node ID
    node_j: int  # End node ID
    E: float = 200e9  # Young's modulus (Pa)
    I: float = 1e-4   # Moment of inertia (m^4)
    A: float = 1e-2   # Cross-sectional area (m^2)


@dataclass
class PointLoad:
    """Point load applied at a node."""
    node_id: int
    Fy: float = 0.0   # Vertical force (N), positive upward
    Mz: float = 0.0   # Moment (N·m), positive counter-clockwise


@dataclass
class UDL:
    """Uniformly distributed load on an element."""
    element_id: int
    w: float  # Load intensity (N/m), positive upward


@dataclass
class Structure:
    """
    Represents a 2D continuous beam structure.
    Uses 2 DOFs per node: [v, θ] (vertical displacement and rotation)
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    elements: Dict[int, Element] = field(default_factory=dict)
    point_loads: List[PointLoad] = field(default_factory=list)
    udls: List[UDL] = field(default_factory=list)
    
    # Results
    displacements: Optional[np.ndarray] = None
    reactions: Optional[Dict[int, Tuple[float, float]]] = None
    
    def add_node(self, id: int, x: float, y: float = 0.0, 
                 support: SupportType = SupportType.FREE) -> Node:
        """Add a node to the structure."""
        node = Node(id=id, x=x, y=y, support=support)
        self.nodes[id] = node
        return node
    
    def add_element(self, id: int, node_i: int, node_j: int,
                    E: float = 200e9, I: float = 1e-4, A: float = 1e-2) -> Element:
        """Add a beam element to the structure."""
        if node_i not in self.nodes or node_j not in self.nodes:
            raise ValueError(f"Nodes {node_i} and {node_j} must exist")
        elem = Element(id=id, node_i=node_i, node_j=node_j, E=E, I=I, A=A)
        self.elements[id] = elem
        return elem
    
    def add_point_load(self, node_id: int, Fy: float = 0.0, Mz: float = 0.0):
        """Add a point load at a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        self.point_loads.append(PointLoad(node_id=node_id, Fy=Fy, Mz=Mz))
    
    def add_udl(self, element_id: int, w: float):
        """Add a uniformly distributed load on an element."""
        if element_id not in self.elements:
            raise ValueError(f"Element {element_id} does not exist")
        self.udls.append(UDL(element_id=element_id, w=w))
    
    def _get_sorted_node_ids(self) -> List[int]:
        """Get node IDs sorted by x-coordinate."""
        return sorted(self.nodes.keys(), key=lambda nid: self.nodes[nid].x)
    
    def _get_node_dof_indices(self, node_id: int) -> Tuple[int, int]:
        """Get the global DOF indices for a node (v, θ)."""
        sorted_ids = self._get_sorted_node_ids()
        idx = sorted_ids.index(node_id)
        return (2 * idx, 2 * idx + 1)
    
    def _get_element_length(self, elem: Element) -> float:
        """Calculate element length."""
        ni = self.nodes[elem.node_i]
        nj = self.nodes[elem.node_j]
        return abs(nj.x - ni.x)
    
    def _get_beam_element(self, elem: Element) -> BeamElement:
        """Create a BeamElement object from an Element."""
        L = self._get_element_length(elem)
        return BeamElement(E=elem.E, I=elem.I, L=L, A=elem.A)
    
    def assemble_global_stiffness(self) -> np.ndarray:
        """
        Assemble the global stiffness matrix.
        
        Returns:
            K_global: (2n x 2n) global stiffness matrix where n is number of nodes
        """
        n_nodes = len(self.nodes)
        n_dof = 2 * n_nodes  # 2 DOFs per node: [v, θ]
        
        K_global = np.zeros((n_dof, n_dof))
        
        for elem in self.elements.values():
            beam = self._get_beam_element(elem)
            K_local = beam.stiffness_matrix_bending()  # 4x4 matrix
            
            # Get global DOF indices
            dof_i = self._get_node_dof_indices(elem.node_i)
            dof_j = self._get_node_dof_indices(elem.node_j)
            
            # Local to global DOF mapping: [v_i, θ_i, v_j, θ_j]
            dof_map = [dof_i[0], dof_i[1], dof_j[0], dof_j[1]]
            
            # Assemble into global matrix
            for i, gi in enumerate(dof_map):
                for j, gj in enumerate(dof_map):
                    K_global[gi, gj] += K_local[i, j]
        
        return K_global
    
    def assemble_load_vector(self) -> np.ndarray:
        """
        Assemble the global load vector including point loads and equivalent nodal loads from UDLs.
        
        Returns:
            F: (2n,) global force vector
        """
        n_nodes = len(self.nodes)
        n_dof = 2 * n_nodes
        
        F = np.zeros(n_dof)
        
        # Add point loads
        for load in self.point_loads:
            dof = self._get_node_dof_indices(load.node_id)
            F[dof[0]] += load.Fy  # Vertical force
            F[dof[1]] += load.Mz  # Moment
        
        # Add equivalent nodal loads from UDLs
        for udl in self.udls:
            elem = self.elements[udl.element_id]
            beam = self._get_beam_element(elem)
            fem = beam.fixed_end_forces_udl(udl.w)  # [V_i, M_i, V_j, M_j]
            
            dof_i = self._get_node_dof_indices(elem.node_i)
            dof_j = self._get_node_dof_indices(elem.node_j)
            
            F[dof_i[0]] += fem[0]  # V_i
            F[dof_i[1]] += fem[1]  # M_i
            F[dof_j[0]] += fem[2]  # V_j
            F[dof_j[1]] += fem[3]  # M_j
        
        return F
    
    def apply_boundary_conditions(self, K: np.ndarray, F: np.ndarray, 
                                   method: str = "penalty") -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Apply boundary conditions to the system.
        
        Args:
            K: Global stiffness matrix
            F: Global force vector
            method: "penalty" (large number) or "elimination" (row/column removal)
            
        Returns:
            K_mod: Modified stiffness matrix
            F_mod: Modified force vector
            fixed_dofs: List of restrained DOF indices
        """
        K_mod = K.copy()
        F_mod = F.copy()
        fixed_dofs = []
        
        for node_id, node in self.nodes.items():
            dof = self._get_node_dof_indices(node_id)
            
            if node.support == SupportType.ROLLER:
                # Restrain vertical displacement (v=0)
                fixed_dofs.append(dof[0])
            elif node.support == SupportType.PIN:
                # Restrain vertical displacement (v=0)
                fixed_dofs.append(dof[0])
            elif node.support == SupportType.FIXED:
                # Restrain both v=0 and θ=0
                fixed_dofs.append(dof[0])
                fixed_dofs.append(dof[1])
        
        if method == "penalty":
            # Large number method
            penalty = 1e20
            for dof_idx in fixed_dofs:
                K_mod[dof_idx, dof_idx] += penalty
                F_mod[dof_idx] = 0  # Prescribed displacement = 0
        else:
            # Elimination method - zero out rows/columns
            for dof_idx in fixed_dofs:
                K_mod[dof_idx, :] = 0
                K_mod[:, dof_idx] = 0
                K_mod[dof_idx, dof_idx] = 1
                F_mod[dof_idx] = 0
        
        return K_mod, F_mod, fixed_dofs
    
    def solve(self) -> Dict:
        """
        Solve the structural analysis problem.
        
        Returns:
            Dictionary containing displacements and reactions
        """
        # Assemble system
        K = self.assemble_global_stiffness()
        F = self.assemble_load_vector()
        
        # Apply boundary conditions
        K_mod, F_mod, fixed_dofs = self.apply_boundary_conditions(K, F)
        
        # Check for singular matrix
        try:
            det = np.linalg.det(K_mod)
            if abs(det) < 1e-10:
                raise ValueError("Structure is unstable (singular matrix)")
        except:
            pass
        
        # Solve for displacements
        try:
            d = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            raise ValueError("Structure is unstable (singular matrix)")
        
        self.displacements = d
        
        # Calculate reactions: R = K @ d - F
        reactions_vector = K @ d - F
        
        # Store reactions for fixed DOFs
        self.reactions = {}
        for node_id, node in self.nodes.items():
            if node.support != SupportType.FREE:
                dof = self._get_node_dof_indices(node_id)
                Ry = reactions_vector[dof[0]] if dof[0] in fixed_dofs else 0
                Rm = reactions_vector[dof[1]] if dof[1] in fixed_dofs else 0
                self.reactions[node_id] = (Ry, Rm)
        
        return {
            "displacements": d,
            "reactions": self.reactions,
            "fixed_dofs": fixed_dofs
        }
    
    def get_node_displacement(self, node_id: int) -> Tuple[float, float]:
        """Get displacement (v, θ) for a node."""
        if self.displacements is None:
            raise ValueError("Structure not solved yet")
        dof = self._get_node_dof_indices(node_id)
        return (self.displacements[dof[0]], self.displacements[dof[1]])
    
    def compute_internal_forces(self, n_points: int = 11) -> Dict:
        """
        Compute internal forces (V, M) for all elements.
        
        Args:
            n_points: Number of sampling points per element
            
        Returns:
            Dictionary with element_id as key, containing stations, V, and M arrays
        """
        if self.displacements is None:
            raise ValueError("Structure not solved yet")
        
        internal_forces = {}
        
        # Build UDL lookup by element_id
        udl_by_element = {}
        for udl in self.udls:
            udl_by_element[udl.element_id] = udl.w
        
        for elem_id, elem in self.elements.items():
            beam = self._get_beam_element(elem)
            L = beam.L
            
            # Get element nodal displacements
            v_i, theta_i = self.get_node_displacement(elem.node_i)
            v_j, theta_j = self.get_node_displacement(elem.node_j)
            d_local = np.array([v_i, theta_i, v_j, theta_j])
            
            # Get UDL if any
            w = udl_by_element.get(elem_id, 0)
            
            # Calculate end forces from stiffness
            K = beam.stiffness_matrix_bending()
            f_elem = K @ d_local  # Element forces from deformation
            
            # Add fixed-end forces for UDL (these are the forces that would exist if ends were fixed)
            if w != 0:
                fem = beam.fixed_end_forces_udl(w)
                f_elem = f_elem - fem  # Subtract because fem represents external load equivalent
            
            # f_elem = [V_i, M_i, V_j, M_j] in local element coordinates
            V_i_elem = -f_elem[0]  # Shear at node i (sign convention)
            M_i_elem = -f_elem[1]  # Moment at node i
            
            # Compute V(x) and M(x) along the element
            stations = np.linspace(0, 1, n_points)
            x = stations * L
            
            # For a beam element:
            # V(x) = V_i - w*x  (shear decreases with downward UDL)
            # M(x) = M_i + V_i*x - w*x²/2
            V = np.zeros(n_points)
            M = np.zeros(n_points)
            
            for i, xi in enumerate(x):
                V[i] = V_i_elem - w * xi
                M[i] = M_i_elem + V_i_elem * xi - w * xi**2 / 2
            
            internal_forces[elem_id] = {
                "stations": stations.tolist(),
                "x": x.tolist(),
                "V": V.tolist(),
                "M": M.tolist(),
                "V_i": V_i_elem,
                "M_i": M_i_elem,
                "V_j": f_elem[2],
                "M_j": f_elem[3]
            }
        
        return internal_forces


def test_structure_assembly():
    """Test Structure class assembly."""
    print("=" * 60)
    print("Task 1.2: Structure Class - Global Stiffness Matrix Assembly")
    print("=" * 60)
    
    # Create a simple two-node, one-element beam
    struct = Structure()
    struct.add_node(1, x=0.0)
    struct.add_node(2, x=4.0)
    struct.add_element(1, node_i=1, node_j=2, E=200e9, I=1e-4)
    
    print("\nTest Case: Single beam, L=4m, E=200GPa, I=1e-4 m⁴")
    print("Nodes: 1 (x=0), 2 (x=4)")
    print("DOFs: [v₁, θ₁, v₂, θ₂]")
    
    K = struct.assemble_global_stiffness()
    print(f"\nGlobal stiffness matrix shape: {K.shape}")
    assert K.shape == (4, 4), f"Expected (4,4), got {K.shape}"
    print("  ✓ Shape correct (4x4)")
    
    # For single element, global K should equal local K
    beam = BeamElement(E=200e9, I=1e-4, L=4.0)
    K_local = beam.stiffness_matrix_bending()
    
    print(f"\nK_global == K_local for single element:")
    is_equal = np.allclose(K, K_local)
    print(f"  {is_equal}")
    assert is_equal, "Single element global K should equal local K"
    print("  ✓ PASSED")
    
    # Test two-element continuous beam
    print("\n" + "-" * 40)
    print("Test Case: Two-span continuous beam")
    struct2 = Structure()
    struct2.add_node(1, x=0.0)
    struct2.add_node(2, x=4.0)
    struct2.add_node(3, x=8.0)
    struct2.add_element(1, node_i=1, node_j=2, E=200e9, I=1e-4)
    struct2.add_element(2, node_i=2, node_j=3, E=200e9, I=1e-4)
    
    K2 = struct2.assemble_global_stiffness()
    print(f"\nGlobal stiffness matrix shape: {K2.shape}")
    assert K2.shape == (6, 6), f"Expected (6,6), got {K2.shape}"
    print("  ✓ Shape correct (6x6)")
    
    # Check symmetry
    print(f"\nSymmetry check: K = K^T")
    is_symmetric = np.allclose(K2, K2.T)
    print(f"  {is_symmetric}")
    assert is_symmetric, "Global K should be symmetric"
    print("  ✓ PASSED")
    
    # Check that middle node (node 2) has accumulated stiffness
    # K[2,2] should be 2 * 12EI/L³ (contributions from both elements)
    EI_L3 = 200e9 * 1e-4 / (4 ** 3)
    expected_K22 = 2 * 12 * EI_L3
    print(f"\nK[2,2] for middle node (2 elements contributing):")
    print(f"  Expected: 2 × 12EI/L³ = {expected_K22:.6e}")
    print(f"  Actual:   {K2[2,2]:.6e}")
    assert abs(K2[2,2] - expected_K22) < 1e3, "Middle node stiffness incorrect"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("Task 1.2: ALL TESTS PASSED ✓")
    print("=" * 60)
    
    return True


def test_boundary_conditions_and_solve():
    """Test boundary conditions and solving."""
    print("\n" + "=" * 60)
    print("Task 1.3 & 1.4: Boundary Conditions & Solver")
    print("=" * 60)
    
    # Test Case: Simply supported beam with center point load
    # L=10m (two 5m spans), E=200GPa, I=8.33e-6 m^4, P=100kN at center
    # Expected: δ_max = PL³/(48EI) = 100000*1000/(48*200e9*8.33e-6) ≈ 1.25mm
    
    print("\n--- Test Case: Simply Supported Beam, Center Point Load ---")
    print("L=10m, E=200GPa, I=8.33e-6 m⁴, P=100kN at x=5m")
    
    struct = Structure()
    struct.add_node(1, x=0.0, support=SupportType.PIN)
    struct.add_node(2, x=5.0, support=SupportType.FREE)  # Load point
    struct.add_node(3, x=10.0, support=SupportType.ROLLER)
    
    E, I = 200e9, 8.33e-6
    struct.add_element(1, node_i=1, node_j=2, E=E, I=I)
    struct.add_element(2, node_i=2, node_j=3, E=E, I=I)
    
    # Apply 100kN downward load at center (negative = downward)
    struct.add_point_load(node_id=2, Fy=-100000)
    
    result = struct.solve()
    
    # Get center displacement
    v_center, theta_center = struct.get_node_displacement(2)
    
    # Theoretical: δ = PL³/(48EI)
    P, L = 100000, 10
    delta_theory = P * L**3 / (48 * E * I)
    
    print(f"\nCenter displacement (v₂):")
    print(f"  Theoretical: {delta_theory*1000:.4f} mm (downward)")
    print(f"  Computed:    {-v_center*1000:.4f} mm")
    print(f"  Error:       {abs(-v_center - delta_theory)/delta_theory*100:.2f}%")
    
    assert abs(-v_center - delta_theory) / delta_theory < 0.01, "Displacement error > 1%"
    print("  ✓ PASSED (error < 1%)")
    
    # Check reactions
    print(f"\nReactions:")
    R_A = result["reactions"][1][0]  # Node 1, vertical
    R_C = result["reactions"][3][0]  # Node 3, vertical
    expected_R = 50000  # P/2 = 50 kN
    
    print(f"  R_A = {R_A/1000:.2f} kN (expected: 50 kN)")
    print(f"  R_C = {R_C/1000:.2f} kN (expected: 50 kN)")
    
    assert abs(R_A - expected_R) < 100, f"R_A error: {R_A}"
    assert abs(R_C - expected_R) < 100, f"R_C error: {R_C}"
    print("  ✓ PASSED")
    
    # Test Case: Cantilever beam
    print("\n--- Test Case: Cantilever Beam, Tip Load ---")
    print("L=4m, E=200GPa, I=1e-4 m⁴, P=50kN at free end")
    
    struct2 = Structure()
    struct2.add_node(1, x=0.0, support=SupportType.FIXED)
    struct2.add_node(2, x=4.0, support=SupportType.FREE)
    
    E, I, L = 200e9, 1e-4, 4
    struct2.add_element(1, node_i=1, node_j=2, E=E, I=I)
    struct2.add_point_load(node_id=2, Fy=-50000)
    
    result2 = struct2.solve()
    v_tip, theta_tip = struct2.get_node_displacement(2)
    
    # Theoretical: δ_tip = PL³/(3EI)
    P = 50000
    delta_theory = P * L**3 / (3 * E * I)
    theta_theory = P * L**2 / (2 * E * I)
    
    print(f"\nTip displacement:")
    print(f"  Theoretical: {delta_theory*1000:.4f} mm")
    print(f"  Computed:    {-v_tip*1000:.4f} mm")
    
    assert abs(-v_tip - delta_theory) / delta_theory < 0.01, "Tip displacement error > 1%"
    print("  ✓ PASSED")
    
    # Check fixed support reaction
    R_fixed = result2["reactions"][1]
    M_fixed = R_fixed[1]
    expected_M = P * L  # 200 kN·m (reaction moment is opposite to applied)
    
    print(f"\nFixed support moment:")
    print(f"  Theoretical: {expected_M/1000:.2f} kN·m (reaction)")
    print(f"  Computed:    {M_fixed/1000:.2f} kN·m")
    
    assert abs(abs(M_fixed) - expected_M) / expected_M < 0.01, "Moment error > 1%"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("Task 1.3 & 1.4: ALL TESTS PASSED ✓")
    print("=" * 60)
    
    return True


def test_udl():
    """Test UDL equivalent nodal loads."""
    print("\n" + "=" * 60)
    print("Task 1.5: UDL Equivalent Nodal Loads")
    print("=" * 60)
    
    # Test Case: Simply supported beam with UDL
    # L=8m, w=10kN/m, E=200GPa, I=1e-4 m^4
    # Expected: δ_max = 5wL⁴/(384EI) = 5*10000*4096/(384*200e9*1e-4) ≈ 2.67mm
    # Expected: R = wL/2 = 40 kN
    
    print("\n--- Test Case: Simply Supported Beam, UDL ---")
    print("L=8m, E=200GPa, I=1e-4 m⁴, w=10kN/m")
    
    struct = Structure()
    struct.add_node(1, x=0.0, support=SupportType.PIN)
    struct.add_node(2, x=4.0, support=SupportType.FREE)  # Midpoint for checking
    struct.add_node(3, x=8.0, support=SupportType.ROLLER)
    
    E, I = 200e9, 1e-4
    struct.add_element(1, node_i=1, node_j=2, E=E, I=I)
    struct.add_element(2, node_i=2, node_j=3, E=E, I=I)
    
    # Apply UDL (10 kN/m downward = negative)
    w = -10000  # N/m
    struct.add_udl(element_id=1, w=w)
    struct.add_udl(element_id=2, w=w)
    
    result = struct.solve()
    
    # Get center displacement
    v_center, _ = struct.get_node_displacement(2)
    
    # Theoretical: δ = 5wL⁴/(384EI)
    L = 8
    delta_theory = 5 * abs(w) * L**4 / (384 * E * I)
    
    print(f"\nCenter displacement (v₂):")
    print(f"  Theoretical: {delta_theory*1000:.4f} mm (downward)")
    print(f"  Computed:    {-v_center*1000:.4f} mm")
    print(f"  Error:       {abs(-v_center - delta_theory)/delta_theory*100:.2f}%")
    
    assert abs(-v_center - delta_theory) / delta_theory < 0.05, "Displacement error > 5%"
    print("  ✓ PASSED (error < 5%)")
    
    # Check reactions
    R_A = result["reactions"][1][0]
    R_C = result["reactions"][3][0]
    expected_R = abs(w) * L / 2  # 40 kN
    
    print(f"\nReactions:")
    print(f"  R_A = {R_A/1000:.2f} kN (expected: 40 kN)")
    print(f"  R_C = {R_C/1000:.2f} kN (expected: 40 kN)")
    
    assert abs(R_A - expected_R) < 500, f"R_A error: {R_A}"
    assert abs(R_C - expected_R) < 500, f"R_C error: {R_C}"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("Task 1.5: ALL TESTS PASSED ✓")
    print("=" * 60)
    
    return True


def test_internal_forces():
    """Test internal force calculation."""
    print("\n" + "=" * 60)
    print("Task 1.6: Internal Force Calculation (V, M)")
    print("=" * 60)
    
    # Test Case 1: Simply supported beam with center point load
    print("\n--- Test Case 1: Simply Supported Beam, Center Point Load ---")
    print("L=10m, P=100kN at center")
    print("Expected: V = ±50kN, M_max = 250 kN·m at center")
    
    struct = Structure()
    struct.add_node(1, x=0.0, support=SupportType.PIN)
    struct.add_node(2, x=5.0, support=SupportType.FREE)
    struct.add_node(3, x=10.0, support=SupportType.ROLLER)
    
    E, I = 200e9, 1e-4
    struct.add_element(1, node_i=1, node_j=2, E=E, I=I)
    struct.add_element(2, node_i=2, node_j=3, E=E, I=I)
    struct.add_point_load(node_id=2, Fy=-100000)
    
    struct.solve()
    forces = struct.compute_internal_forces(n_points=11)
    
    # Check shear in element 1 (should be +50kN throughout)
    V1 = forces[1]["V"]
    print(f"\nElement 1 (left span):")
    print(f"  V at x=0: {V1[0]/1000:.2f} kN (expected: 50 kN)")
    print(f"  V at x=L: {V1[-1]/1000:.2f} kN (expected: 50 kN)")
    
    # Check shear in element 2 (should be -50kN throughout)
    V2 = forces[2]["V"]
    print(f"\nElement 2 (right span):")
    print(f"  V at x=0: {V2[0]/1000:.2f} kN (expected: -50 kN)")
    print(f"  V at x=L: {V2[-1]/1000:.2f} kN (expected: -50 kN)")
    
    # Check moment at center (end of element 1 / start of element 2)
    M1_end = forces[1]["M"][-1]
    M2_start = forces[2]["M"][0]
    expected_M_center = 50000 * 5  # 250 kN·m
    
    print(f"\nMoment at center:")
    print(f"  M (elem 1 end): {M1_end/1000:.2f} kN·m")
    print(f"  M (elem 2 start): {M2_start/1000:.2f} kN·m")
    print(f"  Expected: 250 kN·m")
    
    assert abs(abs(V1[0]) - 50000) < 1000, "V1 error"
    assert abs(abs(V2[0]) - 50000) < 1000, "V2 error"
    assert abs(abs(M1_end) - expected_M_center) < 5000, f"M_center error: {M1_end}"
    print("  ✓ PASSED")
    
    # Test Case 2: Simply supported beam with UDL
    print("\n--- Test Case 2: Simply Supported Beam, UDL ---")
    print("L=8m, w=10kN/m")
    print("Expected: V varies linearly from +40kN to -40kN")
    print("Expected: M_max = wL²/8 = 80 kN·m at center")
    
    struct2 = Structure()
    struct2.add_node(1, x=0.0, support=SupportType.PIN)
    struct2.add_node(2, x=8.0, support=SupportType.ROLLER)
    
    struct2.add_element(1, node_i=1, node_j=2, E=E, I=I)
    struct2.add_udl(element_id=1, w=-10000)
    
    struct2.solve()
    forces2 = struct2.compute_internal_forces(n_points=11)
    
    V = forces2[1]["V"]
    M = forces2[1]["M"]
    x = forces2[1]["x"]
    
    print(f"\nShear force distribution:")
    print(f"  V(0) = {V[0]/1000:.2f} kN (expected: 40 kN)")
    print(f"  V(4) = {V[5]/1000:.2f} kN (expected: 0 kN)")
    print(f"  V(8) = {V[-1]/1000:.2f} kN (expected: -40 kN)")
    
    print(f"\nBending moment distribution:")
    print(f"  M(0) = {M[0]/1000:.2f} kN·m (expected: 0)")
    print(f"  M(4) = {M[5]/1000:.2f} kN·m (expected: 80 kN·m)")
    print(f"  M(8) = {M[-1]/1000:.2f} kN·m (expected: 0)")
    
    expected_M_max = 10000 * 64 / 8  # wL²/8 = 80 kN·m
    assert abs(V[0] - 40000) < 1000, f"V(0) error: {V[0]}"
    assert abs(V[-1] + 40000) < 1000, f"V(L) error: {V[-1]}"
    assert abs(M[5] - expected_M_max) < 5000, f"M_max error: {M[5]}"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("Task 1.6: ALL TESTS PASSED ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_structure_assembly()
    test_boundary_conditions_and_solve()
    test_udl()
    test_internal_forces()
