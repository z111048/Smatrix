"""
Pydantic models for Smatrix API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SupportType(str, Enum):
    FREE = "free"
    ROLLER = "roller"      # Roller on ground (v=0, can move horizontally)
    ROLLER_X = "roller_x"  # Same as ROLLER
    ROLLER_Y = "roller_y"  # Roller on wall (u=0, can move vertically)
    PIN = "pin"            # Pin support (u=0, v=0)
    FIXED = "fixed"        # Fixed support (u=0, v=0, θ=0)


class NodeInput(BaseModel):
    """Node definition for API input."""
    id: int
    x: float
    y: float = 0.0
    support: SupportType = SupportType.FREE


class ElementInput(BaseModel):
    """Element definition for API input."""
    id: int
    node_i: int
    node_j: int
    E: float = Field(default=200e9, description="Young's modulus (Pa)")
    I: float = Field(default=1e-4, description="Moment of inertia (m^4)")
    A: float = Field(default=1e-2, description="Cross-sectional area (m^2)")


class PointLoadInput(BaseModel):
    """Point load at a node."""
    node_id: int
    Fx: float = Field(default=0.0, description="Horizontal force (N), positive rightward")
    Fy: float = Field(default=0.0, description="Vertical force (N), positive upward")
    Mz: float = Field(default=0.0, description="Moment (N·m), positive counter-clockwise")


class UDLInput(BaseModel):
    """Uniformly distributed load on an element."""
    element_id: int
    w: float = Field(description="Load intensity (N/m), positive upward")


class AnalysisRequest(BaseModel):
    """Request body for structural analysis."""
    nodes: List[NodeInput]
    elements: List[ElementInput]
    point_loads: List[PointLoadInput] = []
    udls: List[UDLInput] = []


class NodeDisplacement(BaseModel):
    """Displacement result for a node."""
    node_id: int
    u: float = Field(default=0.0, description="Horizontal displacement (m)")
    v: float = Field(description="Vertical displacement (m)")
    theta: float = Field(description="Rotation (rad)")


class NodeReaction(BaseModel):
    """Reaction force at a support."""
    node_id: int
    Fx: float = Field(default=0.0, description="Horizontal reaction (N)")
    Fy: float = Field(description="Vertical reaction (N)")
    Mz: float = Field(description="Moment reaction (N·m)")


class ElementInternalForces(BaseModel):
    """Internal forces for an element."""
    element_id: int
    stations: List[float] = Field(description="Normalized positions (0 to 1)")
    x: List[float] = Field(description="Absolute positions along element (m)")
    V: List[float] = Field(description="Shear force at each station (N)")
    M: List[float] = Field(description="Bending moment at each station (N·m)")


class AnalysisResponse(BaseModel):
    """Response from structural analysis."""
    success: bool = True
    displacements: List[NodeDisplacement]
    reactions: List[NodeReaction]
    internal_forces: List[ElementInternalForces]


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
