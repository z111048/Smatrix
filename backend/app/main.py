"""
FastAPI main application for Smatrix - Structural Matrix Analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    AnalysisRequest, AnalysisResponse, ErrorResponse,
    NodeDisplacement, NodeReaction, ElementInternalForces,
    SupportType as APISupportType
)
from .structure_2d import Structure2D, SupportType


app = FastAPI(
    title="Smatrix API",
    description="Structural Matrix Analysis for 2D Beams and Frames",
    version="0.3.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


def map_support_type(api_support: APISupportType) -> SupportType:
    """Map API support type to internal support type."""
    mapping = {
        APISupportType.FREE: SupportType.NONE,
        APISupportType.ROLLER: SupportType.ROLLER_X,
        APISupportType.ROLLER_X: SupportType.ROLLER_X,
        APISupportType.ROLLER_Y: SupportType.ROLLER_Y,
        APISupportType.PIN: SupportType.PIN,
        APISupportType.FIXED: SupportType.FIXED,
    }
    return mapping[api_support]


@app.post("/analyze", response_model=AnalysisResponse, responses={400: {"model": ErrorResponse}})
def analyze_structure(request: AnalysisRequest):
    """
    Analyze a 2D beam/frame structure using Direct Stiffness Method.
    
    Supports:
    - Continuous beams (horizontal)
    - Portal frames
    - Inclined members
    - Various support types
    
    Returns displacements, reactions, and internal forces (V, M).
    """
    try:
        # Validate input
        if not request.nodes:
            raise ValueError("At least one node is required")
        if not request.elements:
            raise ValueError("At least one element is required")
        
        # Build structure using Structure2D for full 2D analysis
        struct = Structure2D()
        
        # Add nodes
        for node in request.nodes:
            struct.add_node(
                node_id=node.id,
                x=node.x,
                y=node.y,
                support=map_support_type(node.support)
            )
        
        # Add elements
        for elem in request.elements:
            struct.add_element(
                elem_id=elem.id,
                node_i_id=elem.node_i,
                node_j_id=elem.node_j,
                E=elem.E,
                A=getattr(elem, 'A', 1e-2),
                I=elem.I
            )
        
        # Add point loads
        for load in request.point_loads:
            struct.add_point_load(
                node_id=load.node_id,
                Fx=getattr(load, 'Fx', 0.0),
                Fy=load.Fy,
                Mz=load.Mz
            )
        
        # Add UDLs (as vertical loads in global Y)
        for udl in request.udls:
            struct.add_element_udl(
                element_id=udl.element_id,
                wx=0.0,
                wy=udl.w
            )
        
        # Solve
        result = struct.solve()
        
        # Compute internal forces
        internal_forces = struct.compute_element_forces()
        
        # Build response
        displacements = []
        for node_id, disp in result["displacements"].items():
            displacements.append(NodeDisplacement(
                node_id=node_id,
                u=disp[0],
                v=disp[1],
                theta=disp[2]
            ))
        
        reactions = []
        for node_id, react in result["reactions"].items():
            reactions.append(NodeReaction(
                node_id=node_id,
                Fx=react[0],
                Fy=react[1],
                Mz=react[2]
            ))
        
        # Build internal forces response
        forces = []
        for elem_id, data in internal_forces.items():
            elem = struct.elements[elem_id]
            node_i = struct.nodes[elem.node_i_id]
            node_j = struct.nodes[elem.node_j_id]
            L = ((node_j.x - node_i.x)**2 + (node_j.y - node_i.y)**2)**0.5
            
            # Convert stations to absolute x positions along element
            x_positions = [s * L for s in data["stations"].tolist()]
            
            forces.append(ElementInternalForces(
                element_id=elem_id,
                stations=data["stations"].tolist(),
                x=x_positions,
                V=data["V"].tolist(),
                M=data["M"].tolist()
            ))
        
        return AnalysisResponse(
            success=True,
            displacements=displacements,
            reactions=reactions,
            internal_forces=forces
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
