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
from .structure import Structure, SupportType


app = FastAPI(
    title="Smatrix API",
    description="Structural Matrix Analysis for 2D Continuous Beams",
    version="0.1.0"
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
        APISupportType.FREE: SupportType.FREE,
        APISupportType.ROLLER: SupportType.ROLLER,
        APISupportType.PIN: SupportType.PIN,
        APISupportType.FIXED: SupportType.FIXED,
    }
    return mapping[api_support]


@app.post("/analyze", response_model=AnalysisResponse, responses={400: {"model": ErrorResponse}})
def analyze_structure(request: AnalysisRequest):
    """
    Analyze a 2D continuous beam structure.
    
    Returns displacements, reactions, and internal forces (V, M).
    """
    try:
        # Build structure from request
        struct = Structure()
        
        # Add nodes
        for node in request.nodes:
            struct.add_node(
                id=node.id,
                x=node.x,
                y=node.y,
                support=map_support_type(node.support)
            )
        
        # Add elements
        for elem in request.elements:
            struct.add_element(
                id=elem.id,
                node_i=elem.node_i,
                node_j=elem.node_j,
                E=elem.E,
                I=elem.I
            )
        
        # Add point loads
        for load in request.point_loads:
            struct.add_point_load(
                node_id=load.node_id,
                Fy=load.Fy,
                Mz=load.Mz
            )
        
        # Add UDLs
        for udl in request.udls:
            struct.add_udl(
                element_id=udl.element_id,
                w=udl.w
            )
        
        # Solve
        result = struct.solve()
        
        # Compute internal forces
        internal_forces = struct.compute_internal_forces(n_points=21)
        
        # Build response
        displacements = []
        for node_id in struct.nodes.keys():
            v, theta = struct.get_node_displacement(node_id)
            displacements.append(NodeDisplacement(
                node_id=node_id,
                v=v,
                theta=theta
            ))
        
        reactions = []
        for node_id, (Fy, Mz) in result["reactions"].items():
            reactions.append(NodeReaction(
                node_id=node_id,
                Fy=Fy,
                Mz=Mz
            ))
        
        forces = []
        for elem_id, data in internal_forces.items():
            forces.append(ElementInternalForces(
                element_id=elem_id,
                stations=data["stations"],
                x=data["x"],
                V=data["V"],
                M=data["M"]
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
