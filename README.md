# Smatrix - Structural Matrix Analysis

A web-based 2D continuous beam structural analysis application using the Direct Stiffness Method.

## Features

- **Interactive Canvas**: Click to add nodes, connect them to create beams
- **Support Types**: Pin, Roller, Fixed
- **Load Types**: Point loads and Uniformly Distributed Loads (UDL)
- **Analysis Results**:
  - Nodal displacements and rotations
  - Support reactions
  - Shear Force Diagram (SFD)
  - Bending Moment Diagram (BMD)
  - Deflection shape visualization

## Tech Stack

### Backend
- Python 3.11+
- FastAPI
- NumPy (matrix operations)
- uv (package management)

### Frontend
- React 18 + TypeScript
- Vite (build tool)
- React-Konva (canvas rendering)
- Zustand (state management)

## Quick Start

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App will be available at `http://localhost:5173`

## API Endpoints

- `GET /health` - Health check
- `POST /analyze` - Analyze structure

### Example Request

```json
{
  "nodes": [
    {"id": 1, "x": 0, "support": "pin"},
    {"id": 2, "x": 5},
    {"id": 3, "x": 10, "support": "roller"}
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4},
    {"id": 2, "node_i": 2, "node_j": 3, "E": 200e9, "I": 1e-4}
  ],
  "point_loads": [
    {"node_id": 2, "Fy": -100000}
  ]
}
```

## Project Structure

```
Smatrix/
├── backend/
│   ├── app/
│   │   ├── beam_element.py  # Element stiffness matrix
│   │   ├── structure.py     # Global assembly & solver
│   │   ├── models.py        # Pydantic models
│   │   └── main.py          # FastAPI app
│   ├── Dockerfile
│   └── pyproject.toml
└── frontend/
    ├── src/
    │   ├── components/      # React components
    │   ├── store/           # Zustand store
    │   ├── api/             # API client
    │   └── types/           # TypeScript types
    └── package.json
```

## Engineering Notes

- Uses 2 DOFs per node: vertical displacement (v) and rotation (θ)
- Beam element stiffness matrix: 4x4 bending-only formulation
- Boundary conditions: Penalty method (large number)
- UDL: Converted to equivalent nodal loads (Fixed-End Forces)
- Deflection visualization: Hermite cubic interpolation

## License

MIT
