# Smatrix - Structural Matrix Analysis

Smatrix is a 2D structural analysis web application based on the Direct Stiffness Method. It combines a FastAPI numerical backend with a React/TypeScript canvas frontend for building and analyzing beams, frames, and inclined members.

## Features

- Interactive modeling: add nodes, connect beam/frame elements, select and edit properties.
- Support types: `free`, `pin`, `roller`, `roller_x`, `roller_y`, and `fixed`.
- Element properties: Young's modulus `E`, moment of inertia `I`, and cross-sectional area `A`.
- Load types:
  - nodal `Fx`, `Fy`, and `Mz`
  - element uniform distributed load `UDL`
  - element point load with position `a`, `Fx`, and `Fy`
- Analysis output:
  - nodal horizontal/vertical displacement and rotation
  - support reactions `Fx`, `Fy`, and `Mz`
  - shear force diagram (SFD)
  - bending moment diagram (BMD)
  - amplified deflection view
- Mobile-friendly UI with bottom toolbar, drawer panel, canvas pan, zoom, and reset controls.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend API | FastAPI, Pydantic |
| Solver | Python, NumPy |
| Frontend | React, TypeScript, Vite |
| State | Zustand |
| Canvas | React-Konva |
| Tests | pytest, ESLint, TypeScript build |

## Quick Start

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

API docs are available at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

If the backend is not on port `8000`, set:

```bash
VITE_API_URL=http://localhost:8010 npm run dev
```

## API Example

`POST /analyze`

```json
{
  "nodes": [
    {"id": 1, "x": 0, "y": 0, "support": "fixed"},
    {"id": 2, "x": 4, "y": 0, "support": "roller_y"}
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "E": 200000000000, "A": 0.02, "I": 0.0001}
  ],
  "point_loads": [
    {"node_id": 2, "Fx": 10000, "Fy": -5000, "Mz": 1000}
  ],
  "udls": [
    {"element_id": 1, "w": -10000}
  ],
  "element_point_loads": [
    {"element_id": 1, "a": 2.0, "Fx": 0, "Fy": -10000}
  ]
}
```

## Development Checks

```bash
cd frontend && npm run lint
cd frontend && npm run build
cd backend && uv run pytest
```

Current backend test suite: 53 tests.

## Documentation

- `docs/SPEC_v0.2.0.md`: engineering specification and implementation status.
- `backend/README.md`: backend API and solver notes.
- `frontend/README.md`: frontend development notes.
- `AGENTS.md`: contributor guide for future coding agents and maintainers.
