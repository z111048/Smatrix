# Smatrix Backend

FastAPI backend and NumPy structural solver for Smatrix.

## Responsibilities

- Expose analysis and health-check API endpoints.
- Validate structural model input with Pydantic.
- Assemble and solve 2D frame stiffness systems.
- Return displacements, reactions, and element internal forces for frontend visualization.

## Commands

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
uv run pytest
```

Swagger UI is available at `http://localhost:8000/docs` while the server is running.

## API Endpoints

- `GET /health`: returns `{"status": "ok"}`.
- `POST /analyze`: solves a 2D structural model.

## Analyze Input Summary

- `nodes`: node id, coordinates, and support type.
- `elements`: node connectivity plus `E`, `A`, and `I`.
- `point_loads`: nodal `Fx`, `Fy`, and `Mz`.
- `udls`: element uniform distributed load `w`, mapped to global vertical load.
- `element_point_loads`: element load position `a` plus global `Fx` and `Fy`.

Support values accepted by the API are `free`, `roller`, `roller_x`, `roller_y`, `pin`, and `fixed`.

## Solver Notes

The active API uses `Structure2D` and `FrameElement2D`, with 3 DOF per node: horizontal displacement `u`, vertical displacement `v`, and rotation `theta`. Inclined members are transformed between local and global coordinates. Boundary conditions are applied with a penalty method.

## Tests

The backend test suite covers API behavior, error handling, structural cases, internal forces, and extended inputs such as `Fx`, `roller_y`, area `A`, and element point loads.

```bash
uv run pytest
```

Current status: 48 passing tests.
