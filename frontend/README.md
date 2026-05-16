# Smatrix Frontend

React + TypeScript + Vite frontend for the Smatrix structural analysis app.

## Main Responsibilities

- Interactive canvas modeling with React-Konva.
- Node, element, support, material, and load editing.
- API calls to the FastAPI backend.
- Result display for displacement, reactions, SFD, BMD, and deflection view.
- Responsive mobile layout with bottom toolbar, drawer panel, pan, zoom, and reset controls.

## Source Layout

- `src/App.tsx`: top-level layout and responsive side panel.
- `src/components/Canvas.tsx`: model drawing, selection, pan/zoom, support/load visualization.
- `src/components/Sidebar.tsx`: node and element property editors.
- `src/components/AnalysisPanel.tsx`: analyze action, result tabs, result tables.
- `src/components/ResultsCanvas.tsx`: deflection, shear, and moment overlays.
- `src/store/index.ts`: Zustand model, load, result, and viewport state.
- `src/api/index.ts`: `/analyze` and `/health` client helpers.
- `src/types/index.ts`: shared frontend data types.

## Commands

```bash
npm install
npm run dev
npm run lint
npm run build
npm run preview
```

Use `VITE_API_URL` when the backend is not running on `http://localhost:8000`:

```bash
VITE_API_URL=http://localhost:8010 npm run dev
```

## Supported Inputs

- Supports: `free`, `pin`, `roller`, `roller_x`, `roller_y`, `fixed`.
- Nodal loads: `Fx`, `Fy`, `Mz`.
- Element properties: `E`, `I`, `A`.
- Element loads: UDL `w`, point load position `a` with `Fx` and `Fy`.

## Notes

The frontend stores values in SI units internally. Several form fields show scaled units for convenience, such as `kN`, `kN.m`, `GPa`, `x10^-6 m4`, and `x10^-4 m2`.
