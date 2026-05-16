# Repository Guidelines

## Project Structure & Module Organization

Smatrix is split into a FastAPI backend and a Vite React frontend. Backend source lives in `backend/app/`, with structural analysis logic in modules such as `structure.py`, `structure_2d.py`, `beam_element.py`, and `frame_element.py`; API schemas and routes are in `models.py` and `main.py`. Backend tests are in `backend/tests/`. Frontend source is under `frontend/src/`, with React components in `frontend/src/components/`, API helpers in `frontend/src/api/`, shared types in `frontend/src/types/`, and Zustand state in `frontend/src/store/`. Project specs and design notes live in `docs/`.

## Build, Test, and Development Commands

- `cd backend && uv sync`: install Python dependencies from `pyproject.toml` and `uv.lock`.
- `cd backend && uv run uvicorn app.main:app --reload --port 8000`: run the API server locally.
- `cd backend && uv run pytest`: run backend unit and API tests.
- `cd frontend && npm install`: install frontend dependencies from `package-lock.json`.
- `cd frontend && npm run dev`: start the Vite development server.
- `cd frontend && npm run build`: type-check and build the production frontend.
- `cd frontend && npm run lint`: run ESLint for TypeScript and React code.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and keep test functions named `test_*`. Prefer typed Pydantic models for API payloads and keep numerical routines deterministic and easy to verify. Frontend code uses TypeScript, React function components, PascalCase component filenames such as `AnalysisPanel.tsx`, and semicolons in TS/TSX files. Keep shared domain shapes in `frontend/src/types/` instead of duplicating ad hoc interfaces.

## Testing Guidelines

Backend tests use `pytest` and FastAPI `TestClient`. Add tests beside existing suites in `backend/tests/`, grouping related cases in classes like `TestAnalyzeEndpoint`. Cover successful analysis cases, invalid input, and numerical edge cases for new structural behavior. When changing API contracts, update both backend tests and frontend types.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries, often with a version prefix, for example `v0.3.2: Improve RWD layout and responsive canvas` or `Fix API and add comprehensive tests`. Keep commits focused on one behavior change. Pull requests should include a concise description, commands run, linked issues when applicable, and screenshots or screen recordings for UI changes.

## Security & Configuration Tips

Do not commit local environment files, generated `dist/` output, or dependency folders. Keep backend and frontend ports aligned with the documented local defaults: API on `8000`, Vite on `5173`.
