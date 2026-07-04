import { useEffect, useState } from 'react';
import Canvas from './components/Canvas';
import ResultsCanvas from './components/ResultsCanvas';
import Toolbar from './components/Toolbar';
import Sidebar from './components/Sidebar';
import AnalysisPanel from './components/AnalysisPanel';
import { useStore } from './store';
import type { EditorMode } from './types';
import './App.css';

const GRID_MOVE_STEP = 0.5;
const FINE_MOVE_STEP = 0.1;

const MODE_HOTKEYS: Record<string, EditorMode> = {
  v: 'select',
  n: 'addNode',
  b: 'addBeam',
  l: 'addPointLoad',
  u: 'addUDL',
  e: 'addElementPointLoad'
};

const isFormControlTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) return false;

  const tagName = target.tagName.toLowerCase();
  return tagName === 'input' || tagName === 'select' || tagName === 'textarea' || target.isContentEditable;
};

const getArrowDelta = (key: string, step: number): { dx: number; dy: number } | null => {
  switch (key) {
    case 'ArrowLeft':
      return { dx: -step, dy: 0 };
    case 'ArrowRight':
      return { dx: step, dy: 0 };
    case 'ArrowUp':
      return { dx: 0, dy: step };
    case 'ArrowDown':
      return { dx: 0, dy: -step };
    default:
      return null;
  }
};

const addCoordinateStep = (value: number, delta: number): number => (
  Number((value + delta).toFixed(10))
);

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => window.matchMedia('(min-width: 992px)').matches);
  const nodes = useStore(state => state.nodes);
  const selectedNodeId = useStore(state => state.selectedNodeId);
  const selectedElementId = useStore(state => state.selectedElementId);
  const moveNode = useStore(state => state.moveNode);
  const deleteNode = useStore(state => state.deleteNode);
  const deleteElement = useStore(state => state.deleteElement);
  const setMode = useStore(state => state.setMode);
  const setSelectedNode = useStore(state => state.setSelectedNode);
  const setSelectedElement = useStore(state => state.setSelectedElement);
  const undo = useStore(state => state.undo);
  const redo = useStore(state => state.redo);

  useEffect(() => {
    const media = window.matchMedia('(min-width: 992px)');
    const handleChange = () => setSidebarOpen(media.matches);

    media.addEventListener('change', handleChange);
    return () => media.removeEventListener('change', handleChange);
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isFormControlTarget(event.target)) {
        return;
      }

      const key = event.key.toLowerCase();

      if (event.ctrlKey && !event.altKey && !event.metaKey) {
        if (key === 'z' && event.shiftKey) {
          event.preventDefault();
          redo();
        } else if (key === 'z') {
          event.preventDefault();
          undo();
        } else if (key === 'y') {
          event.preventDefault();
          redo();
        }
        return;
      }

      if (event.ctrlKey || event.altKey || event.metaKey) {
        return;
      }

      const shortcutMode = MODE_HOTKEYS[key];
      if (shortcutMode) {
        event.preventDefault();
        setMode(shortcutMode);
        return;
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        setMode('select');
        setSelectedNode(null);
        setSelectedElement(null);
        return;
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        if (selectedNodeId !== null) {
          deleteNode(selectedNodeId);
        } else if (selectedElementId !== null) {
          deleteElement(selectedElementId);
        }
        return;
      }

      const step = event.shiftKey ? FINE_MOVE_STEP : GRID_MOVE_STEP;
      const delta = getArrowDelta(event.key, step);
      if (delta && selectedNodeId !== null) {
        const selectedNode = nodes.find(node => node.id === selectedNodeId);
        if (!selectedNode) return;

        event.preventDefault();
        moveNode(
          selectedNode.id,
          addCoordinateStep(selectedNode.x, delta.dx),
          addCoordinateStep(selectedNode.y, delta.dy)
        );
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    deleteElement,
    deleteNode,
    moveNode,
    nodes,
    redo,
    selectedElementId,
    selectedNodeId,
    setMode,
    setSelectedElement,
    setSelectedNode,
    undo
  ]);

  return (
    <div className="app">
      <header className="header">
        <h1>Smatrix</h1>
        <span className="header-subtitle">Structural Matrix Analysis</span>
        <button 
          className="sidebar-toggle mobile-only"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
        >
          {sidebarOpen ? 'X' : '☰'}
        </button>
      </header>
      
      <Toolbar />
      
      <div className="main-content">
        <div className="canvas-container">
          <Canvas />
          <ResultsCanvas />
        </div>
        
        {sidebarOpen && (
          <button
            className="panel-backdrop mobile-only"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close panel"
          />
        )}

        <div className={`right-panel ${sidebarOpen ? 'open' : 'closed'}`}>
          <button 
            className="sidebar-collapse desktop-only"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? '›' : '‹'}
          </button>
          {sidebarOpen && (
            <>
              <button
                className="drawer-close mobile-only"
                onClick={() => setSidebarOpen(false)}
                aria-label="Close properties panel"
              >
                Close / 關閉
              </button>
              <Sidebar />
              <AnalysisPanel />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
