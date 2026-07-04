import { useEffect, useState } from 'react';
import Canvas from './components/Canvas';
import ResultsCanvas from './components/ResultsCanvas';
import Toolbar from './components/Toolbar';
import Sidebar from './components/Sidebar';
import AnalysisPanel from './components/AnalysisPanel';
import { useStore } from './store';
import './App.css';

const isFormControlTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) return false;

  const tagName = target.tagName.toLowerCase();
  return tagName === 'input' || tagName === 'select' || tagName === 'textarea' || target.isContentEditable;
};

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => window.matchMedia('(min-width: 992px)').matches);
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
      if (!event.ctrlKey || event.altKey || event.metaKey || isFormControlTarget(event.target)) {
        return;
      }

      const key = event.key.toLowerCase();
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
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [redo, undo]);

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
