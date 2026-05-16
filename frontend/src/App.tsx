import { useEffect, useState } from 'react';
import Canvas from './components/Canvas';
import ResultsCanvas from './components/ResultsCanvas';
import Toolbar from './components/Toolbar';
import Sidebar from './components/Sidebar';
import AnalysisPanel from './components/AnalysisPanel';
import './App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => window.matchMedia('(min-width: 992px)').matches);

  useEffect(() => {
    const media = window.matchMedia('(min-width: 992px)');
    const handleChange = () => setSidebarOpen(media.matches);

    media.addEventListener('change', handleChange);
    return () => media.removeEventListener('change', handleChange);
  }, []);

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
