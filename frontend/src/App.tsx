import Canvas from './components/Canvas';
import ResultsCanvas from './components/ResultsCanvas';
import Toolbar from './components/Toolbar';
import Sidebar from './components/Sidebar';
import AnalysisPanel from './components/AnalysisPanel';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Smatrix - Structural Matrix Analysis</h1>
      </header>
      
      <Toolbar />
      
      <div className="main-content">
        <div className="canvas-container">
          <Canvas />
          <ResultsCanvas />
        </div>
        
        <div className="right-panel">
          <Sidebar />
          <AnalysisPanel />
        </div>
      </div>
    </div>
  );
}

export default App;
