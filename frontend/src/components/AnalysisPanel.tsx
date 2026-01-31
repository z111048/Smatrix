// Analysis controls and results panel

import React from 'react';
import { useStore } from '../store';
import { analyzeStructure } from '../api';
import type { ViewMode } from '../types';

const AnalysisPanel: React.FC = () => {
  const {
    nodes, elements, pointLoads, udls,
    result, isLoading, error,
    viewMode, setViewMode,
    setResult, setLoading, setError
  } = useStore();

  const handleAnalyze = async () => {
    if (nodes.length < 2 || elements.length < 1) {
      setError('Need at least 2 nodes and 1 element');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const analysisResult = await analyzeStructure(nodes, elements, pointLoads, udls);
      setResult(analysisResult);
      setViewMode('deflection');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const viewModes: { mode: ViewMode; label: string }[] = [
    { mode: 'structure', label: 'Structure' },
    { mode: 'deflection', label: 'Deflection' },
    { mode: 'sfd', label: 'Shear (SFD)' },
    { mode: 'bmd', label: 'Moment (BMD)' },
  ];

  return (
    <div className="analysis-panel">
      <div className="analyze-section">
        <button
          className="analyze-btn"
          onClick={handleAnalyze}
          disabled={isLoading}
        >
          {isLoading ? '⏳ Analyzing...' : '▶ Analyze'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {result && (
        <>
          <div className="view-tabs">
            {viewModes.map(vm => (
              <button
                key={vm.mode}
                className={`view-tab ${viewMode === vm.mode ? 'active' : ''}`}
                onClick={() => setViewMode(vm.mode)}
              >
                {vm.label}
              </button>
            ))}
          </div>

          <div className="results-section">
            <h3>Results</h3>
            
            <div className="result-group">
              <h4>Displacements</h4>
              <table>
                <thead>
                  <tr>
                    <th>Node</th>
                    <th>v (mm)</th>
                    <th>θ (mrad)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.displacements.map(d => (
                    <tr key={d.node_id}>
                      <td>{d.node_id}</td>
                      <td>{(d.v * 1000).toFixed(3)}</td>
                      <td>{(d.theta * 1000).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="result-group">
              <h4>Reactions</h4>
              <table>
                <thead>
                  <tr>
                    <th>Node</th>
                    <th>Fy (kN)</th>
                    <th>Mz (kN·m)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.reactions.map(r => (
                    <tr key={r.node_id}>
                      <td>{r.node_id}</td>
                      <td>{(r.Fy / 1000).toFixed(2)}</td>
                      <td>{(r.Mz / 1000).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default AnalysisPanel;
