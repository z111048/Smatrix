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

  // Count unsupported nodes
  const freeNodes = nodes.filter(n => n.support === 'free');
  const hasSupport = nodes.length > 0 && freeNodes.length < nodes.length;

  const handleAnalyze = async () => {
    if (nodes.length < 2) {
      setError('需要至少 2 個節點 / Need at least 2 nodes');
      return;
    }
    
    if (elements.length < 1) {
      setError('需要至少 1 個桿件 / Need at least 1 element');
      return;
    }
    
    // Check if at least one node has a support
    if (!hasSupport) {
      const nodeIds = freeNodes.map(n => n.id).join(', ');
      setError(`所有節點都沒有支承 (節點 ${nodeIds}) / All nodes are free. Set support for at least one node.`);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const analysisResult = await analyzeStructure(nodes, elements, pointLoads, udls);
      setResult(analysisResult);
      setViewMode('deflection');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      // Translate common error messages
      let displayError = errorMessage;
      if (errorMessage.includes('singular') || errorMessage.includes('unstable')) {
        displayError = '結構不穩定 / Structure is unstable (check supports and connections)';
      } else if (errorMessage.includes('Length must be positive')) {
        displayError = '桿件長度必須為正值 / Element length must be positive';
      }
      setError(displayError);
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
      {/* Warning for unsupported nodes */}
      {nodes.length > 0 && !hasSupport && (
        <div className="warning-message">
          ⚠️ 所有節點都沒有支承 / No supports defined
          <br />
          <small>選取節點並設定支承類型 / Select a node and set support type</small>
        </div>
      )}
      
      {/* Warning for free nodes */}
      {hasSupport && freeNodes.length > 0 && freeNodes.length < nodes.length && (
        <div className="info-message">
          ℹ️ {freeNodes.length} 個自由節點 (橘色) / {freeNodes.length} free node(s) (orange)
        </div>
      )}

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
