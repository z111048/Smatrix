// Analysis controls and results panel

import React from 'react';
import { useStore } from '../store';
import { analyzeStructure } from '../api';
import type { NodeDisplacement, NodeReaction, ViewMode } from '../types';

type ExportRow = Array<string | number>;

const displacementHeaders = ['Node', 'u (mm)', 'v (mm)', 'θ (mrad)'];
const reactionHeaders = ['Node', 'Fx (kN)', 'Fy (kN)', 'Mz (kN·m)'];

const formatDisplacementRows = (displacements: NodeDisplacement[]): ExportRow[] =>
  displacements.map(d => [
    d.node_id,
    ((d.u ?? 0) * 1000).toFixed(3),
    (d.v * 1000).toFixed(3),
    (d.theta * 1000).toFixed(3),
  ]);

const formatReactionRows = (reactions: NodeReaction[]): ExportRow[] =>
  reactions.map(r => [
    r.node_id,
    ((r.Fx ?? 0) / 1000).toFixed(2),
    (r.Fy / 1000).toFixed(2),
    (r.Mz / 1000).toFixed(2),
  ]);

const buildTsv = (headers: string[], rows: ExportRow[]) =>
  [headers, ...rows].map(row => row.join('\t')).join('\n');

const escapeCsvCell = (value: string | number) => {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

const buildCsv = (headers: string[], rows: ExportRow[]) =>
  [headers, ...rows]
    .map(row => row.map(escapeCsvCell).join(','))
    .join('\n');

const downloadCsv = (filename: string, headers: string[], rows: ExportRow[]) => {
  const blob = new Blob([buildCsv(headers, rows)], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

const AnalysisPanel: React.FC = () => {
  const {
    nodes, elements, pointLoads, udls, elementPointLoads,
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
      const analysisResult = await analyzeStructure(nodes, elements, pointLoads, udls, elementPointLoads);
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

  const handleCopyTable = async (headers: string[], rows: ExportRow[]) => {
    try {
      await navigator.clipboard.writeText(buildTsv(headers, rows));
    } catch {
      setError('複製失敗 / Copy failed');
    }
  };

  const displacementRows = result ? formatDisplacementRows(result.displacements) : [];
  const reactionRows = result ? formatReactionRows(result.reactions) : [];

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
              <div className="result-group-header">
                <h4>Displacements</h4>
                <div className="result-actions">
                  <button
                    type="button"
                    className="result-action-btn"
                    onClick={() => void handleCopyTable(displacementHeaders, displacementRows)}
                  >
                    複製 Copy
                  </button>
                  <button
                    type="button"
                    className="result-action-btn"
                    onClick={() => downloadCsv('smatrix-displacements.csv', displacementHeaders, displacementRows)}
                  >
                    匯出 CSV
                  </button>
                </div>
              </div>
              <table>
                <thead>
                  <tr>
                    {displacementHeaders.map(header => (
                      <th key={header}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {displacementRows.map(row => (
                    <tr key={row[0]}>
                      {row.map((cell, index) => (
                        <td key={`${row[0]}-${displacementHeaders[index]}`}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="result-group">
              <div className="result-group-header">
                <h4>Reactions</h4>
                <div className="result-actions">
                  <button
                    type="button"
                    className="result-action-btn"
                    onClick={() => void handleCopyTable(reactionHeaders, reactionRows)}
                  >
                    複製 Copy
                  </button>
                  <button
                    type="button"
                    className="result-action-btn"
                    onClick={() => downloadCsv('smatrix-reactions.csv', reactionHeaders, reactionRows)}
                  >
                    匯出 CSV
                  </button>
                </div>
              </div>
              <table>
                <thead>
                  <tr>
                    {reactionHeaders.map(header => (
                      <th key={header}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {reactionRows.map(row => (
                    <tr key={row[0]}>
                      {row.map((cell, index) => (
                        <td key={`${row[0]}-${reactionHeaders[index]}`}>{cell}</td>
                      ))}
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
