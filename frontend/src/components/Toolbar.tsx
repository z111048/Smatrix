// Toolbar component

import React, { useRef, useState } from 'react';
import { createProjectDocument, parseProjectDocument, useStore } from '../store';
import type { EditorMode } from '../types';

const Toolbar: React.FC = () => {
  const {
    mode,
    nodes,
    elements,
    pointLoads,
    udls,
    elementPointLoads,
    past,
    future,
    setMode,
    clearAll,
    replaceProject,
    undo,
    redo
  } = useStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [projectMessage, setProjectMessage] = useState<string | null>(null);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const canUndo = past.length > 0;
  const canRedo = future.length > 0;
  const shortcutsPanelId = 'keyboard-shortcuts-panel';

  const tools: { mode: EditorMode; label: string; icon: string; hotkey: string }[] = [
    { mode: 'select', label: 'Select', icon: '↖', hotkey: 'V' },
    { mode: 'addNode', label: 'Add Node', icon: '○', hotkey: 'N' },
    { mode: 'addBeam', label: 'Add Beam', icon: '━', hotkey: 'B' },
    { mode: 'addPointLoad', label: 'Point Load', icon: '↓', hotkey: 'L' },
    { mode: 'addUDL', label: 'UDL', icon: '⇊', hotkey: 'U' },
    { mode: 'addElementPointLoad', label: 'Element Load', icon: '↧', hotkey: 'E' },
  ];

  const shortcutItems: { keys: string; action: string }[] = [
    { keys: 'V', action: 'Select mode' },
    { keys: 'N', action: 'Add node' },
    { keys: 'B', action: 'Add beam' },
    { keys: 'L', action: 'Point load' },
    { keys: 'U', action: 'UDL' },
    { keys: 'E', action: 'Element load' },
    { keys: 'Arrow keys', action: 'Move selected node 0.5 m' },
    { keys: 'Shift + Arrow', action: 'Move selected node 0.1 m' },
    { keys: 'Delete / Backspace', action: 'Delete selection' },
    { keys: 'Esc', action: 'Clear selection' },
    { keys: 'Ctrl + Z / Y', action: 'Undo / redo' },
  ];

  const setStatus = (message: string, isError = false) => {
    if (isError) {
      setProjectError(message);
      setProjectMessage(null);
    } else {
      setProjectMessage(message);
      setProjectError(null);
    }
  };

  const handleExport = () => {
    const project = createProjectDocument({
      nodes,
      elements,
      pointLoads,
      udls,
      elementPointLoads
    });
    const json = JSON.stringify(project, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const date = new Date().toISOString().slice(0, 10);

    link.href = url;
    link.download = `smatrix-project-${date}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setStatus('已匯出 Exported project JSON');
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';

    if (!file) return;

    try {
      const text = await file.text();
      const parsedProject: unknown = JSON.parse(text);
      const result = parseProjectDocument(parsedProject);

      if (!result.ok) {
        setStatus(`匯入失敗 Import failed: ${result.error}`, true);
        return;
      }

      replaceProject(result.project);
      setStatus('已匯入 Imported project JSON');
    } catch (error) {
      const message = error instanceof SyntaxError ? 'Invalid JSON file' : 'Could not read file';
      setStatus(`匯入失敗 Import failed: ${message}`, true);
    }
  };

  return (
    <div className="toolbar" role="toolbar" aria-label="Editor tools and project actions">
      <div className="tool-group">
        {tools.map(tool => (
          <button
            key={tool.mode}
            type="button"
            className={`tool-btn ${mode === tool.mode ? 'active' : ''}`}
            onClick={() => setMode(tool.mode)}
            title={`${tool.label} (${tool.hotkey})`}
            aria-label={`${tool.label} mode (${tool.hotkey})`}
            aria-pressed={mode === tool.mode}
          >
            <span className="icon">{tool.icon}</span>
            <span className="label">{tool.label}</span>
          </button>
        ))}
      </div>
      <div className="tool-group shortcuts-group">
        <button
          type="button"
          className="tool-btn"
          onClick={() => setShortcutsOpen(open => !open)}
          title="快捷鍵 Shortcuts"
          aria-label="Show keyboard shortcuts"
          aria-expanded={shortcutsOpen}
          aria-controls={shortcutsPanelId}
        >
          快捷鍵 Shortcuts
        </button>
        {shortcutsOpen && (
          <div
            id={shortcutsPanelId}
            className="shortcuts-panel"
            role="note"
            aria-label="Keyboard shortcuts"
          >
            <h2>快捷鍵 Shortcuts</h2>
            <dl>
              {shortcutItems.map(item => (
                <div key={`${item.keys}-${item.action}`} className="shortcut-row">
                  <dt>{item.keys}</dt>
                  <dd>{item.action}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </div>
      <div className="tool-group">
        <button
          type="button"
          className="tool-btn"
          onClick={undo}
          disabled={!canUndo}
          title="復原 Undo (Ctrl+Z)"
          aria-label="Undo (Ctrl+Z)"
        >
          復原 Undo
        </button>
        <button
          type="button"
          className="tool-btn"
          onClick={redo}
          disabled={!canRedo}
          title="重做 Redo (Ctrl+Y / Ctrl+Shift+Z)"
          aria-label="Redo (Ctrl+Y or Ctrl+Shift+Z)"
        >
          重做 Redo
        </button>
      </div>
      <div className="tool-group">
        <button
          type="button"
          className="tool-btn danger"
          onClick={clearAll}
          title="Clear All"
          aria-label="Clear all model data"
        >
          🗑️ Clear
        </button>
      </div>
      <div className="tool-group project-controls">
        <button
          type="button"
          className="tool-btn"
          onClick={handleExport}
          title="Export JSON"
          aria-label="Export project JSON"
        >
          匯出 Export
        </button>
        <button
          type="button"
          className="tool-btn"
          onClick={handleImportClick}
          title="Import JSON"
          aria-label="Import project JSON"
        >
          匯入 Import
        </button>
        <input
          ref={fileInputRef}
          className="project-file-input"
          type="file"
          accept="application/json,.json"
          onChange={handleImport}
          aria-label="Project JSON file"
        />
      </div>
      {(projectMessage || projectError) && (
        <div
          className={`project-status ${projectError ? 'error' : ''}`}
          role={projectError ? 'alert' : 'status'}
          aria-live="polite"
        >
          {projectError || projectMessage}
        </div>
      )}
    </div>
  );
};

export default Toolbar;
