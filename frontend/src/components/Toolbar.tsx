// Toolbar component

import React from 'react';
import { useStore } from '../store';
import type { EditorMode } from '../types';

const Toolbar: React.FC = () => {
  const { mode, setMode, clearAll } = useStore();

  const tools: { mode: EditorMode; label: string; icon: string }[] = [
    { mode: 'select', label: 'Select', icon: '↖' },
    { mode: 'addNode', label: 'Add Node', icon: '○' },
    { mode: 'addBeam', label: 'Add Beam', icon: '━' },
    { mode: 'addPointLoad', label: 'Point Load', icon: '↓' },
    { mode: 'addUDL', label: 'UDL', icon: '⇊' },
  ];

  return (
    <div className="toolbar">
      <div className="tool-group">
        {tools.map(tool => (
          <button
            key={tool.mode}
            className={`tool-btn ${mode === tool.mode ? 'active' : ''}`}
            onClick={() => setMode(tool.mode)}
            title={tool.label}
          >
            <span className="icon">{tool.icon}</span>
            <span className="label">{tool.label}</span>
          </button>
        ))}
      </div>
      <div className="tool-group">
        <button className="tool-btn danger" onClick={clearAll} title="Clear All">
          🗑️ Clear
        </button>
      </div>
    </div>
  );
};

export default Toolbar;
