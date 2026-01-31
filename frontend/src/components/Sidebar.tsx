// Properties sidebar component

import React, { useState, useEffect } from 'react';
import { useStore } from '../store';
import type { SupportType } from '../types';

const Sidebar: React.FC = () => {
  const {
    nodes, elements, pointLoads, udls,
    selectedNodeId, selectedElementId, mode,
    updateNode, deleteNode,
    updateElement, deleteElement,
    addPointLoad, deletePointLoad,
    addUDL, deleteUDL
  } = useStore();

  const selectedNode = selectedNodeId !== null ? nodes.find(n => n.id === selectedNodeId) : null;
  const selectedElement = selectedElementId !== null ? elements.find(e => e.id === selectedElementId) : null;

  // Local state for editing
  const [nodeX, setNodeX] = useState('');
  const [nodeY, setNodeY] = useState('');
  const [nodeSupport, setNodeSupport] = useState<SupportType>('free');
  
  const [elemE, setElemE] = useState('');
  const [elemI, setElemI] = useState('');
  
  const [loadFy, setLoadFy] = useState('');
  const [loadMz, setLoadMz] = useState('');
  
  const [udlW, setUdlW] = useState('');

  // Update local state when selection changes
  useEffect(() => {
    if (selectedNode) {
      setNodeX(selectedNode.x.toString());
      setNodeY(selectedNode.y.toString());
      setNodeSupport(selectedNode.support);
      
      const load = pointLoads.find(p => p.nodeId === selectedNode.id);
      setLoadFy(load ? (load.Fy / 1000).toString() : '0');
      setLoadMz(load ? (load.Mz / 1000).toString() : '0');
    }
  }, [selectedNode, pointLoads]);

  useEffect(() => {
    if (selectedElement) {
      setElemE((selectedElement.E / 1e9).toString());
      setElemI((selectedElement.I * 1e6).toString());
      
      const udl = udls.find(u => u.elementId === selectedElement.id);
      setUdlW(udl ? (udl.w / 1000).toString() : '0');
    }
  }, [selectedElement, udls]);

  const handleNodeUpdate = () => {
    if (selectedNode) {
      updateNode(selectedNode.id, {
        x: parseFloat(nodeX) || 0,
        y: parseFloat(nodeY) || 0,
        support: nodeSupport
      });
    }
  };

  const handleElementUpdate = () => {
    if (selectedElement) {
      updateElement(selectedElement.id, {
        E: (parseFloat(elemE) || 200) * 1e9,
        I: (parseFloat(elemI) || 100) * 1e-6
      });
    }
  };

  const handleLoadUpdate = () => {
    if (selectedNode) {
      const fy = (parseFloat(loadFy) || 0) * 1000;
      const mz = (parseFloat(loadMz) || 0) * 1000;
      if (fy === 0 && mz === 0) {
        deletePointLoad(selectedNode.id);
      } else {
        addPointLoad(selectedNode.id, fy, mz);
      }
    }
  };

  const handleUdlUpdate = () => {
    if (selectedElement) {
      const w = (parseFloat(udlW) || 0) * 1000;
      if (w === 0) {
        deleteUDL(selectedElement.id);
      } else {
        addUDL(selectedElement.id, w);
      }
    }
  };

  return (
    <div className="sidebar">
      <h2>Properties</h2>
      
      {/* Mode indicator */}
      <div className="mode-indicator">
        Mode: <strong>{mode}</strong>
      </div>

      {/* Node properties */}
      {selectedNode && (
        <div className="property-group">
          <h3>Node {selectedNode.id}</h3>
          
          <div className="form-row">
            <label>X (m):</label>
            <input
              type="number"
              value={nodeX}
              onChange={(e) => setNodeX(e.target.value)}
              onBlur={handleNodeUpdate}
              step="0.5"
            />
          </div>
          
          <div className="form-row">
            <label>Y (m):</label>
            <input
              type="number"
              value={nodeY}
              onChange={(e) => setNodeY(e.target.value)}
              onBlur={handleNodeUpdate}
              step="0.5"
            />
          </div>
          
          <div className="form-row">
            <label>Support:</label>
            <select
              value={nodeSupport}
              onChange={(e) => {
                setNodeSupport(e.target.value as SupportType);
                updateNode(selectedNode.id, { support: e.target.value as SupportType });
              }}
            >
              <option value="free">Free</option>
              <option value="pin">Pin</option>
              <option value="roller">Roller</option>
              <option value="fixed">Fixed</option>
            </select>
          </div>
          
          <h4>Point Load</h4>
          <div className="form-row">
            <label>Fy (kN):</label>
            <input
              type="number"
              value={loadFy}
              onChange={(e) => setLoadFy(e.target.value)}
              onBlur={handleLoadUpdate}
              placeholder="↓ negative"
            />
          </div>
          <div className="form-row">
            <label>Mz (kN·m):</label>
            <input
              type="number"
              value={loadMz}
              onChange={(e) => setLoadMz(e.target.value)}
              onBlur={handleLoadUpdate}
            />
          </div>
          
          <button className="delete-btn" onClick={() => deleteNode(selectedNode.id)}>
            Delete Node
          </button>
        </div>
      )}

      {/* Element properties */}
      {selectedElement && (
        <div className="property-group">
          <h3>Element {selectedElement.id}</h3>
          <p className="info">
            Nodes: {selectedElement.nodeI} → {selectedElement.nodeJ}
          </p>
          
          <div className="form-row">
            <label>E (GPa):</label>
            <input
              type="number"
              value={elemE}
              onChange={(e) => setElemE(e.target.value)}
              onBlur={handleElementUpdate}
            />
          </div>
          
          <div className="form-row">
            <label>I (×10⁻⁶ m⁴):</label>
            <input
              type="number"
              value={elemI}
              onChange={(e) => setElemI(e.target.value)}
              onBlur={handleElementUpdate}
            />
          </div>
          
          <h4>Uniform Distributed Load</h4>
          <div className="form-row">
            <label>w (kN/m):</label>
            <input
              type="number"
              value={udlW}
              onChange={(e) => setUdlW(e.target.value)}
              onBlur={handleUdlUpdate}
              placeholder="↓ negative"
            />
          </div>
          
          <button className="delete-btn" onClick={() => deleteElement(selectedElement.id)}>
            Delete Element
          </button>
        </div>
      )}

      {/* No selection */}
      {!selectedNode && !selectedElement && (
        <div className="no-selection">
          <p>Select a node or element to view properties</p>
          <ul>
            <li><strong>Add Node:</strong> Click on canvas</li>
            <li><strong>Add Beam:</strong> Click two nodes</li>
            <li><strong>Select:</strong> Click node/beam</li>
          </ul>
        </div>
      )}

      {/* Structure summary */}
      <div className="summary">
        <h3>Structure Summary</h3>
        <p>Nodes: {nodes.length}</p>
        <p>Elements: {elements.length}</p>
        <p>Point Loads: {pointLoads.length}</p>
        <p>UDLs: {udls.length}</p>
      </div>
    </div>
  );
};

export default Sidebar;
