// Properties sidebar component

import React, { useState } from 'react';
import { useStore } from '../store';
import type { Element, ElementPointLoad, Node, PointLoad, SupportType, UDL } from '../types';

interface NodeEditorProps {
  node: Node;
  load: PointLoad | undefined;
  updateNode: (id: number, updates: Partial<Node>) => void;
  deleteNode: (id: number) => void;
  addPointLoad: (nodeId: number, Fx: number, Fy: number, Mz?: number) => void;
  deletePointLoad: (nodeId: number) => void;
}

const NodeEditor: React.FC<NodeEditorProps> = ({
  node,
  load,
  updateNode,
  deleteNode,
  addPointLoad,
  deletePointLoad
}) => {
  const [nodeX, setNodeX] = useState(node.x.toString());
  const [nodeY, setNodeY] = useState(node.y.toString());
  const [nodeSupport, setNodeSupport] = useState<SupportType>(node.support);
  const [loadFx, setLoadFx] = useState(load ? ((load.Fx || 0) / 1000).toString() : '0');
  const [loadFy, setLoadFy] = useState(load ? (load.Fy / 1000).toString() : '0');
  const [loadMz, setLoadMz] = useState(load ? (load.Mz / 1000).toString() : '0');

  const handleNodeUpdate = () => {
    updateNode(node.id, {
      x: parseFloat(nodeX) || 0,
      y: parseFloat(nodeY) || 0,
      support: nodeSupport
    });
  };

  const handleLoadUpdate = () => {
    const fx = (parseFloat(loadFx) || 0) * 1000;
    const fy = (parseFloat(loadFy) || 0) * 1000;
    const mz = (parseFloat(loadMz) || 0) * 1000;
    if (fx === 0 && fy === 0 && mz === 0) {
      deletePointLoad(node.id);
    } else {
      addPointLoad(node.id, fx, fy, mz);
    }
  };

  return (
    <div className="property-group">
      <h3>Node {node.id}</h3>

      <div className="form-row">
        <label>X (m):</label>
        <input
          type="number"
          value={nodeX}
          onChange={(e) => setNodeX(e.target.value)}
          onBlur={handleNodeUpdate}
          step="0.5"
          inputMode="decimal"
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
          inputMode="decimal"
        />
      </div>

      <div className="form-row">
        <label>Support:</label>
        <select
          value={nodeSupport}
          onChange={(e) => {
            const support = e.target.value as SupportType;
            setNodeSupport(support);
            updateNode(node.id, { support });
          }}
        >
          <option value="free">Free</option>
          <option value="pin">Pin</option>
          <option value="roller">Roller</option>
          <option value="roller_x">Roller X (v=0)</option>
          <option value="roller_y">Roller Y (u=0)</option>
          <option value="fixed">Fixed</option>
        </select>
      </div>

      <h4>Point Load / 節點載重</h4>
      <div className="form-row">
        <label>Fx (kN):</label>
        <input
          type="number"
          value={loadFx}
          onChange={(e) => setLoadFx(e.target.value)}
          onBlur={handleLoadUpdate}
          placeholder="→ positive"
          inputMode="decimal"
        />
      </div>
      <div className="form-row">
        <label>Fy (kN):</label>
        <input
          type="number"
          value={loadFy}
          onChange={(e) => setLoadFy(e.target.value)}
          onBlur={handleLoadUpdate}
          placeholder="↓ negative"
          inputMode="decimal"
        />
      </div>
      <div className="form-row">
        <label>Mz (kN.m):</label>
        <input
          type="number"
          value={loadMz}
          onChange={(e) => setLoadMz(e.target.value)}
          onBlur={handleLoadUpdate}
          inputMode="decimal"
        />
      </div>

      <button className="delete-btn" onClick={() => deleteNode(node.id)}>
        Delete Node / 刪除節點
      </button>
    </div>
  );
};

interface ElementEditorProps {
  element: Element;
  udl: UDL | undefined;
  elementPointLoad: ElementPointLoad | undefined;
  updateElement: (id: number, updates: Partial<Element>) => void;
  deleteElement: (id: number) => void;
  addUDL: (elementId: number, w: number) => void;
  deleteUDL: (elementId: number) => void;
  addElementPointLoad: (elementId: number, a: number, Fx: number, Fy: number) => void;
  deleteElementPointLoad: (elementId: number) => void;
}

const ElementEditor: React.FC<ElementEditorProps> = ({
  element,
  udl,
  elementPointLoad,
  updateElement,
  deleteElement,
  addUDL,
  deleteUDL,
  addElementPointLoad,
  deleteElementPointLoad
}) => {
  const [elemE, setElemE] = useState((element.E / 1e9).toString());
  const [elemI, setElemI] = useState((element.I * 1e6).toString());
  const [elemA, setElemA] = useState((((element.A || 1e-2) / 1e-4)).toString());
  const [udlW, setUdlW] = useState(udl ? (udl.w / 1000).toString() : '0');
  const [pointA, setPointA] = useState(elementPointLoad ? elementPointLoad.a.toString() : '0');
  const [pointFx, setPointFx] = useState(elementPointLoad ? (elementPointLoad.Fx / 1000).toString() : '0');
  const [pointFy, setPointFy] = useState(elementPointLoad ? (elementPointLoad.Fy / 1000).toString() : '0');

  const handleElementUpdate = () => {
    updateElement(element.id, {
      E: (parseFloat(elemE) || 200) * 1e9,
      I: (parseFloat(elemI) || 100) * 1e-6,
      A: (parseFloat(elemA) || 100) * 1e-4
    });
  };

  const handleUdlUpdate = () => {
    const w = (parseFloat(udlW) || 0) * 1000;
    if (w === 0) {
      deleteUDL(element.id);
    } else {
      addUDL(element.id, w);
    }
  };

  const handleElementPointLoadUpdate = () => {
    const a = parseFloat(pointA) || 0;
    const fx = (parseFloat(pointFx) || 0) * 1000;
    const fy = (parseFloat(pointFy) || 0) * 1000;

    if (fx === 0 && fy === 0) {
      deleteElementPointLoad(element.id);
    } else {
      addElementPointLoad(element.id, a, fx, fy);
    }
  };

  return (
    <div className="property-group">
      <h3>Element {element.id}</h3>
      <p className="info">
        Nodes: {element.nodeI} {'->'} {element.nodeJ}
      </p>

      <div className="form-row">
        <label>E (GPa):</label>
        <input
          type="number"
          value={elemE}
          onChange={(e) => setElemE(e.target.value)}
          onBlur={handleElementUpdate}
          inputMode="decimal"
        />
      </div>

      <div className="form-row">
        <label>I (x10^-6 m4):</label>
        <input
          type="number"
          value={elemI}
          onChange={(e) => setElemI(e.target.value)}
          onBlur={handleElementUpdate}
          inputMode="decimal"
        />
      </div>

      <div className="form-row">
        <label>A (x10^-4 m2):</label>
        <input
          type="number"
          value={elemA}
          onChange={(e) => setElemA(e.target.value)}
          onBlur={handleElementUpdate}
          inputMode="decimal"
        />
      </div>

      <h4>UDL / 均佈載重</h4>
      <div className="form-row">
        <label>w (kN/m):</label>
        <input
          type="number"
          value={udlW}
          onChange={(e) => setUdlW(e.target.value)}
          onBlur={handleUdlUpdate}
          placeholder="↓ negative"
          inputMode="decimal"
        />
      </div>

      <h4>Element Point Load / 桿件集中載重</h4>
      <div className="form-row">
        <label>a (m):</label>
        <input
          type="number"
          value={pointA}
          onChange={(e) => setPointA(e.target.value)}
          onBlur={handleElementPointLoadUpdate}
          placeholder="from node i"
          inputMode="decimal"
        />
      </div>
      <div className="form-row">
        <label>Fx (kN):</label>
        <input
          type="number"
          value={pointFx}
          onChange={(e) => setPointFx(e.target.value)}
          onBlur={handleElementPointLoadUpdate}
          placeholder="→ positive"
          inputMode="decimal"
        />
      </div>
      <div className="form-row">
        <label>Fy (kN):</label>
        <input
          type="number"
          value={pointFy}
          onChange={(e) => setPointFy(e.target.value)}
          onBlur={handleElementPointLoadUpdate}
          placeholder="↓ negative"
          inputMode="decimal"
        />
      </div>

      <button className="delete-btn" onClick={() => deleteElement(element.id)}>
        Delete Element / 刪除桿件
      </button>
    </div>
  );
};

const Sidebar: React.FC = () => {
  const {
    nodes, elements, pointLoads, udls, elementPointLoads,
    selectedNodeId, selectedElementId, mode,
    updateNode, deleteNode,
    updateElement, deleteElement,
    addPointLoad, deletePointLoad,
    addUDL, deleteUDL,
    addElementPointLoad, deleteElementPointLoad
  } = useStore();

  const selectedNode = selectedNodeId !== null ? nodes.find(n => n.id === selectedNodeId) : null;
  const selectedElement = selectedElementId !== null ? elements.find(e => e.id === selectedElementId) : null;

  return (
    <div className="sidebar">
      <h2>Properties / 屬性</h2>

      {/* Mode indicator */}
      <div className="mode-indicator">
        Mode / 模式: <strong>{mode}</strong>
      </div>

      {/* Node properties */}
      {selectedNode && (
        <NodeEditor
          key={`node-${selectedNode.id}`}
          node={selectedNode}
          load={pointLoads.find(p => p.nodeId === selectedNode.id)}
          updateNode={updateNode}
          deleteNode={deleteNode}
          addPointLoad={addPointLoad}
          deletePointLoad={deletePointLoad}
        />
      )}

      {/* Element properties */}
      {selectedElement && (
        <ElementEditor
          key={`element-${selectedElement.id}`}
          element={selectedElement}
          udl={udls.find(u => u.elementId === selectedElement.id)}
          elementPointLoad={elementPointLoads.find(load => load.elementId === selectedElement.id)}
          updateElement={updateElement}
          deleteElement={deleteElement}
          addUDL={addUDL}
          deleteUDL={deleteUDL}
          addElementPointLoad={addElementPointLoad}
          deleteElementPointLoad={deleteElementPointLoad}
        />
      )}

      {/* No selection */}
      {!selectedNode && !selectedElement && (
        <div className="no-selection">
          <p>Select a node or element to view properties / 選取節點或桿件以編輯屬性</p>
          <ul>
            <li><strong>Add Node:</strong> Click canvas / 點擊畫布</li>
            <li><strong>Add Beam:</strong> Click two nodes / 點兩個節點</li>
            <li><strong>Loads:</strong> Select node or element / 選取節點或桿件</li>
            <li><strong>Select:</strong> Click node or beam / 點選模型</li>
          </ul>
        </div>
      )}

      {/* Structure summary */}
      <div className="summary">
        <h3>Structure Summary / 結構摘要</h3>
        <p>Nodes: {nodes.length}</p>
        <p>Elements: {elements.length}</p>
        <p>Point Loads: {pointLoads.length}</p>
        <p>UDLs: {udls.length}</p>
        <p>Element Point Loads: {elementPointLoads.length}</p>
      </div>
    </div>
  );
};

export default Sidebar;
