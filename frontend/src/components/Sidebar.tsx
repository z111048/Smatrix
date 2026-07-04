// Properties sidebar component

import React, { useState } from 'react';
import { useStore } from '../store';
import type { Element, ElementPointLoad, Node, PointLoad, SupportType, UDL } from '../types';

type NumberValidator = (value: number) => string | null;

type ParsedNumber = {
  ok: true;
  value: number;
} | {
  ok: false;
  error: string;
};

const finiteNumber: NumberValidator = () => null;
const positiveNumber: NumberValidator = (value) => (
  value > 0 ? null : 'Must be > 0 / 需大於 0'
);
const nonNegativeNumber: NumberValidator = (value) => (
  value >= 0 ? null : 'Must be >= 0 / 需大於等於 0'
);

const parseValidatedNumber = (
  rawValue: string,
  validate: NumberValidator = finiteNumber
): ParsedNumber => {
  const trimmedValue = rawValue.trim();

  if (trimmedValue === '') {
    return { ok: false, error: 'Required / 必填' };
  }

  const value = Number(trimmedValue);
  if (!Number.isFinite(value)) {
    return { ok: false, error: 'Enter a number / 請輸入數字' };
  }

  const validationError = validate(value);
  if (validationError) {
    return { ok: false, error: validationError };
  }

  return { ok: true, value };
};

interface ValidatedNumberInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  validate?: NumberValidator;
  onBlur?: () => void;
  placeholder?: string;
}

const ValidatedNumberInput: React.FC<ValidatedNumberInputProps> = ({
  label,
  value,
  onChange,
  validate = finiteNumber,
  onBlur,
  placeholder
}) => {
  const result = parseValidatedNumber(value, validate);
  const error = result.ok ? null : result.error;

  return (
    <div className={`form-row ${error ? 'has-error' : ''}`}>
      <label>{label}</label>
      <div className="field-control">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onBlur={onBlur}
          placeholder={placeholder}
          inputMode="decimal"
          aria-invalid={error ? 'true' : 'false'}
        />
        {error && <div className="field-error">{error}</div>}
      </div>
    </div>
  );
};

const formatNumber = (value: number): string => {
  if (value === 0) return '0';

  const absValue = Math.abs(value);
  if (absValue >= 1000 || absValue < 0.001) {
    return value.toExponential(2);
  }

  return Number(value.toFixed(3)).toString();
};

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
  const [loadFx, setLoadFx] = useState(load ? ((load.Fx ?? 0) / 1000).toString() : '0');
  const [loadFy, setLoadFy] = useState(load ? (load.Fy / 1000).toString() : '0');
  const [loadMz, setLoadMz] = useState(load ? (load.Mz / 1000).toString() : '0');

  const handleNodeUpdate = () => {
    const x = parseValidatedNumber(nodeX);
    const y = parseValidatedNumber(nodeY);

    if (!x.ok || !y.ok) return;

    updateNode(node.id, {
      x: x.value,
      y: y.value,
      support: nodeSupport
    });
  };

  const handleLoadUpdate = () => {
    const parsedFx = parseValidatedNumber(loadFx);
    const parsedFy = parseValidatedNumber(loadFy);
    const parsedMz = parseValidatedNumber(loadMz);

    if (!parsedFx.ok || !parsedFy.ok || !parsedMz.ok) return;

    const fx = parsedFx.value * 1000;
    const fy = parsedFy.value * 1000;
    const mz = parsedMz.value * 1000;

    if (fx === 0 && fy === 0 && mz === 0) {
      deletePointLoad(node.id);
    } else {
      addPointLoad(node.id, fx, fy, mz);
    }
  };

  return (
    <div className="property-group">
      <h3>Node {node.id}</h3>

      <ValidatedNumberInput
        label="X (m):"
        value={nodeX}
        onChange={setNodeX}
        onBlur={handleNodeUpdate}
      />
      <ValidatedNumberInput
        label="Y (m):"
        value={nodeY}
        onChange={setNodeY}
        onBlur={handleNodeUpdate}
      />

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
      <ValidatedNumberInput
        label="Fx (kN):"
        value={loadFx}
        onChange={setLoadFx}
        onBlur={handleLoadUpdate}
        placeholder="→ positive"
      />
      <ValidatedNumberInput
        label="Fy (kN):"
        value={loadFy}
        onChange={setLoadFy}
        onBlur={handleLoadUpdate}
        placeholder="↓ negative"
      />
      <ValidatedNumberInput
        label="Mz (kN.m):"
        value={loadMz}
        onChange={setLoadMz}
        onBlur={handleLoadUpdate}
      />

      <button className="delete-btn" onClick={() => deleteNode(node.id)}>
        Delete Node / 刪除節點
      </button>
    </div>
  );
};

interface ElementEditorProps {
  element: Element;
  udls: UDL[];
  elementPointLoads: ElementPointLoad[];
  updateElement: (id: number, updates: Partial<Element>) => void;
  deleteElement: (id: number) => void;
  addUDL: (elementId: number, w1: number, w2?: number) => void;
  deleteUDL: (id: number) => void;
  addElementPointLoad: (elementId: number, a: number, Fx: number, Fy: number) => void;
  deleteElementPointLoad: (id: number) => void;
}

const ElementEditor: React.FC<ElementEditorProps> = ({
  element,
  udls,
  elementPointLoads,
  updateElement,
  deleteElement,
  addUDL,
  deleteUDL,
  addElementPointLoad,
  deleteElementPointLoad
}) => {
  const [elemE, setElemE] = useState((element.E / 1e9).toString());
  const [elemI, setElemI] = useState((element.I * 1e6).toString());
  const [elemA, setElemA] = useState((((element.A ?? 1e-2) / 1e-4)).toString());
  const [udlW1, setUdlW1] = useState('0');
  const [udlW2, setUdlW2] = useState('0');
  const [pointA, setPointA] = useState('0');
  const [pointFx, setPointFx] = useState('0');
  const [pointFy, setPointFy] = useState('0');

  const handleElementUpdate = () => {
    const e = parseValidatedNumber(elemE, positiveNumber);
    const i = parseValidatedNumber(elemI, nonNegativeNumber);
    const a = parseValidatedNumber(elemA, positiveNumber);

    if (!e.ok || !i.ok || !a.ok) return;

    updateElement(element.id, {
      E: e.value * 1e9,
      I: i.value * 1e-6,
      A: a.value * 1e-4
    });
  };

  const handleAddUdl = () => {
    const w1 = parseValidatedNumber(udlW1);
    const w2 = parseValidatedNumber(udlW2);
    if (!w1.ok || !w2.ok) return;

    addUDL(element.id, w1.value * 1000, w2.value * 1000);
    setUdlW1('0');
    setUdlW2('0');
  };

  const handleAddElementPointLoad = () => {
    const a = parseValidatedNumber(pointA);
    const fx = parseValidatedNumber(pointFx);
    const fy = parseValidatedNumber(pointFy);

    if (!a.ok || !fx.ok || !fy.ok) return;

    addElementPointLoad(element.id, a.value, fx.value * 1000, fy.value * 1000);
    setPointA('0');
    setPointFx('0');
    setPointFy('0');
  };

  return (
    <div className="property-group">
      <h3>Element {element.id}</h3>
      <p className="info">
        Nodes: {element.nodeI} {'->'} {element.nodeJ}
      </p>

      <ValidatedNumberInput
        label="E (GPa):"
        value={elemE}
        onChange={setElemE}
        validate={positiveNumber}
        onBlur={handleElementUpdate}
      />
      <ValidatedNumberInput
        label="I (x10^-6 m4):"
        value={elemI}
        onChange={setElemI}
        validate={nonNegativeNumber}
        onBlur={handleElementUpdate}
      />
      <ValidatedNumberInput
        label="A (x10^-4 m2):"
        value={elemA}
        onChange={setElemA}
        validate={positiveNumber}
        onBlur={handleElementUpdate}
      />

      <h4>UDL / 均佈載重</h4>
      <div className="load-list">
        {udls.length === 0 ? (
          <p className="load-empty">No UDLs / 尚無均佈載重</p>
        ) : udls.map((load, index) => (
          <div className="load-item" key={load.id}>
            <span>
              #{index + 1} w1: {formatNumber(load.w1 / 1000)} kN/m,
              w2: {formatNumber(load.w2 / 1000)} kN/m
            </span>
            <button type="button" onClick={() => deleteUDL(load.id)}>
              Delete / 刪除
            </button>
          </div>
        ))}
      </div>
      <ValidatedNumberInput
        label="起點 w1 (kN/m):"
        value={udlW1}
        onChange={setUdlW1}
        placeholder="↓ negative"
      />
      <ValidatedNumberInput
        label="終點 w2 (kN/m):"
        value={udlW2}
        onChange={setUdlW2}
        placeholder="↓ negative"
      />
      <button type="button" className="add-load-btn" onClick={handleAddUdl}>
        Add UDL / 新增均佈載重
      </button>

      <h4>Element Point Load / 桿件集中載重</h4>
      <div className="load-list">
        {elementPointLoads.length === 0 ? (
          <p className="load-empty">No element loads / 尚無桿件集中載重</p>
        ) : elementPointLoads.map((load, index) => (
          <div className="load-item" key={load.id}>
            <span>
              #{index + 1} a: {formatNumber(load.a)} m,
              Fx: {formatNumber(load.Fx / 1000)} kN,
              Fy: {formatNumber(load.Fy / 1000)} kN
            </span>
            <button type="button" onClick={() => deleteElementPointLoad(load.id)}>
              Delete / 刪除
            </button>
          </div>
        ))}
      </div>
      <ValidatedNumberInput
        label="a (m):"
        value={pointA}
        onChange={setPointA}
        placeholder="from node i"
      />
      <ValidatedNumberInput
        label="Fx (kN):"
        value={pointFx}
        onChange={setPointFx}
        placeholder="→ positive"
      />
      <ValidatedNumberInput
        label="Fy (kN):"
        value={pointFy}
        onChange={setPointFy}
        placeholder="↓ negative"
      />
      <button type="button" className="add-load-btn" onClick={handleAddElementPointLoad}>
        Add Element Load / 新增桿件載重
      </button>

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
  const selectedNodeLoad = selectedNode ? pointLoads.find(p => p.nodeId === selectedNode.id) : undefined;
  const selectedElementUdls = selectedElement ? udls.filter(u => u.elementId === selectedElement.id) : [];
  const selectedElementPointLoads = selectedElement
    ? elementPointLoads.filter(load => load.elementId === selectedElement.id)
    : [];
  const selectedNodeEditorKey = selectedNode
    ? [
      'node',
      selectedNode.id,
      selectedNode.x,
      selectedNode.y,
      selectedNode.support,
      selectedNodeLoad?.Fx ?? 'none',
      selectedNodeLoad?.Fy ?? 'none',
      selectedNodeLoad?.Mz ?? 'none'
    ].join('-')
    : undefined;
  const selectedElementEditorKey = selectedElement
    ? [
      'element',
      selectedElement.id,
      selectedElement.E,
      selectedElement.I,
      selectedElement.A ?? 'none',
      selectedElementUdls.map(load => `${load.id}:${load.w1}:${load.w2}`).join('|'),
      selectedElementPointLoads.map(load => `${load.id}:${load.a}:${load.Fx}:${load.Fy}`).join('|')
    ].join('-')
    : undefined;

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
          key={selectedNodeEditorKey}
          node={selectedNode}
          load={selectedNodeLoad}
          updateNode={updateNode}
          deleteNode={deleteNode}
          addPointLoad={addPointLoad}
          deletePointLoad={deletePointLoad}
        />
      )}

      {/* Element properties */}
      {selectedElement && (
        <ElementEditor
          key={selectedElementEditorKey}
          element={selectedElement}
          udls={selectedElementUdls}
          elementPointLoads={selectedElementPointLoads}
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
