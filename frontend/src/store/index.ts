// Zustand store for Smatrix state management

import { create } from 'zustand';
import type { Node, Element, PointLoad, UDL, ElementPointLoad, AnalysisResult, EditorMode, ViewMode, SupportType } from '../types';

export const PROJECT_VERSION = 1;
export const PROJECT_STORAGE_KEY = `smatrix:project:v${PROJECT_VERSION}`;

const AUTOSAVE_DELAY_MS = 500;
const HISTORY_LIMIT = 50;
const SUPPORT_TYPES: SupportType[] = ['free', 'roller', 'roller_x', 'roller_y', 'pin', 'fixed'];

export interface ProjectModel {
  nodes: Node[];
  elements: Element[];
  pointLoads: PointLoad[];
  udls: UDL[];
  elementPointLoads: ElementPointLoad[];
}

export interface ProjectDocument extends ProjectModel {
  version: typeof PROJECT_VERSION;
}

type ModelSnapshot = ProjectModel;

type ValidationResult<T> = {
  ok: true;
  value: T;
} | {
  ok: false;
  error: string;
};

type ParsedUDL = Omit<UDL, 'id'> & { id?: number };
type ParsedElementPointLoad = Omit<ElementPointLoad, 'id'> & { id?: number };

export type ProjectValidationResult = {
  ok: true;
  project: ProjectDocument;
} | {
  ok: false;
  error: string;
};

interface StoreState {
  // Structure data
  nodes: Node[];
  elements: Element[];
  pointLoads: PointLoad[];
  udls: UDL[];
  elementPointLoads: ElementPointLoad[];
  
  // UI state
  mode: EditorMode;
  viewMode: ViewMode;
  selectedNodeId: number | null;
  selectedElementId: number | null;
  
  // Analysis results
  result: AnalysisResult | null;
  isLoading: boolean;
  error: string | null;
  
  // Canvas state
  scale: number;
  offsetX: number;
  offsetY: number;
  
  // Beam creation state
  beamStartNodeId: number | null;

  // ID counters
  nextNodeId: number;
  nextElementId: number;
  nextLoadId: number;

  // Model history
  past: ModelSnapshot[];
  future: ModelSnapshot[];
  
  // Default material properties
  defaultE: number;
  defaultI: number;
  defaultA: number;
  
  // Actions
  addNode: (x: number, y: number) => void;
  moveNode: (id: number, x: number, y: number) => void;
  updateNode: (id: number, updates: Partial<Node>) => void;
  deleteNode: (id: number) => void;
  
  addElement: (nodeI: number, nodeJ: number) => void;
  updateElement: (id: number, updates: Partial<Element>) => void;
  deleteElement: (id: number) => void;
  
  addPointLoad: (nodeId: number, Fx: number, Fy: number, Mz?: number) => void;
  updatePointLoad: (nodeId: number, updates: Partial<PointLoad>) => void;
  deletePointLoad: (nodeId: number) => void;
  
  addUDL: (elementId: number, w1: number, w2?: number) => void;
  updateUDL: (id: number, updates: Partial<UDL>) => void;
  deleteUDL: (id: number) => void;

  addElementPointLoad: (elementId: number, a: number, Fx: number, Fy: number) => void;
  updateElementPointLoad: (id: number, updates: Partial<ElementPointLoad>) => void;
  deleteElementPointLoad: (id: number) => void;
  
  setMode: (mode: EditorMode) => void;
  setViewMode: (mode: ViewMode) => void;
  setSelectedNode: (id: number | null) => void;
  setSelectedElement: (id: number | null) => void;
  setBeamStartNode: (id: number | null) => void;
  
  setResult: (result: AnalysisResult | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  setScale: (scale: number) => void;
  setOffset: (x: number, y: number) => void;
  panViewport: (dx: number, dy: number) => void;
  resetViewport: () => void;
  
  replaceProject: (project: ProjectModel) => void;
  clearAll: () => void;
  undo: () => void;
  redo: () => void;
}

const emptyProjectModel = (): ProjectModel => ({
  nodes: [],
  elements: [],
  pointLoads: [],
  udls: [],
  elementPointLoads: []
});

const createModelSnapshot = (model: ProjectModel): ModelSnapshot => ({
  nodes: model.nodes.map(({ id, x, y, support }) => ({ id, x, y, support })),
  elements: model.elements.map(({ id, nodeI, nodeJ, E, I, A, releaseI, releaseJ }) => {
    const element: Element = {
      id,
      nodeI,
      nodeJ,
      E,
      I,
      releaseI: releaseI ?? false,
      releaseJ: releaseJ ?? false
    };
    if (A !== undefined) {
      element.A = A;
    }
    return element;
  }),
  pointLoads: model.pointLoads.map(({ nodeId, Fx, Fy, Mz }) => {
    const load: PointLoad = { nodeId, Fy, Mz };
    if (Fx !== undefined) {
      load.Fx = Fx;
    }
    return load;
  }),
  udls: model.udls.map(({ id, elementId, w1, w2 }) => ({ id, elementId, w1, w2 })),
  elementPointLoads: model.elementPointLoads.map(({ id, elementId, a, Fx, Fy }) => ({
    id,
    elementId,
    a,
    Fx,
    Fy
  }))
});

export const createProjectDocument = (model: ProjectModel): ProjectDocument => ({
  version: PROJECT_VERSION,
  ...createModelSnapshot(model)
});

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

const isPositiveInteger = (value: unknown): value is number => (
  typeof value === 'number' && Number.isInteger(value) && value > 0
);

const isFiniteNumber = (value: unknown): value is number => (
  typeof value === 'number' && Number.isFinite(value)
);

const isSupportType = (value: unknown): value is SupportType => (
  typeof value === 'string' && SUPPORT_TYPES.includes(value as SupportType)
);

const getRequiredArray = (
  record: Record<string, unknown>,
  key: keyof ProjectModel
): ValidationResult<unknown[]> => {
  const value = record[key];
  if (!Array.isArray(value)) {
    return { ok: false, error: `${key} must be an array` };
  }
  return { ok: true, value };
};

const getRequiredNumber = (
  record: Record<string, unknown>,
  key: string,
  label: string
): ValidationResult<number> => {
  const value = record[key];
  if (!isFiniteNumber(value)) {
    return { ok: false, error: `${label}.${key} must be a finite number` };
  }
  return { ok: true, value };
};

const getRequiredId = (
  record: Record<string, unknown>,
  key: string,
  label: string
): ValidationResult<number> => {
  const value = record[key];
  if (!isPositiveInteger(value)) {
    return { ok: false, error: `${label}.${key} must be a positive integer` };
  }
  return { ok: true, value };
};

const getOptionalId = (
  record: Record<string, unknown>,
  key: string,
  label: string
): ValidationResult<number | undefined> => {
  const value = record[key];
  if (value === undefined) {
    return { ok: true, value: undefined };
  }

  if (!isPositiveInteger(value)) {
    return { ok: false, error: `${label}.${key} must be a positive integer` };
  }

  return { ok: true, value };
};

const getOptionalBoolean = (
  record: Record<string, unknown>,
  key: string,
  label: string
): ValidationResult<boolean> => {
  const value = record[key];
  if (value === undefined) {
    return { ok: true, value: false };
  }

  if (typeof value !== 'boolean') {
    return { ok: false, error: `${label}.${key} must be a boolean` };
  }

  return { ok: true, value };
};

const hasDuplicateIds = (ids: number[]): boolean => ids.length !== new Set(ids).size;

const validateNode = (value: unknown, index: number): ValidationResult<Node> => {
  const label = `nodes[${index}]`;
  if (!isRecord(value)) {
    return { ok: false, error: `${label} must be an object` };
  }

  const id = getRequiredId(value, 'id', label);
  if (!id.ok) return id;

  const x = getRequiredNumber(value, 'x', label);
  if (!x.ok) return x;

  const y = getRequiredNumber(value, 'y', label);
  if (!y.ok) return y;

  if (!isSupportType(value.support)) {
    return { ok: false, error: `${label}.support is not supported` };
  }

  return {
    ok: true,
    value: {
      id: id.value,
      x: x.value,
      y: y.value,
      support: value.support
    }
  };
};

const validateElement = (value: unknown, index: number): ValidationResult<Element> => {
  const label = `elements[${index}]`;
  if (!isRecord(value)) {
    return { ok: false, error: `${label} must be an object` };
  }

  const id = getRequiredId(value, 'id', label);
  if (!id.ok) return id;

  const nodeI = getRequiredId(value, 'nodeI', label);
  if (!nodeI.ok) return nodeI;

  const nodeJ = getRequiredId(value, 'nodeJ', label);
  if (!nodeJ.ok) return nodeJ;

  const E = getRequiredNumber(value, 'E', label);
  if (!E.ok) return E;
  if (E.value <= 0) {
    return { ok: false, error: `${label}.E must be greater than 0` };
  }

  const I = getRequiredNumber(value, 'I', label);
  if (!I.ok) return I;
  if (I.value < 0) {
    return { ok: false, error: `${label}.I must be greater than or equal to 0` };
  }

  if (value.A !== undefined && !isFiniteNumber(value.A)) {
    return { ok: false, error: `${label}.A must be a finite number` };
  }
  if (value.A !== undefined && value.A <= 0) {
    return { ok: false, error: `${label}.A must be greater than 0` };
  }

  const releaseI = getOptionalBoolean(value, 'releaseI', label);
  if (!releaseI.ok) return releaseI;

  const releaseJ = getOptionalBoolean(value, 'releaseJ', label);
  if (!releaseJ.ok) return releaseJ;

  const element: Element = {
    id: id.value,
    nodeI: nodeI.value,
    nodeJ: nodeJ.value,
    E: E.value,
    I: I.value,
    releaseI: releaseI.value,
    releaseJ: releaseJ.value
  };

  if (value.A !== undefined) {
    element.A = value.A;
  }

  return { ok: true, value: element };
};

const validatePointLoad = (value: unknown, index: number): ValidationResult<PointLoad> => {
  const label = `pointLoads[${index}]`;
  if (!isRecord(value)) {
    return { ok: false, error: `${label} must be an object` };
  }

  const nodeId = getRequiredId(value, 'nodeId', label);
  if (!nodeId.ok) return nodeId;

  const Fy = getRequiredNumber(value, 'Fy', label);
  if (!Fy.ok) return Fy;

  const Mz = getRequiredNumber(value, 'Mz', label);
  if (!Mz.ok) return Mz;

  if (value.Fx !== undefined && !isFiniteNumber(value.Fx)) {
    return { ok: false, error: `${label}.Fx must be a finite number` };
  }

  const load: PointLoad = {
    nodeId: nodeId.value,
    Fy: Fy.value,
    Mz: Mz.value
  };

  if (value.Fx !== undefined) {
    load.Fx = value.Fx;
  }

  return { ok: true, value: load };
};

const validateUDL = (value: unknown, index: number): ValidationResult<ParsedUDL> => {
  const label = `udls[${index}]`;
  if (!isRecord(value)) {
    return { ok: false, error: `${label} must be an object` };
  }

  const id = getOptionalId(value, 'id', label);
  if (!id.ok) return id;

  const elementId = getRequiredId(value, 'elementId', label);
  if (!elementId.ok) return elementId;

  const legacyW = value.w;
  const hasLegacyW = legacyW !== undefined;
  if (hasLegacyW && !isFiniteNumber(legacyW)) {
    return { ok: false, error: `${label}.w must be a finite number` };
  }

  const w1Value = value.w1 ?? legacyW;
  const w2Value = value.w2 ?? legacyW;

  if (!isFiniteNumber(w1Value)) {
    return { ok: false, error: `${label}.w1 must be a finite number` };
  }

  if (!isFiniteNumber(w2Value)) {
    return { ok: false, error: `${label}.w2 must be a finite number` };
  }

  return {
    ok: true,
    value: {
      ...(id.value !== undefined ? { id: id.value } : {}),
      elementId: elementId.value,
      w1: w1Value,
      w2: w2Value
    }
  };
};

const validateElementPointLoad = (
  value: unknown,
  index: number
): ValidationResult<ParsedElementPointLoad> => {
  const label = `elementPointLoads[${index}]`;
  if (!isRecord(value)) {
    return { ok: false, error: `${label} must be an object` };
  }

  const id = getOptionalId(value, 'id', label);
  if (!id.ok) return id;

  const elementId = getRequiredId(value, 'elementId', label);
  if (!elementId.ok) return elementId;

  const a = getRequiredNumber(value, 'a', label);
  if (!a.ok) return a;

  const Fx = getRequiredNumber(value, 'Fx', label);
  if (!Fx.ok) return Fx;

  const Fy = getRequiredNumber(value, 'Fy', label);
  if (!Fy.ok) return Fy;

  return {
    ok: true,
    value: {
      ...(id.value !== undefined ? { id: id.value } : {}),
      elementId: elementId.value,
      a: a.value,
      Fx: Fx.value,
      Fy: Fy.value
    }
  };
};

const validateItems = <T>(
  values: unknown[],
  validate: (value: unknown, index: number) => ValidationResult<T>
): ValidationResult<T[]> => {
  const items: T[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const result = validate(values[index], index);
    if (!result.ok) return result;
    items.push(result.value);
  }
  return { ok: true, value: items };
};

const materializeLoadIds = (
  udls: ParsedUDL[],
  elementPointLoads: ParsedElementPointLoad[]
): ValidationResult<{ udls: UDL[]; elementPointLoads: ElementPointLoad[] }> => {
  const existingLoadIds = [
    ...udls.map(load => load.id),
    ...elementPointLoads.map(load => load.id)
  ].filter((id): id is number => id !== undefined);

  if (hasDuplicateIds(existingLoadIds)) {
    return { ok: false, error: 'Load ids must be unique' };
  }

  const usedLoadIds = new Set(existingLoadIds);
  let nextLoadId = existingLoadIds.reduce((maxId, id) => Math.max(maxId, id), 0) + 1;
  const getNextLoadId = () => {
    while (usedLoadIds.has(nextLoadId)) {
      nextLoadId += 1;
    }

    const id = nextLoadId;
    usedLoadIds.add(id);
    nextLoadId += 1;
    return id;
  };

  return {
    ok: true,
    value: {
      udls: udls.map(load => ({
        id: load.id ?? getNextLoadId(),
        elementId: load.elementId,
        w1: load.w1,
        w2: load.w2
      })),
      elementPointLoads: elementPointLoads.map(load => ({
        id: load.id ?? getNextLoadId(),
        elementId: load.elementId,
        a: load.a,
        Fx: load.Fx,
        Fy: load.Fy
      }))
    }
  };
};

const deriveNextIds = (nodes: Node[], elements: Element[], udls: UDL[], elementPointLoads: ElementPointLoad[]) => ({
  nextNodeId: nodes.reduce((maxId, node) => Math.max(maxId, node.id), 0) + 1,
  nextElementId: elements.reduce((maxId, element) => Math.max(maxId, element.id), 0) + 1,
  nextLoadId: [
    ...udls.map(load => load.id),
    ...elementPointLoads.map(load => load.id)
  ].reduce((maxId, id) => Math.max(maxId, id), 0) + 1
});

const modelSnapshotsEqual = (left: ProjectModel, right: ProjectModel): boolean => (
  JSON.stringify(createModelSnapshot(left)) === JSON.stringify(createModelSnapshot(right))
);

const pushHistory = (history: ModelSnapshot[], snapshot: ModelSnapshot): ModelSnapshot[] => (
  [...history, snapshot].slice(-HISTORY_LIMIT)
);

const restoreModelSnapshot = (snapshot: ModelSnapshot) => {
  const model = createModelSnapshot(snapshot);
  const ids = deriveNextIds(model.nodes, model.elements, model.udls, model.elementPointLoads);

  return {
    ...model,
    nextNodeId: ids.nextNodeId,
    nextElementId: ids.nextElementId,
    nextLoadId: ids.nextLoadId
  };
};

const withHistory = (state: StoreState, patch: Partial<StoreState>): Partial<StoreState> => {
  const before = createModelSnapshot(state);
  const after = createModelSnapshot({ ...state, ...patch });

  if (modelSnapshotsEqual(before, after)) {
    return patch;
  }

  return {
    ...patch,
    past: pushHistory(state.past, before),
    future: []
  };
};

export const parseProjectDocument = (value: unknown): ProjectValidationResult => {
  if (!isRecord(value)) {
    return { ok: false, error: 'Project file must contain a JSON object' };
  }

  if (value.version !== PROJECT_VERSION) {
    return { ok: false, error: `Unsupported project version: ${String(value.version)}` };
  }

  const nodesArray = getRequiredArray(value, 'nodes');
  if (!nodesArray.ok) return nodesArray;

  const elementsArray = getRequiredArray(value, 'elements');
  if (!elementsArray.ok) return elementsArray;

  const pointLoadsArray = getRequiredArray(value, 'pointLoads');
  if (!pointLoadsArray.ok) return pointLoadsArray;

  const udlsArray = getRequiredArray(value, 'udls');
  if (!udlsArray.ok) return udlsArray;

  const elementPointLoadsArray = getRequiredArray(value, 'elementPointLoads');
  if (!elementPointLoadsArray.ok) return elementPointLoadsArray;

  const nodes = validateItems(nodesArray.value, validateNode);
  if (!nodes.ok) return nodes;

  if (hasDuplicateIds(nodes.value.map(node => node.id))) {
    return { ok: false, error: 'Node ids must be unique' };
  }

  const elements = validateItems(elementsArray.value, validateElement);
  if (!elements.ok) return elements;

  if (hasDuplicateIds(elements.value.map(element => element.id))) {
    return { ok: false, error: 'Element ids must be unique' };
  }

  const pointLoads = validateItems(pointLoadsArray.value, validatePointLoad);
  if (!pointLoads.ok) return pointLoads;

  const udls = validateItems(udlsArray.value, validateUDL);
  if (!udls.ok) return udls;

  const elementPointLoads = validateItems(elementPointLoadsArray.value, validateElementPointLoad);
  if (!elementPointLoads.ok) return elementPointLoads;

  const elementLoads = materializeLoadIds(udls.value, elementPointLoads.value);
  if (!elementLoads.ok) return elementLoads;

  const nodeIds = new Set(nodes.value.map(node => node.id));
  const elementIds = new Set(elements.value.map(element => element.id));

  if (elements.value.some(element => !nodeIds.has(element.nodeI) || !nodeIds.has(element.nodeJ))) {
    return { ok: false, error: 'Every element must reference existing nodes' };
  }

  if (pointLoads.value.some(load => !nodeIds.has(load.nodeId))) {
    return { ok: false, error: 'Every point load must reference an existing node' };
  }

  if (elementLoads.value.udls.some(load => !elementIds.has(load.elementId))) {
    return { ok: false, error: 'Every UDL must reference an existing element' };
  }

  if (elementLoads.value.elementPointLoads.some(load => !elementIds.has(load.elementId))) {
    return { ok: false, error: 'Every element point load must reference an existing element' };
  }

  if (hasDuplicateIds(pointLoads.value.map(load => load.nodeId))) {
    return { ok: false, error: 'Point loads must have unique node ids' };
  }

  return {
    ok: true,
    project: {
      version: PROJECT_VERSION,
      nodes: nodes.value,
      elements: elements.value,
      pointLoads: pointLoads.value,
      udls: elementLoads.value.udls,
      elementPointLoads: elementLoads.value.elementPointLoads
    }
  };
};

const loadStoredProject = (): ProjectDocument => {
  const emptyProject = createProjectDocument(emptyProjectModel());
  if (typeof window === 'undefined') return emptyProject;

  try {
    const storedProject = window.localStorage.getItem(PROJECT_STORAGE_KEY);
    if (!storedProject) return emptyProject;

    const parsedProject: unknown = JSON.parse(storedProject);
    const result = parseProjectDocument(parsedProject);
    if (result.ok) return result.project;
    console.warn(`Unable to restore saved Smatrix project: ${result.error}`);
  } catch (error) {
    console.warn('Unable to restore saved Smatrix project:', error);
  }

  return emptyProject;
};

const initialProject = loadStoredProject();
const initialIds = deriveNextIds(
  initialProject.nodes,
  initialProject.elements,
  initialProject.udls,
  initialProject.elementPointLoads
);

export const useStore = create<StoreState>((set) => ({
  // Initial state
  nodes: initialProject.nodes,
  elements: initialProject.elements,
  pointLoads: initialProject.pointLoads,
  udls: initialProject.udls,
  elementPointLoads: initialProject.elementPointLoads,
  
  mode: 'select',
  viewMode: 'structure',
  selectedNodeId: null,
  selectedElementId: null,
  
  result: null,
  isLoading: false,
  error: null,
  
  scale: 50,  // pixels per meter
  offsetX: 100,
  offsetY: 300,
  
  beamStartNodeId: null,

  nextNodeId: initialIds.nextNodeId,
  nextElementId: initialIds.nextElementId,
  nextLoadId: initialIds.nextLoadId,

  past: [],
  future: [],
  
  defaultE: 200e9,  // 200 GPa (steel)
  defaultI: 1e-4,   // 1e-4 m^4
  defaultA: 1e-2,   // 0.01 m^2
  
  // Node actions
  addNode: (x: number, y: number) => {
    set(state => {
      const id = state.nextNodeId;
      return withHistory(state, {
        nextNodeId: id + 1,
        nodes: [...state.nodes, { id, x, y, support: 'free' as SupportType }],
        selectedNodeId: id,
        selectedElementId: null
      });
    });
  },

  moveNode: (id: number, x: number, y: number) => {
    set(state => withHistory(state, {
      nodes: state.nodes.map(n => n.id === id ? { ...n, x, y } : n)
    }));
  },
  
  updateNode: (id: number, updates: Partial<Node>) => {
    set(state => withHistory(state, {
      nodes: state.nodes.map(n => n.id === id ? { ...n, ...updates } : n)
    }));
  },
  
  deleteNode: (id: number) => {
    set(state => {
      const removedElementIds = new Set(
        state.elements
          .filter(e => e.nodeI === id || e.nodeJ === id)
          .map(e => e.id)
      );

      return withHistory(state, {
        nodes: state.nodes.filter(n => n.id !== id),
        elements: state.elements.filter(e => !removedElementIds.has(e.id)),
        udls: state.udls.filter(u => !removedElementIds.has(u.elementId)),
        elementPointLoads: state.elementPointLoads.filter(load => !removedElementIds.has(load.elementId)),
        pointLoads: state.pointLoads.filter(p => p.nodeId !== id),
        selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
        selectedElementId: (
          state.selectedElementId !== null && removedElementIds.has(state.selectedElementId)
            ? null
            : state.selectedElementId
        )
      });
    });
  },
  
  // Element actions
  addElement: (nodeI: number, nodeJ: number) => {
    set(state => {
      const id = state.nextElementId;
      return withHistory(state, {
        nextElementId: id + 1,
        elements: [
          ...state.elements,
          {
            id,
            nodeI,
            nodeJ,
            E: state.defaultE,
            I: state.defaultI,
            A: state.defaultA,
            releaseI: false,
            releaseJ: false
          }
        ],
        selectedElementId: id,
        selectedNodeId: null,
        beamStartNodeId: null
      });
    });
  },
  
  updateElement: (id: number, updates: Partial<Element>) => {
    set(state => withHistory(state, {
      elements: state.elements.map(e => e.id === id ? { ...e, ...updates } : e)
    }));
  },
  
  deleteElement: (id: number) => {
    set(state => withHistory(state, {
      elements: state.elements.filter(e => e.id !== id),
      udls: state.udls.filter(u => u.elementId !== id),
      elementPointLoads: state.elementPointLoads.filter(load => load.elementId !== id),
      selectedElementId: state.selectedElementId === id ? null : state.selectedElementId
    }));
  },
  
  // Point load actions
  addPointLoad: (nodeId: number, Fx: number, Fy: number, Mz: number = 0) => {
    set(state => {
      const existing = state.pointLoads.find(p => p.nodeId === nodeId);
      if (existing) {
        return withHistory(state, {
          pointLoads: state.pointLoads.map(p => 
            p.nodeId === nodeId ? { ...p, Fx, Fy, Mz } : p
          )
        });
      }
      return withHistory(state, { pointLoads: [...state.pointLoads, { nodeId, Fx, Fy, Mz }] });
    });
  },
  
  updatePointLoad: (nodeId: number, updates: Partial<PointLoad>) => {
    set(state => withHistory(state, {
      pointLoads: state.pointLoads.map(p => 
        p.nodeId === nodeId ? { ...p, ...updates } : p
      )
    }));
  },
  
  deletePointLoad: (nodeId: number) => {
    set(state => withHistory(state, {
      pointLoads: state.pointLoads.filter(p => p.nodeId !== nodeId)
    }));
  },
  
  // UDL actions
  addUDL: (elementId: number, w1: number, w2?: number) => {
    set(state => {
      const id = state.nextLoadId;
      return withHistory(state, {
        nextLoadId: id + 1,
        udls: [...state.udls, { id, elementId, w1, w2: w2 ?? w1 }]
      });
    });
  },
  
  updateUDL: (id: number, updates: Partial<UDL>) => {
    set(state => withHistory(state, {
      udls: state.udls.map(u => u.id === id ? { ...u, ...updates } : u)
    }));
  },
  
  deleteUDL: (id: number) => {
    set(state => withHistory(state, {
      udls: state.udls.filter(u => u.id !== id)
    }));
  },

  // Element point load actions
  addElementPointLoad: (elementId: number, a: number, Fx: number, Fy: number) => {
    set(state => {
      const id = state.nextLoadId;
      return withHistory(state, {
        nextLoadId: id + 1,
        elementPointLoads: [...state.elementPointLoads, { id, elementId, a, Fx, Fy }]
      });
    });
  },

  updateElementPointLoad: (id: number, updates: Partial<ElementPointLoad>) => {
    set(state => withHistory(state, {
      elementPointLoads: state.elementPointLoads.map(load =>
        load.id === id ? { ...load, ...updates } : load
      )
    }));
  },

  deleteElementPointLoad: (id: number) => {
    set(state => withHistory(state, {
      elementPointLoads: state.elementPointLoads.filter(load => load.id !== id)
    }));
  },
  
  // UI actions
  setMode: (mode: EditorMode) => set({ mode, beamStartNodeId: null }),
  setViewMode: (mode: ViewMode) => set({ viewMode: mode }),
  setSelectedNode: (id: number | null) => set({ selectedNodeId: id, selectedElementId: null }),
  setSelectedElement: (id: number | null) => set({ selectedElementId: id, selectedNodeId: null }),
  setBeamStartNode: (id: number | null) => set({ beamStartNodeId: id }),
  
  // Result actions
  setResult: (result: AnalysisResult | null) => set({ result }),
  setLoading: (isLoading: boolean) => set({ isLoading }),
  setError: (error: string | null) => set({ error }),
  
  // Canvas actions
  setScale: (scale: number) => set({ scale: Math.max(10, Math.min(200, scale)) }),
  setOffset: (offsetX: number, offsetY: number) => set({ offsetX, offsetY }),
  panViewport: (dx: number, dy: number) => set(state => ({
    offsetX: state.offsetX + dx,
    offsetY: state.offsetY + dy
  })),
  resetViewport: () => set({ scale: 50, offsetX: 100, offsetY: 300 }),

  replaceProject: (project: ProjectModel) => {
    const ids = deriveNextIds(project.nodes, project.elements, project.udls, project.elementPointLoads);
    set(state => withHistory(state, {
      nodes: project.nodes.map(node => ({ ...node })),
      elements: project.elements.map(element => ({ ...element })),
      pointLoads: project.pointLoads.map(load => ({ ...load })),
      udls: project.udls.map(load => ({ ...load })),
      elementPointLoads: project.elementPointLoads.map(load => ({ ...load })),
      nextNodeId: ids.nextNodeId,
      nextElementId: ids.nextElementId,
      nextLoadId: ids.nextLoadId,
      result: null,
      isLoading: false,
      error: null,
      selectedNodeId: null,
      selectedElementId: null,
      beamStartNodeId: null
    }));
  },
  
  // Clear all
  clearAll: () => {
    set(state => withHistory(state, {
      nodes: [],
      elements: [],
      pointLoads: [],
      udls: [],
      elementPointLoads: [],
      nextNodeId: 1,
      nextElementId: 1,
      nextLoadId: 1,
      result: null,
      isLoading: false,
      error: null,
      selectedNodeId: null,
      selectedElementId: null,
      beamStartNodeId: null
    }));
  },

  undo: () => {
    set(state => {
      const previous = state.past[state.past.length - 1];
      if (!previous) {
        return {};
      }

      return {
        ...restoreModelSnapshot(previous),
        past: state.past.slice(0, -1),
        future: pushHistory(state.future, createModelSnapshot(state))
      };
    });
  },

  redo: () => {
    set(state => {
      const next = state.future[state.future.length - 1];
      if (!next) {
        return {};
      }

      return {
        ...restoreModelSnapshot(next),
        past: pushHistory(state.past, createModelSnapshot(state)),
        future: state.future.slice(0, -1)
      };
    });
  }
}));

if (typeof window !== 'undefined') {
  let autosaveTimer: number | undefined;
  let lastSerializedProject = JSON.stringify(createProjectDocument(initialProject));

  useStore.subscribe(state => {
    const project = createProjectDocument(state);
    const serializedProject = JSON.stringify(project);

    if (serializedProject === lastSerializedProject) {
      return;
    }

    lastSerializedProject = serializedProject;

    if (autosaveTimer !== undefined) {
      window.clearTimeout(autosaveTimer);
    }

    autosaveTimer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(PROJECT_STORAGE_KEY, serializedProject);
      } catch (error) {
        console.warn('Unable to autosave Smatrix project:', error);
      }
    }, AUTOSAVE_DELAY_MS);
  });
}
