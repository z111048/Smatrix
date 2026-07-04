import {
  PROJECT_STORAGE_KEY,
  PROJECT_VERSION,
  createProjectDocument,
  parseProjectDocument,
  useStore
} from './index';
import type { ProjectModel } from './index';

const resetStoreState = () => {
  useStore.setState({
    nodes: [],
    elements: [],
    pointLoads: [],
    udls: [],
    elementPointLoads: [],
    mode: 'select',
    viewMode: 'structure',
    selectedNodeId: null,
    selectedElementId: null,
    result: null,
    isLoading: false,
    error: null,
    scale: 50,
    offsetX: 100,
    offsetY: 300,
    beamStartNodeId: null,
    nextNodeId: 1,
    nextElementId: 1,
    nextLoadId: 1,
    past: [],
    future: [],
    defaultE: 200e9,
    defaultI: 1e-4,
    defaultA: 1e-2
  });
};

const addBasicElement = () => {
  useStore.getState().addNode(0, 0);
  useStore.getState().addNode(4, 0);
  useStore.getState().addElement(1, 2);
};

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  resetStoreState();
  vi.clearAllTimers();
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('useStore model actions', () => {
  it('adds nodes and elements with defaults and sequential ids', () => {
    addBasicElement();

    const state = useStore.getState();
    expect(state.nodes).toHaveLength(2);
    expect(state.nodes.map(node => node.id)).toEqual([1, 2]);
    expect(state.nodes[0]).toMatchObject({ id: 1, x: 0, y: 0, support: 'free' });
    expect(state.elements).toHaveLength(1);
    expect(state.elements[0]).toMatchObject({
      id: 1,
      nodeI: 1,
      nodeJ: 2,
      E: 200e9,
      I: 1e-4,
      A: 1e-2,
      releaseI: false,
      releaseJ: false
    });
    expect(state.selectedElementId).toBe(1);
    expect(state.nextNodeId).toBe(3);
    expect(state.nextElementId).toBe(2);
  });

  it('allows multiple loads per element and cascades them when deleting the element', () => {
    addBasicElement();
    useStore.getState().addUDL(1, -1000);
    useStore.getState().addUDL(1, -2000, -3000);
    useStore.getState().addElementPointLoad(1, 1.25, 500, -1500);
    useStore.getState().addElementPointLoad(1, 2.5, 0, -2500);

    expect(useStore.getState().udls.map(load => load.id)).toEqual([1, 2]);
    expect(useStore.getState().elementPointLoads.map(load => load.id)).toEqual([3, 4]);
    expect(useStore.getState().nextLoadId).toBe(5);

    useStore.getState().deleteElement(1);

    const state = useStore.getState();
    expect(state.nodes).toHaveLength(2);
    expect(state.elements).toHaveLength(0);
    expect(state.udls).toHaveLength(0);
    expect(state.elementPointLoads).toHaveLength(0);
    expect(state.selectedElementId).toBeNull();
    expect(state.nextLoadId).toBe(5);
  });

  it('cascades connected elements and loads when deleting a node', () => {
    useStore.getState().addNode(0, 0);
    useStore.getState().addNode(4, 0);
    useStore.getState().addNode(8, 0);
    useStore.getState().addElement(1, 2);
    useStore.getState().addElement(2, 3);
    useStore.getState().addPointLoad(2, 1000, -2000, 3000);
    useStore.getState().addUDL(1, -1000);
    useStore.getState().addElementPointLoad(2, 2, 0, -1500);

    useStore.getState().deleteNode(2);

    const state = useStore.getState();
    expect(state.nodes.map(node => node.id)).toEqual([1, 3]);
    expect(state.elements).toHaveLength(0);
    expect(state.pointLoads).toHaveLength(0);
    expect(state.udls).toHaveLength(0);
    expect(state.elementPointLoads).toHaveLength(0);
    expect(state.selectedElementId).toBeNull();
  });

  it('updates a node point load instead of duplicating it', () => {
    useStore.getState().addNode(0, 0);

    useStore.getState().addPointLoad(1, 1000, -2000, 0);
    useStore.getState().addPointLoad(1, 3000, -4000, 5000);

    expect(useStore.getState().pointLoads).toEqual([
      { nodeId: 1, Fx: 3000, Fy: -4000, Mz: 5000 }
    ]);
  });

  it('undoes and redoes one move entry for a dragged node', () => {
    useStore.getState().addNode(0, 0);
    useStore.getState().moveNode(1, 2, 3);

    expect(useStore.getState().past).toHaveLength(2);

    useStore.getState().undo();

    expect(useStore.getState().nodes[0]).toMatchObject({ id: 1, x: 0, y: 0 });
    expect(useStore.getState().future).toHaveLength(1);

    useStore.getState().redo();

    expect(useStore.getState().nodes[0]).toMatchObject({ id: 1, x: 2, y: 3 });
    expect(useStore.getState().future).toHaveLength(0);
  });

  it('caps undo history at fifty model snapshots', () => {
    for (let i = 0; i < 55; i += 1) {
      useStore.getState().addNode(i, 0);
    }

    expect(useStore.getState().past).toHaveLength(50);
    expect(useStore.getState().nodes).toHaveLength(55);

    for (let i = 0; i < 50; i += 1) {
      useStore.getState().undo();
    }

    expect(useStore.getState().nodes).toHaveLength(5);

    useStore.getState().undo();

    expect(useStore.getState().nodes).toHaveLength(5);
  });

  it('clears the project model, counters, selection, and analysis state', () => {
    addBasicElement();
    useStore.getState().addUDL(1, -1000);
    useStore.getState().setResult({
      success: true,
      displacements: [{ node_id: 1, u: 0, v: 0, theta: 0 }],
      reactions: [],
      internal_forces: []
    });
    useStore.getState().setLoading(true);
    useStore.getState().setError('analysis failed');

    useStore.getState().clearAll();

    const state = useStore.getState();
    expect(state.nodes).toEqual([]);
    expect(state.elements).toEqual([]);
    expect(state.udls).toEqual([]);
    expect(state.elementPointLoads).toEqual([]);
    expect(state.nextNodeId).toBe(1);
    expect(state.nextElementId).toBe(1);
    expect(state.nextLoadId).toBe(1);
    expect(state.result).toBeNull();
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
    expect(state.selectedNodeId).toBeNull();
    expect(state.selectedElementId).toBeNull();
  });

  it('re-derives id counters after replacing an imported project', () => {
    const importedProject: ProjectModel = {
      nodes: [
        { id: 4, x: 0, y: 0, support: 'pin' },
        { id: 9, x: 5, y: 0, support: 'roller' }
      ],
      elements: [
        { id: 7, nodeI: 4, nodeJ: 9, E: 210e9, I: 2e-4, releaseI: false, releaseJ: false }
      ],
      pointLoads: [],
      udls: [{ id: 10, elementId: 7, w1: -1000, w2: -2000 }],
      elementPointLoads: [{ id: 12, elementId: 7, a: 2, Fx: 0, Fy: -3000 }]
    };

    useStore.getState().replaceProject(importedProject);

    expect(useStore.getState().nextNodeId).toBe(10);
    expect(useStore.getState().nextElementId).toBe(8);
    expect(useStore.getState().nextLoadId).toBe(13);

    useStore.getState().addNode(10, 0);
    useStore.getState().addElement(9, 10);
    useStore.getState().addUDL(8, -4000);

    expect(useStore.getState().nodes.at(-1)?.id).toBe(10);
    expect(useStore.getState().elements.at(-1)?.id).toBe(8);
    expect(useStore.getState().udls.at(-1)?.id).toBe(13);
  });
});

describe('project document validation and migration', () => {
  it('materializes legacy load ids and migrates UDL w to w1 and w2', () => {
    const result = parseProjectDocument({
      version: PROJECT_VERSION,
      nodes: [
        { id: 1, x: 0, y: 0, support: 'pin' },
        { id: 2, x: 4, y: 0, support: 'roller' }
      ],
      elements: [
        { id: 1, nodeI: 1, nodeJ: 2, E: 200e9, I: 1e-4 }
      ],
      pointLoads: [{ nodeId: 2, Fy: -1000, Mz: 0 }],
      udls: [{ elementId: 1, w: -2500 }],
      elementPointLoads: [{ elementId: 1, a: 1.5, Fx: 0, Fy: -3000 }]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      throw new Error(result.error);
    }

    expect(result.project.version).toBe(PROJECT_VERSION);
    expect(result.project.udls).toEqual([{ id: 1, elementId: 1, w1: -2500, w2: -2500 }]);
    expect(result.project.elementPointLoads).toEqual([
      { id: 2, elementId: 1, a: 1.5, Fx: 0, Fy: -3000 }
    ]);
    expect(result.project.elements[0]).toMatchObject({ releaseI: false, releaseJ: false });
  });

  it('rejects invalid imports with duplicate ids or missing references', () => {
    const duplicateNodes = parseProjectDocument({
      version: PROJECT_VERSION,
      nodes: [
        { id: 1, x: 0, y: 0, support: 'pin' },
        { id: 1, x: 4, y: 0, support: 'roller' }
      ],
      elements: [],
      pointLoads: [],
      udls: [],
      elementPointLoads: []
    });

    expect(duplicateNodes.ok).toBe(false);
    if (duplicateNodes.ok) {
      throw new Error('Expected duplicate node ids to fail validation');
    }
    expect(duplicateNodes.error).toContain('Node ids');

    const missingNode = parseProjectDocument({
      version: PROJECT_VERSION,
      nodes: [{ id: 1, x: 0, y: 0, support: 'pin' }],
      elements: [{ id: 1, nodeI: 1, nodeJ: 2, E: 200e9, I: 1e-4 }],
      pointLoads: [],
      udls: [],
      elementPointLoads: []
    });

    expect(missingNode.ok).toBe(false);
    if (missingNode.ok) {
      throw new Error('Expected missing node reference to fail validation');
    }
    expect(missingNode.error).toContain('existing nodes');
  });
});

describe('project persistence', () => {
  it('autosaves project documents to the versioned localStorage key', () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');

    useStore.getState().addNode(1.5, -2);

    expect(localStorage.getItem(PROJECT_STORAGE_KEY)).toBeNull();

    vi.advanceTimersByTime(500);

    expect(setItemSpy).toHaveBeenCalledWith(PROJECT_STORAGE_KEY, expect.any(String));

    const storedProject = localStorage.getItem(PROJECT_STORAGE_KEY);
    expect(storedProject).not.toBeNull();

    const parsedProject = JSON.parse(storedProject ?? '{}');
    expect(parsedProject).toMatchObject({
      version: PROJECT_VERSION,
      nodes: [{ id: 1, x: 1.5, y: -2, support: 'free' }],
      elements: [],
      pointLoads: [],
      udls: [],
      elementPointLoads: []
    });
  });

  it('restores a saved project from localStorage on module initialization', async () => {
    const project = createProjectDocument({
      nodes: [
        { id: 3, x: 0, y: 0, support: 'fixed' },
        { id: 7, x: 6, y: 0, support: 'roller_y' }
      ],
      elements: [
        { id: 5, nodeI: 3, nodeJ: 7, E: 210e9, I: 3e-4, A: 2e-2, releaseI: true, releaseJ: false }
      ],
      pointLoads: [{ nodeId: 7, Fx: 1000, Fy: -2000, Mz: 3000 }],
      udls: [{ id: 8, elementId: 5, w1: -1000, w2: -1500 }],
      elementPointLoads: [{ id: 9, elementId: 5, a: 2.5, Fx: 0, Fy: -4000 }]
    });

    localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(project));
    vi.resetModules();

    const freshStore = await import('./index');
    const state = freshStore.useStore.getState();

    expect(state.nodes).toEqual(project.nodes);
    expect(state.elements).toEqual(project.elements);
    expect(state.pointLoads).toEqual(project.pointLoads);
    expect(state.udls).toEqual(project.udls);
    expect(state.elementPointLoads).toEqual(project.elementPointLoads);
    expect(state.nextNodeId).toBe(8);
    expect(state.nextElementId).toBe(6);
    expect(state.nextLoadId).toBe(10);
  });
});
