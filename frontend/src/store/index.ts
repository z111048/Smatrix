// Zustand store for Smatrix state management

import { create } from 'zustand';
import type { Node, Element, PointLoad, UDL, AnalysisResult, EditorMode, ViewMode, SupportType } from '../types';

interface StoreState {
  // Structure data
  nodes: Node[];
  elements: Element[];
  pointLoads: PointLoad[];
  udls: UDL[];
  
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
  
  // Default material properties
  defaultE: number;
  defaultI: number;
  
  // Actions
  addNode: (x: number, y: number) => void;
  updateNode: (id: number, updates: Partial<Node>) => void;
  deleteNode: (id: number) => void;
  
  addElement: (nodeI: number, nodeJ: number) => void;
  updateElement: (id: number, updates: Partial<Element>) => void;
  deleteElement: (id: number) => void;
  
  addPointLoad: (nodeId: number, Fy: number, Mz?: number) => void;
  updatePointLoad: (nodeId: number, updates: Partial<PointLoad>) => void;
  deletePointLoad: (nodeId: number) => void;
  
  addUDL: (elementId: number, w: number) => void;
  updateUDL: (elementId: number, w: number) => void;
  deleteUDL: (elementId: number) => void;
  
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
  
  clearAll: () => void;
}

let nextNodeId = 1;
let nextElementId = 1;

export const useStore = create<StoreState>((set, get) => ({
  // Initial state
  nodes: [],
  elements: [],
  pointLoads: [],
  udls: [],
  
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
  
  defaultE: 200e9,  // 200 GPa (steel)
  defaultI: 1e-4,   // 1e-4 m^4
  
  // Node actions
  addNode: (x: number, y: number) => {
    const id = nextNodeId++;
    set(state => ({
      nodes: [...state.nodes, { id, x, y, support: 'free' as SupportType }],
      selectedNodeId: id,
      selectedElementId: null
    }));
  },
  
  updateNode: (id: number, updates: Partial<Node>) => {
    set(state => ({
      nodes: state.nodes.map(n => n.id === id ? { ...n, ...updates } : n)
    }));
  },
  
  deleteNode: (id: number) => {
    set(state => ({
      nodes: state.nodes.filter(n => n.id !== id),
      elements: state.elements.filter(e => e.nodeI !== id && e.nodeJ !== id),
      pointLoads: state.pointLoads.filter(p => p.nodeId !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId
    }));
  },
  
  // Element actions
  addElement: (nodeI: number, nodeJ: number) => {
    const { defaultE, defaultI } = get();
    const id = nextElementId++;
    set(state => ({
      elements: [...state.elements, { id, nodeI, nodeJ, E: defaultE, I: defaultI }],
      selectedElementId: id,
      selectedNodeId: null,
      beamStartNodeId: null
    }));
  },
  
  updateElement: (id: number, updates: Partial<Element>) => {
    set(state => ({
      elements: state.elements.map(e => e.id === id ? { ...e, ...updates } : e)
    }));
  },
  
  deleteElement: (id: number) => {
    set(state => ({
      elements: state.elements.filter(e => e.id !== id),
      udls: state.udls.filter(u => u.elementId !== id),
      selectedElementId: state.selectedElementId === id ? null : state.selectedElementId
    }));
  },
  
  // Point load actions
  addPointLoad: (nodeId: number, Fy: number, Mz: number = 0) => {
    set(state => {
      const existing = state.pointLoads.find(p => p.nodeId === nodeId);
      if (existing) {
        return {
          pointLoads: state.pointLoads.map(p => 
            p.nodeId === nodeId ? { ...p, Fy, Mz } : p
          )
        };
      }
      return { pointLoads: [...state.pointLoads, { nodeId, Fy, Mz }] };
    });
  },
  
  updatePointLoad: (nodeId: number, updates: Partial<PointLoad>) => {
    set(state => ({
      pointLoads: state.pointLoads.map(p => 
        p.nodeId === nodeId ? { ...p, ...updates } : p
      )
    }));
  },
  
  deletePointLoad: (nodeId: number) => {
    set(state => ({
      pointLoads: state.pointLoads.filter(p => p.nodeId !== nodeId)
    }));
  },
  
  // UDL actions
  addUDL: (elementId: number, w: number) => {
    set(state => {
      const existing = state.udls.find(u => u.elementId === elementId);
      if (existing) {
        return {
          udls: state.udls.map(u => u.elementId === elementId ? { ...u, w } : u)
        };
      }
      return { udls: [...state.udls, { elementId, w }] };
    });
  },
  
  updateUDL: (elementId: number, w: number) => {
    set(state => ({
      udls: state.udls.map(u => u.elementId === elementId ? { ...u, w } : u)
    }));
  },
  
  deleteUDL: (elementId: number) => {
    set(state => ({
      udls: state.udls.filter(u => u.elementId !== elementId)
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
  
  // Clear all
  clearAll: () => {
    nextNodeId = 1;
    nextElementId = 1;
    set({
      nodes: [],
      elements: [],
      pointLoads: [],
      udls: [],
      result: null,
      error: null,
      selectedNodeId: null,
      selectedElementId: null,
      beamStartNodeId: null
    });
  }
}));
