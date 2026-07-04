// API client for Smatrix backend

import type { Node, Element, PointLoad, UDL, ElementPointLoad, AnalysisResult } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface AnalysisRequest {
  nodes: Array<{
    id: number;
    x: number;
    y: number;
    support: string;
  }>;
  elements: Array<{
    id: number;
    node_i: number;
    node_j: number;
    E: number;
    I: number;
    A?: number;
    release_i: boolean;
    release_j: boolean;
  }>;
  point_loads: Array<{
    node_id: number;
    Fx?: number;
    Fy: number;
    Mz: number;
  }>;
  udls: Array<{
    element_id: number;
    w1: number;
    w2: number;
  }>;
  element_point_loads: Array<{
    element_id: number;
    a: number;
    Fx: number;
    Fy: number;
  }>;
}

export async function analyzeStructure(
  nodes: Node[],
  elements: Element[],
  pointLoads: PointLoad[],
  udls: UDL[],
  elementPointLoads: ElementPointLoad[] = []
): Promise<AnalysisResult> {
  const request: AnalysisRequest = {
    nodes: nodes.map(n => ({
      id: n.id,
      x: n.x,
      y: n.y,
      support: n.support
    })),
    elements: elements.map(e => ({
      id: e.id,
      node_i: e.nodeI,
      node_j: e.nodeJ,
      E: e.E,
      I: e.I,
      A: e.A ?? 1e-2,
      release_i: e.releaseI,
      release_j: e.releaseJ
    })),
    point_loads: pointLoads.map(p => ({
      node_id: p.nodeId,
      Fx: p.Fx ?? 0,
      Fy: p.Fy,
      Mz: p.Mz
    })),
    udls: udls.map(u => ({
      element_id: u.elementId,
      w1: u.w1,
      w2: u.w2
    })),
    element_point_loads: elementPointLoads.map(load => ({
      element_id: load.elementId,
      a: load.a,
      Fx: load.Fx,
      Fy: load.Fy
    }))
  };

  const response = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Analysis failed');
  }

  return response.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
