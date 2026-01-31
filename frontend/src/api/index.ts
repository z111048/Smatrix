// API client for Smatrix backend

import type { Node, Element, PointLoad, UDL, AnalysisResult } from '../types';

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
  }>;
  point_loads: Array<{
    node_id: number;
    Fy: number;
    Mz: number;
  }>;
  udls: Array<{
    element_id: number;
    w: number;
  }>;
}

export async function analyzeStructure(
  nodes: Node[],
  elements: Element[],
  pointLoads: PointLoad[],
  udls: UDL[]
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
      I: e.I
    })),
    point_loads: pointLoads.map(p => ({
      node_id: p.nodeId,
      Fy: p.Fy,
      Mz: p.Mz
    })),
    udls: udls.map(u => ({
      element_id: u.elementId,
      w: u.w
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
