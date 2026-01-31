// Type definitions for Smatrix frontend

export type SupportType = 'free' | 'roller' | 'pin' | 'fixed';

export interface Node {
  id: number;
  x: number;
  y: number;
  support: SupportType;
}

export interface Element {
  id: number;
  nodeI: number;
  nodeJ: number;
  E: number;  // Young's modulus (Pa)
  I: number;  // Moment of inertia (m^4)
}

export interface PointLoad {
  nodeId: number;
  Fy: number;  // Vertical force (N), positive upward
  Mz: number;  // Moment (N·m)
}

export interface UDL {
  elementId: number;
  w: number;  // Load intensity (N/m), positive upward
}

export interface NodeDisplacement {
  node_id: number;
  v: number;   // Vertical displacement (m)
  theta: number;  // Rotation (rad)
}

export interface NodeReaction {
  node_id: number;
  Fy: number;  // Vertical reaction (N)
  Mz: number;  // Moment reaction (N·m)
}

export interface ElementInternalForces {
  element_id: number;
  stations: number[];
  x: number[];
  V: number[];  // Shear force
  M: number[];  // Bending moment
}

export interface AnalysisResult {
  success: boolean;
  displacements: NodeDisplacement[];
  reactions: NodeReaction[];
  internal_forces: ElementInternalForces[];
}

export type EditorMode = 'select' | 'addNode' | 'addBeam' | 'addPointLoad' | 'addUDL';

export type ViewMode = 'structure' | 'deflection' | 'sfd' | 'bmd';
