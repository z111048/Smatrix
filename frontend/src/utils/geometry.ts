import type { AnalysisResult, Node } from '../types';

export interface ViewTransform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export interface ScreenPoint {
  x: number;
  y: number;
}

export const toScreenPoint = (
  wx: number,
  wy: number,
  { scale, offsetX, offsetY }: ViewTransform
): ScreenPoint => ({
  x: offsetX + wx * scale,
  y: offsetY - wy * scale
});

export const toWorldPoint = (
  sx: number,
  sy: number,
  { scale, offsetX, offsetY }: ViewTransform
): ScreenPoint => ({
  x: (sx - offsetX) / scale,
  y: (offsetY - sy) / scale
});

export const maxFiniteAbs = (values: readonly number[]) => values.reduce((max, value) => (
  Number.isFinite(value) ? Math.max(max, Math.abs(value)) : max
), 0);

export const getStructureDiagonal = (nodes: readonly Pick<Node, 'x' | 'y'>[]) => {
  if (nodes.length === 0) {
    return 0;
  }

  let minX = nodes[0].x;
  let maxX = nodes[0].x;
  let minY = nodes[0].y;
  let maxY = nodes[0].y;

  nodes.forEach(node => {
    minX = Math.min(minX, node.x);
    maxX = Math.max(maxX, node.x);
    minY = Math.min(minY, node.y);
    maxY = Math.max(maxY, node.y);
  });

  return Math.hypot(maxX - minX, maxY - minY);
};

export const getAdaptiveScale = (maxValue: number, targetOrdinate: number) => {
  if (maxValue <= 0 || targetOrdinate <= 0) {
    return 0;
  }

  return targetOrdinate / maxValue;
};

export const computeResultDiagramScales = (
  nodes: readonly Pick<Node, 'x' | 'y'>[],
  result: Pick<AnalysisResult, 'displacements' | 'internal_forces'>,
  targetOrdinateRatio: number
) => {
  const structureDiagonal = getStructureDiagonal(nodes);
  const targetOrdinate = structureDiagonal * targetOrdinateRatio;
  const maxDisplacement = maxFiniteAbs(
    result.displacements.map(displacement => Math.hypot(displacement.u ?? 0, displacement.v))
  );
  const maxShear = maxFiniteAbs(result.internal_forces.flatMap(forces => forces.V));
  const maxMoment = maxFiniteAbs(result.internal_forces.flatMap(forces => forces.M));

  return {
    structureDiagonal,
    targetOrdinate,
    maxDisplacement,
    maxShear,
    maxMoment,
    deflectionScale: getAdaptiveScale(maxDisplacement, targetOrdinate),
    forceScale: getAdaptiveScale(maxShear, targetOrdinate),
    momentScale: getAdaptiveScale(maxMoment, targetOrdinate)
  };
};
