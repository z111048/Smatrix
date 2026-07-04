import type { AnalysisResult, Node } from '../types';
import {
  computeResultDiagramScales,
  getAdaptiveScale,
  getStructureDiagonal,
  maxFiniteAbs,
  toScreenPoint,
  toWorldPoint
} from './geometry';

describe('coordinate transforms', () => {
  it('round-trips world and screen points', () => {
    const transform = { scale: 40, offsetX: 100, offsetY: 300 };
    const screen = toScreenPoint(2.25, -1.5, transform);
    const world = toWorldPoint(screen.x, screen.y, transform);

    expect(screen).toEqual({ x: 190, y: 360 });
    expect(world.x).toBeCloseTo(2.25);
    expect(world.y).toBeCloseTo(-1.5);
  });

  it('flips the Y axis for engineering coordinates', () => {
    const transform = { scale: 50, offsetX: 0, offsetY: 250 };

    expect(toScreenPoint(0, 1, transform).y).toBe(200);
    expect(toScreenPoint(0, 0, transform).y).toBe(250);
    expect(toScreenPoint(0, -1, transform).y).toBe(300);
  });
});

describe('result diagram scaling', () => {
  it('computes structure extents and finite maxima', () => {
    const nodes: Node[] = [
      { id: 1, x: -1, y: 2, support: 'free' },
      { id: 2, x: 2, y: -2, support: 'pin' }
    ];

    expect(getStructureDiagonal(nodes)).toBe(5);
    expect(getStructureDiagonal([])).toBe(0);
    expect(maxFiniteAbs([0, -4, Number.NaN, Number.POSITIVE_INFINITY, 2])).toBe(4);
  });

  it('returns zero scale when the ordinate or result maximum is zero', () => {
    expect(getAdaptiveScale(0, 10)).toBe(0);
    expect(getAdaptiveScale(10, 0)).toBe(0);
    expect(getAdaptiveScale(4, 2)).toBe(0.5);
  });

  it('keeps all-zero result diagrams finite', () => {
    const result: AnalysisResult = {
      success: true,
      displacements: [{ node_id: 1, u: 0, v: 0, theta: 0 }],
      reactions: [],
      internal_forces: [
        { element_id: 1, stations: [0, 1], x: [0, 4], V: [0, 0], M: [0, 0] }
      ]
    };

    const scales = computeResultDiagramScales(
      [
        { x: 0, y: 0 },
        { x: 3, y: 4 }
      ],
      result,
      0.15
    );

    expect(scales.structureDiagonal).toBe(5);
    expect(scales.targetOrdinate).toBe(0.75);
    expect(scales.maxDisplacement).toBe(0);
    expect(scales.maxShear).toBe(0);
    expect(scales.maxMoment).toBe(0);
    expect(scales.deflectionScale).toBe(0);
    expect(scales.forceScale).toBe(0);
    expect(scales.momentScale).toBe(0);
    Object.values(scales).forEach(value => {
      expect(Number.isNaN(value)).toBe(false);
    });
  });

  it('computes adaptive scales from result maxima', () => {
    const result: AnalysisResult = {
      success: true,
      displacements: [
        { node_id: 1, u: 0.01, v: 0, theta: 0 },
        { node_id: 2, u: 0, v: -0.02, theta: 0 }
      ],
      reactions: [],
      internal_forces: [
        { element_id: 1, stations: [0, 1], x: [0, 10], V: [1000, -4000], M: [0, 8000] }
      ]
    };

    const scales = computeResultDiagramScales(
      [
        { x: 0, y: 0 },
        { x: 10, y: 0 }
      ],
      result,
      0.2
    );

    expect(scales.targetOrdinate).toBe(2);
    expect(scales.maxDisplacement).toBe(0.02);
    expect(scales.maxShear).toBe(4000);
    expect(scales.maxMoment).toBe(8000);
    expect(scales.deflectionScale).toBe(100);
    expect(scales.forceScale).toBe(0.0005);
    expect(scales.momentScale).toBe(0.00025);
  });
});
