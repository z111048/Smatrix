// Results visualization canvas component

import React, { useCallback } from 'react';
import { Stage, Layer, Line, Circle, Group, Text } from 'react-konva';
import { useStore } from '../store';

// Hermite cubic interpolation for smooth deflection curves
function hermiteInterpolation(
  x0: number, y0: number, m0: number,
  x1: number, y1: number, m1: number,
  numPoints: number = 20
): number[][] {
  const points: number[][] = [];
  const h = x1 - x0;
  
  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints;
    const t2 = t * t;
    const t3 = t2 * t;
    
    // Hermite basis functions
    const h00 = 2*t3 - 3*t2 + 1;
    const h10 = t3 - 2*t2 + t;
    const h01 = -2*t3 + 3*t2;
    const h11 = t3 - t2;
    
    const x = x0 + t * h;
    const y = h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1;
    
    points.push([x, y]);
  }
  
  return points;
}

const ResultsCanvas: React.FC = () => {
  const {
    nodes, elements, result, viewMode,
    scale, offsetX, offsetY
  } = useStore();

  const getNode = (id: number) => nodes.find(n => n.id === id);

  // Convert world coordinates to screen
  const toScreen = useCallback((wx: number, wy: number) => ({
    x: offsetX + wx * scale,
    y: offsetY - wy * scale
  }), [offsetX, offsetY, scale]);

  if (!result || viewMode === 'structure') {
    return null;
  }

  // Get displacement for a node
  const getDisplacement = (nodeId: number) => {
    const d = result.displacements.find(d => d.node_id === nodeId);
    return d ? { v: d.v, theta: d.theta } : { v: 0, theta: 0 };
  };

  // Get internal forces for an element
  const getInternalForces = (elemId: number) => {
    return result.internal_forces.find(f => f.element_id === elemId);
  };

  // Render deflection shape
  const renderDeflection = () => {
    const deflectionScale = 50; // Amplification factor for visibility
    const curves: React.ReactElement[] = [];
    
    elements.forEach(elem => {
      const nodeI = getNode(elem.nodeI);
      const nodeJ = getNode(elem.nodeJ);
      if (!nodeI || !nodeJ) return;

      const dispI = getDisplacement(elem.nodeI);
      const dispJ = getDisplacement(elem.nodeJ);

      // Use Hermite interpolation for smooth curve
      const interpolated = hermiteInterpolation(
        nodeI.x, dispI.v * deflectionScale, dispI.theta * deflectionScale,
        nodeJ.x, dispJ.v * deflectionScale, dispJ.theta * deflectionScale,
        20
      );

      const points: number[] = [];
      interpolated.forEach(([wx, wy]) => {
        const screen = toScreen(wx, wy);
        points.push(screen.x, screen.y);
      });

      curves.push(
        <Line
          key={`defl-${elem.id}`}
          points={points}
          stroke="#ef4444"
          strokeWidth={3}
          lineCap="round"
          lineJoin="round"
        />
      );
    });

    // Draw original structure as dashed line
    const originalLines = elements.map(elem => {
      const nodeI = getNode(elem.nodeI);
      const nodeJ = getNode(elem.nodeJ);
      if (!nodeI || !nodeJ) return null;

      const posI = toScreen(nodeI.x, 0);
      const posJ = toScreen(nodeJ.x, 0);

      return (
        <Line
          key={`orig-${elem.id}`}
          points={[posI.x, posI.y, posJ.x, posJ.y]}
          stroke="#9ca3af"
          strokeWidth={2}
          dash={[8, 4]}
        />
      );
    });

    // Draw deflected nodes
    const deflectedNodes = nodes.map(node => {
      const disp = getDisplacement(node.id);
      const pos = toScreen(node.x, disp.v * deflectionScale);
      
      return (
        <Group key={`node-defl-${node.id}`}>
          <Circle
            x={pos.x}
            y={pos.y}
            radius={6}
            fill="#ef4444"
            stroke="#dc2626"
            strokeWidth={2}
          />
          <Text
            x={pos.x + 10}
            y={pos.y - 20}
            text={`${(disp.v * 1000).toFixed(2)} mm`}
            fill="#dc2626"
            fontSize={11}
          />
        </Group>
      );
    });

    return (
      <>
        {originalLines}
        {curves}
        {deflectedNodes}
      </>
    );
  };

  // Render SFD (Shear Force Diagram)
  const renderSFD = () => {
    const forceScale = 0.003; // Scale for force visualization
    const diagrams: React.ReactElement[] = [];
    
    elements.forEach(elem => {
      const nodeI = getNode(elem.nodeI);
      const nodeJ = getNode(elem.nodeJ);
      if (!nodeI || !nodeJ) return;

      const forces = getInternalForces(elem.id);
      if (!forces) return;

      const baseY = offsetY;
      const points: number[] = [];
      
      // Start from baseline at node I
      const startPos = toScreen(nodeI.x, 0);
      points.push(startPos.x, baseY);

      // Add all force points
      forces.x.forEach((x, i) => {
        const screenX = offsetX + (nodeI.x + x) * scale;
        const screenY = baseY - forces.V[i] * forceScale;
        points.push(screenX, screenY);
      });

      // Close back to baseline
      const endPos = toScreen(nodeJ.x, 0);
      points.push(endPos.x, baseY);

      diagrams.push(
        <Group key={`sfd-${elem.id}`}>
          <Line
            points={points}
            fill="rgba(59, 130, 246, 0.3)"
            stroke="#2563eb"
            strokeWidth={2}
            closed={true}
          />
        </Group>
      );

      // Add value labels at key points
      const maxV = Math.max(...forces.V.map(Math.abs));
      const maxIdx = forces.V.findIndex(v => Math.abs(v) === maxV);
      if (maxIdx >= 0) {
        const labelX = offsetX + (nodeI.x + forces.x[maxIdx]) * scale;
        const labelY = baseY - forces.V[maxIdx] * forceScale;
        diagrams.push(
          <Text
            key={`sfd-label-${elem.id}`}
            x={labelX + 5}
            y={labelY - 15}
            text={`${(forces.V[maxIdx] / 1000).toFixed(1)} kN`}
            fill="#1d4ed8"
            fontSize={11}
            fontStyle="bold"
          />
        );
      }
    });

    // Draw baseline
    const baseline = (
      <Line
        key="sfd-baseline"
        points={[offsetX, offsetY, offsetX + 1000, offsetY]}
        stroke="#374151"
        strokeWidth={2}
      />
    );

    return (
      <>
        {baseline}
        {diagrams}
      </>
    );
  };

  // Render BMD (Bending Moment Diagram)
  const renderBMD = () => {
    const momentScale = 0.0003; // Scale for moment visualization
    const diagrams: React.ReactElement[] = [];
    
    elements.forEach(elem => {
      const nodeI = getNode(elem.nodeI);
      const nodeJ = getNode(elem.nodeJ);
      if (!nodeI || !nodeJ) return;

      const forces = getInternalForces(elem.id);
      if (!forces) return;

      const baseY = offsetY;
      const points: number[] = [];
      
      // Start from baseline at node I
      const startPos = toScreen(nodeI.x, 0);
      points.push(startPos.x, baseY);

      // Add all moment points (positive moment drawn below for engineering convention)
      forces.x.forEach((x, i) => {
        const screenX = offsetX + (nodeI.x + x) * scale;
        const screenY = baseY + forces.M[i] * momentScale; // + for tension on bottom
        points.push(screenX, screenY);
      });

      // Close back to baseline
      const endPos = toScreen(nodeJ.x, 0);
      points.push(endPos.x, baseY);

      diagrams.push(
        <Group key={`bmd-${elem.id}`}>
          <Line
            points={points}
            fill="rgba(234, 88, 12, 0.3)"
            stroke="#ea580c"
            strokeWidth={2}
            closed={true}
          />
        </Group>
      );

      // Add value labels at key points
      const maxM = Math.max(...forces.M.map(Math.abs));
      const maxIdx = forces.M.findIndex(m => Math.abs(m) === maxM);
      if (maxIdx >= 0) {
        const labelX = offsetX + (nodeI.x + forces.x[maxIdx]) * scale;
        const labelY = baseY + forces.M[maxIdx] * momentScale;
        diagrams.push(
          <Text
            key={`bmd-label-${elem.id}`}
            x={labelX + 5}
            y={labelY + 5}
            text={`${(Math.abs(forces.M[maxIdx]) / 1000).toFixed(1)} kN·m`}
            fill="#c2410c"
            fontSize={11}
            fontStyle="bold"
          />
        );
      }
    });

    // Draw baseline (beam location)
    const baseline = (
      <Line
        key="bmd-baseline"
        points={[offsetX, offsetY, offsetX + 1000, offsetY]}
        stroke="#374151"
        strokeWidth={2}
      />
    );

    return (
      <>
        {baseline}
        {diagrams}
      </>
    );
  };

  return (
    <Stage
      width={1200}
      height={600}
      style={{ 
        position: 'absolute', 
        top: 0, 
        left: 0, 
        pointerEvents: 'none',
        background: 'rgba(255,255,255,0.95)'
      }}
    >
      <Layer>
        {/* Title */}
        <Text
          x={20}
          y={20}
          text={
            viewMode === 'deflection' ? 'Deflection Shape (amplified)' :
            viewMode === 'sfd' ? 'Shear Force Diagram (SFD)' :
            viewMode === 'bmd' ? 'Bending Moment Diagram (BMD)' : ''
          }
          fontSize={18}
          fontStyle="bold"
          fill="#1f2937"
        />

        {viewMode === 'deflection' && renderDeflection()}
        {viewMode === 'sfd' && renderSFD()}
        {viewMode === 'bmd' && renderBMD()}

        {/* Legend */}
        <Group x={20} y={550}>
          {viewMode === 'deflection' && (
            <>
              <Line points={[0, 10, 40, 10]} stroke="#9ca3af" strokeWidth={2} dash={[8, 4]} />
              <Text x={50} y={3} text="Original" fill="#6b7280" fontSize={12} />
              <Line points={[120, 10, 160, 10]} stroke="#ef4444" strokeWidth={3} />
              <Text x={170} y={3} text="Deflected" fill="#6b7280" fontSize={12} />
            </>
          )}
          {viewMode === 'sfd' && (
            <>
              <Line points={[0, 0, 40, 0, 40, 20, 0, 20]} closed fill="rgba(59, 130, 246, 0.3)" stroke="#2563eb" />
              <Text x={50} y={3} text="Shear Force (+ = clockwise)" fill="#6b7280" fontSize={12} />
            </>
          )}
          {viewMode === 'bmd' && (
            <>
              <Line points={[0, 0, 40, 0, 40, 20, 0, 20]} closed fill="rgba(234, 88, 12, 0.3)" stroke="#ea580c" />
              <Text x={50} y={3} text="Bending Moment (drawn on tension side)" fill="#6b7280" fontSize={12} />
            </>
          )}
        </Group>
      </Layer>
    </Stage>
  );
};

export default ResultsCanvas;
