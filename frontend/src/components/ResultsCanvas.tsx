// Results visualization canvas component

import React, { useCallback, useState, useEffect, useRef } from 'react';
import { Stage, Layer, Line, Circle, Group, Text } from 'react-konva';
import { useStore } from '../store';

const ResultsCanvas: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  
  const {
    nodes, elements, result, viewMode,
    scale, offsetX, offsetY
  } = useStore();

  // Responsive canvas sizing
  useEffect(() => {
    const updateSize = () => {
      // Find the canvas-container parent
      const container = containerRef.current?.parentElement;
      if (container) {
        const { clientWidth, clientHeight } = container;
        setDimensions({
          width: clientWidth,
          height: Math.max(clientHeight, 300)
        });
      }
    };
    
    updateSize();
    window.addEventListener('resize', updateSize);
    
    const resizeObserver = new ResizeObserver(updateSize);
    const container = containerRef.current?.parentElement;
    if (container) {
      resizeObserver.observe(container);
    }
    
    return () => {
      window.removeEventListener('resize', updateSize);
      resizeObserver.disconnect();
    };
  }, []);

  const getNode = (id: number) => nodes.find(n => n.id === id);

  // Convert world coordinates to screen
  const toScreen = useCallback((wx: number, wy: number) => ({
    x: offsetX + wx * scale,
    y: offsetY - wy * scale
  }), [offsetX, offsetY, scale]);

  if (!result || viewMode === 'structure') {
    return <div ref={containerRef} style={{ display: 'none' }} />;
  }

  // Get displacement for a node
  const getDisplacement = (nodeId: number) => {
    const d = result.displacements.find(d => d.node_id === nodeId);
    return d ? { u: d.u || 0, v: d.v, theta: d.theta } : { u: 0, v: 0, theta: 0 };
  };

  // Get internal forces for an element
  const getInternalForces = (elemId: number) => {
    return result.internal_forces.find(f => f.element_id === elemId);
  };

  // Render deflection shape
  const renderDeflection = () => {
    const deflectionScale = 50; // Amplification factor for visibility
    const curves: React.ReactElement[] = [];
    const originalLines: React.ReactElement[] = [];
    
    elements.forEach(elem => {
      const nodeI = getNode(elem.nodeI);
      const nodeJ = getNode(elem.nodeJ);
      if (!nodeI || !nodeJ) return;

      const dispI = getDisplacement(elem.nodeI);
      const dispJ = getDisplacement(elem.nodeJ);

      // Draw original structure as dashed line
      const origI = toScreen(nodeI.x, nodeI.y);
      const origJ = toScreen(nodeJ.x, nodeJ.y);
      originalLines.push(
        <Line
          key={`orig-${elem.id}`}
          points={[origI.x, origI.y, origJ.x, origJ.y]}
          stroke="#9ca3af"
          strokeWidth={2}
          dash={[8, 4]}
        />
      );

      // Calculate element geometry
      const dx = nodeJ.x - nodeI.x;
      const dy = nodeJ.y - nodeI.y;
      const L = Math.sqrt(dx * dx + dy * dy);
      if (L === 0) return;

      // For 2D deflection, we need to show both u and v displacements
      // Use simplified linear interpolation for now (Hermite is for 1D bending)
      const numPoints = 20;
      const points: number[] = [];
      
      for (let i = 0; i <= numPoints; i++) {
        const t = i / numPoints;
        
        // Interpolate position along element
        const baseX = nodeI.x + dx * t;
        const baseY = nodeI.y + dy * t;
        
        // Interpolate displacements (linear for simplicity)
        const u = dispI.u + (dispJ.u - dispI.u) * t;
        const v = dispI.v + (dispJ.v - dispI.v) * t;
        
        // Apply deflection scale
        const deflectedX = baseX + u * deflectionScale;
        const deflectedY = baseY + v * deflectionScale;
        
        const screen = toScreen(deflectedX, deflectedY);
        points.push(screen.x, screen.y);
      }

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

    // Draw deflected nodes
    const deflectedNodes = nodes.map(node => {
      const disp = getDisplacement(node.id);
      const deflectedX = node.x + disp.u * deflectionScale;
      const deflectedY = node.y + disp.v * deflectionScale;
      const pos = toScreen(deflectedX, deflectedY);
      
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
            text={`δ=${Math.sqrt(disp.u**2 + disp.v**2) * 1000 > 0.01 
              ? (Math.sqrt(disp.u**2 + disp.v**2) * 1000).toFixed(2) + ' mm' 
              : '0 mm'}`}
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
    const structureLines: React.ReactElement[] = [];
    
    elements.forEach(elem => {
      const nodeI = getNode(elem.nodeI);
      const nodeJ = getNode(elem.nodeJ);
      if (!nodeI || !nodeJ) return;

      const forces = getInternalForces(elem.id);
      if (!forces) return;

      // Draw the actual structure line as baseline
      const startPos = toScreen(nodeI.x, nodeI.y);
      const endPos = toScreen(nodeJ.x, nodeJ.y);
      
      structureLines.push(
        <Line
          key={`struct-${elem.id}`}
          points={[startPos.x, startPos.y, endPos.x, endPos.y]}
          stroke="#374151"
          strokeWidth={3}
        />
      );

      // Calculate element geometry
      const dx = nodeJ.x - nodeI.x;
      const dy = nodeJ.y - nodeI.y;
      const L = Math.sqrt(dx * dx + dy * dy);
      if (L === 0) return;
      
      // Unit normal vector (perpendicular to element, pointing "up" in local coords)
      const nx = -dy / L;
      const ny = dx / L;

      const points: number[] = [];
      
      // Start from node I on baseline
      points.push(startPos.x, startPos.y);

      // Add all force points offset perpendicular to element
      forces.x.forEach((xLocal, i) => {
        const ratio = xLocal / L;
        const wx = nodeI.x + dx * ratio;
        const wy = nodeI.y + dy * ratio;
        // Offset by shear force in normal direction
        const offset = forces.V[i] * forceScale;
        const screenPos = toScreen(wx + nx * offset / scale, wy + ny * offset / scale);
        points.push(screenPos.x, screenPos.y);
      });

      // Close back to node J on baseline
      points.push(endPos.x, endPos.y);

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
      if (maxIdx >= 0 && maxV > 0) {
        const ratio = forces.x[maxIdx] / L;
        const wx = nodeI.x + dx * ratio;
        const wy = nodeI.y + dy * ratio;
        const offset = forces.V[maxIdx] * forceScale;
        const labelPos = toScreen(wx + nx * offset / scale, wy + ny * offset / scale);
        
        diagrams.push(
          <Text
            key={`sfd-label-${elem.id}`}
            x={labelPos.x + 5}
            y={labelPos.y - 15}
            text={`${(forces.V[maxIdx] / 1000).toFixed(1)} kN`}
            fill="#1d4ed8"
            fontSize={11}
            fontStyle="bold"
          />
        );
      }
    });

    return (
      <>
        {structureLines}
        {diagrams}
      </>
    );
  };

  // Render BMD (Bending Moment Diagram)
  const renderBMD = () => {
    const momentScale = 0.0003; // Scale for moment visualization
    const diagrams: React.ReactElement[] = [];
    const structureLines: React.ReactElement[] = [];
    
    elements.forEach(elem => {
      const nodeI = getNode(elem.nodeI);
      const nodeJ = getNode(elem.nodeJ);
      if (!nodeI || !nodeJ) return;

      const forces = getInternalForces(elem.id);
      if (!forces) return;

      // Draw the actual structure line as baseline
      const startPos = toScreen(nodeI.x, nodeI.y);
      const endPos = toScreen(nodeJ.x, nodeJ.y);
      
      structureLines.push(
        <Line
          key={`struct-${elem.id}`}
          points={[startPos.x, startPos.y, endPos.x, endPos.y]}
          stroke="#374151"
          strokeWidth={3}
        />
      );

      // Calculate element geometry
      const dx = nodeJ.x - nodeI.x;
      const dy = nodeJ.y - nodeI.y;
      const L = Math.sqrt(dx * dx + dy * dy);
      if (L === 0) return;
      
      // Unit normal vector (perpendicular to element)
      // For BMD, positive moment (tension on bottom) draws on tension side
      const nx = -dy / L;
      const ny = dx / L;

      const points: number[] = [];
      
      // Start from node I on baseline
      points.push(startPos.x, startPos.y);

      // Add all moment points offset perpendicular to element
      // Positive moment drawn on tension side (typically below for simple beams)
      forces.x.forEach((xLocal, i) => {
        const ratio = xLocal / L;
        const wx = nodeI.x + dx * ratio;
        const wy = nodeI.y + dy * ratio;
        // Offset by moment in negative normal direction (tension side)
        const offset = -forces.M[i] * momentScale;
        const screenPos = toScreen(wx + nx * offset / scale, wy + ny * offset / scale);
        points.push(screenPos.x, screenPos.y);
      });

      // Close back to node J on baseline
      points.push(endPos.x, endPos.y);

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
      if (maxIdx >= 0 && maxM > 0) {
        const ratio = forces.x[maxIdx] / L;
        const wx = nodeI.x + dx * ratio;
        const wy = nodeI.y + dy * ratio;
        const offset = -forces.M[maxIdx] * momentScale;
        const labelPos = toScreen(wx + nx * offset / scale, wy + ny * offset / scale);
        
        diagrams.push(
          <Text
            key={`bmd-label-${elem.id}`}
            x={labelPos.x + 5}
            y={labelPos.y + 5}
            text={`${(Math.abs(forces.M[maxIdx]) / 1000).toFixed(1)} kN·m`}
            fill="#c2410c"
            fontSize={11}
            fontStyle="bold"
          />
        );
      }
    });

    return (
      <>
        {structureLines}
        {diagrams}
      </>
    );
  };

  return (
    <div ref={containerRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
      <Stage
        width={dimensions.width}
        height={dimensions.height}
        style={{ 
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
        <Group x={20} y={dimensions.height - 50}>
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
    </div>
  );
};

export default ResultsCanvas;
