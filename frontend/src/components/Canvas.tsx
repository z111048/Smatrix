// Canvas component using React-Konva

import React, { useRef, useCallback } from 'react';
import { Stage, Layer, Line, Circle, Group, Text, Arrow } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useStore } from '../store';
import type { Node, Element } from '../types';

const GRID_SIZE = 50;
const NODE_RADIUS = 8;

// Support visualization components
const SupportPin: React.FC<{ x: number; y: number }> = ({ x, y }) => (
  <Group>
    <Line points={[x, y, x - 15, y + 25, x + 15, y + 25, x, y]} closed stroke="#2563eb" strokeWidth={2} />
    <Line points={[x - 20, y + 25, x + 20, y + 25]} stroke="#2563eb" strokeWidth={2} />
  </Group>
);

const SupportRoller: React.FC<{ x: number; y: number }> = ({ x, y }) => (
  <Group>
    <Line points={[x, y, x - 15, y + 20, x + 15, y + 20, x, y]} closed stroke="#16a34a" strokeWidth={2} />
    <Circle x={x - 10} y={y + 28} radius={5} stroke="#16a34a" strokeWidth={2} />
    <Circle x={x + 10} y={y + 28} radius={5} stroke="#16a34a" strokeWidth={2} />
    <Line points={[x - 20, y + 35, x + 20, y + 35]} stroke="#16a34a" strokeWidth={2} />
  </Group>
);

const SupportFixed: React.FC<{ x: number; y: number }> = ({ x, y }) => (
  <Group>
    <Line points={[x - 15, y - 20, x - 15, y + 20]} stroke="#dc2626" strokeWidth={3} />
    {[-15, -5, 5, 15].map((dy, i) => (
      <Line key={i} points={[x - 15, y + dy, x - 25, y + dy + 10]} stroke="#dc2626" strokeWidth={2} />
    ))}
  </Group>
);

const Canvas: React.FC = () => {
  const stageRef = useRef<any>(null);
  
  const {
    nodes, elements, pointLoads, udls,
    mode, selectedNodeId, selectedElementId,
    scale, offsetX, offsetY,
    beamStartNodeId,
    addNode, setSelectedNode, setSelectedElement,
    setBeamStartNode, addElement,
    setScale
  } = useStore();

  // Convert world coordinates to screen
  const toScreen = useCallback((wx: number, wy: number) => ({
    x: offsetX + wx * scale,
    y: offsetY - wy * scale  // Flip Y for engineering convention
  }), [offsetX, offsetY, scale]);

  // Convert screen coordinates to world
  const toWorld = useCallback((sx: number, sy: number) => ({
    x: (sx - offsetX) / scale,
    y: (offsetY - sy) / scale
  }), [offsetX, offsetY, scale]);

  // Handle stage click
  const handleStageClick = (e: KonvaEventObject<MouseEvent>) => {
    if (e.target === e.target.getStage()) {
      if (mode === 'addNode') {
        const pos = e.target.getStage()?.getPointerPosition();
        if (pos) {
          const world = toWorld(pos.x, pos.y);
          // Snap to grid
          const snappedX = Math.round(world.x * 2) / 2;
          const snappedY = Math.round(world.y * 2) / 2;
          addNode(snappedX, snappedY);
        }
      } else if (mode === 'select') {
        setSelectedNode(null);
        setSelectedElement(null);
      }
    }
  };

  // Handle wheel for zoom
  const handleWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const delta = e.evt.deltaY > 0 ? -5 : 5;
    setScale(scale + delta);
  };

  // Handle node click
  const handleNodeClick = (node: Node, e: KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true;
    
    if (mode === 'addBeam') {
      if (beamStartNodeId === null) {
        setBeamStartNode(node.id);
      } else if (beamStartNodeId !== node.id) {
        addElement(beamStartNodeId, node.id);
      }
    } else {
      setSelectedNode(node.id);
    }
  };

  // Handle element click
  const handleElementClick = (elem: Element, e: KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true;
    if (mode === 'select' || mode === 'addUDL') {
      setSelectedElement(elem.id);
    }
  };

  // Find node by ID
  const getNode = (id: number) => nodes.find(n => n.id === id);

  // Draw grid
  const drawGrid = () => {
    const lines = [];
    const width = 1200;
    const height = 600;
    
    for (let x = 0; x < width; x += GRID_SIZE) {
      lines.push(
        <Line key={`v${x}`} points={[x, 0, x, height]} stroke="#e5e7eb" strokeWidth={1} />
      );
    }
    for (let y = 0; y < height; y += GRID_SIZE) {
      lines.push(
        <Line key={`h${y}`} points={[0, y, width, y]} stroke="#e5e7eb" strokeWidth={1} />
      );
    }
    return lines;
  };

  // Draw supports
  const renderSupport = (node: Node, screenPos: { x: number; y: number }) => {
    switch (node.support) {
      case 'pin':
        return <SupportPin x={screenPos.x} y={screenPos.y} />;
      case 'roller':
        return <SupportRoller x={screenPos.x} y={screenPos.y} />;
      case 'fixed':
        return <SupportFixed x={screenPos.x} y={screenPos.y} />;
      default:
        return null;
    }
  };

  // Draw point loads
  const renderPointLoad = (nodeId: number) => {
    const load = pointLoads.find(p => p.nodeId === nodeId);
    if (!load || load.Fy === 0) return null;
    
    const node = getNode(nodeId);
    if (!node) return null;
    
    const pos = toScreen(node.x, node.y);
    const arrowLength = Math.min(80, Math.abs(load.Fy) / 10000 * 40 + 40);
    const isDown = load.Fy < 0;
    
    return (
      <Group key={`load-${nodeId}`}>
        <Arrow
          points={isDown ? [pos.x, pos.y - arrowLength, pos.x, pos.y - 5] : [pos.x, pos.y + arrowLength, pos.x, pos.y + 5]}
          stroke="#ef4444"
          strokeWidth={3}
          pointerLength={10}
          pointerWidth={8}
          fill="#ef4444"
        />
        <Text
          x={pos.x + 10}
          y={isDown ? pos.y - arrowLength - 10 : pos.y + arrowLength - 10}
          text={`${Math.abs(load.Fy / 1000).toFixed(0)} kN`}
          fill="#ef4444"
          fontSize={12}
        />
      </Group>
    );
  };

  // Draw UDL
  const renderUDL = (elementId: number) => {
    const udl = udls.find(u => u.elementId === elementId);
    if (!udl || udl.w === 0) return null;
    
    const elem = elements.find(e => e.id === elementId);
    if (!elem) return null;
    
    const nodeI = getNode(elem.nodeI);
    const nodeJ = getNode(elem.nodeJ);
    if (!nodeI || !nodeJ) return null;
    
    const posI = toScreen(nodeI.x, nodeI.y);
    const posJ = toScreen(nodeJ.x, nodeJ.y);
    
    const arrows = [];
    const numArrows = 8;
    const isDown = udl.w < 0;
    const arrowLen = 30;
    
    for (let i = 0; i <= numArrows; i++) {
      const t = i / numArrows;
      const x = posI.x + (posJ.x - posI.x) * t;
      const y = posI.y + (posJ.y - posI.y) * t;
      arrows.push(
        <Arrow
          key={`udl-${elementId}-${i}`}
          points={isDown ? [x, y - arrowLen, x, y - 5] : [x, y + arrowLen, x, y + 5]}
          stroke="#f97316"
          strokeWidth={2}
          pointerLength={6}
          pointerWidth={5}
          fill="#f97316"
        />
      );
    }
    
    // Top line connecting arrows
    arrows.push(
      <Line
        key={`udl-line-${elementId}`}
        points={[posI.x, posI.y - arrowLen, posJ.x, posJ.y - arrowLen]}
        stroke="#f97316"
        strokeWidth={2}
      />
    );
    
    return <Group>{arrows}</Group>;
  };

  return (
    <Stage
      ref={stageRef}
      width={1200}
      height={600}
      onClick={handleStageClick}
      onWheel={handleWheel}
      style={{ border: '1px solid #d1d5db', borderRadius: '8px', background: '#fafafa' }}
    >
      <Layer>
        {/* Grid */}
        {drawGrid()}
        
        {/* Elements (beams) */}
        {elements.map(elem => {
          const nodeI = getNode(elem.nodeI);
          const nodeJ = getNode(elem.nodeJ);
          if (!nodeI || !nodeJ) return null;
          
          const posI = toScreen(nodeI.x, nodeI.y);
          const posJ = toScreen(nodeJ.x, nodeJ.y);
          const isSelected = selectedElementId === elem.id;
          
          return (
            <Group key={`elem-${elem.id}`}>
              <Line
                points={[posI.x, posI.y, posJ.x, posJ.y]}
                stroke={isSelected ? '#2563eb' : '#374151'}
                strokeWidth={isSelected ? 6 : 4}
                onClick={(e) => handleElementClick(elem, e)}
                hitStrokeWidth={20}
              />
              {renderUDL(elem.id)}
            </Group>
          );
        })}
        
        {/* Beam creation preview */}
        {mode === 'addBeam' && beamStartNodeId !== null && (
          (() => {
            const startNode = getNode(beamStartNodeId);
            if (!startNode) return null;
            const pos = toScreen(startNode.x, startNode.y);
            return (
              <Circle
                x={pos.x}
                y={pos.y}
                radius={NODE_RADIUS + 4}
                stroke="#2563eb"
                strokeWidth={2}
                dash={[5, 5]}
              />
            );
          })()
        )}
        
        {/* Nodes */}
        {nodes.map(node => {
          const pos = toScreen(node.x, node.y);
          const isSelected = selectedNodeId === node.id;
          const isBeamStart = beamStartNodeId === node.id;
          
          return (
            <Group key={`node-${node.id}`}>
              {/* Support symbol */}
              {renderSupport(node, pos)}
              
              {/* Node circle */}
              <Circle
                x={pos.x}
                y={pos.y}
                radius={NODE_RADIUS}
                fill={isSelected ? '#2563eb' : isBeamStart ? '#16a34a' : '#1f2937'}
                stroke={isSelected ? '#1d4ed8' : '#374151'}
                strokeWidth={2}
                onClick={(e) => handleNodeClick(node, e)}
              />
              
              {/* Node label */}
              <Text
                x={pos.x + 12}
                y={pos.y - 8}
                text={`${node.id}`}
                fill="#6b7280"
                fontSize={12}
              />
              
              {/* Point load */}
              {renderPointLoad(node.id)}
            </Group>
          );
        })}
        
        {/* Coordinate axes */}
        <Arrow
          points={[50, 550, 120, 550]}
          stroke="#9ca3af"
          strokeWidth={2}
          pointerLength={8}
          pointerWidth={6}
          fill="#9ca3af"
        />
        <Text x={125} y={543} text="X" fill="#9ca3af" fontSize={14} />
        <Arrow
          points={[50, 550, 50, 480]}
          stroke="#9ca3af"
          strokeWidth={2}
          pointerLength={8}
          pointerWidth={6}
          fill="#9ca3af"
        />
        <Text x={43} y={465} text="Y" fill="#9ca3af" fontSize={14} />
      </Layer>
    </Stage>
  );
};

export default Canvas;
