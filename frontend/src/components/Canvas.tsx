// Canvas component using React-Konva

import React, { useRef, useCallback, useState, useEffect } from 'react';
import { Stage, Layer, Line, Circle, Group, Text, Arrow } from 'react-konva';
import type { Stage as KonvaStage } from 'konva/lib/Stage';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useStore } from '../store';
import type { Node, Element } from '../types';
import { toScreenPoint, toWorldPoint } from '../utils/geometry';

const GRID_SIZE = 50;
const NODE_RADIUS = 8;
const HINGE_RADIUS = 6;
const HINGE_OFFSET = NODE_RADIUS + 10;
const CANVAS_ARIA_LABEL = [
  'Structure canvas.',
  'Keyboard controls: V Select, N Add Node, B Add Beam, L Point Load, U UDL, E Element Load.',
  'Arrow keys move the selected node 0.5 meters; Shift plus arrow keys move 0.1 meters.',
  'Delete or Backspace deletes the selected node or element. Escape clears selection.'
].join(' ');

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

const SupportRollerY: React.FC<{ x: number; y: number }> = ({ x, y }) => (
  <Group>
    <Line points={[x, y, x - 20, y - 15, x - 20, y + 15, x, y]} closed stroke="#16a34a" strokeWidth={2} />
    <Circle x={x - 28} y={y - 10} radius={5} stroke="#16a34a" strokeWidth={2} />
    <Circle x={x - 28} y={y + 10} radius={5} stroke="#16a34a" strokeWidth={2} />
    <Line points={[x - 35, y - 20, x - 35, y + 20]} stroke="#16a34a" strokeWidth={2} />
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
  const stageRef = useRef<KonvaStage | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPointer, setLastPointer] = useState<{ x: number; y: number } | null>(null);
  const [didPan, setDidPan] = useState(false);

  useEffect(() => {
    const container = stageRef.current?.container();
    container?.setAttribute('aria-label', CANVAS_ARIA_LABEL);
  }, []);
  
  // Responsive canvas sizing
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const { clientWidth, clientHeight } = containerRef.current;
        setDimensions({
          width: clientWidth,
          height: Math.max(clientHeight, 300)
        });
      }
    };
    
    updateSize();
    window.addEventListener('resize', updateSize);
    
    // ResizeObserver for container size changes
    const resizeObserver = new ResizeObserver(updateSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    
    return () => {
      window.removeEventListener('resize', updateSize);
      resizeObserver.disconnect();
    };
  }, []);
  
  const {
    nodes, elements, pointLoads, udls, elementPointLoads,
    mode, selectedNodeId, selectedElementId,
    scale, offsetX, offsetY,
    beamStartNodeId,
    addNode, setSelectedNode, setSelectedElement,
    setBeamStartNode, addElement, moveNode,
    setScale, panViewport, resetViewport
  } = useStore();

  // Convert world coordinates to screen
  const toScreen = useCallback(
    (wx: number, wy: number) => toScreenPoint(wx, wy, { offsetX, offsetY, scale }),
    [offsetX, offsetY, scale]
  );

  // Convert screen coordinates to world
  const toWorld = useCallback(
    (sx: number, sy: number) => toWorldPoint(sx, sy, { offsetX, offsetY, scale }),
    [offsetX, offsetY, scale]
  );

  const modeHint = (() => {
    if (mode === 'addNode') return 'Add Node / 新增節點：tap canvas';
    if (mode === 'addBeam' && beamStartNodeId === null) return 'Add Beam / 新增桿件：tap first node';
    if (mode === 'addBeam') return `Add Beam / 新增桿件：start node ${beamStartNodeId}, tap end node`;
    if (mode === 'addPointLoad') return 'Point Load / 節點載重：select node, edit panel';
    if (mode === 'addUDL') return 'UDL / 均佈載重：select element, edit panel';
    if (mode === 'addElementPointLoad') return 'Element Load / 桿件集中載重：select element, edit panel';
    return 'Select / 選取：tap model, drag blank canvas to pan';
  })();

  // Handle stage click
  const handleStageClick = (e: KonvaEventObject<MouseEvent>) => {
    if (didPan) {
      setDidPan(false);
      return;
    }

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

  const beginPan = (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (mode !== 'select' || e.target !== e.target.getStage()) return;

    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos) return;

    setIsPanning(true);
    setLastPointer(pos);
    setDidPan(false);
  };

  const focusCanvas = useCallback(() => {
    stageRef.current?.container().focus({ preventScroll: true });
  }, []);

  const handleCanvasMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    focusCanvas();
    beginPan(e);
  };

  const handleCanvasTouchStart = (e: KonvaEventObject<TouchEvent>) => {
    focusCanvas();
    beginPan(e);
  };

  const updatePan = (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (!isPanning) return;

    e.evt.preventDefault();
    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos || !lastPointer) return;

    const dx = pos.x - lastPointer.x;
    const dy = pos.y - lastPointer.y;
    if (Math.abs(dx) + Math.abs(dy) > 1) {
      panViewport(dx, dy);
      setDidPan(true);
    }
    setLastPointer(pos);
  };

  const endPan = () => {
    setIsPanning(false);
    setLastPointer(null);
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

  const handleNodeDragStart = (node: Node, e: KonvaEventObject<DragEvent>) => {
    e.cancelBubble = true;
    setSelectedNode(node.id);
  };

  const handleNodeDragEnd = (node: Node, e: KonvaEventObject<DragEvent>) => {
    e.cancelBubble = true;

    const dragOffset = e.target.position();
    const nodeScreen = toScreen(node.x, node.y);
    const world = toWorld(nodeScreen.x + dragOffset.x, nodeScreen.y + dragOffset.y);
    const snappedX = Math.round(world.x * 2) / 2;
    const snappedY = Math.round(world.y * 2) / 2;

    e.target.position({ x: 0, y: 0 });

    if (snappedX !== node.x || snappedY !== node.y) {
      moveNode(node.id, snappedX, snappedY);
    }
  };

  // Handle element click
  const handleElementClick = (elem: Element, e: KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true;
    if (mode === 'select' || mode === 'addUDL' || mode === 'addElementPointLoad') {
      setSelectedElement(elem.id);
    }
  };

  // Find node by ID
  const getNode = (id: number) => nodes.find(n => n.id === id);

  // Draw grid
  const drawGrid = () => {
    const lines = [];
    const { width, height } = dimensions;
    
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
      case 'roller_x':
        return <SupportRoller x={screenPos.x} y={screenPos.y} />;
      case 'roller_y':
        return <SupportRollerY x={screenPos.x} y={screenPos.y} />;
      case 'fixed':
        return <SupportFixed x={screenPos.x} y={screenPos.y} />;
      default:
        return null;
    }
  };

  const renderHingeSymbols = (
    elem: Element,
    posI: { x: number; y: number },
    posJ: { x: number; y: number },
    isSelected: boolean
  ) => {
    if (!elem.releaseI && !elem.releaseJ) return null;

    const dx = posJ.x - posI.x;
    const dy = posJ.y - posI.y;
    const length = Math.hypot(dx, dy);
    if (length === 0) return null;

    const unitX = dx / length;
    const unitY = dy / length;
    const offset = Math.min(HINGE_OFFSET, length * 0.3);
    const stroke = isSelected ? '#1d4ed8' : '#111827';
    const hinges: React.ReactElement[] = [];

    if (elem.releaseI) {
      hinges.push(
        <Circle
          key={`hinge-i-${elem.id}`}
          x={posI.x + unitX * offset}
          y={posI.y + unitY * offset}
          radius={HINGE_RADIUS}
          stroke={stroke}
          strokeWidth={2}
          fill="#fafafa"
          listening={false}
        />
      );
    }

    if (elem.releaseJ) {
      hinges.push(
        <Circle
          key={`hinge-j-${elem.id}`}
          x={posJ.x - unitX * offset}
          y={posJ.y - unitY * offset}
          radius={HINGE_RADIUS}
          stroke={stroke}
          strokeWidth={2}
          fill="#fafafa"
          listening={false}
        />
      );
    }

    return hinges;
  };

  // Draw point loads
  const renderPointLoad = (nodeId: number) => {
    const load = pointLoads.find(p => p.nodeId === nodeId);
    if (!load || ((load.Fx ?? 0) === 0 && load.Fy === 0)) return null;
    
    const node = getNode(nodeId);
    if (!node) return null;
    
    const pos = toScreen(node.x, node.y);
    const fx = load.Fx ?? 0;
    const fy = load.Fy;
    const forceMagnitude = Math.max(Math.abs(fx), Math.abs(fy));
    const arrowLength = Math.min(80, forceMagnitude / 10000 * 40 + 40);
    const isDown = load.Fy < 0;
    const isLeft = fx < 0;
    
    return (
      <Group key={`load-${nodeId}`}>
        {fy !== 0 && (
          <Arrow
            points={isDown ? [pos.x, pos.y - arrowLength, pos.x, pos.y - 5] : [pos.x, pos.y + arrowLength, pos.x, pos.y + 5]}
            stroke="#ef4444"
            strokeWidth={3}
            pointerLength={10}
            pointerWidth={8}
            fill="#ef4444"
          />
        )}
        {fx !== 0 && (
          <Arrow
            points={isLeft ? [pos.x + arrowLength, pos.y, pos.x + 5, pos.y] : [pos.x - arrowLength, pos.y, pos.x - 5, pos.y]}
            stroke="#dc2626"
            strokeWidth={3}
            pointerLength={10}
            pointerWidth={8}
            fill="#dc2626"
          />
        )}
        <Text
          x={pos.x + 10}
          y={pos.y - arrowLength - 18}
          text={[
            fx !== 0 ? `Fx ${Math.abs(fx / 1000).toFixed(0)} kN` : '',
            fy !== 0 ? `Fy ${Math.abs(fy / 1000).toFixed(0)} kN` : ''
          ].filter(Boolean).join('\n')}
          fill="#ef4444"
          fontSize={12}
        />
      </Group>
    );
  };

  // Draw UDL
  const renderUDL = (elementId: number) => {
    const elementUdls = udls.filter(u => u.elementId === elementId && (u.w1 !== 0 || u.w2 !== 0));
    if (elementUdls.length === 0) return null;
    
    const elem = elements.find(e => e.id === elementId);
    if (!elem) return null;
    
    const nodeI = getNode(elem.nodeI);
    const nodeJ = getNode(elem.nodeJ);
    if (!nodeI || !nodeJ) return null;
    
    const posI = toScreen(nodeI.x, nodeI.y);
    const posJ = toScreen(nodeJ.x, nodeJ.y);
    const screenDx = posJ.x - posI.x;
    const screenDy = posJ.y - posI.y;
    const screenLength = Math.hypot(screenDx, screenDy) || 1;
    const normal = {
      x: screenDy / screenLength,
      y: -screenDx / screenLength
    };

    const loadShapes: React.ReactElement[] = [];
    const numArrows = 8;
    const endGap = 5;

    elementUdls.forEach((udl, loadIndex) => {
      const maxIntensity = Math.max(Math.abs(udl.w1), Math.abs(udl.w2));
      if (maxIntensity === 0) return;

      const maxArrowLength = 34 + loadIndex * 12;
      const boundaryPoints: number[] = [];

      for (let i = 0; i <= numArrows; i++) {
        const t = i / numArrows;
        const x = posI.x + (posJ.x - posI.x) * t;
        const y = posI.y + (posJ.y - posI.y) * t;
        const intensity = udl.w1 + (udl.w2 - udl.w1) * t;
        const magnitude = Math.abs(intensity);
        const directionSign = intensity >= 0 ? 1 : -1;
        const direction = {
          x: normal.x * directionSign,
          y: normal.y * directionSign
        };
        const arrowLength = magnitude === 0 ? 0 : Math.max(10, (magnitude / maxIntensity) * maxArrowLength);
        const start = {
          x: x - direction.x * arrowLength,
          y: y - direction.y * arrowLength
        };
        const end = {
          x: x - direction.x * endGap,
          y: y - direction.y * endGap
        };

        boundaryPoints.push(start.x, start.y);
        if (magnitude === 0) continue;

        loadShapes.push(
          <Arrow
            key={`udl-${udl.id}-${i}`}
            points={[start.x, start.y, end.x, end.y]}
            stroke="#f97316"
            strokeWidth={2}
            pointerLength={6}
            pointerWidth={5}
            fill="#f97316"
          />
        );
      }

      loadShapes.push(
        <Line
          key={`udl-line-${udl.id}`}
          points={boundaryPoints}
          stroke="#f97316"
          strokeWidth={2}
        />
      );
      loadShapes.push(
        <Line
          key={`udl-side-i-${udl.id}`}
          points={[posI.x, posI.y, boundaryPoints[0], boundaryPoints[1]]}
          stroke="#f97316"
          strokeWidth={1}
        />
      );
      loadShapes.push(
        <Line
          key={`udl-side-j-${udl.id}`}
          points={[
            posJ.x,
            posJ.y,
            boundaryPoints[boundaryPoints.length - 2],
            boundaryPoints[boundaryPoints.length - 1]
          ]}
          stroke="#f97316"
          strokeWidth={1}
        />
      );
    });
    
    return <Group>{loadShapes}</Group>;
  };

  const renderElementPointLoad = (elementId: number) => {
    const loads = elementPointLoads.filter(p => (
      p.elementId === elementId && (p.Fx !== 0 || p.Fy !== 0)
    ));
    if (loads.length === 0) return null;

    const elem = elements.find(e => e.id === elementId);
    if (!elem) return null;

    const nodeI = getNode(elem.nodeI);
    const nodeJ = getNode(elem.nodeJ);
    if (!nodeI || !nodeJ) return null;

    const dx = nodeJ.x - nodeI.x;
    const dy = nodeJ.y - nodeI.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    if (length === 0) return null;

    return (
      <Group>
        {loads.map((load, loadIndex) => {
          const ratio = Math.max(0, Math.min(1, load.a / length));
          const wx = nodeI.x + dx * ratio;
          const wy = nodeI.y + dy * ratio;
          const basePos = toScreen(wx, wy);
          const posI = toScreen(nodeI.x, nodeI.y);
          const posJ = toScreen(nodeJ.x, nodeJ.y);
          const screenDx = posJ.x - posI.x;
          const screenDy = posJ.y - posI.y;
          const screenLength = Math.hypot(screenDx, screenDy) || 1;
          const stackOffset = loadIndex * 10;
          const pos = {
            x: basePos.x - (screenDy / screenLength) * stackOffset,
            y: basePos.y + (screenDx / screenLength) * stackOffset
          };
          const forceMagnitude = Math.max(Math.abs(load.Fx), Math.abs(load.Fy));
          const arrowLength = Math.min(80, forceMagnitude / 10000 * 40 + 40);
          const isDown = load.Fy < 0;
          const isLeft = load.Fx < 0;

          return (
            <Group key={`element-point-load-${load.id}`}>
              {load.Fy !== 0 && (
                <Arrow
                  points={isDown ? [pos.x, pos.y - arrowLength, pos.x, pos.y - 5] : [pos.x, pos.y + arrowLength, pos.x, pos.y + 5]}
                  stroke="#7c3aed"
                  strokeWidth={3}
                  pointerLength={10}
                  pointerWidth={8}
                  fill="#7c3aed"
                />
              )}
              {load.Fx !== 0 && (
                <Arrow
                  points={isLeft ? [pos.x + arrowLength, pos.y, pos.x + 5, pos.y] : [pos.x - arrowLength, pos.y, pos.x - 5, pos.y]}
                  stroke="#6d28d9"
                  strokeWidth={3}
                  pointerLength={10}
                  pointerWidth={8}
                  fill="#6d28d9"
                />
              )}
              <Text
                x={pos.x + 10}
                y={pos.y - arrowLength - 18}
                text={[
                  load.Fx !== 0 ? `Px ${Math.abs(load.Fx / 1000).toFixed(0)} kN` : '',
                  load.Fy !== 0 ? `Py ${Math.abs(load.Fy / 1000).toFixed(0)} kN` : ''
                ].filter(Boolean).join('\n')}
                fill="#6d28d9"
                fontSize={12}
              />
            </Group>
          );
        })}
      </Group>
    );
  };

  return (
    <div ref={containerRef} className="canvas-shell">
      <div className="canvas-status">
        <span>{modeHint}</span>
        <span>{Math.round(scale)} px/m</span>
      </div>
      <div className="viewport-controls" aria-label="Canvas zoom controls">
        <button type="button" onClick={() => setScale(scale + 10)} title="Zoom in / 放大">+</button>
        <button type="button" onClick={() => setScale(scale - 10)} title="Zoom out / 縮小">-</button>
        <button type="button" onClick={resetViewport} title="Reset view / 重設視圖">Reset</button>
      </div>
      <Stage
        ref={stageRef}
        width={dimensions.width}
        height={dimensions.height}
        className="konva-stage"
        role="application"
        tabIndex={0}
        title="Structure canvas keyboard controls"
        onClick={handleStageClick}
        onWheel={handleWheel}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={updatePan}
        onMouseUp={endPan}
        onMouseLeave={endPan}
        onTouchStart={handleCanvasTouchStart}
        onTouchMove={updatePan}
        onTouchEnd={endPan}
        style={{ background: '#fafafa', cursor: isPanning ? 'grabbing' : mode === 'select' ? 'grab' : 'crosshair' }}
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
              {renderHingeSymbols(elem, posI, posJ, isSelected)}
              {renderUDL(elem.id)}
              {renderElementPointLoad(elem.id)}
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
            <Group
              key={`node-${node.id}`}
              x={0}
              y={0}
              draggable={mode === 'select'}
              onDragStart={(e) => handleNodeDragStart(node, e)}
              onDragEnd={(e) => handleNodeDragEnd(node, e)}
            >
              {/* Support symbol */}
              {renderSupport(node, pos)}
              
              {/* Node circle - orange for unsupported nodes */}
              <Circle
                x={pos.x}
                y={pos.y}
                radius={NODE_RADIUS}
                fill={
                  isSelected ? '#2563eb' : 
                  isBeamStart ? '#16a34a' : 
                  node.support === 'free' ? '#f97316' :  // Orange for free nodes
                  '#1f2937'
                }
                stroke={
                  isSelected ? '#1d4ed8' : 
                  node.support === 'free' ? '#ea580c' :  // Dark orange border
                  '#374151'
                }
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
        
        {/* Coordinate axes - positioned relative to canvas size */}
        <Arrow
          points={[50, dimensions.height - 50, 120, dimensions.height - 50]}
          stroke="#9ca3af"
          strokeWidth={2}
          pointerLength={8}
          pointerWidth={6}
          fill="#9ca3af"
        />
        <Text x={125} y={dimensions.height - 57} text="X" fill="#9ca3af" fontSize={14} />
        <Arrow
          points={[50, dimensions.height - 50, 50, dimensions.height - 120]}
          stroke="#9ca3af"
          strokeWidth={2}
          pointerLength={8}
          pointerWidth={6}
          fill="#9ca3af"
        />
        <Text x={43} y={dimensions.height - 135} text="Y" fill="#9ca3af" fontSize={14} />
      </Layer>
    </Stage>
    </div>
  );
};

export default Canvas;
