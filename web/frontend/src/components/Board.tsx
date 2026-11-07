import React, { useEffect, useMemo } from 'react';
import { ActionJson, GameState, OverlayItem } from '../api/types';
import {
  boardDimensions,
  cellCenter,
  cellRect,
  wallSlotRect,
  DEFAULT_GEOMETRY,
} from '../lib/geometry';
import { buildOverlayLookup, getHeatmapColor } from './Heatmap';

export type ActionMode =
  | 'MOVE'
  | 'HOP'
  | 'WALL_H'
  | 'WALL_V'
  | 'REPLACE'
  | 'BLOCK'
  | null;

export interface BoardProps {
  state: GameState;
  mode?: ActionMode;
  overlay?: OverlayItem[];
  showPaths?: boolean;
  onAction?: (action: ActionJson) => void;
  interactive?: boolean;
}

export const Board: React.FC<BoardProps> = ({
  state,
  mode = null,
  overlay,
  showPaths = true,
  onAction,
  interactive = true,
}) => {
  const n = state.n;
  const { width, height } = useMemo(() => boardDimensions(n), [n]);
  const overlayLookup = useMemo(() => buildOverlayLookup(overlay), [overlay]);

  // Keyboard navigation for MOVE
  useEffect(() => {
    if (!interactive || !onAction || state.terminal.over) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const p0Pos = state.pawns[state.to_move];
      if (!p0Pos) return;

      let targetR = p0Pos[0];
      let targetC = p0Pos[1];

      if (e.key === 'ArrowUp') targetR += 1;
      else if (e.key === 'ArrowDown') targetR -= 1;
      else if (e.key === 'ArrowLeft') targetC -= 1;
      else if (e.key === 'ArrowRight') targetC += 1;
      else return;

      const legalMoves = state.legal.MOVE || [];
      const match = legalMoves.find(
        ([r, c]) => r === targetR && c === targetC
      );
      if (match) {
        e.preventDefault();
        onAction(['MOVE', match]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state, interactive, onAction]);

  // Lookup sets for active mode targets
  const targetCells = useMemo(() => {
    const map = new Set<string>();
    if (!interactive) return map;
    if (mode === 'MOVE') {
      (state.legal.MOVE || []).forEach(([r, c]) => map.add(`${r},${c}`));
    } else if (mode === 'HOP') {
      (state.legal.HOP || []).forEach(([r, c]) => map.add(`${r},${c}`));
    } else if (mode === 'BLOCK') {
      (state.legal.BLOCK || []).forEach(([r, c]) => map.add(`${r},${c}`));
    }
    return map;
  }, [state.legal, mode, interactive]);

  const targetWallSlots = useMemo(() => {
    const map = new Set<string>();
    if (!interactive) return map;
    if (mode === 'WALL_H') {
      (state.legal.WALL || [])
        .filter(([, , o]) => o === 'H')
        .forEach(([r, c]) => map.add(`${r},${c},H`));
    } else if (mode === 'WALL_V') {
      (state.legal.WALL || [])
        .filter(([, , o]) => o === 'V')
        .forEach(([r, c]) => map.add(`${r},${c},V`));
    } else if (mode === 'REPLACE') {
      (state.legal.REPLACE || []).forEach(([r, c, o]) => map.add(`${r},${c},${o}`));
    }
    return map;
  }, [state.legal, mode, interactive]);

  // Blocked cell map
  const blockedMap = useMemo(() => {
    const map = new Map<string, number>();
    (state.blocked || []).forEach((b) => map.set(`${b.r},${b.c}`, b.remaining));
    return map;
  }, [state.blocked]);

  const handleCellClick = (r: number, c: number) => {
    if (!interactive || !onAction || state.terminal.over) return;
    if (mode === 'MOVE') {
      onAction(['MOVE', [r, c]]);
    } else if (mode === 'HOP') {
      onAction(['HOP', [r, c]]);
    } else if (mode === 'BLOCK') {
      onAction(['BLOCK', [r, c]]);
    }
  };

  const handleWallSlotClick = (r: number, c: number, o: 'H' | 'V') => {
    if (!interactive || !onAction || state.terminal.over) return;
    if (mode === 'WALL_H' || mode === 'WALL_V') {
      onAction(['WALL', [r, c, o]]);
    } else if (mode === 'REPLACE') {
      onAction(['REPLACE', [r, c, o]]);
    }
  };

  return (
    <svg
      className="svg-board"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      data-testid="wallz-board"
    >
      <defs>
        <pattern
          id="block-hatch"
          width="8"
          height="8"
          patternTransform="rotate(45 0 0)"
          patternUnits="userSpaceOnUse"
        >
          <line
            x1="0"
            y1="0"
            x2="0"
            y2="8"
            stroke="#c084fc"
            strokeWidth="2.5"
          />
        </pattern>
        <filter id="latest-wall-glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Grid Cells */}
      {Array.from({ length: n }).map((_, r) =>
        Array.from({ length: n }).map((__, c) => {
          const rect = cellRect(r, c, n);
          const cellKey = `${r},${c}`;
          const isTarget = targetCells.has(cellKey);
          const blockRemaining = blockedMap.get(cellKey);
          const overlayItem = overlayLookup.cells.get(cellKey);
          const heatmapBg = getHeatmapColor(overlayItem);

          // Goal rows: P1 goal is row 0; P0 goal is row n-1
          let cellFill = '#1e293b';
          if (r === 0) cellFill = 'rgba(244, 63, 94, 0.08)';
          if (r === n - 1) cellFill = 'rgba(56, 189, 248, 0.08)';
          if (heatmapBg) cellFill = heatmapBg;
          if (isTarget) cellFill = 'rgba(16, 185, 129, 0.25)';

          return (
            <g
              key={`cell-${cellKey}`}
              onClick={() => handleCellClick(r, c)}
              style={{ cursor: isTarget ? 'pointer' : 'default' }}
            >
              <rect
                x={rect.x}
                y={rect.y}
                width={rect.width}
                height={rect.height}
                rx={6}
                fill={cellFill}
                stroke={isTarget ? '#10b981' : '#334155'}
                strokeWidth={isTarget ? 2 : 1}
              />

              {/* Blocked Cell Hatching */}
              {blockRemaining !== undefined && (
                <>
                  <rect
                    x={rect.x}
                    y={rect.y}
                    width={rect.width}
                    height={rect.height}
                    rx={6}
                    fill="url(#block-hatch)"
                    opacity={0.65}
                  />
                  <circle
                    cx={rect.x + rect.width - 12}
                    cy={rect.y + 12}
                    r={9}
                    fill="#a855f7"
                  />
                  <text
                    x={rect.x + rect.width - 12}
                    y={rect.y + 16}
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize={11}
                    fontWeight="bold"
                  >
                    {blockRemaining}
                  </text>
                </>
              )}

              {/* Heatmap overlay label */}
              {overlayItem && (
                <text
                  x={rect.x + rect.width / 2}
                  y={rect.y + rect.height - 8}
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize={10}
                  fontWeight="600"
                >
                  {overlayItem.label}
                </text>
              )}
            </g>
          );
        })
      )}

      {/* Shortest Path Polylines */}
      {showPaths &&
        state.paths &&
        state.paths.map((path, playerIdx) => {
          if (!path || path.length < 2) return null;
          const points = path
            .map(([r, c]) => {
              const pt = cellCenter(r, c, n);
              return `${pt.x},${pt.y}`;
            })
            .join(' ');
          const strokeColor = playerIdx === 0 ? '#38bdf8' : '#f43f5e';
          return (
            <polyline
              key={`path-${playerIdx}`}
              points={points}
              fill="none"
              stroke={strokeColor}
              strokeWidth={3}
              strokeDasharray="4 4"
              opacity={0.65}
              strokeLinecap="round"
            />
          );
        })}

      {/* Wall Placement & Replacement Slots (Ghosts) */}
      {Array.from({ length: n - 1 }).map((_, r) =>
        Array.from({ length: n - 1 }).map((__, c) => (
          <React.Fragment key={`slots-${r}-${c}`}>
            {/* Horizontal Slot */}
            {(() => {
              const slot = wallSlotRect(r, c, 'H', n);
              const key = `${r},${c},H`;
              const isTarget = targetWallSlots.has(key);
              const overlayItem = overlayLookup.walls.get(key);
              const color = getHeatmapColor(overlayItem);

              return (
                <rect
                  x={slot.x}
                  y={slot.y}
                  width={slot.width}
                  height={slot.height}
                  rx={3}
                  fill={color || (isTarget ? '#10b981' : 'transparent')}
                  opacity={isTarget || color ? 0.75 : 0}
                  stroke={isTarget ? '#10b981' : 'none'}
                  strokeWidth={1}
                  onClick={() => handleWallSlotClick(r, c, 'H')}
                  style={{ cursor: isTarget ? 'pointer' : 'default' }}
                />
              );
            })()}

            {/* Vertical Slot */}
            {(() => {
              const slot = wallSlotRect(r, c, 'V', n);
              const key = `${r},${c},V`;
              const isTarget = targetWallSlots.has(key);
              const overlayItem = overlayLookup.walls.get(key);
              const color = getHeatmapColor(overlayItem);

              return (
                <rect
                  x={slot.x}
                  y={slot.y}
                  width={slot.width}
                  height={slot.height}
                  rx={3}
                  fill={color || (isTarget ? '#10b981' : 'transparent')}
                  opacity={isTarget || color ? 0.75 : 0}
                  stroke={isTarget ? '#10b981' : 'none'}
                  strokeWidth={1}
                  onClick={() => handleWallSlotClick(r, c, 'V')}
                  style={{ cursor: isTarget ? 'pointer' : 'default' }}
                />
              );
            })()}
          </React.Fragment>
        ))
      )}

      {/* Placed Walls */}
      {(state.walls || []).map((w, idx) => {
        const slot = wallSlotRect(w.r, w.c, w.o, n);
        const wallColor = w.owner === 0 ? '#0284c7' : '#e11d48';

        return (
          <g
            key={`wall-${idx}`}
            aria-label={w.latest && state.rules.A ? 'Replaceable Wall' : undefined}
          >
            <rect
              x={slot.x}
              y={slot.y}
              width={slot.width}
              height={slot.height}
              rx={3}
              fill={wallColor}
              stroke={w.latest && state.rules.A ? '#facc15' : '#0f172a'}
              strokeWidth={w.latest && state.rules.A ? 2 : 1}
              filter={w.latest && state.rules.A ? 'url(#latest-wall-glow)' : undefined}
            />
            {w.latest && state.rules.A && (
              <title>Replaceable Wall (Rule A active)</title>
            )}
          </g>
        );
      })}

      {/* Pawns */}
      {state.pawns &&
        state.pawns.map(([r, c], pIdx) => {
          const pt = cellCenter(r, c, n);
          const pColor = pIdx === 0 ? '#38bdf8' : '#f43f5e';
          const pText = `P${pIdx}`;

          return (
            <g key={`pawn-${pIdx}`}>
              <circle
                cx={pt.x}
                cy={pt.y}
                r={DEFAULT_GEOMETRY.cellSize * 0.36}
                fill={pColor}
                stroke="#ffffff"
                strokeWidth={2.5}
              />
              <text
                x={pt.x}
                y={pt.y + 5}
                textAnchor="middle"
                fill="#ffffff"
                fontSize={13}
                fontWeight="bold"
              >
                {pText}
              </text>
            </g>
          );
        })}
    </svg>
  );
};
