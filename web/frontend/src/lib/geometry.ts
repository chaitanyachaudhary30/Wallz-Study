/**
 * Pure SVG geometry calculations for the Modified Wallz board.
 *
 * NOTE: Drawing only. Single-source guard ensures no game logic or rule checking
 * exists in frontend code.
 */

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface BoardGeometryConfig {
  n: number;
  cellSize: number;
  wallGap: number;
  padding: number;
}

export const DEFAULT_GEOMETRY = {
  cellSize: 50,
  wallGap: 14,
  padding: 24,
};

export function boardDimensions(
  n: number,
  cellSize: number = DEFAULT_GEOMETRY.cellSize,
  wallGap: number = DEFAULT_GEOMETRY.wallGap,
  padding: number = DEFAULT_GEOMETRY.padding
): { width: number; height: number } {
  const inner = n * cellSize + (n - 1) * wallGap;
  const total = inner + 2 * padding;
  return { width: total, height: total };
}

export function cellRect(
  r: number,
  c: number,
  n: number,
  cellSize: number = DEFAULT_GEOMETRY.cellSize,
  wallGap: number = DEFAULT_GEOMETRY.wallGap,
  padding: number = DEFAULT_GEOMETRY.padding
): Rect {
  // Row 0 drawn at the bottom (C-GEOM)
  const rowFromTop = n - 1 - r;
  const x = padding + c * (cellSize + wallGap);
  const y = padding + rowFromTop * (cellSize + wallGap);
  return { x, y, width: cellSize, height: cellSize };
}

export function cellCenter(
  r: number,
  c: number,
  n: number,
  cellSize: number = DEFAULT_GEOMETRY.cellSize,
  wallGap: number = DEFAULT_GEOMETRY.wallGap,
  padding: number = DEFAULT_GEOMETRY.padding
): Point {
  const rect = cellRect(r, c, n, cellSize, wallGap, padding);
  return {
    x: rect.x + rect.width / 2,
    y: rect.y + rect.height / 2,
  };
}

export function wallSlotRect(
  r: number,
  c: number,
  o: 'H' | 'V',
  n: number,
  cellSize: number = DEFAULT_GEOMETRY.cellSize,
  wallGap: number = DEFAULT_GEOMETRY.wallGap,
  padding: number = DEFAULT_GEOMETRY.padding
): Rect {
  if (o === 'H') {
    // Spans horizontally across columns c and c+1, located in the gap between row r and r+1
    const x = padding + c * (cellSize + wallGap);
    const rowFromTop = n - 2 - r; // Row r+1's top index
    const y = padding + rowFromTop * (cellSize + wallGap) + cellSize;
    const width = 2 * cellSize + wallGap;
    const height = wallGap;
    return { x, y, width, height };
  } else {
    // Spans vertically across rows r and r+1, located in the gap between column c and c+1
    const x = padding + c * (cellSize + wallGap) + cellSize;
    const rowFromTop = n - 2 - r;
    const y = padding + rowFromTop * (cellSize + wallGap);
    const width = wallGap;
    const height = 2 * cellSize + wallGap;
    return { x, y, width, height };
  }
}
