import { OverlayItem } from '../api/types';

export interface OverlayLookup {
  cells: Map<string, OverlayItem>; // "r,c" -> item
  walls: Map<string, OverlayItem>; // "r,c,o" -> item
}

export function buildOverlayLookup(items?: OverlayItem[]): OverlayLookup {
  const cells = new Map<string, OverlayItem>();
  const walls = new Map<string, OverlayItem>();

  if (!items) return { cells, walls };

  for (const item of items) {
    const [actType, val] = item.action;
    if (actType === 'MOVE' || actType === 'HOP' || actType === 'BLOCK') {
      const key = `${val[0]},${val[1]}`;
      // In case of multiple actions on same cell, prefer the highest value
      const existing = cells.get(key);
      if (!existing || (item.value !== null && (existing.value === null || item.value > existing.value))) {
        cells.set(key, item);
      }
    } else if (actType === 'WALL' || actType === 'REPLACE') {
      const [r, c, o] = val;
      const key = `${r},${c},${o}`;
      walls.set(key, item);
    }
  }

  return { cells, walls };
}

export function getHeatmapColor(item?: OverlayItem): string | null {
  if (!item) return null;

  if (item.value === null) {
    if (item.label.toLowerCase().includes('win')) {
      return '#f59e0b'; // Gold for winning moves
    }
    return '#94a3b8'; // Grey for unanalysed
  }

  const v = item.value;
  if (v > 0) {
    // Green scale
    const intensity = Math.min(1, Math.max(0.2, v / 3.0));
    return `rgba(16, 185, 129, ${intensity.toFixed(2)})`;
  } else if (v < 0) {
    // Red scale
    const intensity = Math.min(1, Math.max(0.2, Math.abs(v) / 3.0));
    return `rgba(239, 68, 68, ${intensity.toFixed(2)})`;
  } else {
    // Neutral grey
    return 'rgba(148, 163, 184, 0.4)';
  }
}
