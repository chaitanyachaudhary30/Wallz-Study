import { OverlayItem } from '../api/types';

export interface OverlayLookup {
  cells: Map<string, OverlayItem>; // "r,c" -> item
  walls: Map<string, OverlayItem>; // "r,c,o" -> item
  scale: number; // max |value| over the overlay, so every agent's values use the full colour range
}

export function buildOverlayLookup(items?: OverlayItem[]): OverlayLookup {
  const cells = new Map<string, OverlayItem>();
  const walls = new Map<string, OverlayItem>();

  if (!items) return { cells, walls, scale: 1 };

  let scale = 0;
  for (const item of items) {
    // Forced wins/losses (minimax 'win in k', solver 'WIN in k') get fixed colours and
    // would otherwise swamp the scale of ordinary evaluations.
    if (isForced(item)) continue;
    if (item.value !== null && Number.isFinite(item.value)) scale = Math.max(scale, Math.abs(item.value));
  }
  if (scale === 0) scale = 1;

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

  return { cells, walls, scale };
}

function isForced(item: OverlayItem): boolean {
  const label = item.label.toLowerCase();
  return label.startsWith('win') || label.startsWith('loss');
}

export function getHeatmapColor(item?: OverlayItem, scale: number = 3.0): string | null {
  if (!item) return null;

  if (isForced(item)) {
    return item.label.toLowerCase().startsWith('win') ? '#f59e0b' : 'rgba(239, 68, 68, 0.9)';
  }

  if (item.value === null) {
    const label = item.label.toLowerCase();
    if (label.includes('win')) {
      return '#f59e0b'; // Gold for winning moves
    }
    if (label.includes('loss')) {
      return 'rgba(239, 68, 68, 0.9)'; // Strong red for immediately losing moves
    }
    return '#94a3b8'; // Grey for unanalysed
  }

  const v = item.value;
  if (v > 0) {
    // Green scale
    const intensity = Math.min(1, Math.max(0.2, v / scale));
    return `rgba(16, 185, 129, ${intensity.toFixed(2)})`;
  } else if (v < 0) {
    // Red scale
    const intensity = Math.min(1, Math.max(0.2, Math.abs(v) / scale));
    return `rgba(239, 68, 68, ${intensity.toFixed(2)})`;
  } else {
    // Neutral grey
    return 'rgba(148, 163, 184, 0.4)';
  }
}
