import { describe, expect, it } from 'vitest';
import { cellRect, wallSlotRect } from '../lib/geometry';

describe('Board Geometry (C-GEOM contract)', () => {
  it('places row 0 at the bottom for all N in {3, 5, 9}', () => {
    for (const n of [3, 5, 9]) {
      const row0 = cellRect(0, 0, n);
      const rowTop = cellRect(n - 1, 0, n);
      expect(row0.y).toBeGreaterThan(rowTop.y);
    }
  });

  it('aligns horizontal wall slot between row r and row r+1 spanning columns c and c+1', () => {
    for (const n of [3, 9]) {
      for (let r = 0; r < n - 1; r++) {
        for (let c = 0; c < n - 1; c++) {
          const wSlot = wallSlotRect(r, c, 'H', n);
          const cBottom = cellRect(r, c, n);
          const cTop = cellRect(r + 1, c, n);
          const cRight = cellRect(r, c + 1, n);

          // Horizontal wall is between cTop and cBottom
          expect(wSlot.y).toBe(cTop.y + cTop.height);
          expect(wSlot.y + wSlot.height).toBe(cBottom.y);

          // X coordinates match column c left and column c+1 right
          expect(wSlot.x).toBe(cBottom.x);
          expect(wSlot.x + wSlot.width).toBe(cRight.x + cRight.width);
        }
      }
    }
  });

  it('aligns vertical wall slot between column c and column c+1 spanning rows r and r+1', () => {
    for (const n of [3, 9]) {
      for (let r = 0; r < n - 1; r++) {
        for (let c = 0; c < n - 1; c++) {
          const wSlot = wallSlotRect(r, c, 'V', n);
          const cLeft = cellRect(r, c, n);
          const cRight = cellRect(r, c + 1, n);
          const cTop = cellRect(r + 1, c, n);

          // Vertical wall is between cLeft and cRight
          expect(wSlot.x).toBe(cLeft.x + cLeft.width);
          expect(wSlot.x + wSlot.width).toBe(cRight.x);

          // Y coordinates match row r+1 top and row r bottom
          expect(wSlot.y).toBe(cTop.y);
          expect(wSlot.y + wSlot.height).toBe(cLeft.y + cLeft.height);
        }
      }
    }
  });
});
