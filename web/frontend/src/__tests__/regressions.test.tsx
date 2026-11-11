import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Board } from '../components/Board';
import { buildOverlayLookup, getHeatmapColor } from '../components/Heatmap';
import { TerminalBanner } from '../components/TerminalBanner';
import stateInitial from './fixtures/state_initial_5x5.json';

describe('Regression fixes (bug hunt 2026-11-08)', () => {
  it('heatmap scale adapts to the overlay range (DQN Q-values in [-1, 1] are visible)', () => {
    const items = [
      { action: ['MOVE', [1, 2]] as any, value: 0.05, label: 'Q=+0.050' },
      { action: ['MOVE', [0, 1]] as any, value: -0.02, label: 'Q=-0.020' },
    ];
    const lookup = buildOverlayLookup(items);
    expect(lookup.scale).toBeCloseTo(0.05);
    expect(getHeatmapColor(items[0], lookup.scale)).toBe('rgba(16, 185, 129, 1.00)');
  });

  it('forced wins/losses do not dominate the scale and get fixed colours', () => {
    const items = [
      { action: ['MOVE', [1, 2]] as any, value: 99996, label: 'win in 4' },
      { action: ['MOVE', [0, 1]] as any, value: 2, label: '+2.00' },
      { action: ['MOVE', [0, 3]] as any, value: null, label: 'loss' },
    ];
    const lookup = buildOverlayLookup(items);
    expect(lookup.scale).toBe(2);
    expect(getHeatmapColor(items[0], lookup.scale)).toBe('#f59e0b');
    expect(getHeatmapColor(items[2], lookup.scale)).toBe('rgba(239, 68, 68, 0.9)');
  });

  it('shortest-path lines and pawns never intercept clicks', () => {
    const { container } = render(<Board state={stateInitial as any} interactive={true} showPaths={true} />);
    container.querySelectorAll('polyline').forEach((pl) => expect(pl.getAttribute('pointer-events')).toBe('none'));
    const pawnGroup = screen.getByText('P0').closest('g');
    expect(pawnGroup?.getAttribute('pointer-events')).toBe('none');
  });

  it('a ply-cap stop is shown as truncated, not as a draw', () => {
    render(<TerminalBanner terminal={{ over: true, winner: null, reason: 'truncated' }} />);
    const text = screen.getByTestId('terminal-banner').textContent || '';
    expect(text).toContain('truncated');
    expect(text).not.toMatch(/^Draw/);
  });

  it('in Wall-H mode only horizontal slots accept clicks', () => {
    const { container } = render(<Board state={stateInitial as any} interactive={true} mode={'WALL_H'} />);
    const slots = Array.from(container.querySelectorAll('rect[pointer-events]'));
    const clickable = slots.filter((r) => r.getAttribute('pointer-events') === 'all');
    expect(clickable.length).toBeGreaterThan(0);
    // horizontal slots are wider than tall
    clickable.forEach((r) => expect(Number(r.getAttribute('width'))).toBeGreaterThan(Number(r.getAttribute('height'))));
  });
});
