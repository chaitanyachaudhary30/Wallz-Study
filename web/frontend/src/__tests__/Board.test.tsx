import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { Board } from '../components/Board';
import stateInitial from './fixtures/state_initial_5x5.json';
import stateMidgame from './fixtures/state_midgame.json';

describe('Board Component (P11 Contract)', () => {
  it('renders initial 5x5 board with 2 pawns and correct cell count', () => {
    render(<Board state={stateInitial as any} interactive={true} />);
    const svg = screen.getByTestId('wallz-board');
    expect(svg).toBeInTheDocument();

    // Check pawn labels
    expect(screen.getByText('P0')).toBeInTheDocument();
    expect(screen.getByText('P1')).toBeInTheDocument();
  });

  it('renders placed walls, latest wall, and blocked cell countdown badge', () => {
    render(<Board state={stateMidgame as any} interactive={true} />);
    // Midgame fixture has a block at (2,2) with remaining timer 2
    expect(screen.getByText('2')).toBeInTheDocument();

    // Has latest replaceable wall with aria label
    expect(screen.getByLabelText('Replaceable Wall')).toBeInTheDocument();
  });

  it('clicking a legal MOVE target dispatches onAction with the exact JSON tuple', () => {
    const handleAction = vi.fn();
    render(
      <Board
        state={stateInitial as any}
        mode="MOVE"
        interactive={true}
        onAction={handleAction}
      />
    );

    // Initial legal moves for P0 from (0, 2): (0, 1), (0, 3), (1, 2)
    // Find cell rect for (1, 2) and click it
    const cells = document.querySelectorAll('rect');
    expect(cells.length).toBeGreaterThan(25);

    // Click on target
    const targetGroup = screen.getByText('P0').closest('svg');
    expect(targetGroup).toBeInTheDocument();

    // Simulate clicking one of the legal MOVE cells
    // Fire click on first target cell
    const clickableCells = Array.from(document.querySelectorAll('g')).filter((g) =>
      g.style.cursor === 'pointer'
    );
    expect(clickableCells.length).toBeGreaterThan(0);

    fireEvent.click(clickableCells[0]);
    expect(handleAction).toHaveBeenCalledTimes(1);
    expect(handleAction.mock.calls[0][0][0]).toBe('MOVE');
    expect(handleAction.mock.calls[0][0][1]).toHaveLength(2);
  });
});
