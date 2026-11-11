import React from 'react';
import { TerminalState } from '../api/types';

interface TerminalBannerProps {
  terminal: TerminalState;
}

export const TerminalBanner: React.FC<TerminalBannerProps> = ({ terminal }) => {
  if (!terminal.over) return null;

  let msg = 'Game Over';
  let bannerClass = 'terminal-banner';

  if (terminal.winner !== null) {
    bannerClass += terminal.winner === 0 ? ' win-p0' : ' win-p1';
    const reasonText = terminal.reason === 'goal' ? 'reached goal line' : terminal.reason;
    msg = `Player ${terminal.winner} Wins (${reasonText})!`;
  } else if (terminal.reason === 'truncated') {
    // The demo's ply cap stops the game; the team reports this apart from rule draws
    bannerClass += ' draw';
    msg = 'Stopped — ply cap reached (truncated, not a rule draw)';
  } else {
    bannerClass += ' draw';
    if (terminal.reason === 'no_legal_actions') {
      msg = 'Draw — the player to move has no legal action';
    } else if (terminal.reason === 'repetition') {
      msg = 'Draw — the same position repeated (repetition limit reached)';
    } else {
      msg = `Game Drawn (${terminal.reason || 'rules'})`;
    }
  }

  return (
    <div className={bannerClass} data-testid="terminal-banner">
      {msg}
    </div>
  );
};
