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
  } else {
    bannerClass += ' draw';
    if (terminal.reason === 'no_legal_actions') {
      msg = 'Draw — No legal moves available (Rule R3)';
    } else if (terminal.reason === 'repetition') {
      msg = 'Draw — Fivefold state repetition reached';
    } else if (terminal.reason === 'truncated') {
      msg = 'Draw — Maximum ply cap reached';
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
