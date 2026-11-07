import React from 'react';
import { GameState } from '../api/types';

interface PlayerCardProps {
  player: 0 | 1;
  name: string;
  state: GameState;
}

export const PlayerCard: React.FC<PlayerCardProps> = ({ player, name, state }) => {
  const isToMove = state.to_move === player && !state.terminal.over;
  const inv = state.inventory[player];
  const totalM = state.rules.M;
  const hopOk = state.hop[player];
  const blockOk = state.block[player];
  const path = state.paths ? state.paths[player] : [];
  const dist = path && path.length > 0 ? path.length - 1 : '?';

  return (
    <div
      className={`card player-card ${
        isToMove ? (player === 0 ? 'active-p0' : 'active-p1') : ''
      }`}
      data-testid={`player-card-${player}`}
    >
      <div className="card-title">
        <span>
          <strong style={{ color: player === 0 ? 'var(--p0-color)' : 'var(--p1-color)' }}>
            Player {player}
          </strong>{' '}
          ({name})
        </span>
        {isToMove && <span className="btn btn-primary" style={{ padding: '0.15rem 0.5rem', fontSize: '0.75rem' }}>Turn</span>}
      </div>

      <div className="stat-row">
        <span>Wall Inventory:</span>
        <span className="stat-val">
          {inv} / {totalM}
        </span>
      </div>

      <div className="stat-row">
        <span>Hop Available:</span>
        <span className="stat-val" style={{ color: hopOk ? '#10b981' : '#ef4444' }}>
          {hopOk ? 'Yes' : 'Used'}
        </span>
      </div>

      <div className="stat-row">
        <span>Block Available:</span>
        <span className="stat-val" style={{ color: blockOk ? '#10b981' : '#ef4444' }}>
          {blockOk ? 'Yes' : 'Used'}
        </span>
      </div>

      <div className="stat-row">
        <span>Shortest Path:</span>
        <span className="stat-val">{dist} steps</span>
      </div>
    </div>
  );
};
