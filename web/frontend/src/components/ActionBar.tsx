import React from 'react';
import { GameState } from '../api/types';
import { ActionMode } from './Board';

interface ActionBarProps {
  state: GameState;
  currentMode: ActionMode;
  onSelectMode: (mode: ActionMode) => void;
  disabled?: boolean;
}

export const ActionBar: React.FC<ActionBarProps> = ({
  state,
  currentMode,
  onSelectMode,
  disabled = false,
}) => {
  const legal = state.legal || { MOVE: [], HOP: [], WALL: [], REPLACE: [], BLOCK: [] };

  const nMove = legal.MOVE?.length || 0;
  const nHop = legal.HOP?.length || 0;
  const nWallH = (legal.WALL || []).filter(([, , o]) => o === 'H').length;
  const nWallV = (legal.WALL || []).filter(([, , o]) => o === 'V').length;
  const nReplace = legal.REPLACE?.length || 0;
  const nBlock = legal.BLOCK?.length || 0;

  return (
    <div className="action-bar" data-testid="action-bar">
      {nMove > 0 && (
        <button
          className={`btn ${currentMode === 'MOVE' ? 'active' : ''}`}
          onClick={() => onSelectMode(currentMode === 'MOVE' ? null : 'MOVE')}
          disabled={disabled}
        >
          Move ({nMove})
        </button>
      )}

      {nHop > 0 && (
        <button
          className={`btn ${currentMode === 'HOP' ? 'active' : ''}`}
          onClick={() => onSelectMode(currentMode === 'HOP' ? null : 'HOP')}
          disabled={disabled}
        >
          Hop ({nHop})
        </button>
      )}

      {nWallH > 0 && (
        <button
          className={`btn ${currentMode === 'WALL_H' ? 'active' : ''}`}
          onClick={() => onSelectMode(currentMode === 'WALL_H' ? null : 'WALL_H')}
          disabled={disabled}
        >
          Wall H ({nWallH})
        </button>
      )}

      {nWallV > 0 && (
        <button
          className={`btn ${currentMode === 'WALL_V' ? 'active' : ''}`}
          onClick={() => onSelectMode(currentMode === 'WALL_V' ? null : 'WALL_V')}
          disabled={disabled}
        >
          Wall V ({nWallV})
        </button>
      )}

      {nReplace > 0 && (
        <button
          className={`btn ${currentMode === 'REPLACE' ? 'active' : ''}`}
          onClick={() => onSelectMode(currentMode === 'REPLACE' ? null : 'REPLACE')}
          disabled={disabled}
        >
          Replace ({nReplace})
        </button>
      )}

      {nBlock > 0 && (
        <button
          className={`btn ${currentMode === 'BLOCK' ? 'active' : ''}`}
          onClick={() => onSelectMode(currentMode === 'BLOCK' ? null : 'BLOCK')}
          disabled={disabled}
        >
          Block ({nBlock})
        </button>
      )}
    </div>
  );
};
