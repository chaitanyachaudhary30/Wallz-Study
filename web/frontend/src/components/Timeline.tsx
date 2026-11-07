import React from 'react';
import { HistoryEntry } from '../api/types';

interface TimelineProps {
  history: HistoryEntry[];
  onUndo?: () => void;
  canUndo?: boolean;
}

export const Timeline: React.FC<TimelineProps> = ({ history, onUndo, canUndo = false }) => {
  const formatAction = (entry: HistoryEntry) => {
    const [actType, val] = entry.action;
    if (actType === 'MOVE' || actType === 'HOP' || actType === 'BLOCK') {
      return `${actType} (${val[0]}, ${val[1]})`;
    } else if (actType === 'WALL' || actType === 'REPLACE') {
      return `${actType} (${val[0]}, ${val[1]}, ${val[2]})`;
    }
    return String(entry.action);
  };

  return (
    <div className="card" style={{ marginTop: '1rem' }} data-testid="timeline">
      <div className="card-title">
        <span>Action Timeline</span>
        {onUndo && (
          <button
            className="btn"
            style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}
            onClick={onUndo}
            disabled={!canUndo}
          >
            Undo
          </button>
        )}
      </div>

      <div className="timeline-list">
        {history.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>
            No moves yet.
          </div>
        ) : (
          history.map((entry, idx) => (
            <div
              key={`ply-${idx}`}
              className={`timeline-item ${entry.player === 0 ? 'p0' : 'p1'}`}
            >
              <span>
                <strong>{entry.ply}.</strong> P{entry.player}: {formatAction(entry)}
              </span>
              {entry.decision_seconds > 0 && (
                <span style={{ color: 'var(--text-muted)' }}>
                  {(entry.decision_seconds * 1000).toFixed(1)}ms
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
