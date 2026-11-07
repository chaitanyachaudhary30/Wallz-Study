import React, { useState } from 'react';
import { Explanation } from '../api/types';

interface ExplanationPanelProps {
  explanation: Explanation | null;
  showOverlay: boolean;
  onToggleOverlay: (show: boolean) => void;
  onAnalyze?: (agentSpec: string) => void;
  isHumanTurn?: boolean;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
  explanation,
  showOverlay,
  onToggleOverlay,
  onAnalyze,
  isHumanTurn = false,
}) => {
  const [selectedAgent, setSelectedAgent] = useState('greedy');

  if (!explanation) {
    return (
      <div className="card" style={{ marginTop: '1rem' }} data-testid="explanation-panel">
        <div className="card-title">Model Thinking</div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          No model decision explanation yet.
        </p>
        {isHumanTurn && onAnalyze && (
          <div style={{ marginTop: '0.75rem' }}>
            <label className="form-label">Ask AI Hint:</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <select
                className="form-select"
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
              >
                <option value="greedy">Greedy</option>
                <option value="minimax:depth=2">Minimax (Depth 2)</option>
              </select>
              <button
                className="btn btn-primary"
                onClick={() => onAnalyze(selectedAgent)}
              >
                Analyze
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  const { agent, summary, details } = explanation;

  return (
    <div className="card" style={{ marginTop: '1rem' }} data-testid="explanation-panel">
      <div className="card-title">
        <span>How {agent} Thinks</span>
        <label
          style={{
            fontSize: '0.75rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={showOverlay}
            onChange={(e) => onToggleOverlay(e.target.checked)}
          />
          Heatmap Overlay
        </label>
      </div>

      <div
        style={{
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderLeft: '3px solid var(--primary)',
          padding: '0.5rem 0.75rem',
          borderRadius: '4px',
          fontSize: '0.85rem',
          marginBottom: '0.75rem',
        }}
      >
        {summary}
      </div>

      {/* Greedy View */}
      {details?.objective && (
        <div style={{ fontSize: '0.8rem' }}>
          <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
            Objective: {details.objective}
          </div>
          {details.rows && details.rows.length > 0 && (
            <div style={{ maxHeight: '140px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-color)' }}>
                    <th style={{ textAlign: 'left', padding: '2px' }}>Action</th>
                    <th style={{ textAlign: 'right', padding: '2px' }}>H</th>
                    <th style={{ textAlign: 'right', padding: '2px' }}>d_self</th>
                    <th style={{ textAlign: 'right', padding: '2px' }}>d_opp</th>
                  </tr>
                </thead>
                <tbody>
                  {details.rows.slice(0, 10).map((r: any, idx: number) => (
                    <tr
                      key={idx}
                      style={{
                        backgroundColor: idx === 0 ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      }}
                    >
                      <td style={{ padding: '2px' }}>
                        {r.action[0]} {JSON.stringify(r.action[1])}
                      </td>
                      <td style={{ textAlign: 'right', padding: '2px' }}>
                        {r.immediate_win ? 'Win' : r.H !== null ? r.H : '—'}
                      </td>
                      <td style={{ textAlign: 'right', padding: '2px' }}>{r.d_self ?? '—'}</td>
                      <td style={{ textAlign: 'right', padding: '2px' }}>{r.d_opp ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Minimax View */}
      {details?.nodes_expanded !== undefined && (
        <div style={{ fontSize: '0.8rem' }}>
          <div className="stat-row">
            <span>Depth:</span>
            <span className="stat-val">{details.depth_completed}</span>
          </div>
          <div className="stat-row">
            <span>Nodes Expanded:</span>
            <span className="stat-val">{details.nodes_expanded.toLocaleString()}</span>
          </div>
          <div className="stat-row">
            <span>Alpha-Beta Cutoffs:</span>
            <span className="stat-val">{details.cutoffs}</span>
          </div>
          <div className="stat-row">
            <span>Root Eval:</span>
            <span className="stat-val">{details.root_value}</span>
          </div>

          {details.features_chosen && Object.keys(details.features_chosen).length > 0 && (
            <div style={{ marginTop: '0.5rem' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.2rem' }}>
                Child Features:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.25rem', fontSize: '0.72rem' }}>
                {Object.entries(details.features_chosen).map(([k, v]) => (
                  <div key={k} style={{ color: 'var(--text-muted)' }}>
                    {k.replace('f', 'F')}: <strong style={{ color: 'var(--text-main)' }}>{String(v)}</strong>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* DQN View */}
      {details?.q_max !== undefined && (
        <div style={{ fontSize: '0.8rem' }}>
          <div className="stat-row">
            <span>Chosen Q:</span>
            <span className="stat-val">{details.q_chosen}</span>
          </div>
          <div className="stat-row">
            <span>Max Q:</span>
            <span className="stat-val">{details.q_max}</span>
          </div>
          {details.top5 && (
            <div style={{ marginTop: '0.4rem' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-muted)' }}>Top Actions:</div>
              {details.top5.map((t: any, i: number) => (
                <div key={i} className="stat-row" style={{ fontSize: '0.75rem' }}>
                  <span>{t.action[0]} {JSON.stringify(t.action[1])}:</span>
                  <span className="stat-val">{t.q}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Exact View */}
      {details?.status && (
        <div style={{ fontSize: '0.8rem' }}>
          <div className="stat-row">
            <span>Game Status:</span>
            <span
              className="stat-val"
              style={{
                color:
                  details.status === 'WIN'
                    ? '#10b981'
                    : details.status === 'LOSS'
                    ? '#ef4444'
                    : '#a855f7',
              }}
            >
              {details.status}
            </span>
          </div>
          {details.rank !== null && (
            <div className="stat-row">
              <span>Distance to Terminal:</span>
              <span className="stat-val">{details.rank} plies</span>
            </div>
          )}
          <div className="stat-row">
            <span>States in Graph:</span>
            <span className="stat-val">{details.states_in_graph?.toLocaleString()}</span>
          </div>
        </div>
      )}

      {isHumanTurn && onAnalyze && (
        <div style={{ marginTop: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem' }}>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <select
              className="form-select"
              value={selectedAgent}
              onChange={(e) => setSelectedAgent(e.target.value)}
              style={{ fontSize: '0.75rem', padding: '0.2rem' }}
            >
              <option value="greedy">Greedy</option>
              <option value="minimax:depth=2">Minimax (Depth 2)</option>
            </select>
            <button
              className="btn btn-primary"
              style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
              onClick={() => onAnalyze(selectedAgent)}
            >
              Ask AI
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
