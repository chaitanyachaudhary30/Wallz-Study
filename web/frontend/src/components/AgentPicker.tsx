import React from 'react';
import { AgentInfo } from '../api/types';

interface AgentPickerProps {
  agents: AgentInfo[];
  p0: string;
  p1: string;
  onSelectP0: (spec: string) => void;
  onSelectP1: (spec: string) => void;
  disabled?: boolean;
}

export const AgentPicker: React.FC<AgentPickerProps> = ({
  agents,
  p0,
  p1,
  onSelectP0,
  onSelectP1,
  disabled = false,
}) => {
  return (
    <div className="card" style={{ marginTop: '1rem' }} data-testid="agent-picker">
      <div className="card-title">Players & Agents</div>

      <div className="form-group">
        <label className="form-label" style={{ color: 'var(--p0-color)' }}>
          Player 0 (Bottom)
        </label>
        <select
          className="form-select"
          value={p0}
          onChange={(e) => onSelectP0(e.target.value)}
          disabled={disabled}
        >
          {agents.map((ag) => (
            <option
              key={`p0-${ag.spec}`}
              value={ag.spec}
              disabled={!ag.available}
              title={ag.reason || undefined}
            >
              {ag.label} {!ag.available ? `(${ag.reason})` : ''}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label className="form-label" style={{ color: 'var(--p1-color)' }}>
          Player 1 (Top)
        </label>
        <select
          className="form-select"
          value={p1}
          onChange={(e) => onSelectP1(e.target.value)}
          disabled={disabled}
        >
          {agents.map((ag) => (
            <option
              key={`p1-${ag.spec}`}
              value={ag.spec}
              disabled={!ag.available}
              title={ag.reason || undefined}
            >
              {ag.label} {!ag.available ? `(${ag.reason})` : ''}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};
