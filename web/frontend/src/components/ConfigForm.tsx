import React from 'react';
import { ConfigDefaults, GameConfig } from '../api/types';

interface ConfigFormProps {
  config: GameConfig;
  defaults?: ConfigDefaults | null;
  onChange: (cfg: GameConfig) => void;
  onNewGame: () => void;
  disabled?: boolean;
}

export const ConfigForm: React.FC<ConfigFormProps> = ({
  config,
  defaults,
  onChange,
  onNewGame,
  disabled = false,
}) => {
  const handlePresetSelect = (presetId: string) => {
    const preset = defaults?.presets?.find((p) => p.id === presetId);
    if (preset) {
      onChange({ ...config, ...preset.config });
    }
  };

  return (
    <div className="card" data-testid="config-form">
      <div className="card-title">Game Configuration</div>

      {defaults?.presets && defaults.presets.length > 0 && (
        <div className="form-group">
          <label className="form-label">Presets</label>
          <select
            className="form-select"
            onChange={(e) => handlePresetSelect(e.target.value)}
            disabled={disabled}
            defaultValue=""
          >
            <option value="" disabled>
              Select preset...
            </option>
            {defaults.presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="form-group">
        <label className="form-label">Board Size (N)</label>
        <select
          className="form-select"
          value={config.board_size}
          onChange={(e) => onChange({ ...config, board_size: Number(e.target.value) })}
          disabled={disabled}
        >
          {[3, 4, 5, 7, 9].map((size) => (
            <option key={size} value={size}>
              {size} × {size}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label className="form-label">Wall Inventory (M)</label>
        <input
          type="number"
          className="form-input"
          min={0}
          max={20}
          value={config.M}
          onChange={(e) => onChange({ ...config, M: Number(e.target.value) })}
          disabled={disabled}
        />
      </div>

      <div className="form-group">
        <label className="form-label">Block Duration (K)</label>
        <input
          type="number"
          className="form-input"
          min={1}
          max={9}
          value={config.K}
          onChange={(e) => onChange({ ...config, K: Number(e.target.value) })}
          disabled={disabled}
        />
      </div>

      <div className="form-group" style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
        <label style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <input
            type="checkbox"
            checked={config.A}
            onChange={(e) => onChange({ ...config, A: e.target.checked })}
            disabled={disabled}
          />
          A (Replace)
        </label>
        <label style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <input
            type="checkbox"
            checked={config.B}
            onChange={(e) => onChange({ ...config, B: e.target.checked })}
            disabled={disabled}
          />
          B (Hop)
        </label>
        <label style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <input
            type="checkbox"
            checked={config.C}
            onChange={(e) => onChange({ ...config, C: e.target.checked })}
            disabled={disabled}
          />
          C (Block)
        </label>
      </div>

      <button
        className="btn btn-primary"
        style={{ width: '100%', marginTop: '0.75rem' }}
        onClick={onNewGame}
        disabled={disabled}
      >
        Start New Game
      </button>
    </div>
  );
};
