import React, { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../api/client';

export const TrainingPage: React.FC = () => {
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [runData, setRunData] = useState<any | null>(null);
  const [summaryData, setSummaryData] = useState<any | null>(null);
  const [trainerConfig, setTrainerConfig] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [liveStatus, setLiveStatus] = useState<any | null>(null);
  const [liveStarting, setLiveStarting] = useState<boolean>(false);

  useEffect(() => {
    Promise.all([
      api.fetchTrainingRuns(),
      api.fetchTrainingSummary(),
      api.fetchTrainingConfig(),
      api.fetchLiveStatus(),
    ])
      .then(([runsList, summary, cfg, live]) => {
        setRuns(runsList);
        setSummaryData(summary?.summary || {});
        setTrainerConfig(cfg);
        setLiveStatus(live);
        if (runsList.length > 0) {
          const defaultRun = runsList.find((r: any) => r.has_log) || runsList[0];
          setSelectedRunId(defaultRun.id);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    api.fetchTrainingRun(selectedRunId)
      .then((data) => setRunData(data))
      .catch(console.error);
  }, [selectedRunId]);

  // Combine curve arrays into Recharts data points
  const chartData = React.useMemo(() => {
    if (!runData?.curves?.env_steps) return [];
    const steps = runData.curves.env_steps;
    const wr = runData.curves.win_rate_rolling100 || [];
    const loss = runData.curves.avg_loss || [];
    const eps = runData.curves.epsilon || [];

    return steps.map((s: number, i: number) => ({
      step: s,
      winRate: wr[i] !== null ? wr[i] * 100 : null,
      loss: loss[i],
      epsilon: eps[i],
    }));
  }, [runData]);

  const handleStartLive = async () => {
    setLiveStarting(true);
    try {
      await api.startLiveTraining(2000, 42);
      const st = await api.fetchLiveStatus();
      setLiveStatus(st);
    } catch (err: any) {
      alert(err.message || 'Failed to start live training');
    } finally {
      setLiveStarting(false);
    }
  };

  const handleStopLive = async () => {
    try {
      await api.stopLiveTraining();
      const st = await api.fetchLiveStatus();
      setLiveStatus(st);
    } catch (err: any) {
      alert(err.message || 'Failed to stop live training');
    }
  };

  if (loading) {
    return <div className="main-content">Loading training metrics...</div>;
  }

  return (
    <div className="main-content" data-testid="training-page">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h2>DQN Reinforcement Learning & Empirical Verification</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label className="form-label" style={{ margin: 0 }}>Select Run:</label>
          <select
            className="form-select"
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            style={{ width: '240px' }}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} {r.release ? '★ (Release)' : ''} {!r.has_log ? '(weights only)' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Win Rate Learning Curve */}
        <div className="card">
          <div className="card-title">Rolling Win Rate (%)</div>
          <div style={{ width: '100%', height: 260 }}>
            {chartData.length > 0 ? (
              <ResponsiveContainer>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <YAxis stroke="#94a3b8" domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                  <Legend />
                  <Line type="monotone" dataKey="winRate" stroke="#10b981" strokeWidth={2} dot={false} name="Win Rate %" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ textAlign: 'center', padding: '3rem', color: '#94a3b8' }}>
                No episode log curves available for this run.
              </div>
            )}
          </div>
        </div>

        {/* TD-Loss Curve */}
        <div className="card">
          <div className="card-title">Average TD-Loss (Huber)</div>
          <div style={{ width: '100%', height: 260 }}>
            {chartData.length > 0 ? (
              <ResponsiveContainer>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                  <Legend />
                  <Line type="monotone" dataKey="loss" stroke="#38bdf8" strokeWidth={2} dot={false} name="Loss" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ textAlign: 'center', padding: '3rem', color: '#94a3b8' }}>
                No loss records for this run.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Live Mini-Training Card (P15) */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="card-title" style={{ margin: 0 }}>Interactive Live Mini-Training (N=3)</div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', margin: '0.25rem 0 0 0' }}>
              Launch a live Double-DQN self-play training run in the background. Stream training logs and inspect checkpoints.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            {liveStatus?.active ? (
              <>
                <span style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 600 }}>
                  ● Training Active ({liveStatus.elapsed_seconds || 0}s)
                </span>
                <button className="btn" onClick={handleStopLive}>
                  Stop Training
                </button>
              </>
            ) : (
              <button
                className="btn btn-primary"
                disabled={liveStarting}
                onClick={handleStartLive}
              >
                {liveStarting ? 'Starting...' : '⚡ Start 2,000-Step Mini-Training'}
              </button>
            )}
          </div>
        </div>

        {liveStatus?.checkpoints && liveStatus.checkpoints.length > 0 && (
          <div style={{ marginTop: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem', fontSize: '0.85rem' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-color)' }}>Generated Checkpoints: </span>
            <span style={{ color: 'var(--primary)', fontFamily: 'monospace' }}>
              {liveStatus.checkpoints.join(', ')}
            </span>
          </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Architecture & Run Meta */}
        <div className="card">
          <div className="card-title">Network Architecture & Meta</div>
          {runData?.meta ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.85rem' }}>
              <div className="stat-row"><span>Board Size:</span> <span className="stat-val">{runData.meta.N} × {runData.meta.N}</span></div>
              <div className="stat-row"><span>Residual Blocks:</span> <span className="stat-val">{runData.meta.blocks}</span></div>
              <div className="stat-row"><span>Conv Channels:</span> <span className="stat-val">{runData.meta.channels}</span></div>
              <div className="stat-row"><span>Hidden Dimension:</span> <span className="stat-val">{runData.meta.hidden}</span></div>
              <div className="stat-row"><span>Double DQN:</span> <span className="stat-val">{String(runData.meta.double)}</span></div>
              <div className="stat-row"><span>Discount (γ):</span> <span className="stat-val">{runData.meta.gamma}</span></div>
              <div className="stat-row"><span>Codec Version:</span> <span className="stat-val">v{runData.meta.codec_version}</span></div>
              <div className="stat-row"><span>Obs Version:</span> <span className="stat-val">v{runData.meta.obs_version}</span></div>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>Metadata unavailable.</p>
          )}

          {runData?.opponents && (
            <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem' }}>
              <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.4rem' }}>Self-Play Opponent Pool Distribution</div>
              <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem' }}>
                <div>Current: <strong>{runData.opponents.current}</strong></div>
                <div>Snapshot: <strong>{runData.opponents.snapshot}</strong></div>
                <div>Greedy: <strong>{runData.opponents.greedy}</strong></div>
              </div>
            </div>
          )}
        </div>

        {/* Canonical Hyperparameters */}
        <div className="card">
          <div className="card-title">Canonical DQN Trainer Hyperparameters</div>
          {trainerConfig ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem', fontSize: '0.825rem' }}>
              <div className="stat-row"><span>Learning Rate:</span> <span className="stat-val">{trainerConfig.lr}</span></div>
              <div className="stat-row"><span>Batch Size:</span> <span className="stat-val">{trainerConfig.batch_size}</span></div>
              <div className="stat-row"><span>Buffer Capacity:</span> <span className="stat-val">{trainerConfig.buffer_capacity?.toLocaleString()}</span></div>
              <div className="stat-row"><span>Target Update Freq:</span> <span className="stat-val">{trainerConfig.target_update_freq} steps</span></div>
              <div className="stat-row"><span>Epsilon Range:</span> <span className="stat-val">{trainerConfig.epsilon_start} → {trainerConfig.epsilon_end}</span></div>
              <div className="stat-row"><span>Decay Steps:</span> <span className="stat-val">{trainerConfig.epsilon_decay_steps?.toLocaleString()}</span></div>
              <div className="stat-row"><span>Gradient Clip:</span> <span className="stat-val">{trainerConfig.grad_clip}</span></div>
              <div className="stat-row"><span>Huber Delta:</span> <span className="stat-val">{trainerConfig.huber_delta}</span></div>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>Trainer config unavailable.</p>
          )}

          {summaryData?.greedy_campaign && (
            <div style={{ marginTop: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem', fontSize: '0.8rem' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.2rem' }}>Baseline Campaign Summary:</div>
              <div>Greedy matches: <strong>{summaryData.greedy_campaign.campaign_results?.length || 9}</strong> configurations evaluated.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
