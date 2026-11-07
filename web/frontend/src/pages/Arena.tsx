import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { AgentInfo, GameConfig, OverlayItem, SessionData } from '../api/types';
import { Board } from '../components/Board';
import { TerminalBanner } from '../components/TerminalBanner';

export const ArenaPage: React.FC = () => {
  const [config, setConfig] = useState<GameConfig>({
    board_size: 5,
    A: true,
    B: true,
    C: true,
    K: 3,
    M: 10,
  });
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [p0, setP0] = useState<string>('greedy');
  const [p1, setP1] = useState<string>('random:seed=1');

  const [session, setSession] = useState<SessionData | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speedMs, setSpeedMs] = useState<number>(400);

  const [batchN, setBatchN] = useState<number>(10);
  const [batchResults, setBatchResults] = useState<any | null>(null);
  const [isBatchRunning, setIsBatchRunning] = useState<boolean>(false);

  const [p0Explanation, setP0Explanation] = useState<any | null>(null);
  const [p1Explanation, setP1Explanation] = useState<any | null>(null);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);

  // Load available agents
  useEffect(() => {
    api.fetchAgents(config).then((list) => {
      const nonHuman = list.filter((a) => a.family !== 'human');
      setAgents(nonHuman);
      if (!nonHuman.find((a) => a.spec === p0)?.available) setP0('greedy');
      if (!nonHuman.find((a) => a.spec === p1)?.available) setP1('random:seed=1');
    }).catch(console.error);
  }, [config]);

  // Start a fresh arena game
  const initGame = async () => {
    setIsPlaying(false);
    setP0Explanation(null);
    setP1Explanation(null);
    try {
      const newSession = await api.createGame(config, p0, p1);
      setSession(newSession);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    initGame();
  }, [config, p0, p1]); // eslint-disable-line react-hooks/exhaustive-deps

  // Step game once
  const stepOnce = async () => {
    if (!session || session.state.terminal.over) return;
    try {
      const mover = session.state.to_move;
      const res = await api.stepGame(session.id, true);
      setSession(res.session);
      if (res.explanation) {
        if (mover === 0) setP0Explanation(res.explanation);
        else setP1Explanation(res.explanation);
      }
    } catch (err) {
      console.error(err);
      setIsPlaying(false);
    }
  };

  // Autoplay loop
  useEffect(() => {
    if (!isPlaying || !session || session.state.terminal.over) return;
    const timer = setTimeout(stepOnce, speedMs);
    return () => clearTimeout(timer);
  }, [isPlaying, session, speedMs]); // eslint-disable-line react-hooks/exhaustive-deps

  // Run batch tournament
  const handleRunBatch = async () => {
    setIsBatchRunning(true);
    setBatchResults(null);
    try {
      const results = await api.runArenaBatch(config, p0, p1, batchN, 42);
      setBatchResults(results);
    } catch (err: any) {
      alert(err.message || 'Batch run failed');
    } finally {
      setIsBatchRunning(false);
    }
  };

  // Active overlay items
  const activeOverlay: OverlayItem[] | undefined = showOverlay
    ? (session?.state.to_move === 0 ? p0Explanation?.overlay : p1Explanation?.overlay)
    : undefined;

  return (
    <div className="main-content" data-testid="arena-page">
      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 340px', gap: '1.5rem', alignItems: 'start' }}>
        {/* Controls Column */}
        <div>
          <div className="card">
            <div className="card-title">Arena Controls</div>

            <div className="form-group">
              <label className="form-label">Board Size</label>
              <select
                className="form-select"
                value={config.board_size}
                onChange={(e) => setConfig({ ...config, board_size: Number(e.target.value) })}
              >
                {[3, 4, 5, 7, 9].map((s) => (
                  <option key={s} value={s}>
                    {s} × {s}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" style={{ color: 'var(--p0-color)' }}>Player 0 Agent</label>
              <select className="form-select" value={p0} onChange={(e) => setP0(e.target.value)}>
                {agents.map((ag) => (
                  <option key={ag.spec} value={ag.spec} disabled={!ag.available}>
                    {ag.label} {!ag.available ? `(${ag.reason})` : ''}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" style={{ color: 'var(--p1-color)' }}>Player 1 Agent</label>
              <select className="form-select" value={p1} onChange={(e) => setP1(e.target.value)}>
                {agents.map((ag) => (
                  <option key={ag.spec} value={ag.spec} disabled={!ag.available}>
                    {ag.label} {!ag.available ? `(${ag.reason})` : ''}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Speed: {(speedMs / 1000).toFixed(2)}s / ply</label>
              <input
                type="range"
                min="100"
                max="1500"
                step="50"
                value={speedMs}
                onChange={(e) => setSpeedMs(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
              <button
                className={`btn ${isPlaying ? 'btn-primary' : ''}`}
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={!session || session.state.terminal.over}
              >
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button className="btn" onClick={stepOnce} disabled={isPlaying || !session || session.state.terminal.over}>
                Step
              </button>
              <button className="btn" onClick={initGame}>
                Reset
              </button>
            </div>
          </div>

          {/* Batch Tournaments Card */}
          <div className="card" style={{ marginTop: '1rem' }}>
            <div className="card-title">Batch Matches</div>
            <div className="form-group">
              <label className="form-label">Games (n):</label>
              <select className="form-select" value={batchN} onChange={(e) => setBatchN(Number(e.target.value))}>
                <option value={6}>6 games (quick)</option>
                <option value={10}>10 games</option>
                <option value={20}>20 games</option>
                <option value={50}>50 games</option>
              </select>
            </div>
            <button
              className="btn btn-primary"
              style={{ width: '100%' }}
              onClick={handleRunBatch}
              disabled={isBatchRunning}
            >
              {isBatchRunning ? 'Running Batch...' : 'Run Tournament Batch'}
            </button>
          </div>
        </div>

        {/* Center Board Column */}
        <div className="board-container">
          {session && <TerminalBanner terminal={session.state.terminal} />}

          {session && (
            <Board
              state={session.state}
              overlay={activeOverlay}
              showPaths={true}
              interactive={false}
            />
          )}

          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.85rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showOverlay}
                onChange={(e) => setShowOverlay(e.target.checked)}
              />
              Show Heatmap Overlay
            </label>
          </div>
        </div>

        {/* Right Info Column: Think Panels */}
        <div>
          {/* P0 Think Box */}
          <div className="card" style={{ borderLeft: '4px solid var(--p0-color)', marginBottom: '1rem' }}>
            <div className="card-title" style={{ color: 'var(--p0-color)' }}>
              P0: {p0} {session?.state.to_move === 0 && !session.state.terminal.over && '● Thinking'}
            </div>
            {p0Explanation ? (
              <div style={{ fontSize: '0.85rem' }}>
                <p>{p0Explanation.summary}</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '0.3rem' }}>
                  Decision time: {(p0Explanation.decision_seconds * 1000).toFixed(1)}ms
                </p>
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Waiting for P0 move...</p>
            )}
          </div>

          {/* P1 Think Box */}
          <div className="card" style={{ borderLeft: '4px solid var(--p1-color)' }}>
            <div className="card-title" style={{ color: 'var(--p1-color)' }}>
              P1: {p1} {session?.state.to_move === 1 && !session.state.terminal.over && '● Thinking'}
            </div>
            {p1Explanation ? (
              <div style={{ fontSize: '0.85rem' }}>
                <p>{p1Explanation.summary}</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '0.3rem' }}>
                  Decision time: {(p1Explanation.decision_seconds * 1000).toFixed(1)}ms
                </p>
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Waiting for P1 move...</p>
            )}
          </div>

          {/* Tournament Batch Results Card */}
          {batchResults && (
            <div className="card" style={{ marginTop: '1rem' }}>
              <div className="card-title">Tournament Results ({batchResults.games} games)</div>
              {Object.entries(batchResults.by_spec).map(([sp, st]: any) => (
                <div key={sp} style={{ marginBottom: '0.5rem', fontSize: '0.85rem' }}>
                  <div>
                    <strong>{sp}</strong>: {st.wins}W - {st.draws}D - {st.losses}L ({(st.win_rate * 100).toFixed(1)}%)
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                    95% Wilson CI: [{(st.ci[0] * 100).toFixed(1)}%, {(st.ci[1] * 100).toFixed(1)}%]
                  </div>
                </div>
              ))}
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                Avg Length: {batchResults.mean_length} plies
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
