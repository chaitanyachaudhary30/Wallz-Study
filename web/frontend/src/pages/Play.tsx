import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../api/client';
import {
  ActionJson,
  AgentInfo,
  ConfigDefaults,
  Explanation,
  GameConfig,
  SessionData,
} from '../api/types';
import { ActionBar } from '../components/ActionBar';
import { AgentPicker } from '../components/AgentPicker';
import { ActionMode, Board } from '../components/Board';
import { ConfigForm } from '../components/ConfigForm';
import { ExplanationPanel } from '../components/ExplanationPanel';
import { PlayerCard } from '../components/PlayerCard';
import { TerminalBanner } from '../components/TerminalBanner';
import { Timeline } from '../components/Timeline';

export const PlayPage: React.FC = () => {
  const [config, setConfig] = useState<GameConfig>({
    board_size: 5,
    A: true,
    B: true,
    C: true,
    K: 3,
    M: 10,
  });
  const [defaults, setDefaults] = useState<ConfigDefaults | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [p0, setP0] = useState<string>('human');
  const [p1, setP1] = useState<string>('greedy');

  const [session, setSession] = useState<SessionData | null>(null);
  const [currentMode, setCurrentMode] = useState<ActionMode>('MOVE');
  const [showPaths, setShowPaths] = useState<boolean>(true);
  const [showOverlay, setShowOverlay] = useState<boolean>(false);
  const [latestExplanation, setLatestExplanation] = useState<Explanation | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const steppingRef = useRef(false);

  // Load defaults & agents on mount
  useEffect(() => {
    api.fetchConfigDefaults().then((res) => {
      setDefaults(res);
      if (res.defaults) {
        setConfig((prev) => ({ ...prev, ...res.defaults }));
      }
    }).catch(console.error);
  }, []);

  // Reload agents whenever config changes
  useEffect(() => {
    api.fetchAgents(config).then((list) => {
      setAgents(list);
      // Fallback if selected agent is no longer available
      const p0Avail = list.find((a) => a.spec === p0)?.available;
      if (p0Avail === false) setP0('human');
      const p1Avail = list.find((a) => a.spec === p1)?.available;
      if (p1Avail === false) setP1('greedy');
    }).catch(console.error);
  }, [config, p0, p1]);

  // Start new game
  const handleStartGame = useCallback(async () => {
    try {
      setErrorMessage(null);
      setLatestExplanation(null);
      const newSession = await api.createGame(config, p0, p1);
      setSession(newSession);
      setCurrentMode('MOVE');
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to start game');
    }
  }, [config, p0, p1]);

  // Start initial game once
  useEffect(() => {
    handleStartGame();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle user action
  const handleAction = async (action: ActionJson) => {
    if (!session || session.state.terminal.over) return;
    try {
      setErrorMessage(null);
      const updated = await api.sendAction(session.id, action);
      setSession(updated);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setErrorMessage(err.data?.error || err.message);
        // Sync state if returned
        if (err.data?.state) {
          setSession((prev) => (prev ? { ...prev, state: err.data.state } : null));
        }
      } else {
        setErrorMessage(err.message);
      }
    }
  };

  // AI Stepping Loop with 300ms pacing
  useEffect(() => {
    if (!session || session.state.terminal.over || steppingRef.current) return;

    const mover = session.state.to_move;
    const moverSpec = session.specs[String(mover)];

    if (moverSpec && moverSpec !== 'human') {
      steppingRef.current = true;
      const timer = setTimeout(async () => {
        try {
          const stepRes = await api.stepGame(session.id, true);
          setSession(stepRes.session);
          if (stepRes.explanation) {
            setLatestExplanation(stepRes.explanation);
          }
        } catch (err: any) {
          setErrorMessage(err.message || 'AI step failed');
        } finally {
          steppingRef.current = false;
        }
      }, 300);

      return () => clearTimeout(timer);
    }
  }, [session]);

  // Undo
  const handleUndo = async () => {
    if (!session) return;
    try {
      setErrorMessage(null);
      const updated = await api.undoGame(session.id, 1, true);
      setSession(updated);
    } catch (err: any) {
      setErrorMessage(err.message || 'Undo failed');
    }
  };

  // Ask AI analysis for human turn
  const handleAnalyze = async (spec: string) => {
    if (!session) return;
    try {
      const exp = await api.analyzePosition(session.id, spec);
      setLatestExplanation(exp);
      setShowOverlay(true);
    } catch (err: any) {
      setErrorMessage(err.message || 'Analysis failed');
    }
  };

  if (!session) {
    return <div className="main-content">Loading game session...</div>;
  }

  const isHumanTurn = session.specs[String(session.state.to_move)] === 'human';

  return (
    <div className="main-content" data-testid="play-page">
      <div className="play-grid">
        {/* Left Column: Config & Agents */}
        <div>
          <ConfigForm
            config={config}
            defaults={defaults}
            onChange={setConfig}
            onNewGame={handleStartGame}
          />
          <AgentPicker
            agents={agents}
            p0={p0}
            p1={p1}
            onSelectP0={setP0}
            onSelectP1={setP1}
          />
        </div>

        {/* Center Column: Board & Controls */}
        <div className="board-container">
          <TerminalBanner terminal={session.state.terminal} />

          <Board
            state={session.state}
            mode={currentMode}
            overlay={showOverlay && latestExplanation ? latestExplanation.overlay : undefined}
            showPaths={showPaths}
            onAction={handleAction}
            interactive={isHumanTurn && !session.state.terminal.over}
          />

          <ActionBar
            state={session.state}
            currentMode={currentMode}
            onSelectMode={setCurrentMode}
            disabled={!isHumanTurn || session.state.terminal.over}
          />

          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showPaths}
                onChange={(e) => setShowPaths(e.target.checked)}
              />
              Show Shortest Paths
            </label>
          </div>
        </div>

        {/* Right Column: Player Cards, Timeline, Explanations */}
        <div>
          <PlayerCard player={0} name={session.specs['0'] || 'P0'} state={session.state} />
          <PlayerCard player={1} name={session.specs['1'] || 'P1'} state={session.state} />

          <Timeline
            history={session.history}
            onUndo={handleUndo}
            canUndo={session.history.length > 0 && !session.state.terminal.over}
          />

          <ExplanationPanel
            explanation={latestExplanation}
            showOverlay={showOverlay}
            onToggleOverlay={setShowOverlay}
            onAnalyze={handleAnalyze}
            isHumanTurn={isHumanTurn}
          />
        </div>
      </div>

      {/* Toast Error Popup */}
      {errorMessage && (
        <div className="toast-error" onClick={() => setErrorMessage(null)}>
          <span>⚠ {errorMessage}</span>
          <button style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}>
            ✕
          </button>
        </div>
      )}
    </div>
  );
};
