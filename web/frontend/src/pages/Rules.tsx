import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { ScenarioDetail } from '../api/types';
import { Board } from '../components/Board';
import { TerminalBanner } from '../components/TerminalBanner';

export const RulesPage: React.FC = () => {
  const [scenarios, setScenarios] = useState<any[]>([]);
  const [selectedName, setSelectedName] = useState<string>('move_basics');
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [currentPly, setCurrentPly] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [rulesDoc, setRulesDoc] = useState<string>('');
  const [showDoc, setShowDoc] = useState<boolean>(false);
  const navigate = useNavigate();

  useEffect(() => {
    api.fetchScenarios().then((list) => {
      setScenarios(list);
    }).catch(console.error);

    api.fetchRules().then((res) => {
      setRulesDoc(res.rules);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedName) return;
    setLoading(true);
    setCurrentPly(0);
    api.fetchScenario(selectedName)
      .then((data) => setDetail(data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedName]);

  const handlePlayFromHere = async () => {
    if (!selectedName) return;
    try {
      const created = await api.createGameFromScenario(selectedName, 'human', 'greedy');
      // Hand the scenario game to the Play page; without this it started a fresh default game
      navigate('/', { state: { sessionId: created.id } });
    } catch (err: any) {
      alert(err.message || 'Failed to initialize playable game');
    }
  };

  const frames = detail?.frames || [];
  const maxPly = frames.length > 0 ? frames.length - 1 : 0;
  const currentFrame = frames[currentPly];

  return (
    <div className="main-content" data-testid="rules-page">
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Left Column: 9 Canonical Scenarios Directory */}
        <div className="card">
          <button
            className="btn"
            style={{ width: '100%', marginBottom: '1rem', fontSize: '0.8rem' }}
            onClick={() => setShowDoc(!showDoc)}
          >
            {showDoc ? 'Hide Official Rules Doc' : '📖 View Rules Spec (Markdown)'}
          </button>

          <div className="card-title">Rule Scenarios ({scenarios.length})</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {scenarios.map((s) => (
              <div
                key={s.name}
                onClick={() => setSelectedName(s.name)}
                style={{
                  padding: '0.6rem 0.8rem',
                  borderRadius: '6px',
                  backgroundColor: selectedName === s.name ? 'var(--primary)' : 'rgba(255,255,255,0.03)',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ fontWeight: 600 }}>{s.title}</div>
                <div style={{ fontSize: '0.75rem', color: selectedName === s.name ? '#ffffff' : 'var(--text-muted)' }}>
                  {s.description.slice(0, 60)}...
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Scenario Player */}
        <div className="board-container">
          {showDoc && (
            <div className="card" style={{ width: '100%', marginBottom: '1.5rem', maxHeight: '500px', overflowY: 'auto' }}>
              <div className="card-title">Official Rules Specification (docs/game_rules.md)</div>
              <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.8rem', lineHeight: '1.4', color: 'var(--text-color)' }}>
                {rulesDoc || 'Loading rules...'}
              </pre>
            </div>
          )}

          {detail?.scenario && (
            <div className="card" style={{ width: '100%', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: 0 }}>{detail.scenario.title}</h3>
                <button className="btn btn-primary" onClick={handlePlayFromHere}>
                  Play From This State →
                </button>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.4rem' }}>
                {detail.scenario.description}
              </p>
            </div>
          )}

          {loading && <p>Executing scenario verification...</p>}

          {currentFrame && (
            <>
              <TerminalBanner terminal={currentFrame.state.terminal} />

              <Board
                state={currentFrame.state}
                showPaths={true}
                interactive={false}
              />

              {/* Scrubber Controls */}
              <div className="card" style={{ width: '100%', maxWidth: '500px', marginTop: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span>
                    Ply: <strong>{currentPly}</strong> / {maxPly}{' '}
                    {currentFrame.action ? `(${currentFrame.action[0]})` : '(Start)'}
                  </span>
                  <div style={{ display: 'flex', gap: '0.4rem' }}>
                    <button className="btn" onClick={() => setCurrentPly(Math.max(0, currentPly - 1))} disabled={currentPly === 0}>
                      ◀ Prev
                    </button>
                    <button className="btn" onClick={() => setCurrentPly(Math.min(maxPly, currentPly + 1))} disabled={currentPly >= maxPly}>
                      Next ▶
                    </button>
                  </div>
                </div>

                <input
                  type="range"
                  min="0"
                  max={maxPly}
                  value={currentPly}
                  onChange={(e) => setCurrentPly(Number(e.target.value))}
                  style={{ width: '100%' }}
                />

                {/* Expectation Check Badges */}
                {currentFrame.checks && currentFrame.checks.length > 0 && (
                  <div style={{ marginTop: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                      Rule Invariant Checks:
                    </div>
                    {currentFrame.checks.map((chk, i) => (
                      <div
                        key={i}
                        style={{
                          fontSize: '0.75rem',
                          color: chk.ok ? '#10b981' : '#ef4444',
                          marginTop: '0.2rem',
                        }}
                      >
                        {chk.ok ? '✓' : '✗'} {chk.message}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
