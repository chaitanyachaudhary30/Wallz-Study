import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { Board } from '../components/Board';
import { TerminalBanner } from '../components/TerminalBanner';

export const ReplaysPage: React.FC = () => {
  const [replays, setReplays] = useState<any[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [query, setQuery] = useState<string>('');
  const [selectedItem, setSelectedItem] = useState<any | null>(null);

  const [activeReplayData, setActiveReplayData] = useState<any | null>(null);
  const [currentPly, setCurrentPly] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  // Load replays
  useEffect(() => {
    api.fetchReplays(0, 50, query)
      .then((res) => {
        setReplays(res.items);
        setTotal(res.total);
        if (res.items.length > 0 && !selectedItem) {
          loadReplay(res.items[0]);
        }
      })
      .catch(console.error);
  }, [query]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadReplay = async (item: any) => {
    setSelectedItem(item);
    setLoading(true);
    setIsPlaying(false);
    setCurrentPly(0);
    try {
      const data = await api.fetchReplayItem(item.file, item.index);
      setActiveReplayData(data);
    } catch (err: any) {
      alert(err.message || 'Failed to load replay');
    } finally {
      setLoading(false);
    }
  };

  // Upload custom replay
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        setLoading(true);
        const data = await api.uploadReplay(json);
        setActiveReplayData(data);
        setSelectedItem({ match_id: json.match_id || file.name, length: json.length || json.replay?.length });
        setCurrentPly(0);
      } catch (err: any) {
        alert(err.message || 'Failed to parse replay file');
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  };

  // Scrubber playback
  const maxPly = activeReplayData?.frames ? activeReplayData.frames.length - 1 : 0;
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      setCurrentPly((prev) => {
        if (prev >= maxPly) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 400);
    return () => clearInterval(timer);
  }, [isPlaying, maxPly]);

  const currentFrameState = activeReplayData?.frames?.[currentPly];

  return (
    <div className="main-content" data-testid="replays-page">
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Left Column: Replay Catalog */}
        <div className="card">
          <div className="card-title">
            <span>Experiment Replays ({total})</span>
            <label className="btn" style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', cursor: 'pointer' }}>
              Upload
              <input type="file" accept=".json" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
          </div>

          <div className="form-group">
            <input
              type="text"
              className="form-input"
              placeholder="Filter by filename or ID..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div style={{ maxHeight: '600px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {replays.map((rep, idx) => {
              const isSelected = selectedItem?.file === rep.file && selectedItem?.index === rep.index;
              return (
                <div
                  key={`${rep.file}-${rep.index}-${idx}`}
                  onClick={() => loadReplay(rep)}
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: '6px',
                    backgroundColor: isSelected ? 'var(--primary)' : 'rgba(255,255,255,0.03)',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {rep.match_id}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: isSelected ? '#ffffff' : 'var(--text-muted)', fontSize: '0.75rem' }}>
                    <span>{rep.config_label}</span>
                    <span>{rep.length} plies</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Replay Viewer & Scrubber */}
        <div className="board-container">
          {loading && <p>Loading frames and verifying replay...</p>}

          {activeReplayData?.verified && (
            <div
              style={{
                padding: '0.5rem 1rem',
                borderRadius: '6px',
                fontSize: '0.85rem',
                fontWeight: 600,
                marginBottom: '0.5rem',
                backgroundColor: activeReplayData.verified.ok ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                color: activeReplayData.verified.ok ? '#10b981' : '#ef4444',
                border: `1px solid ${activeReplayData.verified.ok ? '#10b981' : '#ef4444'}`,
              }}
            >
              {activeReplayData.verified.ok ? '✓ Deterministic Replay Verified' : `⚠ ${activeReplayData.verified.message}`}
            </div>
          )}

          {currentFrameState && (
            <>
              <TerminalBanner terminal={currentFrameState.terminal} />

              <Board
                state={currentFrameState}
                showPaths={true}
                interactive={false}
              />

              {/* Scrubber Controls */}
              <div className="card" style={{ width: '100%', maxWidth: '500px', marginTop: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span>Ply: <strong>{currentPly}</strong> / {maxPly}</span>
                  <div style={{ display: 'flex', gap: '0.4rem' }}>
                    <button className="btn" onClick={() => setCurrentPly(Math.max(0, currentPly - 1))} disabled={currentPly === 0}>
                      ◀
                    </button>
                    <button className="btn btn-primary" onClick={() => setIsPlaying(!isPlaying)}>
                      {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button className="btn" onClick={() => setCurrentPly(Math.min(maxPly, currentPly + 1))} disabled={currentPly >= maxPly}>
                      ▶
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
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
