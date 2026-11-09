import React, { Suspense, lazy } from 'react';
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom';
import { PlayPage } from './pages/Play';

// Only the Play page is in the initial bundle; the other pages (and recharts, used by
// Training) load on first visit, which keeps the first paint fast.
const ArenaPage = lazy(() => import('./pages/Arena').then((m) => ({ default: m.ArenaPage })));
const ReplaysPage = lazy(() => import('./pages/Replays').then((m) => ({ default: m.ReplaysPage })));
const RulesPage = lazy(() => import('./pages/Rules').then((m) => ({ default: m.RulesPage })));
const TrainingPage = lazy(() => import('./pages/Training').then((m) => ({ default: m.TrainingPage })));

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="app-container">
        <header className="navbar">
          <div className="nav-brand">
            <span>♟ Modified Wallz</span>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 400 }}>
              Multi-Agent Spatial AI
            </span>
          </div>
          <nav className="nav-links">
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Play
            </NavLink>
            <NavLink to="/arena" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Arena
            </NavLink>
            <NavLink to="/training" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Training
            </NavLink>
            <NavLink to="/replays" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Replays
            </NavLink>
            <NavLink to="/rules" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Rules
            </NavLink>
          </nav>
        </header>

        <Suspense fallback={<div className="main-content">Loading…</div>}>
        <Routes>
          <Route path="/" element={<PlayPage />} />
          <Route path="/arena" element={<ArenaPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/replays" element={<ReplaysPage />} />
          <Route path="/rules" element={<RulesPage />} />
          <Route path="/play" element={<PlayPage />} />
          <Route
            path="*"
            element={
              <div className="main-content">
                <h2>Page not found</h2>
                <p>
                  Use the navigation above, or go to <NavLink to="/">Play</NavLink>.
                </p>
              </div>
            }
          />
        </Routes>
        </Suspense>
      </div>
    </BrowserRouter>
  );
};
