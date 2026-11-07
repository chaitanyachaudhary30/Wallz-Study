import React from 'react';
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom';
import { ArenaPage } from './pages/Arena';
import { PlayPage } from './pages/Play';
import { ReplaysPage } from './pages/Replays';
import { RulesPage } from './pages/Rules';
import { TrainingPage } from './pages/Training';

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
      </div>
    </BrowserRouter>
  );
};
