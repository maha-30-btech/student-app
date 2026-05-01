import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import './index.css';

import Dashboard  from './pages/Dashboard';
import Predict    from './pages/Predict';
import Models     from './pages/Models';
import History    from './pages/History';
import Comparison from './pages/Comparison';
import About      from './pages/About';

const NAV = [
  { to: '/',        icon: '⬡', label: 'Dashboard'  },
  { to: '/predict', icon: '◈', label: 'Predict'    },
  { to: '/models',  icon: '◉', label: 'Models'     },
  { to: '/compare', icon: '⇄', label: 'Compare'    },
  { to: '/history', icon: '▦', label: 'History'    },
  { to: '/about',   icon: '◇', label: 'About'      },
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-tag">IEEE TKDE 2024</div>
        <div className="logo-title">SHAP-DCatBoost<br/>Student Predictor</div>
      </div>
      <nav className="nav-section">
        <div className="nav-label">Navigation</div>
        {NAV.map(({ to, icon, label }) => (
          <NavLink key={to} to={to} end={to === '/'}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <span className="nav-icon">{icon}</span>{label}
          </NavLink>
        ))}
      </nav>
      <div style={{ marginTop:'auto', padding:'16px 20px', borderTop:'1px solid var(--border)' }}>
        <div style={{ fontSize:11, color:'var(--text3)', fontFamily:'var(--mono)', lineHeight:1.8 }}>
          Base: DCatBoostF<br/>
          Novelty: SHAP-Adaptive<br/>
          Eqs N1 · N2 · N3
        </div>
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/"        element={<Dashboard />}  />
            <Route path="/predict" element={<Predict />}    />
            <Route path="/models"  element={<Models />}     />
            <Route path="/compare" element={<Comparison />} />
            <Route path="/history" element={<History />}    />
            <Route path="/about"   element={<About />}      />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
