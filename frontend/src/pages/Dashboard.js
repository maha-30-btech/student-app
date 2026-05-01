import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getStats, getComparisons } from '../services/api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, Radar, Legend,
} from 'recharts';

const TT = {
  contentStyle: { background:'var(--bg3)', border:'1px solid var(--border)', borderRadius:8, fontSize:12 },
  labelStyle:   { color:'var(--text)' },
};

export default function Dashboard() {
  const [stats,   setStats]   = useState(null);
  const [comps,   setComps]   = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([getStats(), getComparisons()])
      .then(([sr, cr]) => {
        setStats(sr.data);
        setComps(cr.data.comparisons || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display:'flex', gap:10, color:'var(--text2)', padding:20 }}>
      <div className="spinner" /> Loading...
    </div>
  );

  const wins = comps.filter(c => c.improvement?.novelty_wins).length;

  const maeBars = comps.map(c => ({
    name:    `${c.dataset}/${c.target}`,
    Base:    c.base?.metrics?.MAE    || 0,
    Novelty: c.novelty?.metrics?.MAE || 0,
  }));

  const f1Bars = comps.map(c => ({
    name:    `${c.dataset}/${c.target}`,
    Base:    c.base?.classification?.f1    || 0,
    Novelty: c.novelty?.classification?.f1 || 0,
  }));

  const avgOf = (arr, key) =>
    arr.length ? arr.reduce((s, c) => s + (c[key] || 0), 0) / arr.length : 0;

  const radarData = [
    { m:'MAE↓',    Base: avgOf(comps.map(c=>c.base?.metrics),'MAE'),       Novelty: avgOf(comps.map(c=>c.novelty?.metrics),'MAE') },
    { m:'RMSE↓',   Base: avgOf(comps.map(c=>c.base?.metrics),'RMSE'),      Novelty: avgOf(comps.map(c=>c.novelty?.metrics),'RMSE') },
    { m:'F1×100',  Base: avgOf(comps.map(c=>c.base?.classification),'f1')*100,       Novelty: avgOf(comps.map(c=>c.novelty?.classification),'f1')*100 },
    { m:'ACC×100', Base: avgOf(comps.map(c=>c.base?.classification),'accuracy')*100, Novelty: avgOf(comps.map(c=>c.novelty?.classification),'accuracy')*100 },
    { m:'MAC×100', Base: avgOf(comps.map(c=>c.base?.metrics),'MAC')*100,             Novelty: avgOf(comps.map(c=>c.novelty?.metrics),'MAC')*100 },
  ];

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">Dashboard</div>
        <div className="page-sub">
          SHAP-Guided Adaptive vs DCatBoostF (IEEE TKDE 2024)&nbsp;
          <span className="badge badge-green">Novelty wins {wins}/{comps.length}</span>
        </div>
      </div>

      {/* Tiles */}
      <div className="grid-4" style={{ marginBottom:20 }}>
        {[
          { label:'Trained Models',   val: stats?.totalModels || 9, color:'var(--accent)',  sub:'Base + Novelty' },
          { label:'Novelty Wins',     val: `${wins}/${comps.length}`, color:'var(--accent2)', sub:'by MAE' },
          { label:'Predictions Made', val: stats?.totalPredictions || 0, color:'var(--purple)' },
          { label:'Novel Equations',  val: 'N1–N3', color:'var(--accent4)', sub:'SHAP · Adaptive θ · Fusion' },
        ].map(({ label, val, color, sub }) => (
          <div key={label} className="card" style={{ borderTop:`3px solid ${color}` }}>
            <div style={{ fontSize:28, fontWeight:700, fontFamily:'var(--mono)', color }}>{val}</div>
            <div style={{ color:'var(--text2)', fontSize:13, marginTop:3 }}>{label}</div>
            {sub && <div style={{ color:'var(--text3)', fontSize:11 }}>{sub}</div>}
          </div>
        ))}
      </div>

      {/* MAE + F1 charts */}
      <div className="grid-2" style={{ marginBottom:20 }}>
        <div className="card">
          <div style={{ fontWeight:600, marginBottom:12, display:'flex', justifyContent:'space-between', alignItems:'center' }}>
            MAE — Base vs Novelty <span className="badge badge-blue">↓ lower better</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={maeBars} margin={{ left:-20, bottom:22 }}>
              <XAxis dataKey="name" tick={{ fontSize:8, fill:'var(--text3)' }} angle={-35} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize:10, fill:'var(--text3)' }} />
              <Tooltip {...TT} />
              <Legend wrapperStyle={{ fontSize:11 }} />
              <Bar dataKey="Base"    fill="#00d4ff" opacity={0.8} radius={[3,3,0,0]} />
              <Bar dataKey="Novelty" fill="#4fffb0" opacity={0.9} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div style={{ fontWeight:600, marginBottom:12, display:'flex', justifyContent:'space-between', alignItems:'center' }}>
            F1 Score — Base vs Novelty <span className="badge badge-green">↑ higher better</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={f1Bars} margin={{ left:-20, bottom:22 }}>
              <XAxis dataKey="name" tick={{ fontSize:8, fill:'var(--text3)' }} angle={-35} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize:10, fill:'var(--text3)' }} domain={[0,1]} />
              <Tooltip {...TT} />
              <Legend wrapperStyle={{ fontSize:11 }} />
              <Bar dataKey="Base"    fill="#00d4ff" opacity={0.8} radius={[3,3,0,0]} />
              <Bar dataKey="Novelty" fill="#4fffb0" opacity={0.9} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Radar + improvement table */}
      <div className="grid-2" style={{ marginBottom:20 }}>
        <div className="card">
          <div style={{ fontWeight:600, marginBottom:10 }}>Average Performance Radar</div>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData} margin={{ top:10, right:30, bottom:5, left:30 }}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="m" tick={{ fontSize:10, fill:'var(--text2)' }} />
              <Radar name="Base Paper" dataKey="Base"    stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.1}  strokeWidth={2} />
              <Radar name="Novelty"    dataKey="Novelty" stroke="#4fffb0" fill="#4fffb0" fillOpacity={0.15} strokeWidth={2} />
              <Legend wrapperStyle={{ fontSize:11 }} />
              <Tooltip {...TT} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div style={{ fontWeight:600, marginBottom:12 }}>Per-Model Improvement</div>
          <table className="data-table">
            <thead><tr><th>Model</th><th>ΔMAE</th><th>ΔF1</th><th>%</th><th></th></tr></thead>
            <tbody>
              {comps.map(c => {
                const imp = c.improvement || {};
                const mWin = imp.MAE_delta >= 0;
                const fWin = imp.F1_delta  >= 0;
                return (
                  <tr key={c.model_id}>
                    <td style={{ fontFamily:'var(--mono)', fontSize:11 }}>{c.model_id}</td>
                    <td style={{ fontFamily:'var(--mono)', fontSize:12,
                        color: mWin ? 'var(--accent)':'var(--accent3)' }}>
                      {mWin?'+':''}{imp.MAE_delta}
                    </td>
                    <td style={{ fontFamily:'var(--mono)', fontSize:12,
                        color: fWin ? 'var(--accent)':'var(--accent3)' }}>
                      {fWin?'+':''}{imp.F1_delta}
                    </td>
                    <td style={{ fontFamily:'var(--mono)', fontSize:11 }}>{imp.MAE_pct}%</td>
                    <td>{imp.novelty_wins
                      ? <span className="badge badge-green">✓</span>
                      : <span className="badge badge-red">✗</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ marginTop:14, display:'flex', gap:8 }}>
            <button className="btn btn-primary" onClick={() => navigate('/compare')}
              style={{ flex:1, justifyContent:'center', fontSize:13 }}>⇄ Full Comparison</button>
            <button className="btn btn-ghost" onClick={() => navigate('/predict')}
              style={{ flex:1, justifyContent:'center', fontSize:13 }}>◈ Predict</button>
          </div>
        </div>
      </div>

      {/* Novelty equation cards */}
      <div className="card">
        <div style={{ fontWeight:600, marginBottom:14 }}>Novelty Algorithm — 3 New Equations</div>
        <div style={{ display:'flex', gap:10, flexWrap:'wrap' }}>
          {[
            { eq:'Eq.N1', title:'Manual SHAP Importance',
              desc:'φᵢ = (1/T)Σₜ Σ_{n:feat=i} [nL/N · nR/N]   Coverage-weighted gain per feature across RF trees',
              color:'var(--accent)' },
            { eq:'Eq.N2', title:'Adaptive Threshold',
              desc:'θₗ = clip(μₗ + γ·σₗ, 0.03, 0.40)   Replaces fixed θ=0.05 with data-driven SHAP distribution',
              color:'var(--accent2)' },
            { eq:'Eq.N3', title:'SHAP-Weighted Feature Fusion',
              desc:'fs₂ ← [fs₂, softmax(φᵢ)·fy]   Replaces plain concat Eq.6 with importance-weighted fusion',
              color:'var(--purple)' },
          ].map(({ eq, title, desc, color }) => (
            <div key={eq} style={{ flex:1, minWidth:200, background:'var(--bg3)',
              border:`1px solid ${color}44`, borderRadius:8, padding:14 }}>
              <div style={{ background:`${color}20`, color, borderRadius:4, padding:'2px 8px',
                fontFamily:'var(--mono)', fontSize:11, fontWeight:700, display:'inline-block', marginBottom:8 }}>
                {eq}
              </div>
              <div style={{ fontWeight:600, fontSize:13, marginBottom:5 }}>{title}</div>
              <div style={{ fontSize:11, color:'var(--text2)', fontFamily:'var(--mono)', lineHeight:1.7 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
