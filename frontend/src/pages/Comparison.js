import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  LineChart, Line, CartesianGrid,
} from 'recharts';
import { getComparisons } from '../services/api';

const BASE_CLR   = '#00d4ff';
const NOVEL_CLR  = '#4fffb0';
const WARN_CLR   = '#ffd166';

// ── Confusion Matrix component ────────────────────────────────────────────────
function ConfusionMatrix({ cm, title, color }) {
  if (!cm) return null;
  const [[tn, fp], [fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  const cells = [
    { label: 'TN', val: tn, bg: 'rgba(0,212,255,0.15)', r: 'Pred Neg', c: 'Act Neg' },
    { label: 'FP', val: fp, bg: 'rgba(255,107,107,0.15)', r: 'Pred Pos', c: 'Act Neg' },
    { label: 'FN', val: fn, bg: 'rgba(255,107,107,0.15)', r: 'Pred Neg', c: 'Act Pos' },
    { label: 'TP', val: tp, bg: 'rgba(79,255,176,0.15)', r: 'Pred Pos', c: 'Act Pos' },
  ];
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 6, letterSpacing: 1 }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
        {cells.map(cl => (
          <div key={cl.label} style={{
            background: cl.bg, border: '1px solid var(--border)',
            borderRadius: 6, padding: '10px 6px', textAlign: 'center'
          }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 18, fontWeight: 700, color }}>{cl.val}</div>
            <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>{cl.label}</div>
            <div style={{ fontSize: 9, color: 'var(--text3)' }}>{(cl.val/total*100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Delta badge ───────────────────────────────────────────────────────────────
function DeltaBadge({ value, lowerBetter = true, suffix = '' }) {
  const good = lowerBetter ? value < 0 : value > 0;
  const col  = good ? 'var(--accent)' : value === 0 ? 'var(--text3)' : 'var(--accent3)';
  const bg   = good ? 'rgba(79,255,176,0.1)' : 'rgba(255,107,107,0.1)';
  return (
    <span style={{ background: bg, color: col, border: `1px solid ${col}`,
      borderRadius: 4, padding: '2px 7px', fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700 }}>
      {value > 0 ? '+' : ''}{value}{suffix}
    </span>
  );
}

// ── MetricBar comparing two values ────────────────────────────────────────────
function MetricCompare({ label, base, novelty, lowerBetter = true }) {
  const max  = Math.max(base, novelty, 0.001);
  const bPct = (base / max) * 100;
  const nPct = (novelty / max) * 100;
  const delta= round4(novelty - base);
  const good = lowerBetter ? delta < 0 : delta > 0;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: 'var(--text2)', fontWeight: 600 }}>{label}</span>
        <DeltaBadge value={delta} lowerBetter={lowerBetter} />
      </div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: BASE_CLR, width: 46, textAlign: 'right', fontFamily: 'var(--mono)' }}>{base}</span>
        <div style={{ flex: 1, position: 'relative', height: 8 }}>
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 8, background: 'var(--bg3)', borderRadius: 4 }} />
          <div style={{ position: 'absolute', top: 0, left: 0, width: `${bPct}%`, height: 8, background: BASE_CLR, borderRadius: 4, opacity: 0.5 }} />
          <div style={{ position: 'absolute', top: 0, left: 0, width: `${nPct}%`, height: 8, background: NOVEL_CLR, borderRadius: 4, opacity: 0.8, marginTop: 0 }} />
        </div>
        <span style={{ fontSize: 11, color: NOVEL_CLR, width: 46, fontFamily: 'var(--mono)' }}>{novelty}</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text3)', marginTop: 2 }}>
        <span>BASE</span><span>NOVELTY</span>
      </div>
    </div>
  );
}

function round4(v) { return Math.round(v * 10000) / 10000; }

const TOOLTIP_STYLE = {
  contentStyle: { background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: 'var(--text)' }
};

export default function Comparison() {
  const [data,      setData]      = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [selected,  setSelected]  = useState(null);
  const [activeTab, setActiveTab] = useState('regression'); // regression | classification | visual

  useEffect(() => {
    getComparisons().then(r => {
      setData(r.data.comparisons || []);
      if (r.data.comparisons?.length) setSelected(r.data.comparisons[0]);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display:'flex', gap:10, color:'var(--text2)', padding:24 }}>
      <div className="spinner" /> Loading comparison data...
    </div>
  );

  const sel = selected;

  // Build bar chart data for all models
  const barData = data.map(d => ({
    name:    `${d.dataset}/${d.target}`,
    Base_MAE:    d.base?.metrics?.MAE    || 0,
    Novelty_MAE: d.novelty?.metrics?.MAE || 0,
    Base_F1:     d.base?.classification?.f1    || 0,
    Novelty_F1:  d.novelty?.classification?.f1 || 0,
    Base_ACC:    d.base?.classification?.accuracy    || 0,
    Novelty_ACC: d.novelty?.classification?.accuracy || 0,
  }));

  // Radar data for selected model
  const radarData = sel ? [
    { m: 'MAE↓',   base: sel.base.metrics.MAE,                   novelty: sel.novelty.metrics.MAE },
    { m: 'RMSE↓',  base: sel.base.metrics.RMSE,                  novelty: sel.novelty.metrics.RMSE },
    { m: 'MAC↑',   base: sel.base.metrics.MAC * 100,              novelty: sel.novelty.metrics.MAC * 100 },
    { m: 'F1↑',    base: sel.base.classification.f1 * 100,        novelty: sel.novelty.classification.f1 * 100 },
    { m: 'ACC↑',   base: sel.base.classification.accuracy * 100,  novelty: sel.novelty.classification.accuracy * 100 },
  ] : [];

  // Improvement summary
  const wins    = data.filter(d => d.improvement?.novelty_wins).length;
  const avgDMAE = data.length ? (data.reduce((s,d)=>s+(d.improvement?.MAE_delta||0),0)/data.length).toFixed(4) : 0;
  const avgDF1  = data.length ? (data.reduce((s,d)=>s+(d.improvement?.F1_delta||0),0)/data.length).toFixed(4) : 0;

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">Model Comparison</div>
        <div className="page-sub">
          DCatBoostF (Base Paper) vs SHAPAdaptiveDCatBoostF (Novelty)
        </div>
      </div>

      {/* Summary tiles */}
      <div className="grid-4" style={{ marginBottom: 20 }}>
        {[
          { label: 'Novelty Wins (MAE)', val: `${wins}/${data.length}`, color: 'var(--accent)' },
          { label: 'Avg ΔMAE (↓ better)',  val: `${avgDMAE}`, color: 'var(--accent2)' },
          { label: 'Avg ΔF1  (↑ better)',  val: `${avgDF1}`,  color: 'var(--purple)' },
          { label: 'Novel Equations',       val: '3 (N1–N3)', color: 'var(--accent4)' },
        ].map(({ label, val, color }) => (
          <div key={label} className="card" style={{ borderTop:`3px solid ${color}`, textAlign:'center', padding:'14px 12px' }}>
            <div style={{ fontSize:22, fontWeight:700, fontFamily:'var(--mono)', color }}>{val}</div>
            <div style={{ fontSize:12, color:'var(--text2)', marginTop:4 }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Model selector */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ fontWeight:600, marginBottom:10, fontSize:13 }}>Select Model to Inspect</div>
        <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
          {data.map(d => (
            <button key={d.model_id}
              onClick={() => setSelected(d)}
              className={`btn ${selected?.model_id===d.model_id ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize:12, padding:'6px 12px' }}>
              {d.dataset}/{d.target}
              {d.improvement?.novelty_wins &&
                <span style={{ marginLeft:5, fontSize:10 }}>✓</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Global charts row */}
      <div className="grid-2" style={{ marginBottom: 20 }}>
        {/* MAE comparison bar chart */}
        <div className="card">
          <div style={{ fontWeight:600, marginBottom:12, fontSize:13, display:'flex', justifyContent:'space-between' }}>
            MAE — All Models <span className="badge badge-blue">lower = better</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={barData} margin={{ left:-20, bottom:20 }}>
              <XAxis dataKey="name" tick={{ fontSize:9, fill:'var(--text3)' }} angle={-35} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize:10, fill:'var(--text3)' }} />
              <Tooltip {...TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize:11 }} />
              <Bar dataKey="Base_MAE"    name="Base"    fill={BASE_CLR}  opacity={0.8} radius={[3,3,0,0]} />
              <Bar dataKey="Novelty_MAE" name="Novelty" fill={NOVEL_CLR} opacity={0.9} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* F1 comparison bar chart */}
        <div className="card">
          <div style={{ fontWeight:600, marginBottom:12, fontSize:13, display:'flex', justifyContent:'space-between' }}>
            F1 Score — All Models <span className="badge badge-green">higher = better</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={barData} margin={{ left:-20, bottom:20 }}>
              <XAxis dataKey="name" tick={{ fontSize:9, fill:'var(--text3)' }} angle={-35} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize:10, fill:'var(--text3)' }} domain={[0,1]} />
              <Tooltip {...TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize:11 }} />
              <Bar dataKey="Base_F1"    name="Base"    fill={BASE_CLR}  opacity={0.8} radius={[3,3,0,0]} />
              <Bar dataKey="Novelty_F1" name="Novelty" fill={NOVEL_CLR} opacity={0.9} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Selected model detail */}
      {sel && (
        <div className="fade-in">
          {/* Tab bar */}
          <div style={{ display:'flex', gap:8, marginBottom:16 }}>
            {['regression','classification','visual'].map(t => (
              <button key={t} onClick={() => setActiveTab(t)}
                className={`btn ${activeTab===t ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize:13, padding:'7px 16px', textTransform:'capitalize' }}>
                {t === 'regression' ? '📊 Regression Metrics'
                 : t === 'classification' ? '🎯 Classification Metrics'
                 : '📈 Visual Analysis'}
              </button>
            ))}
          </div>

          {/* ── REGRESSION TAB ── */}
          {activeTab === 'regression' && (
            <div className="grid-2">
              <div className="card">
                <div style={{ fontWeight:600, marginBottom:16 }}>
                  Metric Comparison — {sel.model_id}
                </div>
                <MetricCompare label="MAE  (↓ better)"  base={sel.base.metrics.MAE}  novelty={sel.novelty.metrics.MAE}  lowerBetter />
                <MetricCompare label="RMSE (↓ better)"  base={sel.base.metrics.RMSE} novelty={sel.novelty.metrics.RMSE} lowerBetter />
                <MetricCompare label="SD   (↓ better)"  base={sel.base.metrics.SD}   novelty={sel.novelty.metrics.SD}   lowerBetter />
                <MetricCompare label="MAC  (↑ better)"  base={sel.base.metrics.MAC}  novelty={sel.novelty.metrics.MAC}  lowerBetter={false} />

                <div style={{ background:'var(--bg3)', borderRadius:8, padding:12, marginTop:16, fontSize:12 }}>
                  <div style={{ color:'var(--text3)', marginBottom:8, fontFamily:'var(--mono)', letterSpacing:1 }}>IMPROVEMENT SUMMARY</div>
                  {[
                    ['MAE reduction', sel.improvement.MAE_delta, true],
                    ['RMSE reduction', sel.improvement.RMSE_delta, true],
                    ['MAE % gain', sel.improvement.MAE_pct, true, '%'],
                  ].map(([label, val, lb, suf='']) => (
                    <div key={label} style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
                      <span style={{ color:'var(--text2)' }}>{label}</span>
                      <DeltaBadge value={val} lowerBetter={lb} suffix={suf} />
                    </div>
                  ))}
                </div>
              </div>

              {/* Radar */}
              <div className="card">
                <div style={{ fontWeight:600, marginBottom:12 }}>Performance Radar</div>
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart data={radarData} margin={{ top:10, right:30, bottom:10, left:30 }}>
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis dataKey="m" tick={{ fontSize:11, fill:'var(--text2)' }} />
                    <Radar name="Base Paper"   dataKey="base"    stroke={BASE_CLR}  fill={BASE_CLR}  fillOpacity={0.1} strokeWidth={2} />
                    <Radar name="Novelty"      dataKey="novelty" stroke={NOVEL_CLR} fill={NOVEL_CLR} fillOpacity={0.15} strokeWidth={2} />
                    <Legend wrapperStyle={{ fontSize:11 }} />
                    <Tooltip {...TOOLTIP_STYLE} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── CLASSIFICATION TAB ── */}
          {activeTab === 'classification' && (
            <div className="grid-2">
              <div className="card">
                <div style={{ fontWeight:600, marginBottom:16 }}>Classification Metrics</div>
                {[
                  { label:'Accuracy',  bv: sel.base.classification.accuracy,  nv: sel.novelty.classification.accuracy,  lb: false },
                  { label:'Precision', bv: sel.base.classification.precision, nv: sel.novelty.classification.precision, lb: false },
                  { label:'Recall',    bv: sel.base.classification.recall,    nv: sel.novelty.classification.recall,    lb: false },
                  { label:'F1 Score',  bv: sel.base.classification.f1,        nv: sel.novelty.classification.f1,        lb: false },
                ].map(({ label, bv, nv, lb }) => (
                  <MetricCompare key={label} label={`${label} (↑ better)`} base={bv} novelty={nv} lowerBetter={lb} />
                ))}

                {/* Class interpretation note */}
                <div style={{ background:'rgba(79,255,176,0.06)', border:'1px solid rgba(79,255,176,0.2)',
                  borderRadius:8, padding:12, marginTop:12, fontSize:12 }}>
                  <div style={{ color:'var(--accent)', fontWeight:600, marginBottom:4 }}>Classification Threshold</div>
                  <div style={{ color:'var(--text2)' }}>
                    {sel.target?.includes('score')
                      ? 'Score ≥ 60 → Pass (1),  Score < 60 → Fail (0)'
                      : 'Grade ≥ 10 → Pass (1),  Grade < 10 → Fail (0)'}
                  </div>
                </div>
              </div>

              {/* Confusion matrices side by side */}
              <div className="card">
                <div style={{ fontWeight:600, marginBottom:16 }}>Confusion Matrices</div>
                <div className="grid-2">
                  <ConfusionMatrix
                    cm={sel.base.confusion_matrix}
                    title="BASE — DCatBoostF"
                    color={BASE_CLR}
                  />
                  <ConfusionMatrix
                    cm={sel.novelty.confusion_matrix}
                    title="NOVELTY — SHAPAdaptive"
                    color={NOVEL_CLR}
                  />
                </div>
                <div style={{ marginTop:16, fontSize:11, color:'var(--text3)', lineHeight:1.8 }}>
                  <span style={{ color:NOVEL_CLR }}>■</span> TN = True Negative (correct Fail) &nbsp;
                  <span style={{ color:NOVEL_CLR }}>■</span> TP = True Positive (correct Pass)<br/>
                  <span style={{ color:'var(--accent3)' }}>■</span> FP = False Positive &nbsp;
                  <span style={{ color:'var(--accent3)' }}>■</span> FN = False Negative
                </div>
              </div>
            </div>
          )}

          {/* ── VISUAL ANALYSIS TAB ── */}
          {activeTab === 'visual' && (
            <div>
              {/* ACC across all models line chart */}
              <div className="card" style={{ marginBottom:16 }}>
                <div style={{ fontWeight:600, marginBottom:12 }}>
                  Accuracy Across All Datasets &nbsp;
                  <span className="badge badge-green">higher = better</span>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={barData} margin={{ left:-15, right:10 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize:9, fill:'var(--text3)' }} angle={-25} textAnchor="end" height={50} />
                    <YAxis tick={{ fontSize:10, fill:'var(--text3)' }} domain={[0,1]} />
                    <Tooltip {...TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontSize:11 }} />
                    <Line type="monotone" dataKey="Base_ACC"    name="Base Acc"    stroke={BASE_CLR}  strokeWidth={2} dot={{ r:4 }} />
                    <Line type="monotone" dataKey="Novelty_ACC" name="Novelty Acc" stroke={NOVEL_CLR} strokeWidth={2} dot={{ r:4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Threshold comparison */}
              {sel.novelty.adaptive_thresholds && (
                <div className="card" style={{ marginBottom:16 }}>
                  <div style={{ fontWeight:600, marginBottom:12 }}>
                    Adaptive Thresholds vs Fixed (Eq.N2)
                  </div>
                  <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
                    {[0.05, 0.05, 0.90].map((base, i) => (
                      <div key={i} style={{
                        flex:1, minWidth:120,
                        background:'var(--bg3)', border:'1px solid var(--border)',
                        borderRadius:8, padding:14, textAlign:'center'
                      }}>
                        <div style={{ fontSize:11, color:'var(--text3)', marginBottom:6 }}>Layer {i+1}</div>
                        <div style={{ display:'flex', justifyContent:'center', gap:12 }}>
                          <div>
                            <div style={{ fontSize:10, color:BASE_CLR }}>BASE</div>
                            <div style={{ fontFamily:'var(--mono)', fontSize:18, color:BASE_CLR }}>{base}</div>
                          </div>
                          <div style={{ fontSize:20, color:'var(--text3)', paddingTop:12 }}>→</div>
                          <div>
                            <div style={{ fontSize:10, color:NOVEL_CLR }}>ADAPTIVE</div>
                            <div style={{ fontFamily:'var(--mono)', fontSize:18, color:NOVEL_CLR }}>
                              {sel.novelty.adaptive_thresholds[i]?.toFixed(4) || '—'}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop:10, fontSize:12, color:'var(--text3)' }}>
                    Eq.N2: θₗ = clip(μₗ + γ·σₗ, 0.03, 0.40) where γ=0.5
                    — thresholds adapt to the SHAP distribution of each layer's features.
                  </div>
                </div>
              )}

              {/* Algorithm difference summary */}
              <div className="card">
                <div style={{ fontWeight:600, marginBottom:14 }}>Algorithm Differences</div>
                <table className="data-table">
                  <thead>
                    <tr><th>Aspect</th><th style={{ color:BASE_CLR }}>Base Paper (DCatBoostF)</th><th style={{ color:NOVEL_CLR }}>Novelty (SHAPAdaptive)</th></tr>
                  </thead>
                  <tbody>
                    {[
                      ['Feature Importance', 'RF split-count (Eq.1)', 'Manual TreeSHAP coverage (Eq.N1)'],
                      ['Threshold Selection', 'Fixed: θ₁=0.05, θ₂=0.05, θ₃=0.90', 'Adaptive: θₗ=μₗ+γσₗ (Eq.N2)'],
                      ['Feature Fusion', 'Plain concat: [fs₂, fy] (Eq.6)', 'Softmax-weighted: [fs₂, ŵᵢ·fy] (Eq.N3)'],
                      ['Layer Training', 'All layers train on full y', 'All layers train on full y (same)'],
                      ['Model Count', '3 layers', `${sel.novelty.layers} layers (adaptive)`],
                    ].map(([aspect, base, nov]) => (
                      <tr key={aspect}>
                        <td style={{ fontWeight:600, color:'var(--text)', fontSize:12 }}>{aspect}</td>
                        <td style={{ color:'var(--text2)', fontSize:12 }}>{base}</td>
                        <td style={{ color:NOVEL_CLR, fontSize:12 }}>{nov}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
