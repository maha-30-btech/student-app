import React, { useState, useEffect } from 'react';
import { getModels, getFeatures, predict } from '../services/api';

// ── Feature metadata: labels, types, ranges ───────────────────────────────────
const FEATURE_META = {
  school:     { label:'School',           type:'select', options:['GP (0)','MS (1)'],             vals:[0,1] },
  sex:        { label:'Sex',              type:'select', options:['Female (0)','Male (1)'],        vals:[0,1] },
  age:        { label:'Age',              type:'range',  min:15, max:22, step:1 },
  address:    { label:'Address',          type:'select', options:['Urban (0)','Rural (1)'],        vals:[0,1] },
  famsize:    { label:'Family Size',      type:'select', options:['≤3 (0)','>3 (1)'],              vals:[0,1] },
  Pstatus:    { label:'Parent Status',    type:'select', options:['Together (0)','Apart (1)'],     vals:[0,1] },
  Medu:       { label:"Mother's Edu",     type:'range',  min:0, max:4, step:1 },
  Fedu:       { label:"Father's Edu",     type:'range',  min:0, max:4, step:1 },
  Mjob:       { label:"Mother's Job",     type:'range',  min:0, max:4, step:1 },
  Fjob:       { label:"Father's Job",     type:'range',  min:0, max:4, step:1 },
  reason:     { label:'School Reason',    type:'range',  min:0, max:3, step:1 },
  guardian:   { label:'Guardian',         type:'range',  min:0, max:2, step:1 },
  traveltime: { label:'Travel Time',      type:'range',  min:1, max:4, step:1 },
  studytime:  { label:'Study Time',       type:'range',  min:1, max:4, step:1 },
  failures:   { label:'Past Failures',    type:'range',  min:0, max:3, step:1 },
  schoolsup:  { label:'School Support',   type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  famsup:     { label:'Family Support',   type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  paid:       { label:'Extra Paid Class', type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  activities: { label:'Activities',       type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  nursery:    { label:'Nursery School',   type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  higher:     { label:'Wants Higher Edu', type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  internet:   { label:'Internet Access',  type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  romantic:   { label:'Romantic Rel.',    type:'select', options:['No (0)','Yes (1)'],            vals:[0,1] },
  famrel:     { label:'Family Relations', type:'range',  min:1, max:5, step:1 },
  freetime:   { label:'Free Time',        type:'range',  min:1, max:5, step:1 },
  goout:      { label:'Go Out',           type:'range',  min:1, max:5, step:1 },
  Dalc:       { label:'Workday Alcohol',  type:'range',  min:1, max:5, step:1 },
  Walc:       { label:'Weekend Alcohol',  type:'range',  min:1, max:5, step:1 },
  health:     { label:'Health Status',    type:'range',  min:1, max:5, step:1 },
  absences:   { label:'Absences',         type:'range',  min:0, max:93, step:1 },
  G1:         { label:'Period Grade 1',   type:'range',  min:0, max:20, step:1 },
  G2:         { label:'Period Grade 2',   type:'range',  min:0, max:20, step:1 },
  gender:     { label:'Gender',           type:'select', options:['Female (0)','Male (1)'],       vals:[0,1] },
  race:       { label:'Race/Ethnicity',   type:'range',  min:0, max:4, step:1 },
  parent_edu: { label:'Parent Education', type:'range',  min:0, max:5, step:1 },
  lunch:      { label:'Lunch Type',       type:'select', options:['Standard (0)','Free/Reduced (1)'], vals:[0,1] },
  test_prep:  { label:'Test Prep Course', type:'select', options:['None (0)','Completed (1)'],   vals:[0,1] },
};

function defaultVal(name, min, max) {
  if (name === 'failures')  return 0;
  if (name === 'studytime') return 2;
  if (name === 'health')    return 3;
  if (name === 'higher')    return 1;
  if (min !== undefined && max !== undefined) return Math.round((min + max) / 2);
  return 0;
}

// ── Semicircle gauge ──────────────────────────────────────────────────────────
function ScoreGauge({ value, max = 20 }) {
  const pct   = Math.min((value / max) * 100, 100);
  const color = pct > 70 ? 'var(--accent)' : pct > 40 ? 'var(--accent4)' : 'var(--accent3)';
  return (
    <div style={{ position:'relative', width:160, height:80, margin:'0 auto' }}>
      <svg viewBox="0 0 160 80" width="160" height="80">
        <path d="M10 80 A70 70 0 0 1 150 80" fill="none" stroke="var(--bg3)" strokeWidth="14" strokeLinecap="round"/>
        <path d="M10 80 A70 70 0 0 1 150 80" fill="none" stroke={color} strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${pct * 2.199} 219.9`}
          style={{ transition:'stroke-dasharray 0.8s ease, stroke 0.4s' }}
        />
      </svg>
      <div style={{ position:'absolute', bottom:4, left:0, right:0, textAlign:'center' }}>
        <span style={{ fontFamily:'var(--mono)', fontSize:32, fontWeight:700, color }}>{value}</span>
        <span style={{ color:'var(--text3)', fontSize:15 }}>/{max}</span>
      </div>
    </div>
  );
}

// ── Classification result badge ───────────────────────────────────────────────
function ClassBadge({ value, isScore }) {
  const threshold = isScore ? 60 : 10;
  const pass = value >= threshold;
  return (
    <div style={{
      display:'inline-flex', alignItems:'center', gap:8,
      background: pass ? 'rgba(79,255,176,0.12)' : 'rgba(255,107,107,0.12)',
      border: `1px solid ${pass ? 'var(--accent)' : 'var(--accent3)'}`,
      borderRadius:8, padding:'8px 16px', marginTop:10
    }}>
      <span style={{ fontSize:18 }}>{pass ? '✓' : '⚠'}</span>
      <div>
        <div style={{ fontWeight:700, color: pass ? 'var(--accent)' : 'var(--accent3)', fontSize:14 }}>
          {pass ? 'PASS' : 'FAIL / AT-RISK'}
        </div>
        <div style={{ fontSize:11, color:'var(--text2)' }}>
          {isScore
            ? (value >= 70 ? 'Above average' : value >= 60 ? 'Passing' : 'Below passing threshold (60)')
            : (value >= 15 ? 'Excellent — Grade A' : value >= 10 ? 'Passing grade' : 'Below pass threshold (10)')}
        </div>
      </div>
    </div>
  );
}

export default function Predict() {
  const [models,     setModels]     = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [featData,   setFeatData]   = useState(null);   // { features:[], featureImportances:[] }
  const [values,     setValues]     = useState({});
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [loadingF,   setLoadingF]   = useState(false);
  const [error,      setError]      = useState('');

  // Load model list once
  useEffect(() => {
    getModels()
      .then(r => {
        const list = r.data?.models || [];
        setModels(list);
        if (list.length > 0) setSelectedId(list[0].model_id);
      })
      .catch(err => setError('Failed to load models: ' + err.message));
  }, []);

  // Load features whenever selected model changes
  useEffect(() => {
    if (!selectedId) return;
    setResult(null);
    setError('');
    setFeatData(null);
    setLoadingF(true);

    getFeatures(selectedId)
      .then(r => {
        const data = r.data;
        // Validate response shape
        if (!data || !Array.isArray(data.features)) {
          throw new Error('Invalid features response from server');
        }
        setFeatData(data);

        // Build default values
        const defaults = {};
        data.features.forEach(f => {
          const meta = FEATURE_META[f.name];
          if (meta) {
            defaults[f.name] = defaultVal(f.name, meta.min, meta.max);
          } else {
            // Fallback: midpoint of norm range
            const mn = typeof f.min === 'number' ? f.min : 0;
            const mx = typeof f.max === 'number' ? f.max : 1;
            defaults[f.name] = Math.round((mn + mx) / 2);
          }
        });
        setValues(defaults);
      })
      .catch(err => {
        setError('Failed to load features: ' + err.message);
        setFeatData(null);
      })
      .finally(() => setLoadingF(false));
  }, [selectedId]);

  const handleChange = (name, val) =>
    setValues(v => ({ ...v, [name]: parseFloat(val) }));

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const r = await predict({ modelId: selectedId, features: values });
      if (!r.data?.success) throw new Error(r.data?.error || 'Prediction failed');
      setResult(r.data);
    } catch (e) {
      const msg = e.response?.data?.error || e.message || 'Prediction failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const selectedMeta = models.find(m => m.model_id === selectedId);
  const isScore      = selectedId?.includes('score');
  const maxVal       = isScore ? 100 : 20;
  const featureList  = featData?.features || [];
  const importances  = featData?.featureImportances || [];

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">Make a Prediction</div>
        <div className="page-sub">Enter student features → run inference → get grade prediction</div>
      </div>

      {/* ── Model selector ─────────────────────────────────── */}
      <div className="card" style={{ marginBottom:20 }}>
        <div style={{ fontWeight:600, marginBottom:12 }}>Select Model</div>
        {models.length === 0 ? (
          <div style={{ color:'var(--accent3)', fontSize:13 }}>
            ⚠ No models found. Run <code>python novelty_train.py</code> in the ml/ folder first.
          </div>
        ) : (
          <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
            {models.map(m => (
              <button key={m.model_id}
                onClick={() => setSelectedId(m.model_id)}
                className={`btn ${selectedId === m.model_id ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize:12, padding:'6px 12px' }}>
                {m.dataset}/{m.target}
              </button>
            ))}
          </div>
        )}

        {selectedMeta?.metrics && (
          <div className="metric-row" style={{ marginTop:14 }}>
            {Object.entries(selectedMeta.metrics).map(([k, v]) => (
              <div key={k} className="metric-pill">
                <div className="mp-label">{k}</div>
                <div className="mp-val">{typeof v === 'number' ? v.toFixed(4) : v}</div>
              </div>
            ))}
            {selectedMeta.layers != null && (
              <div className="metric-pill">
                <div className="mp-label">LAYERS</div>
                <div className="mp-val">{selectedMeta.layers}</div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="grid-2">
        {/* ── Feature inputs ──────────────────────────────── */}
        <div className="card">
          <div style={{ fontWeight:600, marginBottom:16 }}>
            Input Features
            {featureList.length > 0 &&
              <span className="badge badge-blue" style={{ marginLeft:8 }}>{featureList.length} features</span>}
          </div>

          {loadingF ? (
            <div style={{ display:'flex', gap:10, color:'var(--text2)', padding:20 }}>
              <div className="spinner" /> Loading features...
            </div>
          ) : featureList.length === 0 ? (
            <div style={{ color:'var(--text2)', fontSize:13, padding:10 }}>
              Select a model above to load its input features.
            </div>
          ) : (
            <>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'14px 20px', marginBottom:20 }}>
                {featureList.map(f => {
                  const meta = FEATURE_META[f.name];
                  const val  = values[f.name] ?? 0;
                  const label = meta?.label || f.name;

                  return (
                    <div key={f.name}>
                      <div style={{ fontSize:12, color:'var(--text2)', marginBottom:5,
                        display:'flex', justifyContent:'space-between' }}>
                        <span>{label}</span>
                        <span style={{ fontFamily:'var(--mono)', color:'var(--accent)', fontSize:11 }}>
                          {val}
                        </span>
                      </div>

                      {meta?.type === 'select' ? (
                        <select
                          value={val}
                          onChange={e => handleChange(f.name, e.target.value)}
                          style={{ width:'100%' }}>
                          {meta.options.map((opt, i) => (
                            <option key={i} value={meta.vals[i]}>{opt}</option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="range"
                          min={meta?.min ?? 0}
                          max={meta?.max ?? 10}
                          step={meta?.step ?? 1}
                          value={val}
                          onChange={e => handleChange(f.name, e.target.value)}
                          style={{ width:'100%', accentColor:'var(--accent)' }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>

              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading}
                style={{ width:'100%', justifyContent:'center', fontSize:15, padding:'12px' }}>
                {loading
                  ? <><div className="spinner" style={{ width:16, height:16 }}/> Running inference...</>
                  : '◈ Predict Student Performance'}
              </button>

              {error && (
                <div style={{ color:'var(--accent3)', fontSize:13, marginTop:10,
                  background:'rgba(255,107,107,0.08)', border:'1px solid rgba(255,107,107,0.3)',
                  borderRadius:6, padding:'8px 12px' }}>
                  ⚠ {error}
                </div>
              )}
            </>
          )}
        </div>

        {/* ── Result + Importances ────────────────────────── */}
        <div>
          {result ? (
            <div className="result-card fade-in" style={{ marginBottom:16 }}>
              <div style={{ fontSize:11, color:'var(--text2)', marginBottom:10,
                fontFamily:'var(--mono)', letterSpacing:1.5 }}>
                PREDICTED {result.target?.toUpperCase()}
              </div>

              <ScoreGauge value={result.prediction} max={maxVal} />
              <ClassBadge value={result.prediction} isScore={isScore} />

              {/* Classification metrics from model */}
              {result.clsMetrics && (
                <div style={{ marginTop:14, display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                  {['accuracy','precision','recall','f1'].map(k => (
                    <div key={k} style={{ background:'var(--bg3)', borderRadius:6, padding:'8px 10px', textAlign:'center' }}>
                      <div style={{ fontSize:10, color:'var(--text3)', marginBottom:2, textTransform:'uppercase', letterSpacing:1 }}>{k}</div>
                      <div style={{ fontFamily:'var(--mono)', fontSize:16, fontWeight:700, color:'var(--accent)' }}>
                        {(result.clsMetrics[k] * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div style={{ marginTop:14, display:'flex', gap:6, justifyContent:'center', flexWrap:'wrap' }}>
                <span className="badge badge-purple">{result.modelType || 'DCatBoostF'}</span>
                <span className="badge badge-blue">{result.layers} layers</span>
                <span className="badge badge-green">✓ Saved to DB</span>
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign:'center', padding:48, marginBottom:16 }}>
              <div style={{ fontSize:52, marginBottom:14, opacity:0.2 }}>◈</div>
              <div style={{ color:'var(--text2)', fontSize:14, lineHeight:1.8 }}>
                Set features on the left<br/>
                then click<br/>
                <strong style={{ color:'var(--text)' }}>Predict Student Performance</strong>
              </div>
            </div>
          )}

          {/* Feature importances */}
          {importances.length > 0 && (
            <div className="card">
              <div style={{ fontWeight:600, marginBottom:14, display:'flex', justifyContent:'space-between' }}>
                Feature Importances
                <span className="badge badge-yellow" style={{ fontSize:10 }}>Eq.N1 SHAP</span>
              </div>
              <div style={{ maxHeight:300, overflowY:'auto' }}>
                {importances.slice(0, 15).map((f, i) => {
                  const maxImp = importances[0]?.importance || 1;
                  const pct    = (f.importance / maxImp) * 100;
                  return (
                    <div key={f.feature} className="imp-bar-wrap">
                      <div className="imp-bar-label">
                        <span className="imp-bar-name">{f.feature}</span>
                        <span className="imp-bar-val">{(f.importance * 100).toFixed(2)}%</span>
                      </div>
                      <div className="imp-bar-track">
                        <div className="imp-bar-fill" style={{ width:`${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
