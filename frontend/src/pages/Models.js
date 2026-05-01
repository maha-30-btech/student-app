import React, { useEffect, useState } from 'react';
import { getModels, getModel } from '../services/api';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';

function LayerDiagram({ layerMeta }) {
  if (!layerMeta || layerMeta.length === 0) return null;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text3)', marginBottom: 8, fontFamily: 'var(--mono)', letterSpacing: 1 }}>
        MULTI-LAYER STRUCTURE
      </div>
      <div className="layer-diagram">
        {layerMeta.map((layer, i) => (
          <React.Fragment key={i}>
            <div className="layer-box">
              <div className="lb-title">Layer {i+1}  θ={layer.threshold}</div>
              <div className="lb-body" style={{ fontSize: 11 }}>
                {layer.feat_orig.length} orig features<br/>
                {layer.n_prev_gen > 0 ? `+ ${layer.n_prev_gen} gen. feat${layer.n_prev_gen>1?'s':''}` : 'No gen. feats'}
              </div>
            </div>
            {i < layerMeta.length - 1 && <div className="layer-arrow">→</div>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

export default function Models() {
  const [models,   setModels]   = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail,   setDetail]   = useState(null);
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    getModels().then(r => {
      setModels(r.data.models || []);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setDetail(null);
    getModel(selected).then(r => setDetail(r.data.model));
  }, [selected]);

  if (loading) return <div style={{ display:'flex',gap:10,color:'var(--text2)' }}><div className="spinner"/>Loading...</div>;

  const sel = selected ? detail : null;

  // Radar chart data
  const radarData = sel ? [
    { metric: 'MAC', value: +(sel.metrics.MAC * 100).toFixed(1) },
    { metric: 'Acc(1-MAE)', value: Math.max(0, +(100 - sel.metrics.MAE * 5).toFixed(1)) },
    { metric: 'Acc(1-RMSE)', value: Math.max(0, +(100 - sel.metrics.RMSE * 5).toFixed(1)) },
    { metric: 'Stability', value: Math.max(0, +(100 - sel.metrics.SD * 5).toFixed(1)) },
    { metric: 'Layers', value: (sel.layer_meta?.length || 1) * 33 },
  ] : [];

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">Trained Models</div>
        <div className="page-sub">DCatBoostF instances — one per dataset × target combination</div>
      </div>

      <div className="grid-2">
        {/* Model list */}
        <div>
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13, color: 'var(--text2)' }}>
            {models.length} models available
          </div>
          {['math','port','exam'].map(ds => {
            const dsModels = models.filter(m => m.dataset === ds);
            if (!dsModels.length) return null;
            return (
              <div key={ds} style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 8 }}>
                  {{ math:'Mathematics Course', port:'Portuguese Course', exam:'Exam Dataset' }[ds]}
                </div>
                {dsModels.map(m => (
                  <div
                    key={m.model_id}
                    onClick={() => setSelected(m.model_id)}
                    style={{
                      cursor: 'pointer',
                      padding: '12px 16px',
                      borderRadius: 'var(--radius)',
                      border: `1px solid ${selected === m.model_id ? 'var(--accent)' : 'var(--border)'}`,
                      background: selected === m.model_id ? 'rgba(79,255,176,0.06)' : 'var(--bg2)',
                      marginBottom: 8,
                      transition: 'all 0.15s',
                    }}
                  >
                    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                      <span style={{ fontFamily:'var(--mono)', fontSize:13, color: selected===m.model_id?'var(--accent)':'var(--text)' }}>
                        {m.dataset}/{m.target}
                      </span>
                      <div style={{ display:'flex', gap:6 }}>
                        <span className="badge badge-green">MAE {m.metrics.MAE}</span>
                        <span className="badge badge-blue">{m.layers} layers</span>
                      </div>
                    </div>
                    <div style={{ marginTop:4, fontSize:12, color:'var(--text3)' }}>
                      RMSE {m.metrics.RMSE} · MAC {(m.metrics.MAC*100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            );
          })}
        </div>

        {/* Detail panel */}
        <div>
          {!selected ? (
            <div className="card" style={{ textAlign:'center', padding:40 }}>
              <div style={{ fontSize:40, opacity:0.2, marginBottom:12 }}>◉</div>
              <div style={{ color:'var(--text2)' }}>Click a model to see details</div>
            </div>
          ) : !detail ? (
            <div className="card" style={{ textAlign:'center', padding:40 }}>
              <div className="spinner" />
            </div>
          ) : (
            <div className="fade-in">
              <div className="card" style={{ marginBottom:14 }}>
                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:14 }}>
                  <div>
                    <div style={{ fontFamily:'var(--mono)', fontSize:16, color:'var(--accent)' }}>{sel?.model_id}</div>
                    <div style={{ color:'var(--text2)', fontSize:13, marginTop:2 }}>
                      Trained: {sel?.trained_at ? new Date(sel.trained_at).toLocaleString() : 'N/A'}
                    </div>
                  </div>
                  <span className="badge badge-purple">{sel?.n_features} features</span>
                </div>

                <div className="metric-row" style={{ marginBottom:16 }}>
                  {Object.entries(sel?.metrics || {}).map(([k,v]) => (
                    <div key={k} className="metric-pill">
                      <div className="mp-label">{k}</div>
                      <div className="mp-val">{typeof v==='number' ? v.toFixed(4) : v}</div>
                    </div>
                  ))}
                </div>

                <div style={{ fontSize:12, color:'var(--text2)', marginBottom:8 }}>
                  Best Params: &nbsp;
                  {Object.entries(sel?.best_params || {}).map(([k,v]) => (
                    <span key={k} className="badge badge-blue" style={{ marginRight:4 }}>
                      {k}: {Array.isArray(v) ? `[${v}]` : v}
                    </span>
                  ))}
                </div>

                <LayerDiagram layerMeta={sel?.layer_meta} />
              </div>

              {/* Radar chart */}
              <div className="card" style={{ marginBottom:14 }}>
                <div style={{ fontWeight:600, marginBottom:10, fontSize:13 }}>Performance Radar</div>
                <ResponsiveContainer width="100%" height={200}>
                  <RadarChart data={radarData} margin={{ top:10, right:30, bottom:10, left:30 }}>
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize:11, fill:'var(--text2)' }} />
                    <Radar dataKey="value" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.15} strokeWidth={2} />
                    <Tooltip contentStyle={{ background:'var(--bg3)', border:'1px solid var(--border)', fontSize:12 }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Top feature importances */}
              <div className="card">
                <div style={{ fontWeight:600, marginBottom:12 }}>
                  Top Feature Importances &nbsp;
                  <span className="badge badge-yellow">Eq. (1): fs = RF(X,y)</span>
                </div>
                {(sel?.feature_importances || []).slice(0,10).map((f, i) => {
                  const max = sel.feature_importances[0].importance;
                  return (
                    <div key={f.feature} className="imp-bar-wrap">
                      <div className="imp-bar-label">
                        <span className="imp-bar-name">{f.feature}</span>
                        <span className="imp-bar-val">{(f.importance*100).toFixed(2)}%</span>
                      </div>
                      <div className="imp-bar-track">
                        <div className="imp-bar-fill" style={{ width:`${(f.importance/max)*100}%` }} />
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
