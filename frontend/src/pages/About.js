import React from 'react';

function Eq({ id, formula, desc, novel }) {
  const color = novel ? 'var(--accent)' : 'var(--accent2)';
  const bc    = novel ? 'rgba(79,255,176,0.06)' : 'rgba(0,212,255,0.04)';
  return (
    <div style={{ background:bc, border:`1px solid ${color}44`, borderLeft:`3px solid ${color}`,
      borderRadius:'var(--radius)', padding:'14px 16px', marginBottom:14 }}>
      <div style={{ display:'flex', gap:10, alignItems:'baseline', marginBottom:6 }}>
        <span style={{ background:`${color}20`, color, borderRadius:4, padding:'2px 8px',
          fontFamily:'var(--mono)', fontSize:11, fontWeight:700, flexShrink:0 }}>
          {novel ? '★ ' : ''}{id}
        </span>
        <code style={{ fontFamily:'var(--mono)', fontSize:13, color, letterSpacing:0.5 }}>{formula}</code>
        {novel && <span className="badge badge-green" style={{ fontSize:9 }}>NOVELTY</span>}
      </div>
      <div style={{ color:'var(--text2)', fontSize:13, lineHeight:1.7 }}>{desc}</div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom:32 }}>
      <div style={{ fontSize:12, fontWeight:600, letterSpacing:2, textTransform:'uppercase',
        color:'var(--text3)', marginBottom:14, paddingBottom:8, borderBottom:'1px solid var(--border)' }}>
        {title}
      </div>
      {children}
    </div>
  );
}

export default function About() {
  return (
    <div className="fade-in" style={{ maxWidth:820 }}>
      <div className="page-header">
        <div className="page-title">About & Algorithm</div>
        <div className="page-sub">All equations — base paper (Eq.1–10) + novelty (Eq.N1–N3)</div>
      </div>

      {/* Paper ref */}
      <div className="card" style={{ marginBottom:28, borderTop:'3px solid var(--accent2)' }}>
        <div style={{ fontFamily:'var(--mono)', fontSize:11, color:'var(--text3)', marginBottom:6 }}>BASE PAPER</div>
        <div style={{ fontWeight:700, fontSize:15, marginBottom:4 }}>
          A Feature Importance-Based Multi-Layer CatBoost for Student Performance Prediction
        </div>
        <div style={{ color:'var(--text2)', fontSize:13, marginBottom:8 }}>
          Zongwen Fan, Jin Gou, Shaoyuan Weng · <em>IEEE TKDE</em>, Vol. 36, No. 11, Nov. 2024
        </div>
        <div style={{ fontFamily:'var(--mono)', fontSize:11, color:'var(--text3)', marginBottom:6, marginTop:14 }}>NOVELTY EXTENSION</div>
        <div style={{ fontWeight:700, fontSize:15, marginBottom:4, color:'var(--accent)' }}>
          SHAP-Guided Adaptive Threshold Multi-Layer CatBoost (SHAPAdaptiveDCatBoostF)
        </div>
        <div className="tag-row">
          <span className="badge badge-green">Manual TreeSHAP (Eq.N1)</span>
          <span className="badge badge-green">Adaptive θ (Eq.N2)</span>
          <span className="badge badge-green">Weighted Fusion (Eq.N3)</span>
          <span className="badge badge-blue">No built-in libraries</span>
        </div>
      </div>

      <Section title="Base Paper Metrics (Equations 7–10)">
        <Eq id="7"  formula="MAE = (1/n) · Σ|ŷᵢ − yᵢ|"
          desc="Mean Absolute Error — primary regression metric. Lower is better." />
        <Eq id="8"  formula="SD = √[(1/n) · Σ(eᵢ − ē)²]"
          desc="Standard deviation of prediction errors. Measures consistency." />
        <Eq id="9"  formula="RMSE = √[(1/n) · Σ(ŷᵢ − yᵢ)²]"
          desc="Root Mean Square Error — penalises large errors more than MAE." />
        <Eq id="10" formula="MAC = (yᵀŷ)² / [(yᵀy)(ŷᵀŷ)]"
          desc="Model Accuracy Criterion — cosine similarity squared. Higher is better (max=1)." />
      </Section>

      <Section title="Base Paper Algorithm (Equations 1–6)">
        <Eq id="1" formula="fs = RF(X, y)"
          desc="Feature importance from Random Forest. 50 trees with default params. Split-count proxy per feature." />
        <Eq id="2" formula="fs = {f₁,…,fₙ}  s.t.  F(f₁) ≤ … ≤ F(fₙ)"
          desc="Sort features ascending by importance — least important first." />
        <Eq id="3" formula="fs₁ ← Σᵢ(F(fᵢ) − θ₁) > 0"
          desc="Accumulate least-important features until cumulative importance exceeds θ₁=0.05." />
        <Eq id="4" formula="fy = CatBoost(X_layer1, y)"
          desc="Train CatBoost on Layer 1 features, generate synthetic feature fy from predictions." />
        <Eq id="5" formula="fs₂ ← Σᵢ(F(fᵢ) − θ₂) > 0"
          desc="Continue accumulation for Layer 2 features, threshold θ₂=0.05." />
        <Eq id="6" formula="fs₂ ← [fs₂, fy]"
          desc="Plain concatenation of original features with generated feature. (Replaced by Eq.N3 in novelty.)" />
      </Section>

      <Section title="★ Novelty Equations (N1–N3) — SHAPAdaptiveDCatBoostF">
        <Eq id="N1" novel
          formula="φᵢ = (1/T) Σₜ Σ_{n∈treeₜ, feat(n)=i} [nL/N · nR/N]"
          desc="Manual TreeSHAP-inspired importance. For every internal node in every Random Forest tree, accumulate the product (nL/N)·(nR/N) — the fraction of samples going left times right. This coverage product measures how much the split disperses data, analogous to the Shapley interaction. Replaces simple split-count (Eq.1). No shap library — fully manual traversal." />
        <Eq id="N2" novel
          formula="θₗ = clip( μₗ + γ·σₗ , 0.03, 0.40 )"
          desc="Adaptive threshold for each layer. μₗ = mean SHAP importance of features in layer l's pool; σₗ = their std; γ = 0.5 (sensitivity). Replaces fixed θ₁=θ₂=0.05. High-variance SHAP distributions get larger thresholds (more features per layer); low-variance gets smaller (fewer features). Clipped to [0.03, 0.40] for stability." />
        <Eq id="N3" novel
          formula="fs₂ ← [fs₂,  softmax(φ₁)·fy₁,  softmax(φ₂)·fy₂, …]"
          desc="SHAP-weighted feature fusion. Generated feature fyₖ from Layer k is multiplied by its softmax-normalised SHAP weight before being passed to the next layer. High-SHAP generated features contribute more. Replaces plain concatenation of Eq.6. softmax computed from scratch: exp(v)/Σexp(v)." />
      </Section>

      <Section title="Classification Metrics (Novelty addition)">
        <div style={{ background:'var(--bg3)', border:'1px solid var(--border)', borderRadius:8, padding:14, marginBottom:14, fontSize:13 }}>
          <div style={{ color:'var(--text2)', lineHeight:1.8 }}>
            Continuous predictions are converted to binary classes:<br/>
            <span style={{ fontFamily:'var(--mono)', color:'var(--accent)' }}>Grade dataset:</span> ≥ 10 → Pass(1), &lt; 10 → Fail(0)<br/>
            <span style={{ fontFamily:'var(--mono)', color:'var(--accent)' }}>Score dataset:</span> ≥ 60 → Pass(1), &lt; 60 → Fail(0)<br/>
          </div>
        </div>
        {[
          ['Accuracy',  'Accuracy  = (TP + TN) / (TP + TN + FP + FN)',   'Overall correct predictions fraction.'],
          ['Precision', 'Precision = TP / (TP + FP)',                     'Of all predicted Pass, how many are truly Pass.'],
          ['Recall',    'Recall    = TP / (TP + FN)',                     'Of all actual Pass, how many were correctly identified.'],
          ['F1',        'F1        = 2 · Precision · Recall / (P + R)',   'Harmonic mean of Precision and Recall. Best single metric.'],
        ].map(([name, formula, desc]) => (
          <Eq key={name} id={name} formula={formula} desc={desc} />
        ))}
      </Section>

      <Section title="Algorithm 1 — Grid Search with 10-Fold CV (Base Paper)">
        <div className="card card-sm" style={{ fontFamily:'var(--mono)', fontSize:12, lineHeight:2, color:'var(--text2)' }}>
          <div>Candidates: n_estimators ∈ [100,300,500],  max_depth ∈ [3,4,5]</div>
          <div style={{ color:'var(--accent)' }}>for i in [100,300,500]:  for j in [3,4,5]:</div>
          <div style={{ paddingLeft:24 }}>10-fold CV → average MAE</div>
          <div>Select (i*, j*) = argmin avg_MAE  →  train final model</div>
          <div>Repeat 20 runs to reduce randomness influence</div>
        </div>
      </Section>

      <Section title="Ordered Boosting (CatBoost core)">
        <div className="card card-sm" style={{ fontFamily:'var(--mono)', fontSize:12, lineHeight:2, color:'var(--text2)' }}>
          <div style={{ color:'var(--accent)' }}>Ordered gradient (avoids target leakage):</div>
          <div>σ = random_permutation(n)</div>
          <div>for sample i at position k in σ:</div>
          <div style={{ paddingLeft:24 }}>F_ordered[i] = mean(y[σ[0..k-1]])</div>
          <div style={{ paddingLeft:24, color:'var(--accent3)' }}>g[i] = F_ordered[i] - y[i]</div>
          <div>XGBTree leaf: value = -G/(H+λ),  gain = 0.5·[GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)]</div>
        </div>
      </Section>

      <Section title="Tech Stack">
        <div className="grid-3">
          {[
            { title:'ML — Python', items:['NumPy only (no sklearn/torch)','Decision Tree from scratch','Random Forest from scratch','Manual TreeSHAP (Eq.N1)','Ordered Boosting (CatBoost)','XGBTree from scratch','JSON model serialization'] },
            { title:'Backend — Node.js', items:['Express 4','MongoDB + Mongoose','JS inference engine','Mirrors Python predict path','REST: 8 endpoints','/api/comparison (new)','Model cache layer'] },
            { title:'Frontend — React', items:['React 18 + Router v6','Recharts (Bar,Radar,Line)','Comparison page (new)','Confusion matrix widget','Classification metrics','Dark sci-fi design system','Axios API calls'] },
          ].map(({ title, items }) => (
            <div key={title} className="card">
              <div style={{ fontWeight:600, marginBottom:10, color:'var(--accent)', fontFamily:'var(--mono)', fontSize:11, letterSpacing:1 }}>{title.toUpperCase()}</div>
              {items.map(it => (
                <div key={it} style={{ fontSize:12, color:'var(--text2)', padding:'3px 0', borderBottom:'1px solid var(--border)' }}>→ {it}</div>
              ))}
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
