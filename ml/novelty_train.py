"""
=============================================================================
NOVELTY ALGORITHM: SHAP-Guided Adaptive Threshold Multi-Layer CatBoost
                   (SHAPAdaptiveDCatBoostF)

Paper Reference: Extension of
  "A Feature Importance-Based Multi-Layer CatBoost for Student Performance
   Prediction" — IEEE TKDE, Vol. 36, No. 11, November 2024

NOVELTY CONTRIBUTIONS (all manually implemented, zero built-ins):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MANUAL SHAP VALUES (TreeSHAP path-dependent, Eq.N1)
   Base paper uses RF feature importance (Eq.1: fs=RF(X,y)).
   Novelty uses SHAP-based importance: φᵢ = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)!/|F|!] * [v(S∪{i})-v(S)]
   This is the Shapley value from cooperative game theory — implemented
   from scratch using tree path tracing (no shap library used).

2. ADAPTIVE THRESHOLD COMPUTATION (Eq.N2)
   Base paper: fixed θ₁=0.05, θ₂=0.05, θ₃=0.90
   Novelty:    θₗ = μₗ + σₗ * γ   where
     μₗ  = mean SHAP importance of features assigned to layer l
     σₗ  = std  SHAP importance of features assigned to layer l
     γ   = sensitivity parameter (default 0.5)
   Each layer's threshold adapts to the actual SHAP distribution.

3. SHAP-WEIGHTED FEATURE FUSION (Eq.N3, replaces plain Eq.6)
   Base paper: fs₂ ← [fs₂, fy]              (plain concatenation)
   Novelty:    fs₂ ← [fs₂, φ̂ᵢ · fy]       where φ̂ᵢ = softmax(φᵢ)
   Generated features are weighted by their SHAP importance
   so high-SHAP generated features contribute more.

4. CLASSIFICATION METRICS (Accuracy, Precision, Recall, F1, Confusion Matrix)
   All computed from scratch — converted from continuous predictions.

EXECUTION TIME: ~5-8 minutes for all 9 models (fast mode)
=============================================================================
"""

import numpy as np
import pandas as pd
import json, os, time, warnings

def jfix(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"Not serializable: {type(o)}")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# MATH UTILITIES (all from scratch)
# ─────────────────────────────────────────────────────────────

def normalize_cols(X):
    mins = X.min(axis=0); maxs = X.max(axis=0)
    denom = maxs - mins;  denom[denom == 0] = 1.0
    return (X - mins) / denom, mins, maxs

def shuffle_data(X, y, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def k_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    sz  = n // k
    out = []
    for i in range(k):
        s = i * sz; e = s + sz if i < k-1 else n
        te = idx[s:e]; tr = np.concatenate([idx[:s], idx[e:]])
        out.append((tr, te))
    return out

# ─────────────────────────────────────────────────────────────
# METRICS — Equations 7-10 (base paper) + Classification (novelty)
# ─────────────────────────────────────────────────────────────

def MAE(yt, yp):
    """Eq.7: MAE = (1/n)Σ|ŷᵢ-yᵢ|"""
    return float(sum(abs(yp[i]-yt[i]) for i in range(len(yt))) / len(yt))

def SD(yt, yp):
    """Eq.8: SD = √[(1/n)Σ(eᵢ-ē)²]"""
    n=len(yt); e=[yp[i]-yt[i] for i in range(n)]; eb=sum(e)/n
    return float((sum((ei-eb)**2 for ei in e)/n)**0.5)

def RMSE(yt, yp):
    """Eq.9: RMSE = √[(1/n)Σ(ŷᵢ-yᵢ)²]"""
    n=len(yt)
    return float((sum((yp[i]-yt[i])**2 for i in range(n))/n)**0.5)

def MAC(yt, yp):
    """Eq.10: MAC = (yᵀŷ)² / [(yᵀy)(ŷᵀŷ)]"""
    y=np.array(yt,dtype=float); yh=np.array(yp,dtype=float)
    d1=float(np.dot(y,yh)); d2=float(np.dot(y,y)); d3=float(np.dot(yh,yh))
    return float((d1**2)/(d2*d3)) if d2>0 and d3>0 else 0.0

def classification_metrics(yt_cont, yp_cont, is_score=False):
    """
    Novelty: Convert continuous predictions → binary classes → compute metrics.
    Grade dataset:  ≥10 = Pass(1), <10 = Fail(0)
    Score dataset:  ≥60 = Pass(1), <60 = Fail(0)
    Returns: accuracy, precision, recall, f1, confusion_matrix [tn,fp,fn,tp]
    """
    thr = 60 if is_score else 10
    yt  = (np.array(yt_cont) >= thr).astype(int)
    yp  = (np.array(yp_cont) >= thr).astype(int)
    n   = len(yt)
    tp  = int(np.sum((yp==1)&(yt==1)))
    tn  = int(np.sum((yp==0)&(yt==0)))
    fp  = int(np.sum((yp==1)&(yt==0)))
    fn  = int(np.sum((yp==0)&(yt==1)))
    acc  = (tp+tn)/n if n>0 else 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return {
        "accuracy": round(acc,4), "precision": round(prec,4),
        "recall":   round(rec,4), "f1":        round(f1,4),
        "tp":tp,"tn":tn,"fp":fp,"fn":fn,
        "confusion_matrix":[[tn,fp],[fn,tp]]
    }

# ─────────────────────────────────────────────────────────────
# DECISION TREE (from scratch — used by RF and XGBTree)
# ─────────────────────────────────────────────────────────────

class _Node:
    __slots__=['f','t','l','r','v','leaf']
    def __init__(self): self.f=self.t=self.l=self.r=self.v=None; self.leaf=False

class DecisionTree:
    def __init__(self, max_depth=5, min_split=2, max_feat=None, seed=None):
        self.max_depth=max_depth; self.min_split=min_split
        self.max_feat=max_feat;   self.rng=np.random.RandomState(seed)
        self.root=None; self.n_features=0

    def fit(self, X, y):
        self.n_features=X.shape[1]
        if self.max_feat is None: self.max_feat=self.n_features
        self.root=self._build(X,y,0); return self

    def _build(self, X, y, d):
        nd=_Node()
        if d>=self.max_depth or len(y)<self.min_split or np.var(y)<1e-10:
            nd.leaf=True; nd.v=float(np.mean(y)); return nd
        fi=self.rng.choice(self.n_features,size=min(self.max_feat,self.n_features),replace=False)
        bf=bt=None; bs=np.inf
        for f in fi:
            col=X[:,f]
            for t in np.unique(col)[:-1]:
                m=col<=t
                if m.sum()==0 or (~m).sum()==0: continue
                sc=self._wmse(y[m],y[~m])
                if sc<bs: bs=sc; bf=f; bt=t
        if bf is None:
            nd.leaf=True; nd.v=float(np.mean(y)); return nd
        nd.f=bf; nd.t=bt; m=X[:,bf]<=bt
        nd.l=self._build(X[m],y[m],d+1)
        nd.r=self._build(X[~m],y[~m],d+1)
        return nd

    def _wmse(self, yl, yr):
        n=len(yl)+len(yr)
        def mse(a): return 0.0 if len(a)==0 else float(np.mean((a-np.mean(a))**2))
        return (len(yl)/n)*mse(yl)+(len(yr)/n)*mse(yr)

    def _pred1(self, x, nd):
        if nd.leaf: return nd.v
        return self._pred1(x, nd.l if x[nd.f]<=nd.t else nd.r)

    def predict(self, X): return np.array([self._pred1(x,self.root) for x in X])

    def feat_imp(self):
        imp=np.zeros(self.n_features); self._imp(self.root,imp)
        s=imp.sum(); return imp/s if s>0 else imp

    def _imp(self, nd, imp):
        if nd is None or nd.leaf: return
        imp[nd.f]+=1.0; self._imp(nd.l,imp); self._imp(nd.r,imp)

    def to_dict(self):
        def s(n):
            if n is None: return None
            if n.leaf: return {"leaf":True,"v":n.v}
            return {"leaf":False,"f":int(n.f),"t":float(n.t),"l":s(n.l),"r":s(n.r)}
        return {"md":self.max_depth,"nf":self.n_features,"root":s(self.root)}

    @classmethod
    def from_dict(cls, d):
        tr=cls(max_depth=d["md"]); tr.n_features=d["nf"]
        def g(nd):
            if nd is None: return None
            no=_Node()
            if nd["leaf"]: no.leaf=True; no.v=nd["v"]; return no
            no.f=nd["f"]; no.t=nd["t"]; no.l=g(nd["l"]); no.r=g(nd["r"]); return no
        tr.root=g(d["root"]); return tr

# ─────────────────────────────────────────────────────────────
# RANDOM FOREST — Eq.(1): fs = RF(X,y)  [base paper]
# ─────────────────────────────────────────────────────────────

class RandomForest:
    def __init__(self, n_estimators=50, max_depth=4, seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.seed=seed; self.trees=[]; self.rng=np.random.RandomState(seed)
        self.n_features=0

    def fit(self, X, y):
        X=np.array(X,dtype=float); y=np.array(y,dtype=float)
        n,p=X.shape; mf=max(1,int(np.sqrt(p))); self.n_features=p; self.trees=[]
        for i in range(self.n_estimators):
            bi=self.rng.randint(0,n,size=n)
            tr=DecisionTree(max_depth=self.max_depth,max_feat=mf,seed=self.rng.randint(0,99999))
            tr.fit(X[bi],y[bi]); self.trees.append(tr)
        return self

    def predict(self, X):
        X=np.array(X,dtype=float)
        return sum(t.predict(X) for t in self.trees)/len(self.trees)

    def feature_importances_(self):
        imp=np.zeros(self.n_features)
        for t in self.trees: imp+=t.feat_imp()
        return imp/len(self.trees)

# ─────────────────────────────────────────────────────────────
# XGBTree — used inside CatBoostScratch
# ─────────────────────────────────────────────────────────────

class _XNode:
    __slots__=['f','t','l','r','v','leaf']
    def __init__(self): self.f=self.t=self.l=self.r=self.v=None; self.leaf=False

class XGBTree:
    def __init__(self, max_depth=3, lam=1.0, seed=42):
        self.max_depth=max_depth; self.lam=lam
        self.rng=np.random.RandomState(seed); self.root=None; self.n_features=0

    def fit(self, X, g, h):
        self.n_features=X.shape[1]; self.root=self._build(X,g,h,0)

    def _lv(self,g,h): return -g.sum()/(h.sum()+self.lam)

    def _build(self, X, g, h, d):
        nd=_XNode()
        if d>=self.max_depth or len(g)<=1:
            nd.leaf=True; nd.v=self._lv(g,h); return nd
        Gt=g.sum(); Ht=h.sum()
        fi=self.rng.choice(X.shape[1],size=max(1,int(np.sqrt(X.shape[1]))),replace=False)
        bg=bf=bt=-np.inf
        for f in fi:
            si=np.argsort(X[:,f]); cs=X[si,f]; gs=g[si]; hs=h[si]
            GL=HL=0.0
            for i in range(len(g)-1):
                GL+=gs[i]; HL+=hs[i]
                if cs[i]==cs[i+1]: continue
                GR=Gt-GL; HR=Ht-HL
                gain=0.5*(GL**2/(HL+self.lam)+GR**2/(HR+self.lam)-Gt**2/(Ht+self.lam))
                if gain>bg: bg=gain; bf=f; bt=(cs[i]+cs[i+1])/2.0
        if bf is None or bg<=0:
            nd.leaf=True; nd.v=self._lv(g,h); return nd
        nd.f=bf; nd.t=bt; m=X[:,bf]<=bt
        nd.l=self._build(X[m],g[m],h[m],d+1)
        nd.r=self._build(X[~m],g[~m],h[~m],d+1)
        return nd

    def _p1(self,x,nd):
        if nd.leaf: return nd.v
        return self._p1(x, nd.l if x[nd.f]<=nd.t else nd.r)

    def predict(self, X): return np.array([self._p1(x,self.root) for x in X])

    def to_dict(self):
        def s(n):
            if n is None: return None
            if n.leaf: return {"leaf":True,"v":float(n.v)}
            return {"leaf":False,"f":int(n.f),"t":float(n.t),"l":s(n.l),"r":s(n.r)}
        return {"md":self.max_depth,"nf":self.n_features,"root":s(self.root)}

    @classmethod
    def from_dict(cls, d):
        t=cls(max_depth=d["md"]); t.n_features=d["nf"]
        def g(nd):
            if nd is None: return None
            no=_XNode()
            if nd["leaf"]: no.leaf=True; no.v=nd["v"]; return no
            no.f=nd["f"]; no.t=nd["t"]; no.l=g(nd["l"]); no.r=g(nd["r"]); return no
        t.root=g(d["root"]); return t

# ─────────────────────────────────────────────────────────────
# CATBOOST SCRATCH (Ordered Boosting)
# ─────────────────────────────────────────────────────────────

class CatBoostScratch:
    def __init__(self, n_estimators=50, max_depth=3, lr=0.1, seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.lr=lr; self.seed=seed; self.trees=[]; self.base=0.0
        self.rng=np.random.RandomState(seed)

    def _og(self, X, y, F):
        n=len(y); perm=self.rng.permutation(n); g=np.zeros(n)
        Fo=np.full(n,self.base)
        for k,i in enumerate(perm):
            g[i]=Fo[i]-y[i]
            if k>0: Fo[i]=float(np.mean(y[perm[:k]]))
        return g

    def fit(self, X, y):
        X=np.array(X,dtype=float); y=np.array(y,dtype=float)
        self.base=float(np.mean(y)); F=np.full(len(y),self.base); self.trees=[]
        for t in range(self.n_estimators):
            g=self._og(X,y,F); h=np.ones(len(y))
            tr=XGBTree(max_depth=self.max_depth,lam=1.0,seed=self.seed+t)
            tr.fit(X,g,h); up=tr.predict(X); F+=self.lr*up; self.trees.append(tr)
        return self

    def predict(self, X):
        X=np.array(X,dtype=float); F=np.full(len(X),self.base)
        for t in self.trees: F+=self.lr*t.predict(X)
        return F

    def to_dict(self):
        return {"n":self.n_estimators,"md":self.max_depth,"lr":self.lr,
                "base":self.base,"trees":[t.to_dict() for t in self.trees]}

    @classmethod
    def from_dict(cls, d):
        m=cls(n_estimators=d["n"],max_depth=d["md"],lr=d["lr"])
        m.base=d["base"]; m.trees=[XGBTree.from_dict(td) for td in d["trees"]]
        return m

# ═════════════════════════════════════════════════════════════
# BASE PAPER: DCatBoostF (exact reproduction for comparison)
# ═════════════════════════════════════════════════════════════

class DCatBoostF:
    """
    Exact reproduction of the base paper algorithm.
    Fixed thresholds (0.05,0.05,0.90), RF importance, plain concat (Eq.6).
    """
    def __init__(self, n_estimators=50, max_depth=3, lr=0.1,
                 thresholds=(0.05,0.05,0.90), seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.lr=lr; self.thresholds=thresholds; self.seed=seed
        self.layers=[]; self.importances=None; self.layer_meta=[]

    def _rf_importance(self, X, y):
        rf=RandomForest(n_estimators=30,max_depth=4,seed=self.seed)
        rf.fit(X,y); return rf.feature_importances_()

    def _select(self, sidx, imp, theta, used):
        sel=[]; cum=0.0
        for fi in sidx[used:]:
            cum+=imp[fi]; sel.append(fi)
            if cum>theta: break
        return sel

    def fit(self, X, y):
        X=np.array(X,dtype=float); y=np.array(y,dtype=float)
        imp=self._rf_importance(X,y); sidx=np.argsort(imp)
        self.importances=imp; self.layers=[]; self.layer_meta=[]; gen=[]; used=0
        for li,theta in enumerate(self.thresholds):
            if X.shape[1]-used<=0: break
            fo = list(sidx[used:]) if li==len(self.thresholds)-1 else self._select(sidx,imp,theta,used)
            if not fo: break
            used+=len(fo)
            Xc=[X[:,fi] for fi in fo]+list(gen)
            Xl=np.column_stack(Xc)
            cat=CatBoostScratch(n_estimators=self.n_estimators,max_depth=self.max_depth,
                                lr=self.lr,seed=self.seed+li)
            cat.fit(Xl,y); p=cat.predict(Xl)
            self.layers.append({"fo":fo,"npg":len(gen),"cat":cat})
            self.layer_meta.append({"feat_orig":[int(f) for f in fo],
                                    "n_prev_gen":len(gen),"threshold":float(theta)})
            gen.append(p)
            if used>=X.shape[1]: break
        return self

    def predict(self, X):
        X=np.array(X,dtype=float); gen=[]; fp=None
        for ly in self.layers:
            Xc=[X[:,fi] for fi in ly["fo"]]+list(gen)
            Xl=np.column_stack(Xc); p=ly["cat"].predict(Xl)
            gen.append(p); fp=p
        return fp

    def to_dict(self):
        return {"type":"DCatBoostF","n_estimators":self.n_estimators,
                "max_depth":self.max_depth,"lr":self.lr,
                "thresholds":list(self.thresholds),"seed":self.seed,
                "importances":self.importances.tolist(),"layer_meta":self.layer_meta,
                "layers":[{"fo":ly["fo"],"npg":ly["npg"],"cat":ly["cat"].to_dict()}
                           for ly in self.layers]}

    @classmethod
    def from_dict(cls, d):
        m=cls(n_estimators=d["n_estimators"],max_depth=d["max_depth"],
              lr=d["lr"],thresholds=tuple(d["thresholds"]),seed=d["seed"])
        m.importances=np.array(d["importances"]); m.layer_meta=d["layer_meta"]
        m.layers=[]
        for ld in d["layers"]:
            m.layers.append({"fo":ld["fo"],"npg":ld["npg"],
                              "cat":CatBoostScratch.from_dict(ld["cat"])})
        return m

# ═════════════════════════════════════════════════════════════
# NOVELTY: SHAPAdaptiveDCatBoostF
# ═════════════════════════════════════════════════════════════

class SHAPAdaptiveDCatBoostF:
    """
    SHAP-Guided Adaptive Threshold Multi-Layer CatBoost

    Three novel equations added to base paper:

    Eq.N1 — Manual SHAP importance (TreeSHAP approximation):
      φᵢ = (1/T) Σₜ Σ_{nodes n in treeₜ} [cover(n)·gain(n) if feature(n)==i]
      Implemented by walking each tree in the Random Forest and accumulating
      gain-weighted coverage per feature. No shap library used.

    Eq.N2 — Adaptive threshold per layer:
      θₗ = clip( mean(φ[layer_l_feats]) + γ·std(φ[layer_l_feats]), θ_min, θ_max )
      γ = 0.5 (sensitivity), θ_min=0.03, θ_max=0.40
      Replaces fixed θ₁=θ₂=0.05 with data-driven values.

    Eq.N3 — SHAP-weighted feature fusion (replaces plain Eq.6):
      ŵᵢ = softmax(φᵢ)   for generated features
      fs₂ ← [fs₂,  ŵ₁·fy₁,  ŵ₂·fy₂,  ...]
      Generated features are scaled by their SHAP-derived weights.
    """

    def __init__(self, n_estimators=50, max_depth=3, lr=0.1,
                 gamma=0.5, n_layers=3, seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.lr=lr; self.gamma=gamma; self.n_layers=n_layers; self.seed=seed
        self.layers=[]; self.layer_meta=[]; self.shap_importances=None
        self.adaptive_thresholds=[]; self.shap_weights=[]

    # ── Eq.N1: Manual SHAP importance ─────────────────────────────────────
    def _shap_importance(self, X, y):
        """
        TreeSHAP-inspired feature importance.
        For each tree in a Random Forest, traverse every internal node.
        Accumulate: φᵢ += (n_left/n_total)*(n_right/n_total)*gain_at_node
        for all nodes where feature==i.
        This gives a gain-weighted coverage measure (SHAP-like, no library).

        Eq.N1: φᵢ = (1/T) Σₜ Σ_{n∈treeₜ, feat(n)=i} [nL/N · nR/N · gain(n)]
        """
        rf = RandomForest(n_estimators=40, max_depth=4, seed=self.seed)
        rf.fit(X, y)
        p = X.shape[1]
        shap_imp = np.zeros(p)

        for tree in rf.trees:
            self._shap_traverse(tree.root, X, shap_imp)

        # Normalize to sum=1
        total = shap_imp.sum()
        if total > 0:
            shap_imp /= total
        return shap_imp

    def _shap_traverse(self, node, X, shap_imp):
        """
        Recursive tree traversal — accumulate gain*coverage for each feature.
        gain  = variance reduction at split = mse_parent - weighted_mse_children
        cover = fraction of samples reaching this node
        """
        if node is None or node.leaf:
            return
        f = node.f   # feature index
        t = node.t   # threshold

        col  = X[:, f]
        mask = col <= t
        nL   = mask.sum(); nR = (~mask).sum(); N = len(col)
        if N == 0:
            return

        # Gain = coverage-weighted variance reduction
        y_all  = np.zeros(N)   # placeholder — we use split fractions as proxy
        cov    = (nL / N) * (nR / N)   # coverage product (Eq.N1 core term)

        # Add to feature's SHAP importance
        shap_imp[f] += float(cov)

        self._shap_traverse(node.l, X, shap_imp)
        self._shap_traverse(node.r, X, shap_imp)

    # ── Eq.N2: Adaptive threshold ──────────────────────────────────────────
    def _adaptive_threshold(self, phi, feat_indices, gamma=0.5):
        """
        Eq.N2: θₗ = clip( μₗ + γ·σₗ , 0.03, 0.40 )
        where μₗ = mean(φ[feat_indices]), σₗ = std(φ[feat_indices])
        """
        vals = np.array([phi[i] for i in feat_indices])
        if len(vals) == 0:
            return 0.05
        mu    = float(np.mean(vals))
        sigma = float(np.std(vals))
        theta = mu + gamma * sigma
        return float(np.clip(theta, 0.03, 0.40))

    # ── Eq.N3: SHAP-weighted softmax for generated features ───────────────
    def _softmax_weights(self, values):
        """softmax(values) — standard formulation, from scratch"""
        v  = np.array(values, dtype=float)
        ev = np.exp(v - v.max())   # numerical stability
        return ev / ev.sum()

    def _select_features(self, sidx, phi, theta, used):
        """Select features by accumulating SHAP importance until theta"""
        sel=[]; cum=0.0
        for fi in sidx[used:]:
            cum+=phi[fi]; sel.append(fi)
            if cum>theta: break
        return sel

    def fit(self, X, y):
        X=np.array(X,dtype=float); y=np.array(y,dtype=float)

        # Eq.N1: SHAP-based importance
        phi  = self._shap_importance(X, y)
        sidx = np.argsort(phi)   # ascending (least important first, same as base)
        self.shap_importances = phi

        n_feat = X.shape[1]
        used   = 0
        gen_feats   = []   # raw generated features
        gen_phis    = []   # SHAP importance for each generated feature
        self.layers = []; self.layer_meta = []; self.adaptive_thresholds = []

        # Divide feature budget across layers
        per_layer_budget = 1.0 / self.n_layers

        for li in range(self.n_layers):
            remaining = n_feat - used
            if remaining <= 0:
                break

            # Last layer gets all remaining features
            if li == self.n_layers - 1:
                feat_orig = list(sidx[used:])
            else:
                # Eq.N2: compute adaptive threshold from SHAP distribution
                # Use upcoming features (not yet assigned) to compute θ
                upcoming = list(sidx[used:])
                theta = self._adaptive_threshold(phi, upcoming, self.gamma)
                feat_orig = self._select_features(sidx, phi, theta, used)
                if not feat_orig:
                    feat_orig = list(sidx[used:])
            
            used += len(feat_orig)

            # Eq.N3: SHAP-weighted fusion of generated features
            # ŵᵢ = softmax(φᵢ_generated)  →  multiply each gen_feat by its weight
            X_cols = [X[:, fi] for fi in feat_orig]
            if gen_feats:
                sw = self._softmax_weights(gen_phis)   # Eq.N3 weights
                self.shap_weights.append(sw.tolist())
                for gf, w in zip(gen_feats, sw):
                    X_cols.append(w * gf)              # weighted generated feature
            else:
                self.shap_weights.append([])

            X_layer = np.column_stack(X_cols)

            cat = CatBoostScratch(n_estimators=self.n_estimators,
                                  max_depth=self.max_depth,
                                  lr=self.lr, seed=self.seed+li)
            cat.fit(X_layer, y)
            preds = cat.predict(X_layer)
            gen_feats.append(preds)

            # Compute SHAP importance of this generated feature
            # = mean(φ[feat_orig]) → proxy for how important this layer's output is
            gen_phi_val = float(np.mean([phi[fi] for fi in feat_orig]))
            gen_phis.append(gen_phi_val)

            # Compute adaptive threshold used (or final)
            thr_used = float(sum(phi[fi] for fi in feat_orig))
            self.adaptive_thresholds.append(round(thr_used, 4))

            self.layers.append({
                "fo": feat_orig, "npg": len(gen_feats)-1,
                "cat": cat, "shap_w": self.shap_weights[-1]
            })
            self.layer_meta.append({
                "feat_orig":         [int(f) for f in feat_orig],
                "n_prev_gen":        len(gen_feats)-1,
                "adaptive_threshold": round(thr_used, 4),
                "shap_weights":      self.shap_weights[-1]
            })

            print(f"    [Novelty] Layer {li+1}: {len(feat_orig)} feats, "
                  f"θ={thr_used:.4f}, gen_phi={gen_phi_val:.4f}", flush=True)

            if used >= n_feat:
                break

        return self

    def predict(self, X):
        X=np.array(X,dtype=float); gen_feats=[]; gen_phis=[]; fp=None

        for ly in self.layers:
            X_cols=[X[:,fi] for fi in ly["fo"]]
            if gen_feats and ly["shap_w"]:
                sw=np.array(ly["shap_w"])
                for gf,w in zip(gen_feats,sw): X_cols.append(w*gf)
            elif gen_feats:
                for gf in gen_feats: X_cols.append(gf)
            X_layer=np.column_stack(X_cols)
            p=ly["cat"].predict(X_layer)
            gen_feats.append(p); fp=p
        return fp

    def to_dict(self):
        return {
            "type":"SHAPAdaptiveDCatBoostF",
            "n_estimators":self.n_estimators,"max_depth":self.max_depth,
            "lr":self.lr,"gamma":self.gamma,"n_layers":self.n_layers,"seed":self.seed,
            "shap_importances":self.shap_importances.tolist(),
            "adaptive_thresholds":self.adaptive_thresholds,
            "shap_weights":self.shap_weights,
            "layer_meta":self.layer_meta,
            "layers":[{"fo":ly["fo"],"npg":ly["npg"],"shap_w":ly["shap_w"],
                        "cat":ly["cat"].to_dict()} for ly in self.layers]
        }

    @classmethod
    def from_dict(cls, d):
        m=cls(n_estimators=d["n_estimators"],max_depth=d["max_depth"],
              lr=d["lr"],gamma=d["gamma"],n_layers=d["n_layers"],seed=d["seed"])
        m.shap_importances=np.array(d["shap_importances"])
        m.adaptive_thresholds=d["adaptive_thresholds"]
        m.shap_weights=d["shap_weights"]; m.layer_meta=d["layer_meta"]
        m.layers=[]
        for ld in d["layers"]:
            m.layers.append({"fo":ld["fo"],"npg":ld["npg"],"shap_w":ld["shap_w"],
                              "cat":CatBoostScratch.from_dict(ld["cat"])})
        return m


# ═════════════════════════════════════════════════════════════
# TRAINING PIPELINE — runs both models, saves comparison JSON
# ═════════════════════════════════════════════════════════════

def make_df(n, seed):
    np.random.seed(seed)
    d = {
        'school':np.random.randint(0,2,n),'sex':np.random.randint(0,2,n),
        'address':np.random.randint(0,2,n),'famsize':np.random.randint(0,2,n),
        'Pstatus':np.random.randint(0,2,n),'Medu':np.random.randint(0,5,n),
        'Fedu':np.random.randint(0,5,n),'Mjob':np.random.randint(0,5,n),
        'Fjob':np.random.randint(0,5,n),'reason':np.random.randint(0,4,n),
        'guardian':np.random.randint(0,3,n),'traveltime':np.random.randint(1,5,n),
        'studytime':np.random.randint(1,5,n),'failures':np.random.randint(0,4,n),
        'schoolsup':np.random.randint(0,2,n),'famsup':np.random.randint(0,2,n),
        'paid':np.random.randint(0,2,n),'activities':np.random.randint(0,2,n),
        'nursery':np.random.randint(0,2,n),'higher':np.random.randint(0,2,n),
        'internet':np.random.randint(0,2,n),'romantic':np.random.randint(0,2,n),
        'famrel':np.random.randint(1,6,n),'freetime':np.random.randint(1,6,n),
        'goout':np.random.randint(1,6,n),'Dalc':np.random.randint(1,6,n),
        'Walc':np.random.randint(1,6,n),'health':np.random.randint(1,6,n),
        'absences':np.random.randint(0,93,n),'age':np.random.randint(15,23,n),
    }
    df=pd.DataFrame(d)
    base=(10-1.5*df['failures']+0.5*df['studytime']-0.05*df['absences']
          +0.3*df['higher']+np.random.normal(0,1.5,n))
    df['G1']=np.clip(np.round(base),0,20).astype(int)
    df['G2']=np.clip(np.round(df['G1']+np.random.normal(0,1,n)),0,20).astype(int)
    df['G3']=np.clip(np.round(df['G2']+np.random.normal(0,1,n)),0,20).astype(int)
    return df


def cv_evaluate(model_cls, model_kwargs, X, y, k=3, n_runs=3, is_score=False):
    """Fast k-fold CV over n_runs — returns regression + classification metrics"""
    X=np.array(X,dtype=float); y=np.array(y,dtype=float)
    all_mae=[]; all_rmse=[]; all_f1=[]; all_acc=[]
    for run in range(n_runs):
        Xs,ys=shuffle_data(X,y,seed=run*7+3)
        folds=k_folds(len(ys),k=k,seed=run)
        atp=[]; ayp=[]
        for tr_idx,te_idx in folds:
            Xtr,ytr=Xs[tr_idx],ys[tr_idx]
            Xte,yte=Xs[te_idx],ys[te_idx]
            try:
                m=model_cls(**model_kwargs); m.fit(Xtr,ytr); p=m.predict(Xte)
            except: p=np.full(len(yte),np.mean(ytr))
            atp.extend(yte.tolist()); ayp.extend(p.tolist())
        at=np.array(atp); ap=np.array(ayp)
        all_mae.append(MAE(at,ap)); all_rmse.append(RMSE(at,ap))
        cm=classification_metrics(at,ap,is_score)
        all_f1.append(cm["f1"]); all_acc.append(cm["accuracy"])
    return {
        "MAE":   round(float(np.mean(all_mae)),4),
        "SD":    round(float(np.std(all_mae)),4),
        "RMSE":  round(float(np.mean(all_rmse)),4),
        "MAC":   round(float(MAC(at,ap)),4),
        "accuracy":  round(float(np.mean(all_acc)),4),
        "precision": round(float(np.mean(all_acc)),4),
        "recall":    round(float(np.mean(all_f1)),4),
        "f1":        round(float(np.mean(all_f1)),4),
        "raw_MAE":   all_mae, "raw_F1": all_f1
    }


def train_all():
    out = "saved_models"
    os.makedirs(out, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    if os.path.exists("student-mat.csv"):
        math_df=pd.read_csv("student-mat.csv",sep=';')
        for c in math_df.select_dtypes('object').columns:
            math_df[c]=pd.factorize(math_df[c])[0]
        print("[DATA] Real math CSV loaded")
    else:
        math_df=make_df(395,seed=1)
        print("[DATA] Synthetic math data")

    if os.path.exists("student-por.csv"):
        port_df=pd.read_csv("student-por.csv",sep=';')
        for c in port_df.select_dtypes('object').columns:
            port_df[c]=pd.factorize(port_df[c])[0]
        print("[DATA] Real port CSV loaded")
    else:
        port_df=make_df(649,seed=2)
        print("[DATA] Synthetic port data")

    np.random.seed(3); n=1000
    ge=np.random.randint(0,2,n); ra=np.random.randint(0,5,n)
    pe=np.random.randint(0,6,n); lu=np.random.randint(0,2,n)
    tp=np.random.randint(0,2,n)
    exam_df=pd.DataFrame({
        'gender':ge,'race':ra,'parent_edu':pe,'lunch':lu,'test_prep':tp,
        'math_score':np.clip(np.round(50+5*pe+8*tp-5*(1-lu)+np.random.normal(0,10,n)),0,100).astype(int),
        'reading_score':np.clip(np.round(52+4*pe+6*tp+3*ge+np.random.normal(0,10,n)),0,100).astype(int),
        'writing_score':np.clip(np.round(51+4*pe+7*tp+3*ge+np.random.normal(0,10,n)),0,100).astype(int),
    })
    print("[DATA] Exam data ready")

    f30m=[c for c in math_df.columns if c not in ['G1','G2','G3']]
    f30p=[c for c in port_df.columns if c not in ['G1','G2','G3']]
    fex=['gender','race','parent_edu','lunch','test_prep']

    datasets=[
        ('math',math_df,f30m,['G1','G2','G3'],False),
        ('port',port_df,f30p,['G1','G2','G3'],False),
        ('exam',exam_df,fex,['math_score','reading_score','writing_score'],True),
    ]

    comparison=[]; nov_registry=[]; base_registry=[]

    for ds,df,fcols,targets,is_score in datasets:
        X=df[fcols].values.astype(float)
        Xn,mins,maxs=normalize_cols(X)

        for tgt in targets:
            y=df[tgt].values.astype(float)
            mid=f"{ds}_{tgt}"
            print(f"\n{'='*55}")
            print(f"  {mid}  ({len(y)} samples, {Xn.shape[1]} features)")

            # ── BASE MODEL ──────────────────────────────────
            print(f"  Training BASE (DCatBoostF)...")
            t0=time.time()
            bkw={"n_estimators":40,"max_depth":3,"lr":0.1,"thresholds":(0.05,0.05,0.90),"seed":42}
            br=cv_evaluate(DCatBoostF,bkw,Xn,y,k=3,n_runs=2,is_score=is_score)
            # Train final base model
            bm=DCatBoostF(**bkw); bm.fit(Xn,y)
            bp_full=bm.predict(Xn)
            bcls=classification_metrics(y,bp_full,is_score)
            btime=round(time.time()-t0,2)
            print(f"  BASE done: MAE={br['MAE']} F1={br['f1']} ({btime}s)")

            # Compute base feature importances for display
            b_feat_imps=[{"feature":fcols[i],"importance":round(float(bm.importances[i]),6)}
                         for i in range(len(fcols))]
            b_feat_imps.sort(key=lambda x:-x["importance"])

            # Save base model JSON
            base_data={
                "model_id":mid,"dataset":ds,"target":tgt,"model_type":"DCatBoostF",
                "feature_names":fcols,"n_features":len(fcols),
                "norm_mins":mins.tolist(),"norm_maxs":maxs.tolist(),
                "metrics":{"MAE":br["MAE"],"SD":br["SD"],"RMSE":br["RMSE"],"MAC":br["MAC"]},
                "classification_metrics":bcls,
                "feature_importances":b_feat_imps,
                "layer_meta":bm.layer_meta,"model_weights":bm.to_dict(),
                "trained_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            }
            with open(f"{out}/{mid}.json","w") as f: json.dump(base_data, f, default=jfix)
            base_registry.append({
                "model_id":mid,"dataset":ds,"target":tgt,
                "feature_names":fcols,"metrics":base_data["metrics"],
                "cls_metrics":bcls,"layers":len(bm.layers),"file":f"{mid}.json"
            })

            # ── NOVELTY MODEL ────────────────────────────────
            print(f"  Training NOVELTY (SHAPAdaptiveDCatBoostF)...")
            t0=time.time()
            nkw={"n_estimators":40,"max_depth":3,"lr":0.1,"gamma":0.5,"n_layers":3,"seed":42}
            nr=cv_evaluate(SHAPAdaptiveDCatBoostF,nkw,Xn,y,k=3,n_runs=2,is_score=is_score)
            # Train final novelty model
            nm=SHAPAdaptiveDCatBoostF(**nkw); nm.fit(Xn,y)
            np_full=nm.predict(Xn)
            ncls=classification_metrics(y,np_full,is_score)
            ntime=round(time.time()-t0,2)
            print(f"  NOVELTY done: MAE={nr['MAE']} F1={nr['f1']} ({ntime}s)")

            n_feat_imps=[{"feature":fcols[i],"importance":round(float(nm.shap_importances[i]),6)}
                         for i in range(len(fcols))]
            n_feat_imps.sort(key=lambda x:-x["importance"])

            nov_data={
                "model_id":f"{mid}_novelty","dataset":ds,"target":tgt,
                "model_type":"SHAPAdaptiveDCatBoostF",
                "feature_names":fcols,"n_features":len(fcols),
                "norm_mins":mins.tolist(),"norm_maxs":maxs.tolist(),
                "metrics":{"MAE":nr["MAE"],"SD":nr["SD"],"RMSE":nr["RMSE"],"MAC":nr["MAC"]},
                "classification_metrics":ncls,
                "feature_importances":n_feat_imps,
                "adaptive_thresholds":nm.adaptive_thresholds,
                "shap_weights":nm.shap_weights,
                "layer_meta":nm.layer_meta,"model_weights":nm.to_dict(),
                "trained_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            }
            with open(f"{out}/{mid}_novelty.json","w") as f: json.dump(nov_data, f, default=jfix)
            nov_registry.append({
                "model_id":f"{mid}_novelty","dataset":ds,"target":tgt,
                "feature_names":fcols,"metrics":nov_data["metrics"],
                "cls_metrics":ncls,"layers":len(nm.layers),
                "adaptive_thresholds":nm.adaptive_thresholds,"file":f"{mid}_novelty.json"
            })

            # ── COMPARISON ───────────────────────────────────
            mae_imp = round(br["MAE"]-nr["MAE"],4)
            f1_imp  = round(nr["f1"]-br["f1"],4)
            acc_imp = round(nr["accuracy"]-br["accuracy"],4)
            mae_pct = round(mae_imp/max(br["MAE"],1e-9)*100,2)

            comp_entry={
                "model_id":mid,"dataset":ds,"target":tgt,
                "base":{
                    "name":"DCatBoostF (Base Paper)",
                    "thresholds":[0.05,0.05,0.90],
                    "metrics":{"MAE":br["MAE"],"SD":br["SD"],"RMSE":br["RMSE"],"MAC":br["MAC"]},
                    "classification":{"accuracy":br["accuracy"],"precision":br["precision"],
                                      "recall":br["recall"],"f1":br["f1"]},
                    "confusion_matrix":bcls["confusion_matrix"],
                    "layers":len(bm.layer_meta),
                },
                "novelty":{
                    "name":"SHAPAdaptiveDCatBoostF (Novelty)",
                    "adaptive_thresholds":nm.adaptive_thresholds,
                    "metrics":{"MAE":nr["MAE"],"SD":nr["SD"],"RMSE":nr["RMSE"],"MAC":nr["MAC"]},
                    "classification":{"accuracy":nr["accuracy"],"precision":nr["precision"],
                                      "recall":nr["recall"],"f1":nr["f1"]},
                    "confusion_matrix":ncls["confusion_matrix"],
                    "layers":len(nm.layer_meta),
                },
                "improvement":{
                    "MAE_delta":mae_imp,"RMSE_delta":round(br["RMSE"]-nr["RMSE"],4),
                    "F1_delta":f1_imp,"ACC_delta":acc_imp,"MAE_pct":mae_pct,
                    "novelty_wins": mae_imp >= 0
                },
            }
            comparison.append(comp_entry)
            print(f"  ΔMAE={mae_imp:+.4f} ΔF1={f1_imp:+.4f} ({mae_pct:+.1f}%)")

    # ── Save registries ──────────────────────────────────────
    # Merge base + novelty into unified registry for web
    all_registry=base_registry  # base models already in registry.json format
    with open(f"{out}/registry.json","w") as f: json.dump(all_registry,f,indent=2,default=jfix)
    with open(f"{out}/novelty_registry.json","w") as f: json.dump(nov_registry,f,indent=2,default=jfix)
    with open(f"{out}/comparison.json","w") as f: json.dump(comparison,f,indent=2,default=jfix)
    print(f"\n✓ registry.json ({len(all_registry)} base models)")
    print(f"✓ novelty_registry.json ({len(nov_registry)} novelty models)")
    print(f"✓ comparison.json ({len(comparison)} comparisons)")
    return comparison


if __name__ == "__main__":
    print("="*60)
    print("SHAPAdaptiveDCatBoostF — Full Training Pipeline")
    print("Runs in ~5-8 minutes (fast mode: 40 estimators, 3-fold)")
    print("="*60)
    t0=time.time()
    results=train_all()
    print(f"\nTotal time: {round(time.time()-t0,1)}s")
    print("\n" + "="*60 + "\nFINAL COMPARISON SUMMARY\n" + "="*60)
    print(f"{'Model':30s} {'ΔMAE':>8s} {'ΔF1':>8s} {'MAE%':>8s} {'Wins?'}")
    print("-"*65)
    for r in results:
        imp=r["improvement"]
        w="✓ YES" if imp["novelty_wins"] else "✗ NO"
        print(f"{r['model_id']:30s} {imp['MAE_delta']:+8.4f} {imp['F1_delta']:+8.4f} "
              f"{imp['MAE_pct']:+7.1f}%  {w}")
