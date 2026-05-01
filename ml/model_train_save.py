"""
=============================================================================
DCatBoostF — Model Training + Saving (JSON serialization, no pickle/joblib)
All algorithms from the paper implemented from scratch (numpy only).
Saves model weights + metadata to JSON for Express API to load.
=============================================================================
"""

import numpy as np
import pandas as pd
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─── Copy all scratch implementations from original code ─────────────────────

def sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def normalize_cols(X):
    mins  = X.min(axis=0)
    maxs  = X.max(axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    return (X - mins) / denom, mins, maxs

def apply_normalization(X, mins, maxs):
    denom = maxs - mins
    denom[denom == 0] = 1.0
    return (X - mins) / denom

def shuffle_dataset(X, y, seed=None):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def k_fold_indices(n, k=10, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k-1 else n
        test_idx  = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, test_idx))
    return folds

# ─── Metrics (Equations 7–10) ─────────────────────────────────────────────────

def MAE(y_true, y_pred):
    """Eq.7: MAE = (1/n)*sum(|y_hat_i - y_i|)"""
    n = len(y_true)
    return sum(abs(y_pred[i] - y_true[i]) for i in range(n)) / n

def SD(y_true, y_pred):
    """Eq.8: SD = sqrt((1/n)*sum((e_i - e_bar)^2))"""
    n = len(y_true)
    errors = [y_pred[i] - y_true[i] for i in range(n)]
    e_bar = sum(errors) / n
    return (sum((e - e_bar)**2 for e in errors) / n) ** 0.5

def RMSE(y_true, y_pred):
    """Eq.9: RMSE = sqrt((1/n)*sum((y_hat_i - y_i)^2))"""
    n = len(y_true)
    return (sum((y_pred[i] - y_true[i])**2 for i in range(n)) / n) ** 0.5

def MAC(y_true, y_pred):
    """Eq.10: MAC = (y^T * y_hat)^2 / ((y^T*y)*(y_hat^T*y_hat))"""
    y  = np.array(y_true,  dtype=float)
    yh = np.array(y_pred, dtype=float)
    dot_yy_hat = float(np.dot(y, yh))
    dot_yy     = float(np.dot(y, y))
    dot_yh_yh  = float(np.dot(yh, yh))
    if dot_yy == 0 or dot_yh_yh == 0:
        return 0.0
    return (dot_yy_hat ** 2) / (dot_yy * dot_yh_yh)

# ─── Decision Tree (used in RF and XGBoost/CatBoost) ─────────────────────────

class DecisionTreeNode:
    __slots__ = ['feature','threshold','left','right','value','is_leaf']
    def __init__(self):
        self.feature = self.threshold = self.left = self.right = self.value = None
        self.is_leaf = False

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, seed=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.root = None
        self.rng  = np.random.RandomState(seed)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        node = DecisionTreeNode()
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) == 0:
            node.is_leaf = True
            node.value   = float(np.mean(y))
            return node
        feat_ids = self.rng.choice(self.n_features,
                                   size=min(self.max_features, self.n_features),
                                   replace=False)
        best_feat, best_thresh, best_score = None, None, np.inf
        for f in feat_ids:
            col = X[:, f]
            for t in np.unique(col)[:-1]:
                mask = col <= t
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue
                score = self._weighted_mse(y[mask], y[~mask])
                if score < best_score:
                    best_score = score; best_feat = f; best_thresh = t
        if best_feat is None:
            node.is_leaf = True; node.value = float(np.mean(y)); return node
        node.feature = best_feat; node.threshold = best_thresh
        mask = X[:, best_feat] <= best_thresh
        node.left  = self._build(X[mask],  y[mask],  depth+1)
        node.right = self._build(X[~mask], y[~mask], depth+1)
        return node

    def _weighted_mse(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        def mse(a): return 0.0 if len(a)==0 else np.mean((a - np.mean(a))**2)
        return (len(y_left)/n)*mse(y_left) + (len(y_right)/n)*mse(y_right)

    def predict_one(self, x, node):
        if node.is_leaf: return node.value
        return self.predict_one(x, node.left if x[node.feature] <= node.threshold else node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])

    def feature_importances(self, n_features):
        imp = np.zeros(n_features)
        self._importance_recurse(self.root, imp)
        total = imp.sum()
        return imp / total if total > 0 else imp

    def _importance_recurse(self, node, imp):
        if node is None or node.is_leaf: return
        imp[node.feature] += 1.0
        self._importance_recurse(node.left, imp)
        self._importance_recurse(node.right, imp)

    # ── Serialization ─────────────────────────────────────────
    def to_dict(self):
        def node_to_dict(node):
            if node is None: return None
            if node.is_leaf: return {"leaf": True, "value": node.value}
            return {
                "leaf": False,
                "feature": int(node.feature),
                "threshold": float(node.threshold),
                "left":  node_to_dict(node.left),
                "right": node_to_dict(node.right)
            }
        return {
            "max_depth": self.max_depth,
            "n_features": self.n_features,
            "root": node_to_dict(self.root)
        }

    @classmethod
    def from_dict(cls, d):
        tree = cls(max_depth=d["max_depth"])
        tree.n_features = d["n_features"]
        def dict_to_node(nd):
            if nd is None: return None
            node = DecisionTreeNode()
            if nd["leaf"]:
                node.is_leaf = True; node.value = nd["value"]; return node
            node.feature   = nd["feature"]
            node.threshold = nd["threshold"]
            node.left      = dict_to_node(nd["left"])
            node.right     = dict_to_node(nd["right"])
            return node
        tree.root = dict_to_node(d["root"])
        return tree

# ─── XGBTree (used inside CatBoost/XGBoost) ──────────────────────────────────

class XGBTreeNode:
    __slots__ = ['feature','threshold','left','right','value','is_leaf']
    def __init__(self):
        self.feature = self.threshold = self.left = self.right = self.value = None
        self.is_leaf = False

class XGBTree:
    def __init__(self, max_depth=4, reg_lambda=1.0, seed=42):
        self.max_depth  = max_depth
        self.reg_lambda = reg_lambda
        self.seed       = seed
        self.root       = None
        self.rng        = np.random.RandomState(seed)

    def fit(self, X, g, h):
        self.n_features = X.shape[1]
        self.root = self._build(X, g, h, 0)

    def _leaf_value(self, g, h):
        return -g.sum() / (h.sum() + self.reg_lambda)

    def _build(self, X, g, h, depth):
        node = XGBTreeNode()
        if depth >= self.max_depth or len(g) <= 1:
            node.is_leaf = True; node.value = self._leaf_value(g, h); return node
        best_gain, best_f, best_t = -np.inf, None, None
        G_total, H_total = g.sum(), h.sum()
        feat_ids = self.rng.choice(X.shape[1], size=max(1, int(np.sqrt(X.shape[1]))), replace=False)
        for f in feat_ids:
            col      = X[:, f]
            sort_idx = np.argsort(col)
            col_s, g_s, h_s = col[sort_idx], g[sort_idx], h[sort_idx]
            G_L = H_L = 0.0
            for i in range(len(g)-1):
                G_L += g_s[i]; H_L += h_s[i]
                if col_s[i] == col_s[i+1]: continue
                G_R = G_total - G_L; H_R = H_total - H_L
                gain = 0.5*(G_L**2/(H_L+self.reg_lambda)+G_R**2/(H_R+self.reg_lambda)-G_total**2/(H_total+self.reg_lambda))
                if gain > best_gain:
                    best_gain = gain; best_f = f; best_t = (col_s[i]+col_s[i+1])/2.0
        if best_f is None or best_gain <= 0:
            node.is_leaf = True; node.value = self._leaf_value(g, h); return node
        node.feature = best_f; node.threshold = best_t
        mask = X[:, best_f] <= best_t
        node.left  = self._build(X[mask],  g[mask],  h[mask],  depth+1)
        node.right = self._build(X[~mask], g[~mask], h[~mask], depth+1)
        return node

    def predict_one(self, x, node):
        if node.is_leaf: return node.value
        return self.predict_one(x, node.left if x[node.feature] <= node.threshold else node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])

    def to_dict(self):
        def nd(node):
            if node is None: return None
            if node.is_leaf: return {"leaf": True, "value": float(node.value)}
            return {"leaf": False, "feature": int(node.feature),
                    "threshold": float(node.threshold),
                    "left": nd(node.left), "right": nd(node.right)}
        return {"max_depth": self.max_depth, "n_features": self.n_features, "root": nd(self.root)}

    @classmethod
    def from_dict(cls, d):
        t = cls(max_depth=d["max_depth"])
        t.n_features = d["n_features"]
        def dn(nd):
            if nd is None: return None
            node = XGBTreeNode()
            if nd["leaf"]: node.is_leaf = True; node.value = nd["value"]; return node
            node.feature = nd["feature"]; node.threshold = nd["threshold"]
            node.left = dn(nd["left"]); node.right = dn(nd["right"]); return node
        t.root = dn(d["root"]); return t

# ─── Random Forest ────────────────────────────────────────────────────────────

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=5, max_features='sqrt',
                 min_samples_split=2, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees = []
        self.rng   = np.random.RandomState(seed)

    def fit(self, X, y):
        X = np.array(X, dtype=float); y = np.array(y, dtype=float)
        n, p = X.shape
        mf = max(1, int(np.sqrt(p))) if self.max_features == 'sqrt' else p
        self.n_features = p
        self.trees = []
        for i in range(self.n_estimators):
            boot_idx = self.rng.randint(0, n, size=n)
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=mf,
                                seed=self.rng.randint(0, 100000))
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = np.zeros(len(X))
        for t in self.trees: preds += t.predict(X)
        return preds / len(self.trees)

    def feature_importances_(self):
        """Eq.(1): fs = RF(X, y)"""
        imp = np.zeros(self.n_features)
        for t in self.trees: imp += t.feature_importances(self.n_features)
        return imp / len(self.trees)

# ─── CatBoost Scratch (ordered boosting) ──────────────────────────────────────

class CatBoostScratch:
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.1, seed=42):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.seed          = seed
        self.trees         = []
        self.base_pred     = 0.0
        self.rng           = np.random.RandomState(seed)

    def _ordered_gradient(self, X, y, F):
        n = len(y)
        perm = self.rng.permutation(n)
        g = np.zeros(n)
        F_ordered = np.full(n, self.base_pred)
        for idx, i in enumerate(perm):
            g[i] = F_ordered[i] - y[i]
            if idx > 0:
                prev = perm[:idx]
                F_ordered[i] = np.mean(y[prev]) if len(prev) > 0 else self.base_pred
        return g

    def fit(self, X, y):
        X = np.array(X, dtype=float); y = np.array(y, dtype=float)
        self.base_pred = float(np.mean(y))
        F = np.full(len(y), self.base_pred)
        self.trees = []
        for t in range(self.n_estimators):
            g = self._ordered_gradient(X, y, F)
            h = np.ones(len(y))
            tree = XGBTree(max_depth=self.max_depth, reg_lambda=1.0, seed=self.seed+t)
            tree.fit(X, g, h)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        F = np.full(len(X), self.base_pred)
        for t in self.trees: F += self.learning_rate * t.predict(X)
        return F

    def to_dict(self):
        return {
            "type": "CatBoostScratch",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "base_pred": self.base_pred,
            "trees": [t.to_dict() for t in self.trees]
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(n_estimators=d["n_estimators"], max_depth=d["max_depth"],
                learning_rate=d["learning_rate"])
        m.base_pred = d["base_pred"]
        m.trees = [XGBTree.from_dict(td) for td in d["trees"]]
        return m

# ─── DCatBoostF (Proposed Model — Algorithm 1, Equations 1–6) ────────────────

class DCatBoostF:
    """
    Feature Importance-Based Multi-Layer CatBoost
    Eq.(1): fs = RF(X, y)
    Eq.(2): Sort ascending
    Eq.(3)/(5): Accumulate features until threshold
    Eq.(4): fy = CatBoost(X, y) — generate new feature
    Eq.(6): fs2 ← [fs2, fy]
    """

    def __init__(self, thresholds=(0.05, 0.05, 0.90),
                 n_estimators=300, max_depth=3,
                 learning_rate=0.1, seed=42):
        self.thresholds    = thresholds
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.seed          = seed
        self.layers        = []
        self.sorted_feat_idx = None
        self.importances   = None

    def _compute_feature_importance(self, X, y):
        """Eq.(1): fs = RF(X, y)"""
        rf = RandomForest(n_estimators=50, max_depth=4, seed=self.seed)
        rf.fit(X, y)
        return rf.feature_importances_()

    def _sort_ascending(self, importances):
        """Eq.(2): sort ascending"""
        return np.argsort(importances)

    def _select_for_layer(self, sorted_idx, importances, threshold, used_count):
        """Eq.(3)/(5): accumulate until threshold"""
        selected, cumulative = [], 0.0
        for fi in sorted_idx[used_count:]:
            cumulative += importances[fi]
            selected.append(fi)
            if cumulative > threshold: break
        return selected

    def fit(self, X, y):
        X = np.array(X, dtype=float); y = np.array(y, dtype=float)
        self.importances     = self._compute_feature_importance(X, y)
        self.sorted_feat_idx = self._sort_ascending(self.importances)
        self.layers          = []
        self.layer_meta      = []   # stores feat indices for each layer
        used_count           = 0
        n_features           = X.shape[1]
        generated_features   = []

        for layer_idx, theta in enumerate(self.thresholds):
            if n_features - used_count <= 0: break

            if layer_idx == len(self.thresholds) - 1:
                layer_feat_orig = list(self.sorted_feat_idx[used_count:])
            else:
                layer_feat_orig = self._select_for_layer(
                    self.sorted_feat_idx, self.importances, theta, used_count)

            if len(layer_feat_orig) == 0: break
            used_count += len(layer_feat_orig)

            # Eq.(6): combine original + generated features
            X_layer_cols = [X[:, fi] for fi in layer_feat_orig]
            for gf in generated_features:
                X_layer_cols.append(gf)
            X_layer = np.column_stack(X_layer_cols)

            cat = CatBoostScratch(n_estimators=self.n_estimators,
                                  max_depth=self.max_depth,
                                  learning_rate=self.learning_rate,
                                  seed=self.seed + layer_idx)
            cat.fit(X_layer, y)

            self.layers.append((layer_feat_orig, len(generated_features), cat))
            self.layer_meta.append({
                "feat_orig": [int(f) for f in layer_feat_orig],
                "n_prev_gen": len(generated_features),
                "threshold": float(theta)
            })

            # Eq.(4): fy = CatBoost(X, y) — generate new feature
            generated_features.append(cat.predict(X_layer))

            if used_count >= n_features: break

        self._generated_features_train = generated_features
        print(f"  DCatBoostF: {len(self.layers)} layers built")
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        final_preds = None
        gen_feats   = []
        for layer_feat_orig, n_prev_gen, cat in self.layers:
            X_cols = [X[:, fi] for fi in layer_feat_orig]
            for gf in gen_feats: X_cols.append(gf)
            X_layer = np.column_stack(X_cols)
            preds = cat.predict(X_layer)
            gen_feats.append(preds)
            final_preds = preds
        return final_preds

    def to_dict(self):
        return {
            "type": "DCatBoostF",
            "thresholds": list(self.thresholds),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "importances": self.importances.tolist(),
            "sorted_feat_idx": self.sorted_feat_idx.tolist(),
            "layer_meta": self.layer_meta,
            "layers": [
                {
                    "feat_orig": [int(f) for f in feat_orig],
                    "n_prev_gen": n_prev_gen,
                    "catboost": cat.to_dict()
                }
                for feat_orig, n_prev_gen, cat in self.layers
            ]
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(thresholds=tuple(d["thresholds"]),
                n_estimators=d["n_estimators"],
                max_depth=d["max_depth"],
                learning_rate=d["learning_rate"],
                seed=d["seed"])
        m.importances      = np.array(d["importances"])
        m.sorted_feat_idx  = np.array(d["sorted_feat_idx"])
        m.layer_meta       = d["layer_meta"]
        m.layers = []
        for ld in d["layers"]:
            cat = CatBoostScratch.from_dict(ld["catboost"])
            m.layers.append((ld["feat_orig"], ld["n_prev_gen"], cat))
        return m

# ─── Grid Search + Cross Validation (Algorithm 1) ────────────────────────────

def grid_search_cv(model_class, param_grid, X, y, k=10, seed=42):
    """Algorithm 1: Grid Search with 10-Fold Cross Validation"""
    X, y = shuffle_dataset(np.array(X, dtype=float), np.array(y, dtype=float), seed=seed)
    folds = k_fold_indices(len(y), k=k, seed=seed)
    best_mae, best_params = np.inf, None

    for params in param_grid:
        fold_maes = []
        for train_idx, test_idx in folds:
            try:
                model = model_class(**params)
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])
                fold_maes.append(MAE(y[test_idx], preds))
            except Exception as e:
                fold_maes.append(999.0)
        avg = np.mean(fold_maes)
        if avg < best_mae:
            best_mae = avg; best_params = params

    return best_params, best_mae

# ─── Synthetic Data (matches paper structure) ─────────────────────────────────

def make_student_data(n=395, seed=1):
    np.random.seed(seed)
    cat_features = {
        'school':     np.random.randint(0,2,n), 'sex': np.random.randint(0,2,n),
        'address':    np.random.randint(0,2,n), 'famsize': np.random.randint(0,2,n),
        'Pstatus':    np.random.randint(0,2,n), 'Medu': np.random.randint(0,5,n),
        'Fedu':       np.random.randint(0,5,n), 'Mjob': np.random.randint(0,5,n),
        'Fjob':       np.random.randint(0,5,n), 'reason': np.random.randint(0,4,n),
        'guardian':   np.random.randint(0,3,n), 'traveltime': np.random.randint(1,5,n),
        'studytime':  np.random.randint(1,5,n), 'failures': np.random.randint(0,4,n),
        'schoolsup':  np.random.randint(0,2,n), 'famsup': np.random.randint(0,2,n),
        'paid':       np.random.randint(0,2,n), 'activities': np.random.randint(0,2,n),
        'nursery':    np.random.randint(0,2,n), 'higher': np.random.randint(0,2,n),
        'internet':   np.random.randint(0,2,n), 'romantic': np.random.randint(0,2,n),
        'famrel':     np.random.randint(1,6,n), 'freetime': np.random.randint(1,6,n),
        'goout':      np.random.randint(1,6,n), 'Dalc': np.random.randint(1,6,n),
        'Walc':       np.random.randint(1,6,n), 'health': np.random.randint(1,6,n),
        'absences':   np.random.randint(0,93,n),'age': np.random.randint(15,23,n),
    }
    df = pd.DataFrame(cat_features)
    base = (10 - 1.5*df['failures'] + 0.5*df['studytime']
            - 0.05*df['absences'] + 0.3*df['higher'] + np.random.normal(0,1.5,n))
    df['G1'] = np.clip(np.round(base), 0, 20).astype(int)
    df['G2'] = np.clip(np.round(df['G1']+np.random.normal(0,1,n)), 0, 20).astype(int)
    df['G3'] = np.clip(np.round(df['G2']+np.random.normal(0,1,n)), 0, 20).astype(int)
    return df

# ─── Train & Save all models ─────────────────────────────────────────────────

def train_and_save_all():
    output_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(output_dir, exist_ok=True)

    math_path = "student-mat.csv"
    port_path = "student-por.csv"

    # Load or generate data
    if os.path.exists(math_path):
        math_df = pd.read_csv(math_path, sep=';')
        for col in math_df.select_dtypes(include='object').columns:
            math_df[col] = pd.factorize(math_df[col])[0]
        print(f"[DATA] Loaded real math dataset: {math_df.shape}")
    else:
        math_df = make_student_data(395, seed=1)
        print("[DATA] Synthetic math dataset (395 samples)")

    if os.path.exists(port_path):
        port_df = pd.read_csv(port_path, sep=';')
        for col in port_df.select_dtypes(include='object').columns:
            port_df[col] = pd.factorize(port_df[col])[0]
        print(f"[DATA] Loaded real portuguese dataset: {port_df.shape}")
    else:
        port_df = make_student_data(649, seed=2)
        print("[DATA] Synthetic portuguese dataset (649 samples)")

    # Exam dataset
    np.random.seed(3); n = 1000
    gender = np.random.randint(0,2,n); race = np.random.randint(0,5,n)
    parent_edu = np.random.randint(0,6,n); lunch = np.random.randint(0,2,n)
    test_prep  = np.random.randint(0,2,n)
    exam_df = pd.DataFrame({
        'gender': gender, 'race': race, 'parent_edu': parent_edu,
        'lunch': lunch, 'test_prep': test_prep,
        'math_score':    np.clip(np.round(50+5*parent_edu+8*test_prep-5*(1-lunch)+np.random.normal(0,10,n)),0,100).astype(int),
        'reading_score': np.clip(np.round(52+4*parent_edu+6*test_prep+3*gender+np.random.normal(0,10,n)),0,100).astype(int),
        'writing_score': np.clip(np.round(51+4*parent_edu+7*test_prep+3*gender+np.random.normal(0,10,n)),0,100).astype(int),
    })

    thresholds = (0.05, 0.05, 0.90)

    # Grid for DCatBoostF (Algorithm 1 — candidate values from Table II)
    dcb_grid = [
        {"thresholds": thresholds, "n_estimators": ne, "max_depth": md, "learning_rate": 0.1}
        for ne in [100, 300] for md in [3, 4]
    ]

    registry = []  # master index saved as JSON

    datasets = [
        ("math",  math_df, [c for c in math_df.columns if c not in ['G1','G2','G3']],
         ['G1','G2','G3']),
        ("port",  port_df, [c for c in port_df.columns if c not in ['G1','G2','G3']],
         ['G1','G2','G3']),
        ("exam",  exam_df, ['gender','race','parent_edu','lunch','test_prep'],
         ['math_score','reading_score','writing_score']),
    ]

    for ds_name, df, feat_cols, targets in datasets:
        X = df[feat_cols].values.astype(float)
        X_norm, mins, maxs = normalize_cols(X)

        for target in targets:
            y = df[target].values.astype(float)
            print(f"\n=== Training DCatBoostF: {ds_name}/{target} ===")
            t0 = time.time()

            best_params, cv_mae = grid_search_cv(
                DCatBoostF, dcb_grid, X_norm, y, k=5, seed=42)
            print(f"  Best params: {best_params}  CV-MAE={cv_mae:.4f}")

            # Train final model on full data
            model = DCatBoostF(**best_params)
            model.fit(X_norm, y)

            # Evaluate metrics
            preds = model.predict(X_norm)
            metrics = {
                "MAE":  round(MAE(y, preds), 4),
                "SD":   round(SD(y, preds), 4),
                "RMSE": round(RMSE(y, preds), 4),
                "MAC":  round(MAC(y, preds), 4)
            }
            elapsed = round(time.time() - t0, 2)
            print(f"  Metrics: {metrics}  Time={elapsed}s")

            # Feature importances
            feat_importances = [
                {"feature": feat_cols[i], "importance": round(float(model.importances[i]), 6)}
                for i in range(len(feat_cols))
            ]
            feat_importances.sort(key=lambda x: x["importance"], reverse=True)

            # Save model
            model_id   = f"{ds_name}_{target}"
            model_file = os.path.join(output_dir, f"{model_id}.json")
            model_data = {
                "model_id":          model_id,
                "dataset":           ds_name,
                "target":            target,
                "feature_names":     feat_cols,
                "n_features":        len(feat_cols),
                "norm_mins":         mins.tolist(),
                "norm_maxs":         maxs.tolist(),
                "best_params":       {k: (list(v) if isinstance(v,tuple) else v)
                                      for k,v in best_params.items()},
                "metrics":           metrics,
                "feature_importances": feat_importances,
                "training_time_s":   elapsed,
                "layer_meta":        model.layer_meta,
                "model_weights":     model.to_dict(),
                "trained_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            with open(model_file, "w") as f:
                json.dump(model_data, f)
            print(f"  Saved → {model_file}")

            registry.append({
                "model_id":      model_id,
                "dataset":       ds_name,
                "target":        target,
                "feature_names": feat_cols,
                "metrics":       metrics,
                "layers":        len(model.layers),
                "file":          f"{model_id}.json"
            })

    # Save registry
    reg_path = os.path.join(output_dir, "registry.json")
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nRegistry saved → {reg_path}")
    print(f"\nAll models saved to: {output_dir}/")
    print(f"Total models: {len(registry)}")
    return registry


if __name__ == "__main__":
    print("="*60)
    print("DCatBoostF — Training & Saving All Models")
    print("IEEE TKDE 2024 — From-scratch implementation")
    print("="*60)
    train_and_save_all()
