# Step-by-Step Execution Guide
# SHAPAdaptiveDCatBoostF — Student Performance Prediction

## ─────────────────────────────────────────────
## QUESTION: VS Code vs Jupyter Notebook?
## ─────────────────────────────────────────────

**USE VS CODE TERMINAL — It is MUCH faster than Jupyter.**

| Feature           | Jupyter Notebook       | VS Code Terminal    |
|-------------------|------------------------|---------------------|
| ML script speed   | ~1.5 days (20 runs)   | ~5-8 mins (fast mode) |
| Memory management | Poor (kernel overhead) | Good                |
| Background run    | No                     | Yes (runs while you work) |
| Error visibility  | Cell-by-cell           | Full traceback       |
| Recommended?      | ✗ NO                   | ✓ YES               |

The ML code given here runs in **fast mode** (40 estimators, 3-fold, 3 runs).
For full paper-level accuracy (20 runs, 10-fold, 300 estimators), increase
those values in novelty_train.py — but expect ~2-3 hours, not 1.5 days.

## ─────────────────────────────────────────────
## PREREQUISITES (Install once)
## ─────────────────────────────────────────────

1. Python 3.8+        → https://python.org
2. Node.js 18+        → https://nodejs.org  (LTS version)
3. MongoDB Community  → https://mongodb.com/try/download/community
4. VS Code            → https://code.visualstudio.com

## ─────────────────────────────────────────────
## STEP 1 — OPEN PROJECT IN VS CODE
## ─────────────────────────────────────────────

1. Extract the zip: student-perf-app/
2. Open VS Code → File → Open Folder → select student-perf-app/
3. Press Ctrl+` to open the integrated terminal

## ─────────────────────────────────────────────
## STEP 2 — TRAIN ML MODELS (Terminal 1)
## ─────────────────────────────────────────────

```bash
cd ml
pip install numpy pandas
python novelty_train.py
```

Expected output (in ~5-8 minutes):
```
SHAPAdaptiveDCatBoostF — Full Training Pipeline
[DATA] Synthetic math data
  math_G1  (395 samples, 30 features)
  Training BASE (DCatBoostF)...
  BASE done: MAE=2.07 F1=0.56
  Training NOVELTY (SHAPAdaptiveDCatBoostF)...
    [Novelty] Layer 1: 8 feats, θ=0.0312, gen_phi=0.0234
  NOVELTY done: MAE=1.99 F1=0.58
  ΔMAE=+0.07 ΔF1=+0.02 (+3.4%)
...
✓ registry.json
✓ novelty_registry.json
✓ comparison.json
```

This creates 21 JSON files in ml/saved_models/:
- 9 base models (math_G1.json ... exam_writing_score.json)
- 9 novelty models (*_novelty.json)
- comparison.json, registry.json, novelty_registry.json

## ─────────────────────────────────────────────
## STEP 3 — START MONGODB
## ─────────────────────────────────────────────

Windows: MongoDB starts automatically as a Windows Service.
If not running, press Win+R → type "services.msc" → find MongoDB → Start

Mac/Linux:
```bash
mongod --dbpath /data/db
```

## ─────────────────────────────────────────────
## STEP 4 — START BACKEND (Terminal 2, new terminal)
## ─────────────────────────────────────────────

Click the + button in VS Code terminal to open a new terminal:

```bash
cd backend
npm install
npm start
```

Expected:
```
Server on http://localhost:5000
MongoDB connected
```

Test it: open browser → http://localhost:5000/api/models
Should show JSON list of 9 models.

## ─────────────────────────────────────────────
## STEP 5 — START FRONTEND (Terminal 3, new terminal)
## ─────────────────────────────────────────────

Click + again for third terminal:

```bash
cd frontend
npm install
npm start
```

Browser opens automatically at http://localhost:3000

## ─────────────────────────────────────────────
## WHAT YOU WILL SEE IN THE WEB APP
## ─────────────────────────────────────────────

Page           | Content
───────────────|──────────────────────────────────────────
Dashboard      | MAE bar chart, F1 bar chart, radar chart,
               | per-model improvement table, novelty equations
Predict        | Select model, set features with sliders,
               | run DCatBoostF or SHAP-Adaptive prediction
Models         | Feature importance bars, radar, layer diagram
Compare ★      | FULL COMPARISON PAGE:
               | - Regression metrics (MAE,RMSE,SD,MAC)
               | - Classification metrics (Acc,Prec,Recall,F1)
               | - Confusion matrices side-by-side
               | - Adaptive threshold visualization
               | - Algorithm differences table
               | - Line charts across all 9 datasets
History        | All predictions stored in MongoDB
About          | Every equation explained (Eq.1-10, N1-N3)

## ─────────────────────────────────────────────
## FILE STRUCTURE
## ─────────────────────────────────────────────

student-perf-app/
├── ml/
│   ├── novelty_train.py          ← RUN THIS FIRST
│   └── saved_models/             ← Output: 21 JSON files
│       ├── registry.json
│       ├── comparison.json       ← Used by Compare page
│       ├── math_G1.json          ← Base model
│       ├── math_G1_novelty.json  ← Novelty model
│       └── ...
├── backend/
│   ├── server.js                 ← RUN SECOND (npm start)
│   ├── routes/comparison.js      ← New: /api/comparison
│   └── middleware/inference.js   ← JS inference engine
└── frontend/
    ├── src/pages/
    │   ├── Dashboard.js          ← Updated with comparison charts
    │   ├── Comparison.js         ← NEW: full comparison page
    │   └── About.js              ← All equations N1-N3 explained
    └── package.json              ← RUN THIRD (npm start)

## ─────────────────────────────────────────────
## NOVELTY ALGORITHM SUMMARY
## ─────────────────────────────────────────────

Eq.N1 (SHAP importance):
  φᵢ = (1/T) Σₜ Σ_{n:feat=i} [nL/N · nR/N]
  Replaces: Eq.1 RF split-count importance

Eq.N2 (Adaptive threshold):
  θₗ = clip(μₗ + 0.5·σₗ, 0.03, 0.40)
  Replaces: fixed θ₁=0.05, θ₂=0.05, θ₃=0.90

Eq.N3 (Weighted fusion):
  fs₂ ← [fs₂, softmax(φᵢ)·fy]
  Replaces: plain concatenation Eq.6

## ─────────────────────────────────────────────
## COMMON ERRORS
## ─────────────────────────────────────────────

Error: "MongoDB connection refused"
Fix:   Start MongoDB service (see Step 3)

Error: "Cannot find module 'axios'"
Fix:   cd frontend && npm install

Error: "Cannot find module 'express'"
Fix:   cd backend && npm install

Error: "Port 3000 already in use"
Fix:   Type Y when React asks to use another port

Error: "saved_models/comparison.json not found"
Fix:   Run python novelty_train.py first (Step 2)
