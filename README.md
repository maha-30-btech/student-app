# DCatBoostF — Student Performance Prediction
## IEEE TKDE 2024 | Full-Stack ML Web Application

> **Paper**: "A Feature Importance-Based Multi-Layer CatBoost for Student Performance Prediction"  
> Zongwen Fan, Jin Gou, Shaoyuan Weng — *IEEE TKDE, Vol. 36, No. 11, November 2024*

---

## Project Structure

```
student-perf-app/
├── ml/
│   ├── model_train_save.py        # From-scratch ML + model serializer
│   └── saved_models/              # 9 trained DCatBoostF models (JSON)
│       ├── registry.json
│       ├── math_G1.json  …  exam_writing_score.json
│
├── backend/
│   ├── server.js                  # Express app
│   ├── middleware/
│   │   ├── inference.js           # JS DCatBoostF inference engine
│   │   └── modelLoader.js         # JSON model cache
│   ├── models/Prediction.js       # Mongoose schema
│   ├── routes/
│   │   ├── models.js              # GET /api/models, /api/models/:id
│   │   ├── predictions.js         # POST /api/predictions
│   │   ├── history.js             # GET /api/history
│   │   └── stats.js               # GET /api/stats
│   └── package.json
│
└── frontend/
    ├── public/index.html
    └── src/
        ├── App.js                 # Router + Sidebar
        ├── index.css              # Full design system
        ├── services/api.js        # Axios API calls
        └── pages/
            ├── Dashboard.js       # Stats + charts
            ├── Predict.js         # Feature input + inference
            ├── Models.js          # Model explorer
            ├── History.js         # Prediction history
            └── About.js           # Paper formulas explained
```

---

## Paper Implementation

All algorithms are implemented **from scratch** (NumPy only, no sklearn/torch):

| Component | File | Paper Reference |
|-----------|------|-----------------|
| Feature importance | `model_train_save.py → RandomForest` | Eq. (1) |
| Sort ascending | `DCatBoostF._sort_ascending()` | Eq. (2) |
| Layer selection | `DCatBoostF._select_for_layer()` | Eq. (3), (5) |
| Feature generation | `DCatBoostF.fit()` | Eq. (4) |
| Feature combination | `DCatBoostF.fit()` | Eq. (6) |
| MAE | `MAE()` | Eq. (7) |
| SD | `SD()` | Eq. (8) |
| RMSE | `RMSE()` | Eq. (9) |
| MAC | `MAC()` | Eq. (10) |
| Grid Search + 10-fold CV | `grid_search_cv()` | Algorithm 1 |
| Wilcoxon Rank-Sum Test | `wilcoxon_rank_sum_test()` | Section III |
| Ordered Boosting | `CatBoostScratch._ordered_gradient()` | CatBoost core |

The JavaScript inference engine (`backend/middleware/inference.js`) exactly mirrors the Python prediction path.

---

## Setup & Run

### Prerequisites
- Python 3.8+ with NumPy, Pandas
- Node.js 18+
- MongoDB (local or Atlas)

### Step 1 — Train Models (already done, skip if using provided saved_models)
```bash
cd ml
pip install numpy pandas
python model_train_save.py
```

This trains 9 DCatBoostF instances and saves them as JSON files:
- **math/G1, G2, G3** — Mathematics course grade prediction
- **port/G1, G2, G3** — Portuguese language course grade prediction
- **exam/math_score, reading_score, writing_score** — Exam score prediction

### Step 2 — Backend
```bash
cd backend
npm install
# Edit .env:
#   MONGO_URI=mongodb://localhost:27017/student_perf
#   PORT=5000
npm start       # production
npm run dev     # development (nodemon)
```

### Step 3 — Frontend
```bash
cd frontend
npm install
npm start       # runs on http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all 9 trained models |
| GET | `/api/models/:id` | Model metadata + feature importances |
| GET | `/api/models/:id/features` | Feature schema for input form |
| POST | `/api/predictions` | Run DCatBoostF inference |
| POST | `/api/predictions/batch` | Batch inference |
| GET | `/api/history` | Paginated prediction history |
| DELETE | `/api/history/:id` | Delete a prediction record |
| GET | `/api/stats` | Dashboard statistics |

### Prediction Request Example
```json
POST /api/predictions
{
  "modelId": "math_G1",
  "features": {
    "failures": 0,
    "studytime": 2,
    "absences": 5,
    "higher": 1,
    "health": 3,
    ...
  }
}
```

### Prediction Response
```json
{
  "success": true,
  "prediction": 14.2,
  "target": "G1",
  "dataset": "math",
  "modelMetrics": { "MAE": 1.23, "SD": 1.45, "RMSE": 1.67, "MAC": 0.89 },
  "layers": 3,
  "recordId": "...",
  "timestamp": "2024-..."
}
```

---

## Datasets

| Dataset | Samples | Features | Targets |
|---------|---------|----------|---------|
| Mathematics (UCI) | 395 | 30 | G1, G2, G3 |
| Portuguese (UCI) | 649 | 30 | G1, G2, G3 |
| Exam Scores | 1000 | 5 | math_score, reading_score, writing_score |

Download real UCI data from: https://archive.ics.uci.edu/ml/datasets/student+performance  
Place `student-mat.csv` and `student-por.csv` in `ml/` and re-run `model_train_save.py`.

---

## Model Architecture — DCatBoostF

```
Input Features (n=30)
        │
        ▼
RF Feature Importance    ← Eq.(1): fs = RF(X, y)
        │
        ▼
Sort Ascending           ← Eq.(2)
        │
   ┌────▼────┐
   │ Layer 1 │  θ₁=0.05  ← Eq.(3): least-important features
   │CatBoost │
   └────┬────┘
        │ fy = CatBoost(X,y)   ← Eq.(4): generate feature
   ┌────▼────┐
   │ Layer 2 │  θ₂=0.05  ← Eq.(5)+(6): next features + fy
   │CatBoost │
   └────┬────┘
        │ fy2 = CatBoost(...)
   ┌────▼────┐
   │ Layer 3 │  θ₃=0.90  ← remaining features + fy + fy2
   │CatBoost │
   └────┬────┘
        │
        ▼
   Final Prediction
```
