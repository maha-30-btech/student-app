/**
 * DCatBoostF + SHAPAdaptiveDCatBoostF Inference Engine
 * Pure JavaScript — no ML libraries.
 * Keys match the exact JSON output of novelty_train.py
 *
 * JSON structure produced by Python:
 *   model_weights.layers[i] = { fo:[...], npg:N, cat:{n,md,lr,base,trees:[]} }
 *   tree node = { leaf:bool, f:int, t:float, l:{...}, r:{...}, v:float }
 *   novelty layers also have: shap_w:[...]
 */

// ── Normalize one input row ───────────────────────────────────────────────────
function normalizeRow(row, mins, maxs) {
  return row.map((val, j) => {
    const denom = maxs[j] - mins[j];
    return denom === 0 ? 0 : (val - mins[j]) / denom;
  });
}

// ── XGBTree: walk one node for one sample ────────────────────────────────────
// Node keys from Python to_dict(): leaf, f, t, l, r, v
function walkNode(x, node) {
  if (!node) return 0;
  if (node.leaf) return node.v;                      // 'v' = value
  return x[node.f] <= node.t                        // 'f' = feature, 't' = threshold
    ? walkNode(x, node.l)                           // 'l' = left
    : walkNode(x, node.r);                          // 'r' = right
}

// ── CatBoost predict for one sample ──────────────────────────────────────────
// catDict keys: { n, md, lr, base, trees:[ {md, nf, root:{...}} ] }
function catPredictOne(x, catDict) {
  let F = catDict.base;                              // 'base' = base_pred
  const lr = catDict.lr;                            // 'lr'   = learning_rate
  for (const tree of catDict.trees) {
    F += lr * walkNode(x, tree.root);
  }
  return F;
}

// ── DCatBoostF predict (base paper model) ────────────────────────────────────
// layer keys: { fo:[feature indices], npg:N, cat:{...} }
function predictDCatBoostF(inputRow, modelWeights, normMins, normMaxs) {
  const xNorm   = normalizeRow(inputRow, normMins, normMaxs);
  const genFeats = [];
  let finalPred  = null;

  for (const layer of modelWeights.layers) {
    const featOrig = layer.fo;                       // 'fo' = feat_orig
    const catDict  = layer.cat;                     // 'cat' = CatBoost model

    // Build feature vector: original features + generated features from prev layers
    const xLayer = [
      ...featOrig.map(fi => xNorm[fi]),
      ...genFeats
    ];

    const pred = catPredictOne(xLayer, catDict);
    genFeats.push(pred);
    finalPred = pred;
  }

  return finalPred;
}

// ── SHAPAdaptiveDCatBoostF predict (novelty model) ───────────────────────────
// novelty layer keys: { fo:[...], npg:N, shap_w:[...], cat:{...} }
// Eq.N3: generated features are multiplied by their softmax SHAP weights
function predictSHAPAdaptive(inputRow, modelWeights, normMins, normMaxs) {
  const xNorm    = normalizeRow(inputRow, normMins, normMaxs);
  const genFeats = [];   // raw generated feature values
  const genAlphas = [];  // SHAP weights for each generated feature
  let finalPred  = null;

  for (const layer of modelWeights.layers) {
    const featOrig = layer.fo;
    const catDict  = layer.cat;
    const shapW    = layer.shap_w || [];  // Eq.N3 weights

    // Build feature vector
    const xLayer = [...featOrig.map(fi => xNorm[fi])];

    // Eq.N3: apply SHAP weight to each generated feature
    if (genFeats.length > 0 && shapW.length > 0) {
      for (let k = 0; k < genFeats.length; k++) {
        xLayer.push((shapW[k] || 1) * genFeats[k]);
      }
    } else {
      // fallback: plain concat (like base paper)
      for (const gf of genFeats) xLayer.push(gf);
    }

    const pred = catPredictOne(xLayer, catDict);
    genFeats.push(pred);
    finalPred = pred;
  }

  return finalPred;
}

// ── Main predict dispatcher ───────────────────────────────────────────────────
// Automatically detects model type from model_weights.type
function runInference(inputRow, modelData) {
  const { model_weights, norm_mins, norm_maxs } = modelData;

  if (!model_weights) throw new Error('model_weights missing from model file');
  if (!norm_mins)     throw new Error('norm_mins missing from model file');
  if (!norm_maxs)     throw new Error('norm_maxs missing from model file');
  if (!model_weights.layers || model_weights.layers.length === 0) {
    throw new Error('model_weights.layers is empty');
  }

  const modelType = model_weights.type || 'DCatBoostF';

  if (modelType === 'SHAPAdaptiveDCatBoostF') {
    return predictSHAPAdaptive(inputRow, model_weights, norm_mins, norm_maxs);
  } else {
    return predictDCatBoostF(inputRow, model_weights, norm_mins, norm_maxs);
  }
}

// ── Metrics (Equations 7-10) ──────────────────────────────────────────────────
function MAE(yTrue, yPred) {
  const n = yTrue.length;
  return yTrue.reduce((s, yi, i) => s + Math.abs(yPred[i] - yi), 0) / n;
}
function SD(yTrue, yPred) {
  const n = yTrue.length;
  const errors = yTrue.map((yi, i) => yPred[i] - yi);
  const eBar   = errors.reduce((a, b) => a + b, 0) / n;
  return Math.sqrt(errors.reduce((s, e) => s + (e - eBar) ** 2, 0) / n);
}
function RMSE(yTrue, yPred) {
  const n = yTrue.length;
  return Math.sqrt(yTrue.reduce((s, yi, i) => s + (yPred[i] - yi) ** 2, 0) / n);
}
function MAC(yTrue, yPred) {
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const d1 = dot(yTrue, yPred), d2 = dot(yTrue, yTrue), d3 = dot(yPred, yPred);
  return (d2 === 0 || d3 === 0) ? 0 : (d1 ** 2) / (d2 * d3);
}

module.exports = { runInference, normalizeRow, MAE, SD, RMSE, MAC };
