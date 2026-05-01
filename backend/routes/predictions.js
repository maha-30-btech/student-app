const express    = require('express');
const router     = express.Router();
const { loadModel } = require('../middleware/modelLoader');
const { runInference } = require('../middleware/inference');
const Prediction = require('../models/Prediction');

/**
 * POST /api/predictions
 * Body: { modelId, features: { featureName: value, ... } }
 */
router.post('/', async (req, res) => {
  try {
    const { modelId, features } = req.body;

    if (!modelId)  return res.status(400).json({ success: false, error: 'modelId is required' });
    if (!features) return res.status(400).json({ success: false, error: 'features is required' });

    const modelData = loadModel(modelId);
    if (!modelData) return res.status(404).json({ success: false, error: `Model "${modelId}" not found` });

    // Build ordered input vector matching feature_names order
    const inputRow = modelData.feature_names.map(name => {
      const val = parseFloat(features[name]);
      if (isNaN(val)) throw new Error(`Missing or invalid feature: "${name}"`);
      return val;
    });

    // Run inference (auto-detects DCatBoostF vs SHAPAdaptiveDCatBoostF)
    const rawPred = runInference(inputRow, modelData);

    if (rawPred === null || rawPred === undefined || isNaN(rawPred)) {
      throw new Error('Model returned invalid prediction (null/NaN)');
    }

    // Clamp to valid range
    const isScore  = modelData.target?.includes('score');
    const maxRange = isScore ? 100 : 20;
    const prediction = Math.min(Math.max(Math.round(rawPred * 10) / 10, 0), maxRange);

    // Save to MongoDB
    const record = await Prediction.create({
      modelId,
      dataset:       modelData.dataset,
      target:        modelData.target,
      inputFeatures: features,
      prediction,
    });

    res.json({
      success:      true,
      prediction,
      target:       modelData.target,
      dataset:      modelData.dataset,
      modelType:    modelData.model_type || 'DCatBoostF',
      modelMetrics: modelData.metrics,
      clsMetrics:   modelData.classification_metrics || null,
      layers:       (modelData.model_weights?.layers || modelData.layer_meta || []).length,
      recordId:     record._id,
      timestamp:    record.timestamp,
    });

  } catch (err) {
    console.error('Prediction error:', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
