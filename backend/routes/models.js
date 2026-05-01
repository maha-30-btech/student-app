const express = require('express');
const router  = express.Router();
const { loadRegistry, loadModel } = require('../middleware/modelLoader');

// GET /api/models — list all available models
router.get('/', (req, res) => {
  try {
    const registry = loadRegistry();
    res.json({ success: true, models: registry });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET /api/models/:modelId — get model metadata + feature importances
router.get('/:modelId', (req, res) => {
  try {
    const model = loadModel(req.params.modelId);
    if (!model) return res.status(404).json({ success: false, error: 'Model not found' });
    // Return everything except the heavy model_weights
    const { model_weights, ...meta } = model;
    res.json({ success: true, model: meta });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET /api/models/:modelId/features — get input feature schema
router.get('/:modelId/features', (req, res) => {
  try {
    const model = loadModel(req.params.modelId);
    if (!model) return res.status(404).json({ success: false, error: 'Model not found' });

    // Build feature schema with ranges from norm_mins/norm_maxs
    const features = model.feature_names.map((name, i) => ({
      name,
      min: model.norm_mins[i],
      max: model.norm_maxs[i],
      importance: model.feature_importances.find(f => f.feature === name)?.importance || 0
    }));

    res.json({
      success: true,
      modelId: model.model_id,
      target: model.target,
      features,
      featureImportances: model.feature_importances
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
