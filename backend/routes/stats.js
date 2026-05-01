// routes/stats.js
const express    = require('express');
const router     = express.Router();
const Prediction = require('../models/Prediction');
const { loadRegistry } = require('../middleware/modelLoader');

// GET /api/stats — dashboard stats
router.get('/', async (req, res) => {
  try {
    const totalPredictions = await Prediction.countDocuments();
    const byDataset = await Prediction.aggregate([
      { $group: { _id: '$dataset', count: { $sum: 1 }, avgPred: { $avg: '$prediction' } } }
    ]);
    const recent = await Prediction.find().sort({ timestamp: -1 }).limit(5);
    const registry = loadRegistry();

    res.json({
      success: true,
      totalPredictions,
      totalModels: registry.length,
      byDataset,
      recentPredictions: recent,
      models: registry.map(m => ({
        modelId: m.model_id,
        dataset: m.dataset,
        target:  m.target,
        metrics: m.metrics,
        layers:  m.layers
      }))
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
