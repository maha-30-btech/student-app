const express = require('express');
const router  = express.Router();
const fs      = require('fs');
const path    = require('path');

const MODELS_DIR = path.join(__dirname, '../../ml/saved_models');

// GET /api/comparison  — all 9 comparison entries
router.get('/', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, 'comparison.json'), 'utf8'));
    res.json({ success: true, comparisons: data });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET /api/comparison/:modelId  — single entry
router.get('/:modelId', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, 'comparison.json'), 'utf8'));
    const entry = data.find(d => d.model_id === req.params.modelId);
    if (!entry) return res.status(404).json({ success: false, error: 'Not found' });
    res.json({ success: true, comparison: entry });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
