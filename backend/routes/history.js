// routes/history.js
const express    = require('express');
const router     = express.Router();
const Prediction = require('../models/Prediction');

// GET /api/history?page=1&limit=20&dataset=math
router.get('/', async (req, res) => {
  try {
    const { page=1, limit=20, dataset, target } = req.query;
    const filter = {};
    if (dataset) filter.dataset = dataset;
    if (target)  filter.target  = target;

    const total = await Prediction.countDocuments(filter);
    const records = await Prediction.find(filter)
      .sort({ timestamp: -1 })
      .skip((page - 1) * limit)
      .limit(parseInt(limit));

    res.json({
      success: true,
      total,
      page: parseInt(page),
      pages: Math.ceil(total / limit),
      records
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// DELETE /api/history/:id
router.delete('/:id', async (req, res) => {
  try {
    await Prediction.findByIdAndDelete(req.params.id);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
