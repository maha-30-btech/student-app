// models/Prediction.js
const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
  modelId:     { type: String, required: true },
  dataset:     { type: String, required: true },
  target:      { type: String, required: true },
  inputFeatures: { type: Object, required: true },
  prediction:  { type: Number, required: true },
  timestamp:   { type: Date,   default: Date.now },
  sessionId:   { type: String }
});

module.exports = mongoose.model('Prediction', PredictionSchema);
