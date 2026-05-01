const fs   = require('fs');
const path = require('path');

const MODELS_DIR = path.join(__dirname, '../../ml/saved_models');
const cache = {};

function loadRegistry() {
  const regPath = path.join(MODELS_DIR, 'registry.json');
  return JSON.parse(fs.readFileSync(regPath, 'utf8'));
}

function loadModel(modelId) {
  if (cache[modelId]) return cache[modelId];
  const filePath = path.join(MODELS_DIR, `${modelId}.json`);
  if (!fs.existsSync(filePath)) return null;
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  cache[modelId] = data;
  return data;
}

module.exports = { loadRegistry, loadModel };
