const express   = require('express');
const mongoose  = require('mongoose');
const cors      = require('cors');
require('dotenv').config();

const app = express();
app.use(cors({ origin: process.env.FRONTEND_URL || 'http://localhost:3000' }));
app.use(express.json());

app.use('/api/models',      require('./routes/models'));
app.use('/api/predictions', require('./routes/predictions'));
app.use('/api/history',     require('./routes/history'));
app.use('/api/stats',       require('./routes/stats'));
app.use('/api/comparison',  require('./routes/comparison'));

mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/student_perf')
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB:', err));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server on http://localhost:${PORT}`));
