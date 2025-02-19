const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();

app.use(express.json());
app.use(cors());

const FLASK_API_URL = 'http://127.0.0.1:5000';

// Route to get crop prediction
app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post(`${FLASK_API_URL}/predict_crop`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Error connecting to Flask API' });
    }
});

// Route to get crop analysis visualization
app.get('/crop_analysis', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_API_URL}/crop_analysis`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Error fetching crop analysis' });
    }
});

const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
