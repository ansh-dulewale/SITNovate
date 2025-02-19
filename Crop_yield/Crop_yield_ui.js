import React, { useState } from 'react';
import axios from 'axios';

const CropPredictor = () => {
    const [formData, setFormData] = useState({ N: '', P: '', K: '', temperature: '', humidity: '', ph: '', rainfall: '' });
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://localhost:3000/predict', formData);
            setPrediction(response.data.recommended_crop);
            setError(null);
        } catch (err) {
            setError('Error predicting crop. Please try again.');
        }
    };

    return (
        <div className="flex flex-col items-center p-6">
            <h1 className="text-2xl font-bold mb-4">Crop Prediction</h1>
            <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4 w-1/2">
                {Object.keys(formData).map((key) => (
                    <input
                        key={key}
                        type="number"
                        name={key}
                        value={formData[key]}
                        onChange={handleChange}
                        placeholder={key.charAt(0).toUpperCase() + key.slice(1)}
                        className="p-2 border rounded"
                        required
                    />
                ))}
                <button type="submit" className="col-span-2 p-2 bg-blue-500 text-white rounded">Predict</button>
            </form>
            {prediction && <p className="mt-4 text-lg font-semibold">Recommended Crop: {prediction}</p>}
            {error && <p className="mt-4 text-red-500">{error}</p>}
        </div>
    );
};

export default CropPredictor;
