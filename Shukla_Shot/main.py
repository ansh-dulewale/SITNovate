from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load your trained model
model = joblib.load("crop_prediction_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from frontend
        features = np.array(data['features']).reshape(1, -1)  # Convert to array
        
        # Make prediction
        prediction = model.predict(features)
        
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
