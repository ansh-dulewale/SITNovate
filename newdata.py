from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Load datasets
crop_production_df = pd.read_csv("crop_production.csv")
crop_recommendation_df = pd.read_csv("Crop_recommendation.csv")

# Initialize Flask app
app = Flask(__name__)

# Preprocess data for crop recommendation
le = LabelEncoder()
crop_recommendation_df['label_encoded'] = le.fit_transform(crop_recommendation_df['label'])
X = crop_recommendation_df.drop(columns=['label', 'label_encoded'])
y = crop_recommendation_df['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("crop_model.pkl", "wb"))

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        features = np.array([[data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']]])
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]
        return jsonify({'recommended_crop': predicted_crop})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/crop_analysis', methods=['GET'])
def crop_analysis():
    try:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=crop_production_df['Crop'], order=crop_production_df['Crop'].value_counts().index)
        plt.xlabel("Count")
        plt.ylabel("Crops")
        plt.title("Crop Production Count")
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        
        return jsonify({'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)