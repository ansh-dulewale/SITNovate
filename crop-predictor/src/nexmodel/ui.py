import numpy as np
import joblib
import gradio as gr
import tensorflow as tf
from tensorflow import keras

# Load the trained model and scaler
model = keras.models.load_model("/mnt/data/crop_yield_model.h5")
scaler = joblib.load("/mnt/data/scaler.pkl")

# Function to make predictions
def predict_yield(*features):
    input_features = np.array([features])
    input_features = scaler.transform(input_features)  # Scale input data
    prediction = model.predict(input_features)[0][0]
    return f"Predicted Yield: {prediction:.2f} hg/ha"

# Define input fields based on model features
feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']  # Update based on dataset
inputs = [gr.Number(label=col) for col in feature_names]

# Define Gradio interface
output = gr.Textbox(label="Predicted Yield")
demo = gr.Interface(fn=predict_yield, inputs=inputs, outputs=output, 
                    title="Crop Yield Prediction",
                    description="Enter the required features to predict the crop yield.")

demo.launch()
