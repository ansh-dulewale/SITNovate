import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
file_path = "/content/drive/MyDrive/SITNnovate/crop_yield/yield_df.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0'])

# Encode categorical variables
label_encoder = LabelEncoder()
df['Area'] = label_encoder.fit_transform(df['Area'])
df['Item'] = label_encoder.fit_transform(df['Item'])

# Define features and target
X = df.drop(columns=['hg/ha_yield'])  # Features
y = df['hg/ha_yield']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "/content/drive/MyDrive/SITNnovate/scaler.pkl")

# Define hyperparameter tuning function
def build_model(learning_rate=0.001, neurons_1=32, neurons_2=16, neurons_3=8):
    model = keras.Sequential([
        layers.Dense(neurons_1, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(neurons_2, activation='relu'),
        layers.Dense(neurons_3, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Experiment with smaller hyperparameters
best_mae = float("inf")
best_params = {}
for lr in [0.01, 0.001]:
    for neurons_1 in [32, 64]:
        for neurons_2 in [16, 32]:
            for neurons_3 in [8, 16]:
                print(f"Training model with lr={lr}, neurons=({neurons_1}, {neurons_2}, {neurons_3})")
                model = build_model(learning_rate=lr, neurons_1=neurons_1, neurons_2=neurons_2, neurons_3=neurons_3)
                history = model.fit(X_train, y_train, epochs=2, batch_size=16, validation_data=(X_test, y_test), verbose=0)
                loss, mae = model.evaluate(X_test, y_test, verbose=0)
                if mae < best_mae:
                    best_mae = mae
                    best_params = {'learning_rate': lr, 'neurons_1': neurons_1, 'neurons_2': neurons_2, 'neurons_3': neurons_3}

# Train final model with best parameters
print(f"Best parameters: {best_params}, Best Test MAE: {best_mae}")
final_model = build_model(**best_params)
history = final_model.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
final_model.save("/content/drive/MyDrive/SITNnovate/crop_yield_model.h5")
print("Model saved successfully.")

# Load the trained model
model = keras.models.load_model("/content/drive/MyDrive/SITNnovate/crop_yield_model.h5",compile = False)
scaler = joblib.load("/content/drive/MyDrive/SITNnovate/scaler.pkl")

def predict_yield(*features):
    input_features = np.array([features])
    input_features = scaler.transform(input_features)  # Scale input data
    prediction = model.predict(input_features)[0][0]
    return f"Predicted Yield: {prediction:.2f} hg/ha"

# Define Gradio interface
inputs = [gr.Number(label=col) for col in X.columns]
output = gr.Textbox(label="Predicted Yield")

demo = gr.Interface(fn=predict_yield, inputs=inputs, outputs=output, title="Crop Yield Prediction",
                     description="Enter the required features to predict the crop yield.")

demo.launch()
