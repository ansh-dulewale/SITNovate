import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset (Replace 'your_dataset.csv' with actual dataset path)
df = pd.read_csv("../Shukla_Shot/crop_prediction_model.pkl")  

# Assuming the last column is the target variable
X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "crop_prediction_model.pkl")

print("Model retrained and saved as crop_prediction_model.pkl")
