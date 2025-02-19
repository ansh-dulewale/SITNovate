import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import joblib  # For saving the model

# Load the dataset
file_path = "datasets/yield_df.csv"
yield_df = pd.read_csv(file_path)

# Drop unnecessary column if present
if "Unnamed: 0" in yield_df.columns:
    yield_df = yield_df.drop(columns=["Unnamed: 0"])

# Encode categorical variables
label_encoder_area = LabelEncoder()
label_encoder_item = LabelEncoder()
yield_df["Area"] = label_encoder_area.fit_transform(yield_df["Area"])
yield_df["Item"] = label_encoder_item.fit_transform(yield_df["Item"])

# Define features and target variable
X = yield_df.drop(columns=["hg/ha_yield"])
y = yield_df["hg/ha_yield"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate performance
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nRMSE: {rmse:.2f}\nR² Score: {r2:.4f}")

# Define parameter grid for tuning
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Perform Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf_model, param_distributions=param_dist, 
    n_iter=10, cv=3, n_jobs=-1, scoring="neg_mean_squared_error", verbose=2, random_state=42
)

# Fit Randomized Search to training data
random_search.fit(X_train, y_train)

# Get best parameters & train the best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Predict on test set
y_pred_tuned = best_model.predict(X_test)

# Evaluate the model
rmse_tuned = mean_squared_error(y_test, y_pred) ** 0.5
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"Tuned Model Performance:\nRMSE: {rmse_tuned:.2f}\nR² Score: {r2_tuned:.4f}")

# Save the trained model
joblib.dump(best_model, "best_crop_yield_model.pkl")
print("Model saved as 'best_crop_yield_model.pkl'")
