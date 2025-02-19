import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
