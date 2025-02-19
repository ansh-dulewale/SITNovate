import crop_yield_model
import joblib
# Load the saved model
loaded_model = joblib.load("best_crop_yield_model.pkl")

# Example: Make predictions with new data
sample_input = X_test.iloc[:5]  # Take first 5 test samples
sample_predictions = loaded_model.predict(sample_input)

print("Sample Predictions:", sample_predictions)
