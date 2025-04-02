import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the trained model
model_filename = "flight.pkl"
with open(model_filename, "rb") as file:
    best_model = pickle.load(file)

# Load test dataset
test_data_file = "archive(2)/Scraped_dataset.csv"
test_data = pd.read_csv(test_data_file)

# Convert 'Price' column to numeric (fix string issue)
test_data["Price"] = test_data["Price"].astype(str).str.replace(",", "").astype(float)

# Separate features (X) and target (y)
y_test = test_data["Price"]
X_test = test_data.drop(columns=["Price"])

# Ensure test features match training features
train_features = best_model.feature_names_in_  # Features used during training
X_test = X_test.reindex(columns=train_features, fill_value=0)

# Make predictions
y_pred_test = best_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Print results
print(f"Test Data - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Save predictions to CSV
output_df = pd.DataFrame({"Actual Price": y_test, "Predicted Price": y_pred_test})
output_df.to_csv("flight_price_predictions.csv", index=False)
print("Predictions saved as 'flight_price_predictions.csv'.")
