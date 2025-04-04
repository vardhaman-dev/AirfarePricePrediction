# Flight Price Prediction Project

## Overview

This project implements a high-accuracy airfare price prediction model using a stacking ensemble. The model is built using historical (cleaned) flight data for training and real-world scraped flight data for testing. Advanced feature engineering—including cyclical encoding for time features and interaction terms—is applied to boost accuracy. The target variable (Fare) is log‑transformed to stabilize variance. The final model is saved and can be integrated into a Flask-based web application for live predictions.

The current implementation includes robust data preprocessing, ensemble modeling using XGBoost, Random Forest, LightGBM (as base models) and CatBoost (as the final estimator), and progress reporting during training.

---

## Detailed Explanation of the Code

### 1. Imports and Setup

The script starts by importing essential libraries for data manipulation, machine learning, and model evaluation. It also sets up warnings suppression and imports `tqdm` for progress reporting.

```python
import pandas as pd
import numpy as np
import re
import joblib
import time
import threading
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm

# Suppress runtime warnings (e.g., "Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

2. Utility Functions

These helper functions convert textual data to numeric (for stops and duration), extract time features, and add cyclical transformations for periodic features.

def convert_stops(x):
    """Convert stop descriptions (e.g., 'non-stop', '1 stop') to numeric values."""
    if not isinstance(x, str):
        return x
    x = x.lower().strip()
    if x == "non-stop":
        return 0
    match = re.search(r'\d+', x)  # Extract the first number
    return int(match.group()) if match else np.nan

def extract_hour(time_str):
    """Extract the hour component from a time string."""
    if pd.isna(time_str):
        return np.nan
    try:
        return int(time_str.split(':')[0])
    except (ValueError, AttributeError):
        return np.nan

def duration_to_hours(duration):
    """Convert a duration string (e.g., '2h 50m') to hours as a float."""
    if not isinstance(duration, str):
        return np.nan
    hours, minutes = 0, 0
    if 'h' in duration:
        try:
            hours = int(re.search(r'(\d+)\s*h', duration).group(1))
        except:
            hours = 0
    if 'm' in duration:
        try:
            minutes = int(re.search(r'(\d+)\s*m', duration).group(1))
        except:
            minutes = 0
    return hours + minutes / 60

def add_cyclical_features(df, col, period):
    """Add sine and cosine transformation for cyclical features."""
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df

3. Training Pipeline

3.1 Loading and Preprocessing Training Data

The training data is loaded from archive(2)/Cleaned_dataset.csv. It is preprocessed by:

Converting stop descriptions to numeric,

Processing date features (including converting dates to ordinal numbers and extracting the day of the week),

Extracting the departure hour,

One-hot encoding categorical variables (e.g., Airline, Class),

Dropping irrelevant columns, and

Log‑transforming the target (Fare).


Missing values are imputed using a KNN imputer.

print("=== Training Phase ===")
train_df = pd.read_csv('archive(2)/Cleaned_dataset.csv')
print("Training Data shape:", train_df.shape)
print(train_df.head(5))

# Standardize column names
train_df.columns = [col.strip() for col in train_df.columns]

# Preprocess training data: Convert stops and handle missing values
train_df['Total_stops'] = train_df['Total_stops'].apply(convert_stops)
train_df['Total_stops'] = train_df['Total_stops'].fillna(train_df['Total_stops'].median())

# Convert date features
train_df['Date_of_journey'] = pd.to_datetime(train_df['Date_of_journey'], format='%d/%m/%Y', errors='coerce')
train_df['Date_of_journey_ordinal'] = train_df['Date_of_journey'].apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
train_df['Journey_day_of_week'] = train_df['Date_of_journey'].dt.dayofweek
train_df['Departure_hour'] = train_df['Departure'].apply(extract_hour)
for col in ['Total_stops', 'Departure_hour']:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# One-hot encode categorical columns: Airline and Class
categorical_cols = ['Airline', 'Class']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

# Drop columns not used in modeling
drop_cols = ['Date_of_journey', 'Flight_code', 'Departure', 'Arrival', 'Journey_day', 'Source', 'Destination']
train_df.drop(columns=drop_cols, inplace=True)

# Prepare features and target (log-transform Fare)
y_train = np.log1p(train_df['Fare'])
X_train_full = train_df.drop(columns=['Fare'])
X_train_full = X_train_full.apply(pd.to_numeric, errors='coerce')

# Impute remaining missing values using KNNImputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train_full_imputed = pd.DataFrame(imputer.fit_transform(X_train_full), columns=X_train_full.columns)
print("NaNs in training data after imputation:", X_train_full_imputed.isna().sum().sum())
print("X_train_full shape:", X_train_full_imputed.shape)

3.2 Advanced Feature Engineering and Scaling

Additional features are engineered by:

Adding cyclical encoding for time features (e.g., departure hour, day of week),

Creating interaction terms (e.g., stops multiplied by duration),

Scaling numeric features using a QuantileTransformer.


# Split training data for internal validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_full_imputed, y_train, test_size=0.2, random_state=42)

# Advanced Feature Engineering
if 'Departure_hour' in X_train_full_imputed.columns:
    X_train_full_imputed = add_cyclical_features(X_train_full_imputed, 'Departure_hour', 24)
if 'Journey_day_of_week' in X_train_full_imputed.columns:
    X_train_full_imputed = add_cyclical_features(X_train_full_imputed, 'Journey_day_of_week', 7)
if 'Total_stops' in X_train_full_imputed.columns and 'Duration_in_hours' in X_train_full_imputed.columns:
    X_train_full_imputed['Stops_Duration'] = X_train_full_imputed['Total_stops'] * X_train_full_imputed['Duration_in_hours']
if 'Departure_hour' in X_train_full_imputed.columns and 'Days_left' in X_train_full_imputed.columns:
    X_train_full_imputed['Hour_DaysLeft'] = X_train_full_imputed['Departure_hour'] * X_train_full_imputed['Days_left']

# Scale numeric features using QuantileTransformer
numeric_cols = ['Total_stops', 'Duration_in_hours', 'Departure_hour', 'Days_left', 
                'Journey_day_of_week', 'Date_of_journey_ordinal', 'Stops_Duration', 'Hour_DaysLeft']
numeric_cols = [col for col in numeric_cols if col in X_train_full_imputed.columns]
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_full_imputed[numeric_cols] = scaler.fit_transform(X_train_full_imputed[numeric_cols])

3.3 Building the Model with Progress Reporting

A stacking ensemble is built using:

Base Estimators: XGBoost, RandomForestRegressor, LGBMRegressor

Final Estimator: CatBoostRegressor


A simple progress bar (using tqdm in a separate thread) shows elapsed training time.

from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm

estimators = [
    ('xgb', XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, objective='reg:squarederror')),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ('lgbm', LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42))
]
final_estimator = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, 
                                     loss_function='RMSE', random_seed=42, verbose=0)
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

print("\nStarting training of stacking ensemble model. Please wait...")

# Progress reporting using a separate thread with tqdm
done_flag = False
def progress_bar():
    from tqdm import tqdm
    with tqdm(desc="Training Progress (sec elapsed)", unit="sec") as pbar:
        while not done_flag:
            time.sleep(1)
            pbar.update(1)

import threading
progress_thread = threading.Thread(target=progress_bar)
progress_thread.start()

start_time = time.time()
stack_model.fit(X_train_split, y_train_split)
done_flag = True  # Signal to stop progress bar
progress_thread.join()
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Validate on validation set
from sklearn.metrics import r2_score
y_val_pred = stack_model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation R² (log scale): {val_r2:.4f}")

4. Testing Phase

4.1 Preprocessing Test Data

The test data is loaded from archive(2)/Scraped_dataset.csv and processed similarly to the training data:

Splitting combined fields, converting duration to hours, processing dates, and imputing missing values.

One-hot encoding is applied, and non-relevant columns are dropped.

The target (Price) is cleaned (commas removed) and log‑transformed.


print("\n=== Testing Phase ===")
scraped_df = pd.read_csv('archive(2)/Scraped_dataset.csv')
print("Scraped Data shape:", scraped_df.shape)

# Split 'Airline-Class' into separate columns if present
if 'Airline-Class' in scraped_df.columns:
    scraped_df['Airline'] = scraped_df['Airline-Class'].str.split('\n').str[0].str.strip()
    scraped_df['Class'] = scraped_df['Airline-Class'].str.split('\n').str[-1].str.strip()

scraped_df['Duration_in_hours'] = scraped_df['Duration'].apply(duration_to_hours)
scraped_df['Date of Booking'] = pd.to_datetime(scraped_df['Date of Booking'], dayfirst=True, errors='coerce')
scraped_df['Date of Journey'] = pd.to_datetime(scraped_df['Date of Journey'], dayfirst=True, errors='coerce')
scraped_df['Days_left'] = (scraped_df['Date of Journey'] - scraped_df['Date of Booking']).dt.days
scraped_df['Journey_day_of_week'] = scraped_df['Date of Journey'].dt.dayofweek
scraped_df['Departure_hour'] = scraped_df['Departure Time'].apply(extract_hour)
scraped_df['Total Stops'] = scraped_df['Total Stops'].apply(convert_stops)
scraped_df['Date_of_Journey_ordinal'] = scraped_df['Date of Journey'].apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
for col in ['Duration_in_hours', 'Departure_hour', 'Total Stops']:
    scraped_df[col] = scraped_df[col].fillna(scraped_df[col].median())

# One-hot encode categorical columns (Airline & Class)
scraped_df = pd.get_dummies(scraped_df, columns=['Airline', 'Class'], drop_first=True)

# Drop columns not used in modeling
drop_test_cols = ['Price', 'Date of Booking', 'Date of Journey', 'Duration', 'Airline-Class', 'Departure Time', 'Arrival Time']
X_test_full = scraped_df.drop(columns=drop_test_cols)
X_test_full = X_test_full.apply(pd.to_numeric, errors='coerce')

# Convert target price using log1p after removing commas
y_test = np.log1p(scraped_df['Price'].str.replace(',', '').astype(float))

# Impute missing values in test data
imputer_test = KNNImputer(n_neighbors=5)
X_test_full_imputed = pd.DataFrame(imputer_test.fit_transform(X_test_full), columns=X_test_full.columns)
print("NaNs in test data after imputation:", X_test_full_imputed.isna().sum().sum())
print("X_test_full shape:", X_test_full_imputed.shape)

4.2 Align Features and Advanced Engineering on Test Data

Extract common columns between training and test sets and apply the same advanced feature engineering and scaling.

common_cols = list(set(X_train_full_imputed.columns) & set(X_test_full_imputed.columns))
print("Common columns:", common_cols)
X_train = X_train_full_imputed[common_cols].copy()
X_test = X_test_full_imputed[common_cols].copy()
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    raise ValueError("Training or testing data is empty after preprocessing.")

if 'Departure_hour' in X_train.columns:
    X_train = add_cyclical_features(X_train, 'Departure_hour', 24)
    X_test = add_cyclical_features(X_test, 'Departure_hour', 24)
if 'Journey_day_of_week' in X_train.columns:
    X_train = add_cyclical_features(X_train, 'Journey_day_of_week', 7)
    X_test = add_cyclical_features(X_test, 'Journey_day_of_week', 7)
if 'Total_stops' in X_train.columns and 'Duration_in_hours' in X_train.columns:
    X_train['Stops_Duration'] = X_train['Total_stops'] * X_train['Duration_in_hours']
    X_test['Stops_Duration'] = X_test['Total_stops'] * X_test['Duration_in_hours']
if 'Departure_hour' in X_train.columns and 'Days_left' in X_train.columns:
    X_train['Hour_DaysLeft'] = X_train['Departure_hour'] * X_train['Days_left']
    X_test['Hour_DaysLeft'] = X_test['Departure_hour'] * X_test['Days_left']

numeric_cols = ['Total_stops', 'Duration_in_hours', 'Departure_hour', 'Days_left', 
                'Journey_day_of_week', 'Date_of_journey_ordinal', 'Stops_Duration', 'Hour_DaysLeft']
numeric_cols = [col for col in numeric_cols if col in X_train.columns]
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

4.3 Testing and Evaluation

Finally, the trained stacking ensemble model is used to predict and evaluate on the test data. The validation R² and test R² (on the log‑transformed target) are printed, and the final model is saved.

# Evaluate on test set
y_test_pred = stack_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test R² (log scale): {test_r2:.4f}")

# Save the final model
joblib.dump(stack_model, 'enhanced_flight_price_model.pkl')
print("Enhanced model saved as 'enhanced_flight_price_model.pkl'")


---

Issues Faced and Their Solutions

1. Performance & Training Time:

Issue: Training on a large dataset (~452,000 rows) on limited hardware was slow.

Solution: Added a progress bar using tqdm in a separate thread to show elapsed training time.



2. RuntimeWarnings and NaN Handling:

Issue: Warnings like "Mean of empty slice" occurred due to missing values.

Solution: Used KNN imputation to fill missing values and suppressed unnecessary warnings.



3. Feature Misalignment:

Issue: Inconsistencies between training and test features caused prediction errors.

Solution: Aligned features by extracting common columns between the training and test datasets and applying identical preprocessing steps.



4. Advanced Feature Engineering:

Issue: Raw features were insufficiently informative.

Solution: Added cyclical encoding for time features and interaction terms (e.g., stops multiplied by duration) to boost model accuracy.



5. Hardware Limitations:

Issue: Training on an Intel i3 8th Gen system was time-consuming.

Solution: Implemented a progress bar to provide real-time feedback and recommended cloud resources for future scalability.





---

Future Enhancements

Dynamic Data Integration: Incorporate live flight data via APIs for real-time price predictions.

Automated Retraining: Set up a pipeline for periodic model retraining with updated data.

Cloud Deployment: Migrate to a cloud platform (e.g., AWS, Heroku) for faster training and scalable deployment.

Enhanced User Interface: Improve the frontend with interactive visualizations and modern design.



---

Conclusion

This project demonstrates a comprehensive and advanced approach to predicting flight prices using robust data preprocessing, advanced feature engineering, and ensemble modeling techniques. Despite hardware limitations, the integration of progress reporting and feature alignment has led to a high-accuracy model. The final stacking ensemble is saved for future deployment and real-time predictions.

Developed by Vardhaman Ganpule
