import pandas as pd
import numpy as np
import re
import joblib
import time
import threading
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm

# Suppress runtime warnings (e.g., mean of empty slice)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Utility Functions ---
def convert_stops(x):
    """Convert stop descriptions to numeric values."""
    if not isinstance(x, str):
        return x
    x = x.lower().strip()
    if x == "non-stop":
        return 0
    match = re.search(r'\d+', x)
    return int(match.group()) if match else np.nan

def extract_hour(time_str):
    """Extract hour from time string."""
    if pd.isna(time_str):
        return np.nan
    try:
        return int(time_str.split(':')[0])
    except (ValueError, AttributeError):
        return np.nan

def duration_to_hours(duration):
    """Convert duration string (e.g., '2h 50m') to hours as float."""
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

# --- Progress Bar Thread Function ---
done_flag = False
def progress_bar():
    with tqdm(desc="Training Progress (seconds elapsed)", unit="sec", total=0) as pbar:
        while not done_flag:
            time.sleep(1)
            pbar.update(1)

# ====================================================
# TRAINING PHASE
# ====================================================
print("=== Training Phase ===")
train_df = pd.read_csv('archive(2)/Cleaned_dataset.csv')
print("Training Data shape:", train_df.shape)
print(train_df.head(5))

# Standardize column names
train_df.columns = [col.strip() for col in train_df.columns]

# Preprocess training data: Convert stops and dates
train_df['Total_stops'] = train_df['Total_stops'].apply(convert_stops)
train_df['Total_stops'] = train_df['Total_stops'].fillna(train_df['Total_stops'].median())

train_df['Date_of_journey'] = pd.to_datetime(train_df['Date_of_journey'], format='%d/%m/%Y', errors='coerce')
train_df['Date_of_journey_ordinal'] = train_df['Date_of_journey'].apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
train_df['Journey_day_of_week'] = train_df['Date_of_journey'].dt.dayofweek
train_df['Departure_hour'] = train_df['Departure'].apply(extract_hour)
for col in ['Total_stops', 'Departure_hour']:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# One-hot encode categorical columns (Airline & Class)
categorical_cols = ['Airline', 'Class']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

# Drop columns not used in modeling
drop_cols = ['Date_of_journey', 'Flight_code', 'Departure', 'Arrival', 'Journey_day', 'Source', 'Destination']
train_df.drop(columns=drop_cols, inplace=True)

# Log-transform the target for stability
y_train = np.log1p(train_df['Fare'])
X_train_full = train_df.drop(columns=['Fare'])
X_train_full = X_train_full.apply(pd.to_numeric, errors='coerce')

# Impute remaining missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train_full_imputed = pd.DataFrame(imputer.fit_transform(X_train_full), columns=X_train_full.columns)
print("NaNs in training data after imputation:", X_train_full_imputed.isna().sum().sum())
print("X_train_full shape:", X_train_full_imputed.shape)

# ====================================================
# TESTING PHASE (Preprocess Scraped Data)
# ====================================================
print("\n=== Testing Phase ===")
scraped_df = pd.read_csv('archive(2)/Scraped_dataset.csv')
print("Scraped Data shape:", scraped_df.shape)

# Split 'Airline-Class' into separate columns if present
if 'Airline-Class' in scraped_df.columns:
    scraped_df['Airline'] = scraped_df['Airline-Class'].str.split('\n').str[0].str.strip()
    scraped_df['Class'] = scraped_df['Airline-Class'].str.split('\n').str[-1].str.strip()

# Convert duration to hours (float)
scraped_df['Duration_in_hours'] = scraped_df['Duration'].apply(duration_to_hours)

# Convert dates and compute days left
scraped_df['Date of Booking'] = pd.to_datetime(scraped_df['Date of Booking'], dayfirst=True, errors='coerce')
scraped_df['Date of Journey'] = pd.to_datetime(scraped_df['Date of Journey'], dayfirst=True, errors='coerce')
scraped_df['Days_left'] = (scraped_df['Date of Journey'] - scraped_df['Date of Booking']).dt.days

# Compute journey day of week and extract departure hour
scraped_df['Journey_day_of_week'] = scraped_df['Date of Journey'].dt.dayofweek
scraped_df['Departure_hour'] = scraped_df['Departure Time'].apply(extract_hour)
scraped_df['Total Stops'] = scraped_df['Total Stops'].apply(convert_stops)
scraped_df['Date_of_Journey_ordinal'] = scraped_df['Date of Journey'].apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
for col in ['Duration_in_hours', 'Departure_hour', 'Total Stops']:
    scraped_df[col] = scraped_df[col].fillna(scraped_df[col].median())

# One-hot encode categorical columns (Airline & Class) for scraped data
scraped_df = pd.get_dummies(scraped_df, columns=['Airline', 'Class'], drop_first=True)

# Define test features; drop columns not used in modeling
drop_test_cols = ['Price', 'Date of Booking', 'Date of Journey', 'Duration', 'Airline-Class', 'Departure Time', 'Arrival Time']
X_test_full = scraped_df.drop(columns=drop_test_cols)
X_test_full = X_test_full.apply(pd.to_numeric, errors='coerce')

# Convert target price from scraped data using log1p after cleaning (remove commas)
y_test = np.log1p(scraped_df['Price'].str.replace(',', '').astype(float))

# Impute missing values in test data
imputer_test = KNNImputer(n_neighbors=5)
X_test_full_imputed = pd.DataFrame(imputer_test.fit_transform(X_test_full), columns=X_test_full.columns)
print("NaNs in test data after imputation:", X_test_full_imputed.isna().sum().sum())
print("X_test_full shape:", X_test_full_imputed.shape)

# ====================================================
# Align Features Between Train and Test
# ====================================================
common_cols = list(set(X_train_full_imputed.columns) & set(X_test_full_imputed.columns))
print("Common columns:", common_cols)
X_train = X_train_full_imputed[common_cols].copy()
X_test = X_test_full_imputed[common_cols].copy()
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    raise ValueError("Training or testing data is empty after preprocessing.")

# ====================================================
# Advanced Feature Engineering
# ====================================================
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

# ====================================================
# Scale Numeric Features using QuantileTransformer
# ====================================================
numeric_cols = ['Total_stops', 'Duration_in_hours', 'Departure_hour', 'Days_left', 
                'Journey_day_of_week', 'Date_of_journey_ordinal', 'Stops_Duration', 'Hour_DaysLeft']
numeric_cols = [col for col in numeric_cols if col in X_train.columns]
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ====================================================
# Split Training Data for Validation
# ====================================================
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ====================================================
# Build a Stacking Ensemble Model
# ====================================================
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

# ====================================================
# Train the Ensemble Model with Progress Reporting
# ====================================================
print("\nStarting training of stacking ensemble model. Please wait...")
done_flag = False
def progress_bar():
    from tqdm import tqdm
    with tqdm(desc="Training Progress (sec elapsed)", unit="sec") as pbar:
        while not done_flag:
            time.sleep(1)
            pbar.update(1)

# Start progress bar thread
progress_thread = threading.Thread(target=progress_bar)
progress_thread.start()

start_time = time.time()
stack_model.fit(X_train_split, y_train_split)
done_flag = True  # Signal progress bar to stop
progress_thread.join()
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Validate on validation set
y_val_pred = stack_model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation R² (log scale): {val_r2:.4f}")

# Test on test set
y_test_pred = stack_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test R² (log scale): {test_r2:.4f}")

# ====================================================
# Save the Final Model
# ====================================================
joblib.dump(stack_model, 'flight.pkl')
print("Enhanced model saved as 'flight.pkl'")
