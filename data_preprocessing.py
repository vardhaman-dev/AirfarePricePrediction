import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load the Historical Dataset
# ------------------------------
df = pd.read_excel('archive/Data_Train.xlsx')
print("Initial dataset info:")
print(df.info())
print(df.head())

# Standardize column names (remove extra spaces)
df.columns = [col.strip() for col in df.columns]

# ------------------------------
# 2. Handle Missing Values & Clean Data
# ------------------------------

# Fill missing values in 'Route' and 'Total_Stops' with mode
df['Route'] = df['Route'].fillna(df['Route'].mode().iloc[0])
df['Total_Stops'] = df['Total_Stops'].fillna(df['Total_Stops'].mode().iloc[0])

# Clean the 'Price' column: remove '$' and ',' then convert to float
df['Price'] = df['Price'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
df['Price'] = df['Price'].fillna(df['Price'].median())

# ------------------------------
# 3. Process the Duration Column
# ------------------------------
# Convert 'Duration' (e.g., "2h 50m") into total minutes.
def convert_duration(duration):
    if isinstance(duration, str):
        try:
            parts = duration.split('h')
            hours = int(parts[0].strip()) if parts[0].strip().isdigit() else 0
            minutes = 0
            if len(parts) > 1:
                min_part = parts[1].replace('m', '').strip()
                minutes = int(min_part) if min_part.isdigit() else 0
            return hours * 60 + minutes
        except Exception:
            return np.nan
    return np.nan

df['Duration_Minutes'] = df['Duration'].apply(convert_duration)
df['Duration_Minutes'] = df['Duration_Minutes'].fillna(df['Duration_Minutes'].median())

# ------------------------------
# 4. Extract Date and Time Features
# ------------------------------
# Convert 'Date_of_Journey' to datetime (format assumed as '%d/%m/%Y')
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y', errors='coerce')
df['Journey_Day'] = df['Date_of_Journey'].dt.day
df['Journey_Month'] = df['Date_of_Journey'].dt.month
df['Journey_Weekday'] = df['Date_of_Journey'].dt.dayofweek
df['Is_Weekend'] = df['Journey_Weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Convert 'Dep_Time' and 'Arrival_Time' to datetime
# Assuming these are in 'HH:MM' format (adjust if needed)
df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce')
df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M', errors='coerce')

df['Dep_Hour'] = df['Dep_Time'].dt.hour
df['Dep_Minute'] = df['Dep_Time'].dt.minute
df['Arrival_Hour'] = df['Arrival_Time'].dt.hour
df['Arrival_Minute'] = df['Arrival_Time'].dt.minute

# ------------------------------
# 5. Innovative Duration Recalculation
# ------------------------------
# Compute duration from Dep_Time and Arrival_Time (in minutes)
def compute_duration(dep, arr):
    if pd.isnull(dep) or pd.isnull(arr):
        return np.nan
    diff = (arr - dep).total_seconds() / 60.0
    if diff < 0:  # Handle flights crossing midnight
        diff += 24 * 60
    return diff

df['Computed_Duration'] = df.apply(lambda row: compute_duration(row['Dep_Time'], row['Arrival_Time']), axis=1)

# Combine the original and computed durations (average if both available)
df['Final_Duration'] = df[['Duration_Minutes', 'Computed_Duration']].mean(axis=1)
df['Final_Duration'] = df['Final_Duration'].fillna(df['Final_Duration'].median())

# ------------------------------
# 6. Categorical Feature Encoding
# ------------------------------
# List of categorical columns to encode
categorical_cols = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ------------------------------
# 7. Feature Scaling (Standardization)
# ------------------------------
# Choose numerical columns to scale
num_cols = ['Price', 'Final_Duration', 'Dep_Hour', 'Dep_Minute', 'Arrival_Hour', 'Arrival_Minute',
            'Journey_Day', 'Journey_Month', 'Journey_Weekday']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ------------------------------
# 8. Cleanup: Drop Redundant Columns
# ------------------------------
# Drop original Duration, raw date/time columns that we no longer need
df.drop(columns=['Duration', 'Duration_Minutes', 'Computed_Duration', 'Dep_Time', 'Arrival_Time', 'Date_of_Journey'], inplace=True)

# ------------------------------
# 9. Final Check and Save Processed Data
# ------------------------------
print("Final processed data columns:")
print(df.columns)
print(df.head())

# Save the processed dataset to a CSV file
df.to_csv('processed_train_data.csv', index=False)
print("Data preprocessing complete. Processed data saved as 'processed_train_data.csv'.")

# ------------------------------
# 10. (Optional) Visualize Key Feature Distributions
# ------------------------------
plt.figure(figsize=(10, 6))
plt.hist(df['Price'], bins=50, color='skyblue', edgecolor='black')
plt.title('Normalized Price Distribution')
plt.xlabel('Price (Normalized)')
plt.ylabel('Frequency')
plt.show()
