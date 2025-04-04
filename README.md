# Flight Price Prediction Project Documentation

Welcome to the comprehensive documentation for the **Flight Price Prediction** project! This guide takes you on a journey from the project's inception to its final deployment, detailing every step with in-depth explanations, code breakdowns, and insights into the development process. Designed for a technical audience, this documentation covers setup instructions, code explanations, and the reasoning behind design choices, ensuring you can understand, replicate, or extend the project with ease.

---

## Introduction to the Project

The **Flight Price Prediction** project is a web-based application that leverages machine learning to predict flight fares based on user-provided inputs. Let’s start by understanding its purpose, goals, and key components.

### Purpose
The application aims to estimate flight prices using historical data and real-time inputs such as departure and arrival details, airline, source, destination, and number of stops. This tool empowers users to make informed travel decisions by providing accurate price predictions.

### Goals
- Deliver reliable flight price estimates to users.
- Create an intuitive web interface for seamless interaction.
- Demonstrate a full machine learning pipeline, from data preprocessing to model deployment.

### Key Components
- **Data Preprocessing**: Cleaning and transforming raw flight data for model training.
- **Model Training**: Building a machine learning model to predict prices.
- **Real-Time Data Fetching**: Integrating live flight data via an API.
- **Web Interface**: A Flask-based application for user interaction and predictions.

---

## Project Structure

Before diving into the code, let’s explore how the project is organized. Understanding the structure helps you navigate the codebase efficiently.

- **`app.py`**: The core Flask application that serves the web interface and handles predictions.
- **`data_preprocessing.py`**: A script to clean and preprocess the historical dataset.
- **`fetch_flights.py`**: A script to fetch real-time flight data from the Skyscanner API.
- **`templates/`**: Directory for HTML templates.
  - `index.html`: The homepage with project overview and metrics.
  - `predict.html`: The prediction page with a user input form and results.
- **`static/`**: Directory for static assets.
  - `style.css`: Stylesheet for the homepage.
  - `enhanced.css`: Stylesheet for the prediction page.
  - `logo.jpg`: Favicon for the web app.
- **`flight.pkl`**: The pre-trained machine learning model.
- **`processed_train_data.csv`**: The cleaned dataset ready for model training.
- **`fetch_flights.log`**: Log file for API requests and responses.
- **`archive/`**: Directory containing the raw dataset (`Data_Train.xlsx`).

This structure separates concerns—data processing, model logic, and presentation—making the project modular and maintainable.

---

## Setup Instructions

To embark on this journey, you’ll need to set up the project locally. Follow these steps to get started.

### Step 1: Clone the Repository
Clone the project from GitHub and navigate into the directory:
```bash
git clone https://github.com/yourusername/flight-price-prediction.git
cd flight-price-prediction
```

### Step 2: Install Dependencies
Install the required Python libraries using pip:
```bash
pip install flask pandas numpy scikit-learn matplotlib requests python-dotenv
```
These libraries power the web framework (Flask), data handling (pandas, NumPy), machine learning (scikit-learn), visualizations (matplotlib), API requests (requests), and environment variables (python-dotenv).

### Step 3: Prepare the Dataset
- Place the raw dataset `Data_Train.xlsx` in the `archive/` folder.
- Run the preprocessing script to generate the cleaned dataset:
  ```bash
  python data_preprocessing.py
  ```
  This creates `processed_train_data.csv`, which we’ll use later.

### Step 4: Set Up API Key (Optional)
To fetch real-time data, sign up for a RapidAPI key for the Skyscanner API. Create a `.env` file in the root directory with:
```
RAPIDAPI_KEY=your_api_key_here
```

### Step 5: Run the Application
Launch the Flask app:
```bash
python app.py
```
Open your browser and visit `http://127.0.0.1:5000/` to see the app in action.

---

## Data Preprocessing (`data_preprocessing.py`)

The journey begins with preparing the raw data. The `data_preprocessing.py` script transforms the messy `Data_Train.xlsx` into a clean, model-ready dataset. Let’s break it down.

### Loading the Dataset
We start by reading the Excel file using pandas:
```python
import pandas as pd
import numpy as np

df = pd.read_excel('archive/Data_Train.xlsx')
```

### Handling Missing Values
Real-world data is rarely perfect. We handle missing values to ensure robustness:
- **Route and Total_Stops**: Fill with the most frequent value (mode).
- **Price**: Convert to float and fill with the median.
```python
df['Route'].fillna(df['Route'].mode()[0], inplace=True)
df['Total_Stops'].fillna(df['Total_Stops'].mode()[0], inplace=True)
df['Price'] = df['Price'].astype(float).fillna(df['Price'].median())
```

### Feature Engineering
We extract meaningful features from raw columns:
- **Duration**: Convert strings like "2h 50m" to total minutes.
```python
def convert_duration(duration):
    if isinstance(duration, str):
        parts = duration.split('h')
        hours = int(parts[0].strip()) if parts[0].strip().isdigit() else 0
        minutes = 0 if len(parts) <= 1 else int(parts[1].replace('m', '').strip() or 0)
        return hours * 60 + minutes
    return np.nan

df['Duration_Minutes'] = df['Duration'].apply(convert_duration)
```
- **Date_of_Journey**: Extract day, month, and weekday.
```python
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
df['Journey_Day'] = df['Date_of_Journey'].dt.day
df['Journey_Month'] = df['Date_of_Journey'].dt.month
df['Journey_Weekday'] = df['Date_of_Journey'].dt.weekday
```
- **Time Features**: Compute duration from departure and arrival times (assumed logic).

### Encoding Categorical Variables
Machine learning models require numerical inputs. We use one-hot encoding for categorical columns:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop='first')
categorical_cols = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']
encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
```

### Scaling Features
To ensure features are on the same scale, we standardize numerical columns:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = ['Duration_Minutes', 'Journey_Day', 'Journey_Month', 'Journey_Weekday']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

### Saving the Processed Data
Finally, we save the cleaned dataset:
```python
df.to_csv('processed_train_data.csv', index=False)
```

This script lays the foundation for model training by turning raw data into a structured, numerical format.

---

## Model Training (Assumed Workflow)

The pre-trained model is stored in `flight.pkl`, but let’s assume how it was created. This step bridges data preprocessing to deployment.

### Dataset and Features
- **Input**: `processed_train_data.csv`.
- **Features**: Numerical columns (e.g., `Duration_Minutes`) and one-hot encoded categorical variables.
- **Target**: `Price`.

### Model Choice
A regression model like Random Forest Regressor is suitable for predicting continuous values like price:
```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle

df = pd.read_csv('processed_train_data.csv')
X = df.drop('Price', axis=1)
y = df['Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open('flight.pkl', 'wb') as f:
    pickle.dump(model, f)
```
- **Why Random Forest?** It handles non-linear relationships and interactions between features well, which is common in flight price data.

---

## Flask Application (`app.py`)

Now, we bring the model to life with a Flask web application. The `app.py` file ties everything together.

### Imports and Setup
```python
from flask import Flask, render_template, request
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)
model = pickle.load(open('flight.pkl', 'rb'))
```

### Homepage Route
The root route displays the project overview:
```python
@app.route('/')
def home():
    return render_template('index.html')
```

### Prediction Route
The `/predict` route handles both form display (GET) and prediction (POST):
```python
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract form inputs
        date_dep = request.form['departure_date']
        dep_time = request.form['departure_time']
        arr_time = request.form['arrival_time']
        airline = request.form['airline']
        source = request.form['source']
        destination = request.form['destination']
        total_stops = int(request.form['total_stops'])

        # Parse date and time
        dep_datetime = pd.to_datetime(f"{date_dep} {dep_time}")
        arr_datetime = pd.to_datetime(f"{date_dep} {arr_time}")
        journey_day = dep_datetime.day
        journey_month = dep_datetime.month
        dep_hour = dep_datetime.hour
        arr_hour = arr_datetime.hour
        duration_minutes = (arr_datetime - dep_datetime).total_seconds() / 60

        # Prepare features (simplified for brevity)
        feature_dict = {
            'Total_Stops': total_stops,
            'Journey_Day': journey_day,
            'Journey_Month': journey_month,
            'Duration_Minutes': duration_minutes,
            f'Airline_{airline}': 1,
            f'Source_{source}': 1,
            f'Destination_{destination}': 1
        }
        feature_df = pd.DataFrame([feature_dict]).reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict
        prediction = model.predict(feature_df)[0]
        return render_template('predict.html', prediction_text=f"Your Flight price is Rs. {round(prediction, 2)}")
    return render_template('predict.html')
```

### Running the App
```python
if __name__ == "__main__":
    app.run(debug=True)
```

This code processes user inputs, aligns them with the model’s expected features, and delivers predictions via the web interface.

---

## Real-Time Data Fetching (`fetch_flights.py`)

To enhance predictions, we fetch live data using the Skyscanner API.

### Setup and Fetching
```python
import requests
import logging
from time import sleep
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('RAPIDAPI_KEY')
logging.basicConfig(filename='fetch_flights.log', level=logging.INFO)

def fetch_flight_data(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            logging.info(f"Successfully fetched data on attempt {attempt + 1}")
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}")
            sleep(5)
    return None

url = "https://skyscanner-api.p.rapidapi.com/v1/flights/search"
headers = {
    "X-RapidAPI-Key": api_key,
    "X-RapidAPI-Host": "skyscanner-api.p.rapidapi.com"
}
data = fetch_flight_data(url, headers)
if data:
    pd.DataFrame(data['flights']).to_csv('flight_prices.csv', index=False)
```

This script ensures robust data retrieval with retries and logging, saving results for analysis or model retraining.

---

## Web Interface (`index.html` and `predict.html`)

The user interacts with the project through two HTML templates.

### `index.html`
- **Purpose**: Introduces the project and displays metrics.
- **Key Elements**:
  ```html
  <h1>Flight Price Prediction</h1>
  <p>A machine learning-powered tool to estimate flight fares.</p>
  <table>
      <tr><th>Model</th><th>Accuracy</th><th>R² Score</th></tr>
      <tr><td>Random Forest</td><td>85%</td><td>0.82</td></tr>
  </table>
  ```
- **Styling**: `style.css` adds a modern layout with particle animations via JavaScript.

### `predict.html`
- **Purpose**: Collects inputs and shows predictions.
- **Form Example**:
  ```html
  <form action="{{ url_for('predict') }}" method="post" id="prediction-form">
      <input type="date" name="departure_date" required>
      <input type="time" name="departure_time" required>
      <input type="time" name="arrival_time" required>
      <select name="airline">
          <option value="Vistara">Vistara</option>
          <!-- More options -->
      </select>
      <input type="number" name="total_stops" min="0" required>
      <input type="submit" value="Predict Price">
  </form>
  <h4>{{ prediction_text }}</h4>
  ```
- **Validation**: JavaScript ensures logical inputs (e.g., arrival after departure).
- **Styling**: `enhanced.css` provides a responsive, animated design.

---

## Challenges and Solutions

The journey wasn’t without obstacles. Here’s how we overcame them:

1. **Performance Issues in Model Training**
   - **Challenge**: Slow training on large datasets.
   - **Solution**: Use cloud platforms or downsample data.

2. **Runtime Warnings in Preprocessing**
   - **Challenge**: NumPy warnings from missing values.
   - **Solution**: Robust imputation and error handling.

3. **Form Data Consistency**
   - **Challenge**: Mismatched categorical names.
   - **Solution**: Standardized naming conventions.

4. **Testing with Real Data**
   - **Challenge**: Validating predictions.
   - **Solution**: Integrated `fetch_flights.py` for real-time comparison.

---

## Conclusion

The **Flight Price Prediction** project is a testament to the power of machine learning and web development working in harmony. From preprocessing raw data to deploying a user-friendly app, this journey showcases a complete pipeline. Future enhancements could include real-time data integration, user accounts, or price trend visualizations.

Thank you for exploring this documentation! You now have the tools to understand, run, and extend this project. Happy coding!
