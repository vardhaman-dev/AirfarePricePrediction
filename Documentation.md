# Airfare Price Prediction Project Documentation

## Project Goal
Develop a system that predicts flight ticket prices using historical flight data. This involves data collection, preprocessing, model training, API development, and creating a dynamic, animated frontend.

## Requirements
- **Data Collection:** Use Skyscanner API or web scraping to fetch flight price data.
- **Data Preprocessing & Feature Engineering:** Clean data, handle missing values, and convert categorical features.
- **Model Training:** Train regression models (Random Forest, XGBoost, Linear Regression) and optimize them.
- **API Development:** Create a Flask API for flight price predictions.
- **Frontend Development:** Build a user-friendly UI using Streamlit integrated with Three.js for animations and gradients.
- **Deployment:** Deploy the solution on cloud platforms (e.g., Heroku, Render, or Streamlit Cloud).

## Technologies
- **Programming Languages & Libraries:** Python, Pandas, scikit-learn, XGBoost, Flask, Streamlit, Three.js.
- **Version Control:** Git and GitHub.

## Project Architecture (Overview)
1. **Data Collection:** Fetch and store flight data.
2. **Data Preprocessing:** Clean and transform data.
3. **Model Training:** Train and optimize machine learning models.
4. **API:** Create a REST API with Flask for serving predictions.
5. **Frontend:** Develop a visually appealing web interface with animations using Streamlit and Three.js.
6. **Deployment:** Deploy the API and the frontend to a cloud service.

## Design Decisions & Rationale
- **Why Multiple Models?**  
  Experimenting with multiple regression models (e.g., Random Forest, XGBoost) can help us compare performance and select the best model.
- **Why Three.js for the Frontend?**  
  Three.js will allow us to add innovative animations and dynamic effects, making the website both professional and engaging.
- **Documentation & GitHub:**  
  Detailed documentation and incremental commits in GitHub ensure that every step is well-tracked and reproducible.
