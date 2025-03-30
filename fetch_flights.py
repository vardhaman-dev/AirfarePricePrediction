import os
import csv
import logging
import requests
from time import sleep
from dotenv import load_dotenv

# Setup logging for production-grade error tracking and debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("fetch_flights.log"),
        logging.StreamHandler()
    ]
)

# Load API key from .env file
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    logging.error("RapidAPI key not found in the .env file.")

# Define the API endpoint and headers
API_URL = "https://skyscanner89.p.rapidapi.com/flights/one-way/list?origin=NYCA&originId=27537542&destination=HNL&destinationId=95673827"
headers = {
    "x-rapidapi-host": "skyscanner89.p.rapidapi.com",
    "x-rapidapi-key": RAPIDAPI_KEY
}

def fetch_flight_data(url, headers, max_retries=3):
    """
    Fetch flight data from the given URL with provided headers.
    Implements retry logic to handle transient errors.
    """
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt+1}: Fetching data from {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            logging.info("Data fetched successfully")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            if attempt < max_retries - 1:
                logging.info("Retrying in 5 seconds...")
                sleep(5)
            else:
                logging.error("Max retries reached. Exiting fetch.")
    return None

def process_data(data):
    """
    Extracts flight information from the API response.
    Navigates into data -> flightQuotes -> results.
    """
    if data is None:
        logging.error("No data provided for processing.")
        return []
    
    results = data.get("data", {}).get("flightQuotes", {}).get("results", [])
    if not results:
        logging.warning("No flight results found in the response. Check the API response structure.")
    return results

def save_to_csv(flights, filename="flight_prices.csv"):
    """
    Saves the flight data to a CSV file.
    Extracts fields: FlightId, Price, Direct flag, and Departure Date.
    """
    if not flights:
        logging.error("No flight data to save.")
        return
    
    # Define CSV headers (adjust these based on the available data)
    fieldnames = ["FlightId", "Price", "Direct", "DepartureDate"]
    try:
        with open(filename, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for flight in flights:
                content = flight.get("content", {})
                outboundLeg = content.get("outboundLeg", {})
                writer.writerow({
                    "FlightId": flight.get("id"),
                    "Price": content.get("rawPrice"),  # You can also use "price" for the formatted string
                    "Direct": content.get("direct"),
                    "DepartureDate": outboundLeg.get("localDepartureDate")
                })
        logging.info(f"Data successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")

if __name__ == "__main__":
    # Fetch the flight data
    flight_data = fetch_flight_data(API_URL, headers)
    
    # Optional: Print the full API response for debugging purposes
    if flight_data:
        print("Full API Response:")
        print(flight_data)
    
    # Process the response to extract flight details
    flights = process_data(flight_data)
    
    # Save the extracted data to a CSV file
    save_to_csv(flights)
