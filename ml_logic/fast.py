import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import logging
import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, mean_squared_error
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prediction_log.txt')
    ]
)
logger = logging.getLogger(__name__)

# Input model with the updated fields
class UserInput(BaseModel):
    ward: int = Field(..., description="Ward identifier")
    date_of_occurrence: str = Field(..., description="Date and time of occurrence in 'MM/DD/YYYY HH:MM' format")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")

# Initialize FastAPI app
app = FastAPI(title="Offense Prediction Model")

# Load the LabelEncoder and models
label_encoder_path = os.path.join(os.path.dirname(__file__),  'label_encoder.pkl')
regressor_path = os.path.join(os.path.dirname(__file__),  'regressor.pkl')
classifier_path = os.path.join(os.path.dirname(__file__), 'classifier.pkl')
label_encoder_ward_path = os.path.join(os.path.dirname(__file__), 'label_encoder_ward.pkl')

try:
    with open(label_encoder_path, 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    with open(classifier_path, 'rb') as model_file:
        classifier = pickle.load(model_file)

    with open(label_encoder_ward_path, 'rb') as model_file:
        label_encoder_ward = pickle.load(model_file)

    with open(regressor_path, 'rb') as model_file:
        regressor = pickle.load(model_file)

    logger.info("LabelEncoder and Models successfully loaded")

except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    classifier = label_encoder_ward = None
except Exception as e:
    logger.error(f"Error loading files: {e}")
    classifier = label_encoder_ward = None


# Haversine formula to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return r * c


# Function to calculate nearest distance to a police district
def calculate_nearest_distance(lat, lon):
    # Example coordinates for a police district
    police_districts = {
        "District 1 (Central)": (41.8345, -87.6216),
        "District 2 (Wentworth)": (41.8027, -87.6185),
        "District 3 (Grand Crossing)": (41.752, -87.6001),
        "District 4 (South Chicago)": (41.7531, -87.5573),
        "District 5 (Calumet)": (41.7365, -87.607),
        "District 6 (Gresham)": (41.7445, -87.6616),
        "District 7 (Englewood)": (41.7843, -87.6745),
        "District 8 (Chicago Lawn)": (41.7794, -87.6864),
        "District 9 (Deering)": (41.827, -87.667),
        "District 10 (Ogden)": (41.8782, -87.7119),
        "District 11 (Harrison)": (41.8589, -87.7107),
        "District 12 (Near West Side)": (41.8844, -87.6456),
        "District 13 (Jefferson Park)": (41.8914, -87.7377),
        "District 14 (Shakespeare)": (41.8986, -87.6743),
        "District 15 (Austin)": (41.8763, -87.7724),
        "District 16 (Albion Park)": (41.9762, -87.7243),
        "District 17 (Woodlawn)": (41.7874, -87.592),
        "District 18 (Pullman)": (41.7317, -87.6079),
        "District 19 (Southwest)": (41.794, -87.74),
        "District 20 (North Lawndale)": (41.8655, -87.7111),
        "District 21 (Near North Side)": (41.9264, -87.6482),
        "District 22 (Lincoln Park)": (41.9252, -87.6549),
    }
    # Calculate the distance to all police stations and select the nearest one
    distances = {district: haversine(lat, lon, *coords) for district, coords in police_districts.items()}
    nearest_district = min(distances, key=distances.get)
    return distances[nearest_district]

# Preprocessing function for input data
def preprocess_input(input_data):
    # Convert 'DATE OF OCCURRENCE' to datetime format
    input_data['DATE'] = pd.to_datetime(input_data['DATE OF OCCURRENCE'])
    input_data['HOUR'] = input_data['DATE'].dt.floor('h')
    # Calculate distance to the nearest police station
    input_data['DISTANCE_TO_POLICE'] = [calculate_nearest_distance(lat, lon) for lat, lon in zip(input_data['LATITUDE'], input_data['LONGITUDE'])]
    input_data['DISTANCE_TO_POLICE'] = input_data.groupby('HOUR')['DISTANCE_TO_POLICE'].transform('mean')
    # Group by 'HOUR' and count occurrences (crime count)
    # input_data['CRIME_COUNT'] = input_data.groupby('HOUR')['LATITUDE'].transform('size')
    crime_count = input_data.groupby('HOUR').size().reset_index(name='CRIME_COUNT')
    input_data = pd.merge(input_data, crime_count, on='HOUR', how='left')
    # Add WARD feature (Label Encoding)
    #label_encoder_ward = LabelEncoder()
    #input_data['WARD_ENCODED'] = label_encoder_ward.fit_transform(input_data['WARD'])
    input_data['WARD'] = input_data.groupby('HOUR')['WARD'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
    # Cyclical features for time
    input_data['TIME_SIN'] = np.sin(2 * np.pi * input_data['HOUR'].dt.hour / 24)
    input_data['TIME_COS'] = np.cos(2 * np.pi * input_data['HOUR'].dt.hour / 24)
    input_data['MONTH_SIN'] = np.sin(2 * np.pi * input_data['HOUR'].dt.month / 12)
    input_data['MONTH_COS'] = np.cos(2 * np.pi * input_data['HOUR'].dt.month / 12)
    # Day of the week (0=Monday, 6=Sunday)
    input_data['DAY_OF_WEEK'] = input_data['HOUR'].dt.dayofweek
    # Rolling 7-day average for crime count
    input_data['ROLLING_7DAY'] = input_data['CRIME_COUNT'].rolling(window=7, min_periods=1).mean().fillna(0)
    # Lag features for crime counts (1 and 24-hour lags)
    input_data['CRIME_COUNT_LAG1'] = input_data['CRIME_COUNT'].shift(1).fillna(0)
    input_data['CRIME_COUNT_LAG24'] = input_data['CRIME_COUNT'].shift(24).fillna(0)
    result_columns = ['TIME_SIN','TIME_COS','CRIME_COUNT_LAG1','CRIME_COUNT_LAG24','ROLLING_7DAY','DISTANCE_TO_POLICE','WARD','DAY_OF_WEEK','MONTH_SIN','MONTH_COS']
    # Return the processed data
    return input_data[result_columns]



# Prediction endpoint
@app.post("/predict")
async def predict_offense(input_data: UserInput):
    if classifier is None or label_encoder_ward is None:
        raise HTTPException(status_code=500, detail="Model or LabelEncoder not loaded")

    try:
        # Convert the input data to a DataFrame
        input_df = pd.DataFrame({
            'WARD': [input_data.ward],
            'DATE OF OCCURRENCE': [input_data.date_of_occurrence],
            'LATITUDE': [input_data.latitude],
            'LONGITUDE': [input_data.longitude]
        })

        # Preprocess the input data
        preprocessed_input = preprocess_input(input_df)

        # Extract features for prediction
        X_input = preprocessed_input[['TIME_SIN', 'TIME_COS', 'CRIME_COUNT_LAG1', 'CRIME_COUNT_LAG24', 'ROLLING_7DAY',
                                      'DISTANCE_TO_POLICE', 'WARD', 'DAY_OF_WEEK', 'MONTH_SIN', 'MONTH_COS']]

        # Generate predictions
        probas = classifier.predict_proba(X_input)[0]

        crime_types_proba = sorted(zip(label_encoder.inverse_transform(range(len(probas))), map(float, probas)), key=lambda x: x[1], reverse=True)

        predicted_crime_counts = regressor.predict(X_input)[0]
        crime_types_count = [(crime_type, round(predicted_crime_counts * probability)) for crime_type, probability in crime_types_proba]

        crime_types_count_sorted = sorted(crime_types_count, key=lambda x: x[1], reverse=True)

        probabilities_dict = {crime_type: round(probability, 4) for crime_type, probability in crime_types_proba}
        counts_dict = {crime_type: count if count > 0 else 0 for crime_type, count in crime_types_count_sorted}

        return {"crime_types_probability": probabilities_dict, "crime_types_count": counts_dict}

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

# To run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
