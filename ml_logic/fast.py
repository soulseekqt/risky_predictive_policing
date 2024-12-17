import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import logging
import sys
import math
import os
import pandas as pd

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

label_encoder_path = os.path.join(os.path.dirname(__file__), 'ml_logic', 'label_encoder.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'ml_logic', 'model.pkl')

# Load the LabelEncoder and model
try:
    label_encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

    with open(label_encoder_path, 'rb') as le_file:
        label_encoder_ward = pickle.load(le_file)

    with open(model_path, 'rb') as model_file:
        classifier = pickle.load(model_file)

    logger.info("LabelEncoder and Random Forest model successfully loaded")

except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    classifier = label_encoder_ward = None
except Exception as e:
    logger.error(f"Error loading files: {e}")
    classifier = label_encoder_ward = None

# Preprocessing function for input data
def preprocess_input(input_data):
    # Convert 'DATE OF OCCURRENCE' to datetime format
    input_data['DATE'] = pd.to_datetime(input_data['DATE OF OCCURRENCE'])
    input_data['HOUR'] = input_data['DATE'].dt.floor('h')

    # Generate cyclical features (sine and cosine of time and month)
    input_data['TIME_SIN'] = np.sin(2 * np.pi * input_data['HOUR'].dt.hour / 24)
    input_data['TIME_COS'] = np.cos(2 * np.pi * input_data['HOUR'].dt.hour / 24)
    input_data['MONTH_SIN'] = np.sin(2 * np.pi * input_data['HOUR'].dt.month / 12)
    input_data['MONTH_COS'] = np.cos(2 * np.pi * input_data['HOUR'].dt.month / 12)

    # Add 'DAY_OF_WEEK' feature
    input_data['DAY_OF_WEEK'] = input_data['HOUR'].dt.dayofweek

    # Add placeholder features (these can be replaced with actual calculations)
    input_data['CRIME_COUNT_LAG1'] = 0
    input_data['CRIME_COUNT_LAG24'] = 0
    input_data['ROLLING_7DAY'] = 0
    input_data['DISTANCE_TO_POLICE'] = 0

    return input_data

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

        # Extract the features for prediction
        X_input = preprocessed_input[['TIME_SIN', 'TIME_COS', 'CRIME_COUNT_LAG1', 'CRIME_COUNT_LAG24', 'ROLLING_7DAY',
                                      'DISTANCE_TO_POLICE', 'WARD', 'DAY_OF_WEEK', 'MONTH_SIN', 'MONTH_COS']]

        # Generate predictions
        probas = classifier.predict_proba(X_input)

        # Get the top 5 predictions
        top_5_idx = np.argsort(probas[0])[-5:][::-1]  # Sort probabilities in descending order
        top_5_classes = classifier.classes_[top_5_idx]
        top_5_probabilities = probas[0][top_5_idx]

        # Map the predicted labels to offense names using the LabelEncoder
        top_5_crimes = {
            label_encoder_ward.inverse_transform([top_5_classes[i]])[0]: top_5_probabilities[i]
            for i in range(5)
        }

        logger.info(f"Top 5 predicted crimes: {top_5_crimes}")
        return {"Top 5 Crimes": top_5_crimes}

    except ValueError as e:
        logger.error(f"Input conversion error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Main block to run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
