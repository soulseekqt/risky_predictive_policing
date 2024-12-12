import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import logging
import sys
import math

# #Example query to run on terminal:
# curl -X POST http://localhost:8000/predict \
#      -H "Content-Type: application/json" \
#      -d '{
#          "ward": "6",
#          "time_category": "Late Evening",
#          "date": "2024-01-15",
#          "weekend": "yes",
#          "latitude": 41.8781,
#          "longitude": -87.6298
#      }'

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

# Predefined mappings for user-friendly inputs
TIME_CATEGORIES = {
    "Late Evening": 0,
    "Early Morning": 1,
    "Late Morning": 2,
    "Early Noon": 3,
    "Late Noon": 4,
    "Early Evening": 5
}

WEEKEND_MAPPING = {
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0
}

def encode_time_category(time_category):
    """Convert time category to numeric encoding"""
    return TIME_CATEGORIES.get(time_category.title())

def encode_weekend(weekend):
    """Convert weekend input to binary"""
    return WEEKEND_MAPPING.get(weekend.lower())

def encode_month(date):
    """
    Convert date to month sine and cosine encoding
    Assumes date is in 'YYYY-MM-DD' format
    """
    import datetime

    # Parse the date
    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")

    # Calculate month angle (2π represents a full year)
    month_angle = (date_obj.month - 1) * (2 * math.pi / 12)

    # Calculate sine and cosine
    month_sin = math.sin(month_angle)
    month_cos = math.cos(month_angle)

    return month_sin, month_cos

# Input model with user-friendly fields
class UserFriendlyInput(BaseModel):
    ward: str = Field(..., description="Ward identifier")
    time_category: str = Field(..., description="Time of day")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    weekend: str = Field(..., description="Is it a weekend?")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")

# Initialize FastAPI app
app = FastAPI(title="User-Friendly Offense Prediction Encoder")

# Load the pre-trained machine learning model
try:
    with open('model.pkl', 'rb') as model_file:
        ml_model = pickle.load(model_file)
    logger.info("Model successfully loaded")
except FileNotFoundError:
    logger.error("model.pkl not found. Please ensure the model file is in the correct directory.")
    ml_model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    ml_model = None

# Prediction endpoint
@app.post("/predict")
async def predict_offense(input_data: UserFriendlyInput):
    if ml_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Validate and encode inputs
        time_category = encode_time_category(input_data.time_category)
        if time_category is None:
            raise ValueError(f"Invalid time category. Must be one of {list(TIME_CATEGORIES.keys())}")

        weekend = encode_weekend(input_data.weekend)
        if weekend is None:
            raise ValueError("Weekend must be 'yes' or 'no'")

        month_sin, month_cos = encode_month(input_data.date)

        # Prepare input features
        input_features = [
            str(input_data.ward),
            time_category,
            month_sin,
            month_cos,
            weekend,
            input_data.latitude,
            input_data.longitude
        ]

        logger.info(f"Processed input features: {input_features}")

        # Convert input to numpy array and reshape
        input_array = np.array(input_features).reshape(1, -1)

        # Make prediction
        prediction = ml_model.predict(input_array)[0]

        logger.info(f"Prediction successful: {prediction}")
        return {
            "offense_prediction": float(prediction),
            "input_details": {
                "ward": input_data.ward,
                "time_category": input_data.time_category,
                "date": input_data.date,
                "weekend": input_data.weekend,
                "latitude": input_data.latitude,
                "longitude": input_data.longitude
            }
        }

    except ValueError as e:
        logger.error(f"Input conversion error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Endpoints for reference
@app.get("/time-categories")
async def get_time_categories():
    return {
        "categories": list(TIME_CATEGORIES.keys()),
        "description": "Predefined time categories for input"
    }

@app.get("/weekend-options")
async def get_weekend_options():
    return {
        "options": ["yes", "no", "y", "n"],
        "description": "Acceptable weekend input values"
    }

# Main block to run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
