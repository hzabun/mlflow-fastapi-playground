import logging
import os
from contextlib import asynccontextmanager
from typing import List

import mlflow.pyfunc
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelService:
    """Service class to manage ML model"""

    def __init__(self):
        self.model = None
        self.model_info = {}

    def load_model(self, mlflow_uri: str, model_name: str, model_version: str):
        """Load model from MLflow"""
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"Loading model {model_name}:{model_version} from {mlflow_uri}")

            self.model = mlflow.pyfunc.load_model(
                f"models:/{model_name}/{model_version}"
            )
            self.model_info = {
                "name": model_name,
                "version": model_version,
                "mlflow_uri": mlflow_uri,
            }
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def predict(self, df: pd.DataFrame):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.predict(df)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    model_service = ModelService()

    # Load model configuration from environment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    model_name = os.getenv("MODEL_NAME", "video_game_sales_model")
    model_version = os.getenv("MODEL_VERSION", "latest")

    # Load the model
    model_service.load_model(mlflow_uri, model_name, model_version)

    # Store in application state
    app.state.model_service = model_service

    yield

    # Shutdown cleanup
    logger.info("Application shutdown")


# Initialize FastAPI app
app = FastAPI(
    title="Video Game Sales Prediction API",
    description="API for predicting video game sales using MLflow models",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models
class GameFeatures(BaseModel):
    Platform: str
    Year: int
    Genre: str
    Publisher: str


class PredictRequest(BaseModel):
    data: List[GameFeatures]


# Dependency to get model service
def get_model_service() -> ModelService:
    """Dependency to get model service from app state"""
    return app.state.model_service


@app.get("/")
def health_check(model_service: ModelService = Depends(get_model_service)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded(),
        "model_info": model_service.model_info,
    }


@app.get("/model_info")
def model_info():
    """Return information about the expected input format"""
    return {
        "expected_features": ["Platform", "Year", "Genre", "Publisher"],
        "feature_types": {
            "Platform": "categorical (string)",
            "Year": "numerical (integer)",
            "Genre": "categorical (string)",
            "Publisher": "categorical (string)",
        },
        "example_input": {
            "Platform": "PS4",
            "Year": 2015,
            "Genre": "Action",
            "Publisher": "Sony",
        },
    }


@app.post("/predict")
def predict(
    request: PredictRequest, model_service: ModelService = Depends(get_model_service)
):
    """Batch prediction endpoint"""
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic models to dictionaries
        input_data = [game.model_dump() for game in request.data]

        # Create DataFrame
        df = pd.DataFrame(input_data)

        # Make predictions
        predictions = model_service.predict(df)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict_single")
def predict_single(
    game: GameFeatures, model_service: ModelService = Depends(get_model_service)
):
    """Single prediction endpoint"""
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create DataFrame with single row
        df = pd.DataFrame([game.model_dump()])

        # Make prediction
        prediction = model_service.predict(df)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
