from typing import List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the pipeline model from MLflow
model = mlflow.pyfunc.load_model("models:/vgs_ridge_pipeline_model/latest")

app = FastAPI()


class GameFeatures(BaseModel):
    Platform: str
    Year: int
    Genre: str
    Publisher: str


class PredictRequest(BaseModel):
    data: List[GameFeatures]


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Convert Pydantic models to dictionaries
        input_data = [game.model_dump() for game in request.data]

        df = pd.DataFrame(input_data)
        predictions = model.predict(df)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict_single")
def predict_single(game: GameFeatures):
    """Convenience endpoint for single predictions"""
    try:

        # Convert Pydantic model to single row dataframe
        df = pd.DataFrame([game.model_dump()])

        prediction = model.predict(df)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.get("/model_info")
def model_info():
    """Return information about the expected input format"""
    return {
        "expected_features": ["Platform", "Year", "Genre", "Publisher"],
        "feature_types": {
            "Platform": "categorical (string)",
            "Year": "numerical (int)",
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
