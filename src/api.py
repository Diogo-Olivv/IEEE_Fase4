import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.config import MODEL_DIR

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]
DUMMY_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


class ChurnInput(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: str = Field(..., examples=["No"])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[1])
    PhoneService: str = Field(..., examples=["No"])
    MultipleLines: str = Field(..., examples=["No phone service"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[29.85])
    TotalCharges: float = Field(..., examples=[29.85])


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    label: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        app.state.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        app.state.columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
        app.state.model_loaded = True
    except Exception as exc:
        app.state.model = None
        app.state.scaler = None
        app.state.columns = None
        app.state.model_loaded = False
        print(f"Failed to load model artifacts: {exc}")
    yield


app = FastAPI(
    title="Telco Churn Predictor",
    description="Inference API for customer churn prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


def transform(payload: ChurnInput, scaler, columns) -> pd.DataFrame:
    df = pd.DataFrame([payload.model_dump()])

    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    df = pd.get_dummies(df, columns=DUMMY_COLS)
    df = df.reindex(columns=columns, fill_value=0)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    return df


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": bool(getattr(app.state, "model_loaded", False))}


@app.post("/predict", response_model=PredictionOutput)
async def predict(payload: ChurnInput):
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    try:
        X = transform(payload, app.state.scaler, app.state.columns)
        prediction = int(app.state.model.predict(X)[0])

        if hasattr(app.state.model, "predict_proba"):
            probability = float(app.state.model.predict_proba(X)[0, 1])
        else:
            probability = float(prediction)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return PredictionOutput(
        prediction=prediction,
        probability=probability,
        label="Churn" if prediction == 1 else "Stay",
    )
