import os

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolate_model_dir(tmp_path, monkeypatch):
    """Impede que preprocess_data sobrescreva os artefatos reais em models/."""
    from src import data_preprocessing

    fake = str(tmp_path / "models")
    os.makedirs(fake, exist_ok=True)
    monkeypatch.setattr(data_preprocessing, "MODEL_DIR", fake)


@pytest.fixture
def raw_df():
    rows = [
        {
            "customerID": "0001-AAAA", "gender": "Female", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No",
            "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
            "TotalCharges": "29.85", "Churn": "No",
        },
        {
            "customerID": "0002-BBBB", "gender": "Male", "SeniorCitizen": 1,
            "Partner": "No", "Dependents": "No", "tenure": 34, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes",
            "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "MonthlyCharges": 56.95,
            "TotalCharges": "1889.5", "Churn": "Yes",
        },
        {
            "customerID": "0003-CCCC", "gender": "Male", "SeniorCitizen": 0,
            "Partner": "No", "Dependents": "Yes", "tenure": 2, "PhoneService": "Yes",
            "MultipleLines": "Yes", "InternetService": "No",
            "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service", "TechSupport": "No internet service",
            "StreamingTV": "No internet service", "StreamingMovies": "No internet service",
            "Contract": "One year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 20.05,
            "TotalCharges": " ", "Churn": "No",
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_payload():
    """Payload Mockado"""
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    }
