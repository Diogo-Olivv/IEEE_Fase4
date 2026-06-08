"""Testes do pré-processamento (src/data_preprocessing.py)."""
import numpy as np

from src.data_preprocessing import preprocess_data


def test_preprocess_returns_X_y_shapes(raw_df):
    X, y = preprocess_data(raw_df.copy())
    assert len(X) == len(y) == 3
    assert "Churn" not in X.columns


def test_no_nan_after_preprocess(raw_df):
    """TotalCharges vazio deve ser imputado; nenhuma coluna pode sair NaN."""
    X, _ = preprocess_data(raw_df.copy())
    assert not X.isna().any().any()


def test_senior_citizen_stays_binary_not_nan(raw_df):
    X, _ = preprocess_data(raw_df.copy())
    assert "SeniorCitizen" in X.columns
    assert not X["SeniorCitizen"].isna().any()
    assert set(X["SeniorCitizen"].unique()).issubset({0, 1})


def test_target_is_binary(raw_df):
    _, y = preprocess_data(raw_df.copy())
    assert set(y.unique()).issubset({0, 1})


def test_numeric_features_are_scaled(raw_df):
    """Após StandardScaler as colunas numéricas sao floats"""
    X, _ = preprocess_data(raw_df.copy())
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        assert np.issubdtype(X[col].dtype, np.floating)
