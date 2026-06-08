""" Testes do treino    """
import os
import mlflow
import pandas as pd

from src import model as model_mod
from src.data_preprocessing import preprocess_data
from src.model import _compute_metrics, split_data, train_all_models


def test_compute_metrics_keys():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.2, 0.9, 0.4, 0.1]
    m = _compute_metrics(y_true, y_pred, y_proba)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(m)
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_train_all_models_returns_best(raw_df, tmp_path, monkeypatch):
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **k: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(model_mod, "MODEL_DIR", str(tmp_path))

    big = pd.concat([raw_df] * 20, ignore_index=True)
    X, y = preprocess_data(big)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results, best_model, best_name = train_all_models(X_train, X_test, y_train, y_test)

    assert isinstance(results, dict) and len(results) == 6
    assert best_model is not None
    assert best_name in results
    for entry in results.values():
        assert "model" in entry and "metrics" in entry
        assert "roc_auc" in entry["metrics"]
    assert os.path.exists(os.path.join(str(tmp_path), "best_model.pkl"))
