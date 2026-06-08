import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import mlflow
import mlflow.sklearn
import joblib

from src.config import MODEL_DIR, OUTPUT_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\Resultados para modelo: {model_name}")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    return out


def _compute_metrics(y_test, y_pred, y_proba):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    return metrics


def train_all_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, cache_size=500),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    }

    best_model = None
    best_name = None
    best_score = 0.0
    results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", MLFLOW_EXPERIMENT))

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.get_params())

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = None

            metrics = _compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train.iloc[:2],
            )

            results[name] = {"model": model, "metrics": metrics}

            score = metrics.get("roc_auc", metrics["f1"])
            print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)
    
    print(f"\nMelhor Modelo: {best_name} (roc_auc={best_score:.4f})")

    with mlflow.start_run(run_name=f"best::{best_name}"):
        mlflow.set_tag("best", True)
        mlflow.set_tag("best_model_name", best_name)
        mlflow.log_metric("best_roc_auc", best_score)

    return results, best_model, best_name


def plot_roc_curves(models, X_test, y_test, save_path=None):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curve Comparison of Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")
    return save_path
