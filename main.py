import argparse

from src.config import DATA_PATH, PROCESSED_DATA_PATH
from src.data_preprocessing import load_data, preprocess_data, log_processed_data
from src.model import split_data, train_all_models, plot_roc_curves


def main():
    parser = argparse.ArgumentParser(description="Train churn models with MLflow tracking.")
    parser.add_argument("--no-plot", action="store_true", help="Skip ROC curve plotting.")
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name (overrides MLFLOW_EXPERIMENT env var).",
    )
    args = parser.parse_args()

    if args.experiment:
        import os
        os.environ["MLFLOW_EXPERIMENT"] = args.experiment

    print("Loading Data")
    df = load_data(DATA_PATH)

    print("Preprocessing")
    X, y = preprocess_data(df)
    log_processed_data(df, PROCESSED_DATA_PATH)

    print("Splitting")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training models")
    results, best_model, best_name = train_all_models(X_train, X_test, y_train, y_test)

    if not args.no_plot:
        print("Plotting ROC curves")
        plot_roc_curves(
            {k: v["model"] for k, v in results.items()}, X_test, y_test
        )

    print(f"\nDone. Best model: {best_name}")


if __name__ == "__main__":
    main()
