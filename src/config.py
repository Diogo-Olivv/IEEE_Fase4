import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data/raw/', 'churn-data.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data/processed/', 'churn-processed.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

MLFLOW_TRACKING_URI = os.environ.get(
    'MLFLOW_TRACKING_URI', f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
)
MLFLOW_EXPERIMENT = os.environ.get('MLFLOW_EXPERIMENT', 'churn-telco')
