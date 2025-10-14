# utils/costants.py

# Paths
S3_ENDPOINT = "http://minio:9000"
S3 = "s3://"

BUCKET_RAW = "data/raw/"
BUCKET_FINAL = "data/final/"
ORIG_DATA_NAME = "healthcare-dataset-stroke-data.csv"
END_DATA_NAME = "stroke_dummies.csv"
PROCESSED_PATH = "processed/stroke/"
OPT_DATASET = "/opt/airflow/dataset/"
DATA_INFO_PATH = "data_info/data.json"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
CATEGORICAL_FEATURES_FILE = "files/categorical_features.txt"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"


# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "target_col_stroke"
TARGET_COLUMN_DEFAULT = "stroke"
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN = "train"
TEST = "test"