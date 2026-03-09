# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(BASE_DIR, "data")
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# os.makedirs(MODELS_DIR, exist_ok=True)

# YIELD_CSV = os.path.join(DATA_DIR, "yield.csv")
# WEATHER_CSV = os.path.join(DATA_DIR, "weather.csv")
# SOIL_CSV = os.path.join(DATA_DIR, "soil.csv")
# NDVI_CSV = os.path.join(DATA_DIR, "ndvi.csv")

# # Model paths
# RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
# XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model_optuna.pkl")
# MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.pkl")
# SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
import os

# config.py is at: backend/src/config.py
# 1st dirname gets: backend/src/
# 2nd dirname gets: backend/ (The root of your backend project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define Data and Model directories relative to the backend root
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Data file paths
YIELD_CSV = os.path.join(DATA_DIR, "yield.csv")
WEATHER_CSV = os.path.join(DATA_DIR, "weather.csv")
SOIL_CSV = os.path.join(DATA_DIR, "soil.csv")
NDVI_CSV = os.path.join(DATA_DIR, "ndvi.csv")

# Model/Scaler file paths
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.pkl") # Keep this if MLP wasn't retrained, or update
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")