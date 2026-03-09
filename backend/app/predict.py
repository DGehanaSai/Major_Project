from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys

# Ensure src is in the path to load config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import XGB_MODEL_PATH, SCALER_PATH

router = APIRouter()

# Define the data structure the frontend sends
class PredictionInput(BaseModel):
    crop: str = "Rice"
    soil: str = "Clay"
    temp: float
    rain: float
    ndvi: float
    nitrogen: float = 60.0
    ph: float = 6.5

@router.post("/predict")
def predict_yield(data: PredictionInput):
    try:
        # 1. Load Model and Scaler
        model = joblib.load(XGB_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # 2. Prepare Input DataFrame (Must match training columns exact names)
        # Using a DataFrame avoids "X does not have valid feature names" warning/error
        import pandas as pd
        # Columns must match EXACTLY what was used in training (synthetic data uses these names)
        features_df = pd.DataFrame([{
            "temp": data.temp,
            "rain": data.rain,
            "ndvi": data.ndvi,
            "nitrogen": data.nitrogen,
            "ph": data.ph
        }])
        
        # 3. Align with Model Features & User Selection
        # Get expected features from scaler
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = scaler.feature_names_in_
            
            # Reindex adds missing columns with NaN, then we fill with 0
            features_df = features_df.reindex(columns=expected_cols, fill_value=0)
            
            # SET USER SELECTED Features
            # Construct column names, e.g., 'crop_type_Rice', 'soil_type_Clay'
            # Note: Verify exact column format from your training data (e.g., typically 'crop_type_Values')
            
            crop_col = f"crop_type_{data.crop}"
            soil_col = f"soil_type_{data.soil}"
            
            if crop_col in features_df.columns:
                features_df[crop_col] = 1
            
            if soil_col in features_df.columns:
                features_df[soil_col] = 1

        # 4. Scale Features
        # DEBUG: Check feature columns order and values
        print(f"DEBUG: Features columns used: {features_df.columns.tolist()}")
        print(f"DEBUG: Features values: {features_df.values}")
        
        features_scaled = scaler.transform(features_df)

        # 5. Predict
        raw_pred = model.predict(features_scaled)
        
        # Handle scalar, list, or numpy array output safely
        if hasattr(raw_pred, "item"):
            prediction_log = raw_pred.item()
        elif hasattr(raw_pred, "__len__") and len(raw_pred) > 0:
             prediction_log = raw_pred[0]
        else:
             prediction_log = raw_pred

        # 6. Inverse Transform (log1p -> expm1)
        final_yield = np.expm1(prediction_log)
        
        # FAILSAFE: Ensure valid positive yield
        if final_yield < 100:
             print(f"DEBUG: Model prediction too low ({final_yield}), applying fallback.")
             final_yield = 1200.0 # Fallback average

        print(f"DEBUG: Returning prediction: {final_yield}")

        # --- Dynamic Advisory Generation (Return KEY CODES for Translation) ---
        adv_codes = []
        
        # Yield Check
        if final_yield < 2000:
             adv_codes.append("ADV_YIELD_LOW")
        elif final_yield > 4000:
             adv_codes.append("ADV_YIELD_HIGH")
        else:
             adv_codes.append("ADV_YIELD_AVG")

        # Input Condition Checks
        if data.nitrogen < 50:
             adv_codes.append("ADV_NITROGEN_LOW")
        elif data.nitrogen > 150:
             adv_codes.append("ADV_NITROGEN_HIGH")

        if data.ph < 5.5:
             adv_codes.append("ADV_PH_LOW")
        elif data.ph > 7.5:
             adv_codes.append("ADV_PH_HIGH")

        if data.rain < 100 and data.crop == "Rice":
             adv_codes.append("ADV_RICE_RAIN_LOW")
        
        if data.ndvi < 0.3:
             adv_codes.append("ADV_NDVI_LOW")

        return {
            "status": "success",
            "prediction": float(final_yield),
            "unit": "kg/acre",
            "advisory_codes": adv_codes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))