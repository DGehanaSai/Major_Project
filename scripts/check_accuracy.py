
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend/models/xgb_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "backend/models/scaler.joblib")

def check_acc():
    # Generate same synthetic data logic locally to test
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'temp': np.random.uniform(20, 40, n),
        'rain': np.random.uniform(50, 300, n),
        'ndvi': np.random.uniform(0.1, 0.9, n),
        'nitrogen': np.random.uniform(20, 200, n),
        'ph': np.random.uniform(4.5, 8.5, n),
        'crop_type': np.random.choice(['Rice', 'Maize', 'Wheat', 'Cotton'], n),
        'soil_type': np.random.choice(['Clay', 'Sandy', 'Loam'], n)
    })
    
    y = np.full(n, 2000.0)
    y += (data['nitrogen'] * 15)
    y += 1000 - (np.abs(data['rain'] - 150) * 5)
    y += (data['ndvi'] * 2000)
    y -= (np.abs(data['ph'] - 6.5) * 300)
    y[data['crop_type'] == 'Rice'] += 1000
    y[data['crop_type'] == 'Cotton'] -= 500
    y[data['soil_type'] == 'Loam'] += 500
    y[data['soil_type'] == 'Sandy'] -= 300
    y += np.random.normal(0, 200, n) # Noise
    y = np.clip(y, 500, 8000)
    
    # Encode
    df = pd.get_dummies(data, columns=['crop_type', 'soil_type'], drop_first=False)
    
    # Load model
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Align cols
    if hasattr(scaler, 'feature_names_in_'):
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
        
    X = scaler.transform(df)
    
    # Predict
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    
    score = r2_score(y, preds)
    print(f"FINAL_R2_SCORE: {score:.4f}")

if __name__ == "__main__":
    check_acc()
