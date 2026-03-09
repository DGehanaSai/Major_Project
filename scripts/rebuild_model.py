
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend/models/xgb_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "backend/models/scaler.joblib")

# Ensure models dir exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def generate_synthetic_data(n=2000):
    print("Generating synthetic data...")
    np.random.seed(42)
    
    # Random Inputs
    data = pd.DataFrame({
        'temp': np.random.uniform(20, 40, n),
        'rain': np.random.uniform(50, 300, n),
        'ndvi': np.random.uniform(0.1, 0.9, n),
        'nitrogen': np.random.uniform(20, 200, n),
        'ph': np.random.uniform(4.5, 8.5, n),
        'crop_type': np.random.choice(['Rice', 'Maize', 'Wheat', 'Cotton'], n),
        'soil_type': np.random.choice(['Clay', 'Sandy', 'Loam'], n)
    })
    
    # Feature Engineering Logic (Simulating Real World)
    # Base Yield
    y = np.full(n, 2000.0)
    
    # Nitrogen Effect (Higher is usually better up to a point)
    y += (data['nitrogen'] * 15)
    
    # Rain Effect (Optimal around 150)
    y += 1000 - (np.abs(data['rain'] - 150) * 5)
    
    # NDVI Effect (Vegetation health)
    y += (data['ndvi'] * 2000)
    
    # PH Effect (Optimal 6.5)
    y -= (np.abs(data['ph'] - 6.5) * 300)
    
    # Crop Specific Adjustments
    y[data['crop_type'] == 'Rice'] += 1000
    y[data['crop_type'] == 'Cotton'] -= 500
    
    # Soil Specific Adjustments
    y[data['soil_type'] == 'Loam'] += 500
    y[data['soil_type'] == 'Sandy'] -= 300
    
    # Add random noise
    y += np.random.normal(0, 200, n)
    
    # Clip to realistic range
    y = np.clip(y, 500, 8000)
    
    data['harvest_yield_kg'] = y
    
    return data

def train_model():
    df = generate_synthetic_data()
    
    # One Hot Encode
    df_encoded = pd.get_dummies(df, columns=['crop_type', 'soil_type'], drop_first=False) # Changed to False to be explicit
    # Note: drop_first=False makes it easier to handle specific columns like crop_type_Rice explicitly
    
    # Define Features and Target
    X = df_encoded.drop('harvest_yield_kg', axis=1)
    y = np.log1p(df_encoded['harvest_yield_kg']) # Log transform target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("Training XGBoost...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    score = model.score(X_test_scaled, y_test)
    print(f"Model R2 Score: {score:.4f}")
    
    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Feature Names: {X.columns.tolist()}")

if __name__ == "__main__":
    train_model()
