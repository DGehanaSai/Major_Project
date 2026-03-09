import joblib
import numpy as np # Import numpy for the square root calculation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .config import RF_MODEL_PATH, SCALER_PATH
from .data_preprocessing import merge_data, build_feature_target

def train_baseline():
    # 1. Load and Prepare Data
    df, encoders = merge_data()
    X, y, feature_cols = build_feature_target(df)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Model
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    # 5. Evaluate Model
    
    # FIX: Remove 'squared=False' and manually calculate RMSE
    # to avoid the TypeError in older scikit-learn versions.
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # Calculate RMSE manually by taking the square root of MSE
    
    r2 = r2_score(y_test, y_pred)

    print(f"[Baseline RF] RMSE={rmse:.3f}, R²={r2:.3f}")

    # 6. Save Model and Scaler
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return rmse, r2

if __name__ == "__main__":
    train_baseline()