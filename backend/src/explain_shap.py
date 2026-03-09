import joblib
import shap
import matplotlib.pyplot as plt
from .config import XGB_MODEL_PATH, SCALER_PATH
from .data_preprocessing import merge_data, build_feature_target
def shap_summary():
    df, _ = merge_data()
    X, y, feature_cols = build_feature_target(df)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    model = joblib.load(XGB_MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_xgb.png", dpi=200)
    print("Saved shap_summary_xgb.png")
if __name__ == "__main__":
    shap_summary()
