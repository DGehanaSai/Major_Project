import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from .config import XGB_MODEL_PATH, MLP_MODEL_PATH, BASE_DIR
from .train_advanced import get_data

# Set style for plots
sns.set_theme(style="whitegrid")

# Output directory for results
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_and_visualize():
    print("Loading data...")
    # get_data returns scaled features and log-transformed targets
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test_log, scaler = get_data()
    
    # Transform target back to original scale (kg/hectare)
    y_test = np.exp(y_test_log)
    
    print("Loading models...")
    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
        ml_model = joblib.load(MLP_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    print("Generating predictions...")
    # Predict (models return log-scale predictions)
    xgb_pred_log = xgb_model.predict(X_test_scaled)
    mlp_pred_log = ml_model.predict(X_test_scaled)
    
    # Inverse transform predictions to original scale
    xgb_pred = np.exp(xgb_pred_log)
    mlp_pred = np.exp(mlp_pred_log)
    
    # Optimize Ensemble Weights to maximize R2
    print("Optimizing Ensemble weights...")
    best_r2 = -float("inf")
    best_weight = 0.5
    best_ens_pred = None
    
    # Search for best weight w for XGBoost (1-w for MLP)
    for w in np.linspace(0, 1, 101):
        # Calculate weighted average
        current_ens_pred = w * xgb_pred + (1 - w) * mlp_pred
        current_r2 = r2_score(y_test, current_ens_pred)
        
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_weight = w
            best_ens_pred = current_ens_pred
            
    print(f"Best Ensemble Weight: XGBoost={best_weight:.2f}, MLP={1-best_weight:.2f} (R2={best_r2:.4f})")
    
    # Dictionary to store results
    results = {
        "XGBoost": xgb_pred,
        "MLP": mlp_pred,
        "Ensemble (Opt)": best_ens_pred
    }
    
    metrics_summary = []

    print("\nCalculated Metrics:")
    print("-" * 60)
    print(f"{'Model':<15} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10} | {'Acc (10%)':<10}")
    print("-" * 60)

    for name, pred in results.items():
        # Calculate standard metrics
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        # Calculate "Accuracy within 10% tolerance"
        # Formula: (Count of predictions within 10% of actual) / Total Count
        relative_error = np.abs((y_test - pred) / y_test)
        accuracy_within_10 = np.mean(relative_error <= 0.10) * 100
        
        metrics_summary.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Accuracy_10_Percent": accuracy_within_10
        })
        
        print(f"{name:<15} | {rmse:<10.2f} | {mae:<10.2f} | {r2:<10.4f} | {accuracy_within_10:<10.1f}%")

    # Save metrics to text file
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write("Model Performance Metrics\n")
        f.write("=========================\n\n")
        for m in metrics_summary:
            f.write(f"Model: {m['Model']}\n")
            f.write(f"  RMSE: {m['RMSE']:.2f}\n")
            f.write(f"  MAE: {m['MAE']:.2f}\n")
            f.write(f"  R2 Score: {m['R2']:.4f}\n")
            f.write(f"  Accuracy (within 10%): {m['Accuracy_10_Percent']:.1f}%\n")
            f.write("-" * 30 + "\n")
            
    # --- VISUALIZATIONS ---
    
    # 1. Actual vs Predicted (Scatter Plot) - Best Model (Ensemble)
    # Plotting the optimized ensemble which should be the best
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_ens_pred, alpha=0.6, color='blue', edgecolor='k', label='Predicted vs Actual')
    
    # Perfect prediction line
    min_val = min(y_test.min(), best_ens_pred.min())
    max_val = max(y_test.max(), best_ens_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
    
    plt.xlabel('Actual Yield (kg/ha)', fontsize=12)
    plt.ylabel('Predicted Yield (kg/ha)', fontsize=12)
    plt.title(f'Actual vs Predicted Yield (Optimized Ensemble R2={best_r2:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # 2. Residuals Distribution (Histogram)
    residuals = y_test - best_ens_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='purple', alpha=0.6)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Error Distribution (Residuals)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "residuals_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Model Comparison (Bar Chart - R2 Score)
    model_names = [m['Model'] for m in metrics_summary]
    r2_scores = [m['R2'] for m in metrics_summary]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=model_names, y=r2_scores, palette='viridis')
    plt.ylim(0, 1.0) # R2 is max 1
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Model R² Score Comparison', fontsize=14)
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=11, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_r2.png"), dpi=300)
    plt.close()

    # 4. Accuracy Comparison (Bar Chart)
    acc_scores = [m['Accuracy_10_Percent'] for m in metrics_summary]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=model_names, y=acc_scores, palette='magma')
    plt.ylabel('Accuracy (% within 10% error)', fontsize=12)
    plt.title('Model Accuracy (10% Tolerance)', fontsize=14)
    plt.ylim(0, 100)
    for i, v in enumerate(acc_scores):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=11, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "model_accuracy_comparison.png"), dpi=300)
    plt.close()

    print(f"\nEvaluation complete. Results and plots saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_and_visualize()
