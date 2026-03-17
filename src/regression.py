"""
Regression Module
- Linear Regression (baseline)
- Random Forest Regressor
- Evaluation: RMSE, MAE, R²
- Predicted-vs-Actual scatter plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FIGURES_DIR = "figures"


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train):
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest_reg(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model, X_test, y_test, name="Model"):
    """Evaluate on log-scale and print metrics in original scale as well."""
    y_pred = model.predict(X_test)

    # Metrics on log-transformed scale
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Metrics back in original shares scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)

    print(f"\n--- {name} ---")
    print(f"[Log scale]      RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  R²: {r2:.4f}")
    print(f"[Original scale] RMSE: {rmse_orig:.2f}  |  MAE: {mae_orig:.2f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred': y_pred}


def plot_predicted_vs_actual(y_test, y_pred, name="Model"):
    """Save a predicted-vs-actual scatter plot (log scale)."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, c='teal')
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual log(shares)', fontsize=12)
    ax.set_ylabel('Predicted log(shares)', fontsize=12)
    ax.set_title(f'Predicted vs Actual — {name}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    safe = name.lower().replace(" ", "_")
    path = os.path.join(FIGURES_DIR, f'fig_reg_{safe}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_regression(X_train, X_test, y_train, y_test):
    """Run full regression pipeline."""
    print("\n" + "=" * 60)
    print("  REGRESSION — Predicting log(shares)")
    print("=" * 60)

    models = {}
    for trainer, name in [(train_linear_regression, "Linear Regression"),
                          (train_ridge_regression, "Ridge Regression"),
                          (train_random_forest_reg, "Random Forest Regressor")]:
        m = trainer(X_train, y_train)
        metrics = evaluate_regressor(m, X_test, y_test, name)
        plot_predicted_vs_actual(y_test, metrics['y_pred'], name)
        models[name] = m

    return models
