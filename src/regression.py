"""
Regression Tasks Module (Phase 3)
==================================
TASK 2: Predicting the Number of Shares an Article Will Receive
  - Type: Supervised - Regression
  - Input: Top N selected features
  - Output: log(1 + shares) continuous value

TASK 5: Optimizing Article Formatting and Media Usage for Maximum Reach
  - Type: Supervised - Regression (focus on feature coefficients)
  - Input: ONLY structural/formatting features
  - Output: Expected shares (log scale)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

FIGURES_DIR = "figures"
RESULTS_DIR = "results"
SEED = 42

# Model groups for analysis
MODEL_GROUPS = {
    'Linear': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'Tree-based': ['Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor'],
    'Distance-based': ['SVR'],
}


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_csv(df, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path)
    print(f"  Saved: {path}")


def get_regression_models():
    """Return a dict of name -> regressor."""
    return {
        "Linear Regression":        LinearRegression(),
        "Ridge Regression":         Ridge(alpha=1.0, random_state=SEED),
        "Lasso Regression":         Lasso(alpha=0.01, random_state=SEED, max_iter=5000),
        "Decision Tree Regressor":  DecisionTreeRegressor(random_state=SEED),
        "Random Forest Regressor":  RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        "SVR":                      SVR(max_iter=5000),
    }


def evaluate_all_regressors(models, X_train, X_test, y_train, y_test, label=""):
    """Train and evaluate all regression models. Returns results DataFrame."""
    rows = []
    trained = {}

    for name, model in models.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Also compute original-scale metrics
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(np.clip(y_pred, None, 20))  # clip to avoid overflow
        rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

        rows.append({
            "Model": name,
            "RMSE_log": round(rmse, 4),
            "MAE_log": round(mae, 4),
            "R2": round(r2, 4),
            "RMSE_orig": round(rmse_orig, 2),
            "Train_Time_s": round(train_time, 3),
        })
        trained[name] = model

    df_results = pd.DataFrame(rows).set_index("Model")
    return df_results, trained


def identify_best_model_group(results_df):
    """Identify which group of models performs best on average R2."""
    group_scores = {}
    for group_name, model_names in MODEL_GROUPS.items():
        present = [m for m in model_names if m in results_df.index]
        if present:
            avg_r2 = results_df.loc[present, 'R2'].mean()
            group_scores[group_name] = round(avg_r2, 4)

    if group_scores:
        best_group = max(group_scores, key=group_scores.get)
        print(f"\n     Model Group Comparison (avg R2):")
        for grp, score in sorted(group_scores.items(), key=lambda x: x[1], reverse=True):
            marker = " <-- BEST" if grp == best_group else ""
            print(f"       {grp:<20s}: {score:.4f}{marker}")
        return best_group, group_scores
    return None, {}


def plot_predicted_vs_actual(y_test, y_pred, name, filename):
    """Save a predicted-vs-actual scatter plot (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, c='teal')
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual log(shares)', fontsize=12)
    ax.set_ylabel('Predicted log(shares)', fontsize=12)
    ax.set_title(f'Predicted vs Actual - {name}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, filename)


def plot_r2_comparison(before_df, after_df, title, filename):
    """Grouped bar chart for R2 scores before vs after."""
    models = before_df.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, before_df["R2"], width,
           label="Before Preprocessing", color="#e74c3c", edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, after_df["R2"], width,
           label="After Preprocessing", color="#3498db", edgecolor="black", alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R2 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, filename)


def plot_metric_heatmap(before_df, after_df, title, filename):
    """Heatmap of metric changes."""
    cols = ["RMSE_log", "MAE_log", "R2"]
    available = [c for c in cols if c in before_df.columns and c in after_df.columns]
    delta = after_df[available] - before_df[available]

    fig, ax = plt.subplots(figsize=(10, max(6, len(delta) * 0.7)))
    sns.heatmap(delta, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Change (After - Before)"})
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  TASK 2: Predicting the Number of Shares
# ══════════════════════════════════════════════════════════════════════

def run_task2_shares_regression(splits):
    """
    TASK 2: Predicting the Number of Shares an Article Will Receive
    Type: Supervised - Regression
    """
    n_features = len(splits['feature_names'])

    print("\n" + "=" * 70)
    print("  TASK 2: Predicting the Number of Shares an Article Will Receive")
    print("  Type: Supervised - Regression")
    print(f"  Input: {n_features} selected features (after feature selection)")
    print("  Output: log(1 + shares) continuous value")
    print("=" * 70)

    # ── BEFORE Preprocessing ──
    print("\n  --- Results BEFORE Preprocessing (raw features, no scaling) ---")
    before_df, trained_before = evaluate_all_regressors(
        get_regression_models(),
        splits['X_train_raw'], splits['X_test_raw'],
        splits['y_train_reg'], splits['y_test_reg'],
        label="Before"
    )
    print(before_df.to_string())
    _save_csv(before_df, "task2_regression_before.csv")

    # ── AFTER Preprocessing ──
    print("\n  --- Results AFTER Preprocessing (scaled features) ---")
    after_df, trained_after = evaluate_all_regressors(
        get_regression_models(),
        splits['X_train_proc'], splits['X_test_proc'],
        splits['y_train_reg'], splits['y_test_reg'],
        label="After"
    )
    print(after_df.to_string())
    _save_csv(after_df, "task2_regression_after.csv")

    # ── Best Model ──
    best_before = before_df['R2'].idxmax()
    best_after = after_df['R2'].idxmax()
    print(f"\n     Best model BEFORE preprocessing: {best_before} (R2: {before_df.loc[best_before, 'R2']:.4f})")
    print(f"     Best model AFTER  preprocessing: {best_after} (R2: {after_df.loc[best_after, 'R2']:.4f})")

    # ── Best Model Group ──
    print("\n     BEFORE preprocessing:")
    identify_best_model_group(before_df)
    print("\n     AFTER preprocessing:")
    best_group, _ = identify_best_model_group(after_df)

    # ── Predicted vs Actual for best model ──
    best_model = trained_after[best_after]
    y_pred_best = best_model.predict(splits['X_test_proc'])
    plot_predicted_vs_actual(splits['y_test_reg'], y_pred_best,
                             best_after, 'fig_task2_pred_vs_actual.png')

    # ── R2 Comparison ──
    plot_r2_comparison(before_df, after_df,
                       'Task 2: R2 Score - Before vs After',
                       'fig_task2_r2_comparison.png')

    # ── Metric Heatmap ──
    plot_metric_heatmap(before_df, after_df,
                        'Task 2: Metric Change After Preprocessing',
                        'fig_task2_heatmap.png')

    print("\n  TASK 2 COMPLETE")
    print("=" * 70)

    return {
        'before': before_df, 'after': after_df,
        'trained_before': trained_before, 'trained_after': trained_after,
        'best_model_name': best_after, 'best_group': best_group,
    }


# ══════════════════════════════════════════════════════════════════════
#  TASK 5: Optimizing Article Formatting and Media Usage
# ══════════════════════════════════════════════════════════════════════

def run_task5_formatting_optimization(df_full, target='shares'):
    """
    TASK 5: Optimizing Article Formatting and Media Usage for Maximum Reach
    Type: Supervised - Regression (focus on feature coefficients / partial dependence)

    Uses ONLY structural/formatting features.
    """
    # Define formatting features
    formatting_features = [
        'n_tokens_title', 'n_tokens_content',
        'num_imgs', 'num_videos',
        'num_hrefs', 'num_self_hrefs',
    ]
    # Keep only features present in the dataframe
    formatting_features = [f for f in formatting_features if f in df_full.columns]

    print("\n" + "=" * 70)
    print("  TASK 5: Optimizing Article Formatting and Media Usage for Maximum Reach")
    print("  Real-World Question: What is the optimal combination of text length,")
    print("    links, and media to drive shares?")
    print("  Type: Supervised - Regression (Feature Coefficients / Partial Dependence)")
    print(f"  Input: {len(formatting_features)} structural features: {formatting_features}")
    print("  Output: log(1 + shares) continuous value")
    print("=" * 70)

    X = df_full[formatting_features].copy()
    y = np.log1p(df_full[target])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Impute + Scale
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train),
                                columns=formatting_features, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test),
                               columns=formatting_features, index=X_test.index)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp),
                                   columns=formatting_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp),
                                  columns=formatting_features, index=X_test.index)

    # Models — use both interpretable (Ridge, Lasso) and powerful (RF, GB)
    models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=SEED),
        "Lasso Regression": Lasso(alpha=0.01, random_state=SEED, max_iter=5000),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    }

    print("\n  --- Model Results (scaled features) ---")
    results_df, trained = evaluate_all_regressors(
        models, X_train_scaled, X_test_scaled, y_train, y_test
    )
    print(results_df.to_string())
    _save_csv(results_df, "task5_formatting_results.csv")

    # ── Interpretable Coefficients (Ridge) ──
    ridge = trained.get("Ridge Regression")
    if ridge is not None:
        print("\n     Ridge Regression Coefficients (Formatting Guidelines):")
        print(f"     {'Feature':<25s} {'Coefficient':>12s}  Interpretation")
        print(f"     {'─'*25} {'─'*12}  {'─'*40}")
        for feat, coef in sorted(zip(formatting_features, ridge.coef_),
                                  key=lambda x: abs(x[1]), reverse=True):
            direction = "increases" if coef > 0 else "decreases"
            print(f"     {feat:<25s} {coef:>12.4f}  1 std increase {direction} log(shares) by {abs(coef):.4f}")

    # ── Lasso Coefficients ──
    lasso = trained.get("Lasso Regression")
    if lasso is not None:
        print("\n     Lasso Regression Coefficients:")
        for feat, coef in sorted(zip(formatting_features, lasso.coef_),
                                  key=lambda x: abs(x[1]), reverse=True):
            status = "ZERO (dropped by Lasso)" if abs(coef) < 1e-6 else f"{coef:.4f}"
            print(f"       {feat:<25s}: {status}")

    # ── RF Feature Importance ──
    rf = trained.get("Random Forest Regressor")
    if rf is not None:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n     Random Forest Feature Importance:")
        for i in indices:
            print(f"       {formatting_features[i]:<25s}: {importances[i]:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_idx = np.argsort(importances)
        ax.barh(range(len(formatting_features)),
                importances[sorted_idx],
                color='#e67e22', edgecolor='black', alpha=0.85)
        ax.set_yticks(range(len(formatting_features)))
        ax.set_yticklabels([formatting_features[i] for i in sorted_idx])
        ax.set_xlabel('Importance')
        ax.set_title("Task 5: Formatting Feature Importance (Random Forest)",
                      fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        _save(fig, 'fig_task5_feature_importance.png')

    # ── Partial Dependence (manual, for key features) ──
    print("\n     Actionable Insights:")
    # Compute mean shares by feature bins
    for feat in ['num_imgs', 'num_videos', 'num_hrefs']:
        if feat in df_full.columns:
            # Create bins
            col = df_full[feat]
            bins = [0, 1, 3, 5, 10, 20, col.max() + 1]
            labels = ['0', '1-3', '4-5', '6-10', '11-20', '20+']
            binned = pd.cut(col, bins=bins, labels=labels, right=False)
            avg_shares = df_full.groupby(binned, observed=False)[target].mean()
            print(f"\n     Average shares by {feat}:")
            for bin_label, avg in avg_shares.items():
                print(f"       {bin_label:>8s}: {avg:>10.0f} shares")

    print("\n  TASK 5 COMPLETE")
    print("=" * 70)

    return {'results': results_df, 'trained': trained}
