
# Figures for the before/after comparison.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

FIGURES_DIR = "figures"


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)

#  1.  ROC CURVES
def plot_roc_curves(trained_models, X_test, y_test, label="After Preprocessing"):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trained_models)))

    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {label}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    safe = label.lower().replace(" ", "_")
    path = os.path.join(FIGURES_DIR, f"fig_roc_{safe}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

#  2.  ACCURACY BAR CHART 
def plot_accuracy_comparison(clf_before, clf_after):
    _ensure_dir()
    models = clf_before.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width / 2, clf_before["Accuracy"], width,
                   label="Before Preprocessing", color="#e74c3c", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, clf_after["Accuracy"], width,
                   label="After Preprocessing", color="#2ecc71", edgecolor="black", alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy — Before vs After Preprocessing",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    path = os.path.join(FIGURES_DIR, "fig_accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

#  3.  METRIC HEATMAP

def plot_metric_heatmap(df_before, df_after, task="Classification"):
    _ensure_dir()

    if task == "Classification":
        cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    else:
        cols = ["RMSE", "MAE", "R2"]

    delta = df_after[cols] - df_before[cols]

    fig, ax = plt.subplots(figsize=(12, max(6, len(delta) * 0.7)))
    sns.heatmap(delta, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Δ (After − Before)"})
    ax.set_title(f"{task} — Metric Change After Preprocessing (Δ)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("")

    safe = task.lower().replace(" ", "_")
    path = os.path.join(FIGURES_DIR, f"fig_heatmap_{safe}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# 4.  FEATURE IMPORTANCE

def plot_rf_feature_importance(trained_models, feature_names, top_n=20):
    _ensure_dir()
    rf = trained_models.get("Random Forest")
    if rf is None:
        print("  ⚠  Random Forest model not found — skipping feature importance plot.")
        return

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_feats = [feature_names[i] for i in indices]
    top_imps  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_feats))
    ax.barh(y_pos, top_imps[::-1], color="teal", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_xlabel("Gini Importance", fontsize=12)
    ax.set_title("Top 20 Feature Importances — Random Forest (After Preprocessing)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    path = os.path.join(FIGURES_DIR, "fig_rf_feature_importance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Also print top features
    print("\n  Top 20 Features (Random Forest Gini Importance):")
    for i, (feat, imp) in enumerate(zip(top_feats, top_imps), 1):
        print(f"    {i:>2}. {feat:<40s} {imp:.4f}")

#  5.  LEARNING CURVE (best classifier)

def plot_learning_curve(trained_models, X_train, y_train, clf_results):
    _ensure_dir()

    best_name = clf_results["Accuracy"].idxmax()
    best_model = trained_models.get(best_name)
    if best_model is None:
        return

    print(f"\n  Generating learning curve for best model: {best_name}...")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X_train, y_train,
            cv=5, n_jobs=-1, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#2ecc71")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#e74c3c")
    ax.plot(train_sizes, train_mean, "o-", color="#2ecc71", lw=2, label="Training score")
    ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", lw=2, label="Validation score")

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Learning Curve — {best_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIGURES_DIR, "fig_learning_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

#  6.  REGRESSION BAR CHART - R² scores
def plot_r2_comparison(reg_before, reg_after):
    _ensure_dir()
    models = reg_before.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, reg_before["R2"], width,
           label="Before Preprocessing", color="#e74c3c", edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, reg_after["R2"], width,
           label="After Preprocessing", color="#3498db", edgecolor="black", alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Regression R² — Before vs After Preprocessing",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(FIGURES_DIR, "fig_r2_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

#  VISUALISATION RUNNER

def run_comparison_visualizations(results):
    print("  GENERATING COMPARISON VISUALIZATIONS")
  
    # ROC curves — before and after
    plot_roc_curves(results["trained_clf_before"],
                    results["X_test_clf_before"], results["y_test_clf"],
                    label="Before Preprocessing")
    plot_roc_curves(results["trained_clf_after"],
                    results["X_test_clf_after"], results["y_test_clf"],
                    label="After Preprocessing")

    # Accuracy comparison bar chart
    plot_accuracy_comparison(results["clf_before"], results["clf_after"])

    # R² comparison bar chart
    plot_r2_comparison(results["reg_before"], results["reg_after"])

    # Metric heatmaps
    plot_metric_heatmap(results["clf_before"], results["clf_after"], "Classification")
    plot_metric_heatmap(results["reg_before"], results["reg_after"], "Regression")

    # Feature importance from Random Forest (after preprocessing)
    plot_rf_feature_importance(results["trained_clf_after"],
                               results["feature_names_after"])

    # Learning curve for best classifier
    plot_learning_curve(results["trained_clf_after"],
                        results["X_test_clf_after"], 
                        results["y_test_clf"],
                        results["clf_after"])

    print("\n  All comparison visualizations saved to figures/")
