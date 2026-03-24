"""
Classification Tasks Module (Phase 3)
======================================
TASK 1: Predicting Whether a News Article Will Be Popular
  - Type: Supervised - Binary Classification
  - Input: Top N selected features
  - Output: Popular (1) / Unpopular (0) based on median shares

TASK 6: Recommending the Optimal Publication Window (Weekday vs. Weekend)
  - Type: Supervised - Binary Classification
  - Input: NLP features + sentiment + content channels
  - Output: Publish on Weekday (0) / Publish on Weekend (1)

Professor's Feedback:
  #2: "Regression is not a task" - we name real-world tasks
  #4: "First identify which model is good"
  #5: "Can be a group of models which can be good"
  #7: "Clear on input and output"
  #12: "Before model training and after model training"
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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve, auc)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

FIGURES_DIR = "figures"
RESULTS_DIR = "results"
SEED = 42

# Model groups for analysis
MODEL_GROUPS = {
    'Linear': ['Logistic Regression'],
    'Distance-based': ['KNN', 'SVM'],
    'Tree-based': ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'AdaBoost'],
    'Probabilistic': ['Naive Bayes'],
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


def get_classification_models():
    """Return a dict of name -> classifier."""
    return {
        "Logistic Regression":  LogisticRegression(random_state=SEED, max_iter=1000),
        "KNN":                  KNeighborsClassifier(n_neighbors=5),
        "SVM":                  SVC(random_state=SEED, probability=True, max_iter=5000),
        "Decision Tree":        DecisionTreeClassifier(random_state=SEED),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "AdaBoost":             AdaBoostClassifier(n_estimators=100, random_state=SEED),
        "Naive Bayes":          GaussianNB(),
    }


def evaluate_all_classifiers(models, X_train, X_test, y_train, y_test, label=""):
    """Train and evaluate all classification models. Returns results DataFrame."""
    rows = []
    trained = {}

    for name, model in models.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc_val = roc_auc_score(y_test, y_prob)
        except Exception:
            auc_val = np.nan

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(auc_val, 4) if not np.isnan(auc_val) else np.nan,
            "Train_Time_s": round(train_time, 3),
        })
        trained[name] = model

    df_results = pd.DataFrame(rows).set_index("Model")
    return df_results, trained


def identify_best_model_group(results_df):
    """Identify which group of models performs best on average."""
    group_scores = {}
    for group_name, model_names in MODEL_GROUPS.items():
        present = [m for m in model_names if m in results_df.index]
        if present:
            avg_acc = results_df.loc[present, 'Accuracy'].mean()
            group_scores[group_name] = round(avg_acc, 4)

    if group_scores:
        best_group = max(group_scores, key=group_scores.get)
        print(f"\n     Model Group Comparison (avg Accuracy):")
        for grp, score in sorted(group_scores.items(), key=lambda x: x[1], reverse=True):
            marker = " <-- BEST" if grp == best_group else ""
            print(f"       {grp:<20s}: {score:.4f}{marker}")
        return best_group, group_scores
    return None, {}


def plot_roc_curves(trained_models, X_test, y_test, title, filename):
    """Overlay ROC curves for all classifiers."""
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
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, filename)


def plot_accuracy_comparison(before_df, after_df, title, filename):
    """Grouped bar chart of accuracy before vs after preprocessing."""
    models = before_df.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width / 2, before_df["Accuracy"], width,
                   label="Before Preprocessing", color="#e74c3c", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, after_df["Accuracy"], width,
                   label="After Preprocessing", color="#2ecc71", edgecolor="black", alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    _save(fig, filename)


def plot_confusion_matrix(y_test, y_pred, name, filename):
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unpopular', 'Popular'],
                yticklabels=['Unpopular', 'Popular'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {name}', fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


def plot_metric_heatmap(before_df, after_df, title, filename):
    """Heatmap showing metric improvement after preprocessing."""
    cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    available_cols = [c for c in cols if c in before_df.columns and c in after_df.columns]
    delta = after_df[available_cols] - before_df[available_cols]

    fig, ax = plt.subplots(figsize=(12, max(6, len(delta) * 0.7)))
    sns.heatmap(delta, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Change (After - Before)"})
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  TASK 1: Predicting Whether a News Article Will Be Popular
# ══════════════════════════════════════════════════════════════════════

def run_task1_popularity_classification(splits):
    """
    TASK 1: Predicting Whether a News Article Will Be Popular
    Type: Supervised - Binary Classification
    """
    n_features = len(splits['feature_names'])

    print("\n" + "=" * 70)
    print("  TASK 1: Predicting Whether a News Article Will Be Popular")
    print("  Type: Supervised - Binary Classification")
    print(f"  Input: {n_features} selected features (after feature selection)")
    print(f"  Output: Binary label (Popular=1 if shares >= {splits['median_val']:.0f}, Unpopular=0)")
    print("=" * 70)

    # ── BEFORE Preprocessing ──
    print("\n  --- Results BEFORE Preprocessing (raw features, no scaling) ---")
    before_df, trained_before = evaluate_all_classifiers(
        get_classification_models(),
        splits['X_train_raw'], splits['X_test_raw'],
        splits['y_train_clf'], splits['y_test_clf'],
        label="Before"
    )
    print(before_df.to_string())
    _save_csv(before_df, "task1_classification_before.csv")

    # ── AFTER Preprocessing ──
    print("\n  --- Results AFTER Preprocessing (scaled features) ---")
    after_df, trained_after = evaluate_all_classifiers(
        get_classification_models(),
        splits['X_train_proc'], splits['X_test_proc'],
        splits['y_train_clf'], splits['y_test_clf'],
        label="After"
    )
    print(after_df.to_string())
    _save_csv(after_df, "task1_classification_after.csv")

    # ── Best Model ──
    best_before = before_df['Accuracy'].idxmax()
    best_after = after_df['Accuracy'].idxmax()
    print(f"\n     Best model BEFORE preprocessing: {best_before} (Acc: {before_df.loc[best_before, 'Accuracy']:.4f})")
    print(f"     Best model AFTER  preprocessing: {best_after} (Acc: {after_df.loc[best_after, 'Accuracy']:.4f})")

    # ── Best Model Group ──
    print("\n     BEFORE preprocessing:")
    identify_best_model_group(before_df)
    print("\n     AFTER preprocessing:")
    best_group, _ = identify_best_model_group(after_df)

    # ── Confusion Matrix for best model ──
    best_model = trained_after[best_after]
    y_pred_best = best_model.predict(splits['X_test_proc'])
    plot_confusion_matrix(splits['y_test_clf'], y_pred_best, best_after,
                          'fig_task1_cm_best.png')

    # ── ROC Curves ──
    plot_roc_curves(trained_before, splits['X_test_raw'], splits['y_test_clf'],
                    'Task 1: ROC Curves (Before Preprocessing)',
                    'fig_task1_roc_before.png')
    plot_roc_curves(trained_after, splits['X_test_proc'], splits['y_test_clf'],
                    'Task 1: ROC Curves (After Preprocessing)',
                    'fig_task1_roc_after.png')

    # ── Accuracy Comparison ──
    plot_accuracy_comparison(before_df, after_df,
                             'Task 1: Classification Accuracy - Before vs After',
                             'fig_task1_accuracy_comparison.png')

    # ── Metric Heatmap ──
    plot_metric_heatmap(before_df, after_df,
                        'Task 1: Metric Change After Preprocessing',
                        'fig_task1_heatmap.png')

    # ── Detailed report for best model ──
    print(f"\n     Detailed Classification Report ({best_after}, After Preprocessing):")
    print(classification_report(splits['y_test_clf'], y_pred_best,
                                target_names=['Unpopular', 'Popular']))

    print("  TASK 1 COMPLETE")
    print("=" * 70)

    return {
        'before': before_df, 'after': after_df,
        'trained_before': trained_before, 'trained_after': trained_after,
        'best_model_name': best_after, 'best_group': best_group,
    }


# ══════════════════════════════════════════════════════════════════════
#  TASK 6: Recommending the Optimal Publication Window
# ══════════════════════════════════════════════════════════════════════

def run_task6_publication_window(df_full, target='shares'):
    """
    TASK 6: Recommending the Optimal Publication Window (Weekday vs. Weekend)
    Type: Supervised - Binary Classification

    Uses ONLY NLP + sentiment + channel features as input.
    Target is is_weekend (0 = weekday, 1 = weekend).
    """
    # Define input features (NLP, sentiment, channels only)
    nlp_features = [
        'global_subjectivity', 'global_sentiment_polarity',
        'global_rate_positive_words', 'global_rate_negative_words',
        'rate_positive_words', 'rate_negative_words',
        'avg_positive_polarity', 'min_positive_polarity', 'max_positive_polarity',
        'avg_negative_polarity', 'min_negative_polarity', 'max_negative_polarity',
        'title_subjectivity', 'title_sentiment_polarity',
        'abs_title_subjectivity', 'abs_title_sentiment_polarity',
    ]
    channel_features = [
        'data_channel_is_lifestyle', 'data_channel_is_entertainment',
        'data_channel_is_bus', 'data_channel_is_socmed',
        'data_channel_is_tech', 'data_channel_is_world',
    ]
    lda_features = ['LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04']

    input_features = nlp_features + channel_features + lda_features
    # Only keep features that exist in the dataframe
    input_features = [f for f in input_features if f in df_full.columns]

    print("\n" + "=" * 70)
    print("  TASK 6: Recommending the Optimal Publication Window")
    print("  Real-World Question: Given the tone and topic of an article,")
    print("    should it be published on a weekday or weekend?")
    print("  Type: Supervised - Binary Classification")
    print(f"  Input: {len(input_features)} NLP + sentiment + channel features")
    print("  Output: Publish on Weekday (0) / Publish on Weekend (1)")
    print("=" * 70)

    if 'is_weekend' not in df_full.columns:
        print("  ERROR: 'is_weekend' column not found. Skipping Task 6.")
        return None

    X = df_full[input_features].copy()
    y = df_full['is_weekend'].astype(int)

    print(f"\n  Class distribution: Weekday={sum(y==0)}, Weekend={sum(y==1)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Impute + Scale
    imputer = SimpleImputer(strategy='median')
    X_train_raw = pd.DataFrame(imputer.fit_transform(X_train),
                                columns=input_features, index=X_train.index)
    X_test_raw = pd.DataFrame(imputer.transform(X_test),
                               columns=input_features, index=X_test.index)

    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(scaler.fit_transform(X_train_raw),
                                 columns=input_features, index=X_train.index)
    X_test_proc = pd.DataFrame(scaler.transform(X_test_raw),
                                columns=input_features, index=X_test.index)

    # Use fewer models for this task
    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "SVM": SVC(random_state=SEED, probability=True, max_iter=5000),
    }

    # ── BEFORE Preprocessing ──
    print("\n  --- Results BEFORE Preprocessing ---")
    before_df, trained_before = evaluate_all_classifiers(
        {k: v for k, v in models.items()},
        X_train_raw, X_test_raw, y_train, y_test, label="Before"
    )
    print(before_df.to_string())

    # ── AFTER Preprocessing ──
    # Need fresh model instances
    models_after = {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "SVM": SVC(random_state=SEED, probability=True, max_iter=5000),
    }

    print("\n  --- Results AFTER Preprocessing ---")
    after_df, trained_after = evaluate_all_classifiers(
        models_after,
        X_train_proc, X_test_proc, y_train, y_test, label="After"
    )
    print(after_df.to_string())

    # ── Best Model ──
    best_name = after_df['Accuracy'].idxmax()
    print(f"\n     Best model: {best_name} (Acc: {after_df.loc[best_name, 'Accuracy']:.4f})")

    # ── Feature Importance for interpretation ──
    rf = trained_after.get("Random Forest")
    if rf is not None:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_feats = [input_features[i] for i in indices]
        top_imps = importances[indices]

        print(f"\n     Top 15 Features for Publication Window Prediction:")
        for i, (feat, imp) in enumerate(zip(top_feats, top_imps), 1):
            print(f"       {i:>2}. {feat:<45s} {imp:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = range(len(top_feats))
        ax.barh(y_pos, top_imps[::-1], color='coral', edgecolor='black', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title('Task 6: Features Predicting Weekday vs Weekend Publication',
                      fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        _save(fig, 'fig_task6_feature_importance.png')

    # ── ROC Curve ──
    plot_roc_curves(trained_after, X_test_proc, y_test,
                    'Task 6: ROC Curves - Publication Window',
                    'fig_task6_roc.png')

    _save_csv(after_df, "task6_publication_window.csv")

    print("\n  TASK 6 COMPLETE")
    print("=" * 70)

    return {'before': before_df, 'after': after_df, 'best_model': best_name}
