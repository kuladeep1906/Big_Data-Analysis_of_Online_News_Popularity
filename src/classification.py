"""
Classification Module
- Logistic Regression (baseline)
- Random Forest Classifier
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix plots
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)

FIGURES_DIR = "figures"


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest_clf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test, name="Model"):
    """Evaluate and print classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    report = classification_report(y_test, y_pred, target_names=['Unpopular', 'Popular'])

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC  : {auc:.4f}")
    print(f"\n{report}")
    return {'accuracy': acc, 'roc_auc': auc, 'y_pred': y_pred}


def plot_confusion_matrix(y_test, y_pred, name="Model"):
    """Save a confusion-matrix heatmap."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unpopular', 'Popular'],
                yticklabels=['Unpopular', 'Popular'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {name}', fontweight='bold')
    safe = name.lower().replace(" ", "_")
    path = os.path.join(FIGURES_DIR, f'fig_cm_{safe}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_classification(X_train, X_test, y_train, y_test):
    """Run full classification pipeline."""
    print("\n" + "=" * 60)
    print("  CLASSIFICATION — Popular vs Unpopular")
    print("=" * 60)

    models = {}
    for trainer, name in [(train_logistic_regression, "Logistic Regression"),
                          (train_random_forest_clf, "Random Forest Classifier")]:
        m = trainer(X_train, y_train)
        metrics = evaluate_classifier(m, X_test, y_test, name)
        plot_confusion_matrix(y_test, metrics['y_pred'], name)
        models[name] = m

    return models
