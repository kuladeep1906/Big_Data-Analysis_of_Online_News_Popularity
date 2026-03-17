"""
ML Pipeline Comparison Module
=============================
Compare model performance BEFORE and AFTER preprocessing.

Models — Classification (8):
    Logistic Regression, KNN, SVM, Decision Tree, Random Forest,
    Gradient Boosting, AdaBoost, Naive Bayes

Models — Regression (7):
    Linear Regression, Ridge, Lasso, Decision Tree Regressor,
    Random Forest Regressor, Gradient Boosting Regressor, SVR
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score,
                              mean_squared_error, mean_absolute_error, r2_score)

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

RESULTS_DIR = "results"
SEED = 42


#  MODEL DICTIONARIES

def get_classification_models():
    """Return a dict of name → classifier (all with random_state where supported)."""
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


def get_regression_models():
    """Return a dict of name → regressor (all with random_state where supported)."""
    return {
        "Linear Regression":        LinearRegression(),
        "Ridge Regression":         Ridge(alpha=1.0, random_state=SEED),
        "Lasso Regression":         Lasso(alpha=0.01, random_state=SEED, max_iter=5000),
        "Decision Tree Regressor":  DecisionTreeRegressor(random_state=SEED),
        "Random Forest Regressor":  RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        "SVR":                      SVR(max_iter=5000),
    }



#  PREPROCESSING  (fit on TRAIN only — no data leakage)


def _remove_high_correlation(X_train, X_test, threshold=0.9):
    """
    Drop one feature from every pair with |r| > threshold.
    Correlation matrix is computed on X_train ONLY.
    """
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_train = X_train.drop(columns=to_drop)
    X_test  = X_test.drop(columns=to_drop)
    return X_train, X_test, to_drop


def preprocess_pipeline(X_train, X_test):
    
    #Full preprocessing pipeline 

    info = {}

    # 1. Missing value Imputation (medain)
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=X_train.columns, index=X_train.index)
    X_test  = pd.DataFrame(imputer.transform(X_test),
                           columns=X_test.columns, index=X_test.index)
    info["missing_imputed"] = int(imputer.statistics_.size)

    # 2. Remove high-correlation features (|r| > 0.9)
    X_train, X_test, dropped_corr = _remove_high_correlation(X_train, X_test, threshold=0.9)
    info["dropped_corr"] = dropped_corr

    # 3. Remove low-variance features (VarianceThreshold)
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_train)
    kept_cols = X_train.columns[selector.get_support()]
    X_train = X_train[kept_cols]
    X_test  = X_test[kept_cols]
    info["dropped_low_var"] = int(X_train.shape[1])  # features remaining

    # 4. StandardScaler normalisation— fit on train
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr  = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train_arr, columns=kept_cols, index=X_train.index)
    X_test  = pd.DataFrame(X_test_arr,  columns=kept_cols, index=X_test.index)
    info["scaler"] = scaler

    return X_train, X_test, info



#  EVALUATION HELPERS


def evaluate_classifiers(models, X_train, X_test, y_train, y_test, run_cv=True):
    """
    Train each classifier, return a DataFrame of metrics.

    Columns: Accuracy, Precision, Recall, F1, ROC_AUC, CV_Accuracy_Mean, Train_Time_s
    """
    rows = []
    trained = {}  # store fitted models for ROC plotting later

    for name, model in models.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)

        # ROC AUC — need probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        # Optional 5-fold cross-validation
        cv_mean = np.nan
        if run_cv:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(model, X_train, y_train,
                                                cv=5, scoring="accuracy", n_jobs=-1)
                    cv_mean = cv_scores.mean()
            except Exception:
                cv_mean = np.nan

        rows.append({
            "Model":            name,
            "Accuracy":         round(acc, 4),
            "Precision":        round(prec, 4),
            "Recall":           round(rec, 4),
            "F1":               round(f1, 4),
            "ROC_AUC":          round(auc, 4) if not np.isnan(auc) else np.nan,
            "CV_Accuracy_Mean": round(cv_mean, 4) if not np.isnan(cv_mean) else np.nan,
            "Train_Time_s":     round(train_time, 3),
        })
        trained[name] = model

    df_results = pd.DataFrame(rows).set_index("Model")
    return df_results, trained


def evaluate_regressors(models, X_train, X_test, y_train, y_test):
    """
    Train each regressor, return a DataFrame of metrics.

    Columns: RMSE, MAE, R2, Train_Time_s
    """
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
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        rows.append({
            "Model":       name,
            "RMSE":        round(rmse, 4),
            "MAE":         round(mae, 4),
            "R2":          round(r2, 4),
            "Train_Time_s": round(train_time, 3),
        })
        trained[name] = model

    df_results = pd.DataFrame(rows).set_index("Model")
    return df_results, trained


# ──────────────────────────────────────────────────────────────
#  COMPARISON TABLE HELPERS
# ──────────────────────────────────────────────────────────────

def _build_comparison_table(df_before, df_after, metric_cols):
    """
    Merge before/after DataFrames and add Δ columns for each metric.
    """
    merged = df_before[metric_cols].add_suffix("_Before").join(
             df_after[metric_cols].add_suffix("_After"), how="outer")

    for col in metric_cols:
        before_col = f"{col}_Before"
        after_col  = f"{col}_After"
        merged[f"Δ_{col}"] = merged[after_col] - merged[before_col]

    # Reorder: Before, After, Δ for each metric
    ordered = []
    for col in metric_cols:
        ordered += [f"{col}_Before", f"{col}_After", f"Δ_{col}"]
    return merged[ordered]


def _save_csv(df, filename):
    """Save DataFrame to results/ directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────

def run_comparison_experiment(df, target="shares"):
    """
    Run the full before/after preprocessing comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with 'shares' column.

    Returns
    -------
    dict with keys: clf_before, clf_after, reg_before, reg_after,
                    clf_comparison, reg_comparison,
                    trained_clf_before, trained_clf_after,
                    trained_reg_before, trained_reg_after,
                    X_test_clf_before, y_test_clf,
                    X_test_clf_after, preprocess_info
    """
    print("\n" + "=" * 70)
    print("  ML PIPELINE COMPARISON — Before vs After Preprocessing")
    print("=" * 70)

    # ── Prepare targets ──────────────────────────────────────
    median_val = df[target].median()
    y_clf = (df[target] >= median_val).astype(int)
    y_reg = np.log1p(df[target])
    X = df.drop(columns=[target])

    print(f"\n  Dataset shape    : {X.shape}")
    print(f"  Classification   : popular = 1 if shares ≥ {median_val:.0f} (median)")
    print(f"  Regression target: log(1 + shares)")
    print(f"  Class balance    : {y_clf.value_counts().to_dict()}")

    # ── Train / test split (SAME split for both experiments) ─
    X_train_raw, X_test_raw, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf
    )
    # Regression uses same row indices
    y_train_reg = y_reg.loc[X_train_raw.index]
    y_test_reg  = y_reg.loc[X_test_raw.index]

    print(f"  Train size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")

    # ══════════════════════════════════════════════════════════
    #  EXPERIMENT 1 — BEFORE preprocessing (baseline)
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  EXPERIMENT 1: WITHOUT Preprocessing (Baseline)")
    print("─" * 70)

    # For baseline, handle NaNs minimally (impute) so models don't crash
    imp_base = SimpleImputer(strategy="median")
    X_train_base = pd.DataFrame(imp_base.fit_transform(X_train_raw),
                                columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_base  = pd.DataFrame(imp_base.transform(X_test_raw),
                                columns=X_test_raw.columns, index=X_test_raw.index)

    print("\n  ▸ Classification (8 models)...")
    clf_before, trained_clf_before = evaluate_classifiers(
        get_classification_models(), X_train_base, X_test_base, y_train_clf, y_test_clf
    )
    print(clf_before.to_string())

    print("\n  ▸ Regression (7 models)...")
    reg_before, trained_reg_before = evaluate_regressors(
        get_regression_models(), X_train_base, X_test_base, y_train_reg, y_test_reg
    )
    print(reg_before.to_string())

    # ══════════════════════════════════════════════════════════
    #  EXPERIMENT 2 — AFTER preprocessing
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  EXPERIMENT 2: WITH Preprocessing")
    print("─" * 70)

    X_train_proc, X_test_proc, preprocess_info = preprocess_pipeline(
        X_train_raw.copy(), X_test_raw.copy()
    )
    dropped = preprocess_info.get("dropped_corr", [])
    print(f"  Dropped {len(dropped)} highly correlated features")
    print(f"  Features remaining after preprocessing: {X_train_proc.shape[1]}")
    print("  NOTE: Scaling is applied to all models. Tree-based models (Decision Tree,")
    print("        Random Forest, Gradient Boosting, AdaBoost) do not require scaling,")
    print("        but it does not negatively affect their performance.")

    print("\n  ▸ Classification (8 models)...")
    clf_after, trained_clf_after = evaluate_classifiers(
        get_classification_models(), X_train_proc, X_test_proc, y_train_clf, y_test_clf
    )
    print(clf_after.to_string())

    print("\n  ▸ Regression (7 models)...")
    reg_after, trained_reg_after = evaluate_regressors(
        get_regression_models(), X_train_proc, X_test_proc, y_train_reg, y_test_reg
    )
    print(reg_after.to_string())

    # ══════════════════════════════════════════════════════════
    #  COMPARISON TABLES
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COMPARISON TABLES — Before vs After Preprocessing")
    print("=" * 70)

    clf_comparison = _build_comparison_table(
        clf_before, clf_after, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    )
    reg_comparison = _build_comparison_table(
        reg_before, reg_after, ["RMSE", "MAE", "R2"]
    )

    print("\n  ▸ Classification Comparison:")
    print(clf_comparison.to_string())

    print("\n  ▸ Regression Comparison:")
    print(reg_comparison.to_string())

    # ── Save to CSV ──────────────────────────────────────────
    print("\n  Saving results to CSV...")
    _save_csv(clf_before,     "classification_before.csv")
    _save_csv(clf_after,      "classification_after.csv")
    _save_csv(reg_before,     "regression_before.csv")
    _save_csv(reg_after,      "regression_after.csv")
    _save_csv(clf_comparison, "classification_comparison.csv")
    _save_csv(reg_comparison, "regression_comparison.csv")

    return {
        "clf_before":           clf_before,
        "clf_after":            clf_after,
        "reg_before":           reg_before,
        "reg_after":            reg_after,
        "clf_comparison":       clf_comparison,
        "reg_comparison":       reg_comparison,
        "trained_clf_before":   trained_clf_before,
        "trained_clf_after":    trained_clf_after,
        "trained_reg_before":   trained_reg_before,
        "trained_reg_after":    trained_reg_after,
        "X_test_clf_before":    X_test_base,
        "X_test_clf_after":     X_test_proc,
        "y_test_clf":           y_test_clf,
        "feature_names_after":  list(X_train_proc.columns),
        "preprocess_info":      preprocess_info,
    }
