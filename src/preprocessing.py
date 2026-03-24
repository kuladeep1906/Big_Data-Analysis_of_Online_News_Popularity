"""
Preprocessing Module
====================
Prepare data splits for classification and regression tasks.
Professor's Feedback #12: "Specify what is before and after model training."

This module provides:
  - Raw (before preprocessing) splits: only imputation, no scaling
  - Preprocessed (after preprocessing) splits: imputation + scaling
  - Both use the SELECTED features from Phase 2
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42


def prepare_splits(df, selected_features, target='shares', test_size=0.2):
    """
    Prepare train/test splits for both classification and regression.

    Returns a dict with raw (before preprocessing) and preprocessed (after) splits:
      - X_train_raw, X_test_raw: imputed only (no scaling)
      - X_train_proc, X_test_proc: imputed + scaled
      - y_train_clf, y_test_clf: binary labels
      - y_train_reg, y_test_reg: log1p(shares)
      - median_val: the threshold used for classification
      - scaler: the fitted StandardScaler
      - feature_names: list of feature names used
    """
    X = df[selected_features].copy()
    y_shares = df[target].copy()

    # Classification target: binary
    median_val = y_shares.median()
    y_clf = (y_shares >= median_val).astype(int)

    # Regression target: log transform
    y_reg = np.log1p(y_shares)

    # Single train/test split (same indices for fair comparison)
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=test_size, random_state=SEED, stratify=y_clf
    )

    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg = y_reg.loc[X_test.index]

    # --- BEFORE preprocessing: impute only ---
    imputer = SimpleImputer(strategy='median')
    X_train_raw = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=selected_features, index=X_train.index
    )
    X_test_raw = pd.DataFrame(
        imputer.transform(X_test),
        columns=selected_features, index=X_test.index
    )

    # --- AFTER preprocessing: impute + scale ---
    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=selected_features, index=X_train.index
    )
    X_test_proc = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=selected_features, index=X_test.index
    )

    print(f"\n  Data Preparation Summary:")
    print(f"    Selected features: {len(selected_features)}")
    print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"    Classification threshold (median shares): {median_val:.0f}")
    print(f"    Class balance: {y_train_clf.value_counts().to_dict()}")
    print(f"    Regression target: log(1 + shares)")

    return {
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train_proc': X_train_proc,
        'X_test_proc': X_test_proc,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'median_val': median_val,
        'scaler': scaler,
        'feature_names': selected_features,
    }
