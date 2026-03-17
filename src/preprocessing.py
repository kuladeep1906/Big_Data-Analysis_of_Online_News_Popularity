"""
Data Preprocessing Module
- Load and clean data
- Drop highly correlated features
- Prepare data for classification (binarize target) and regression (log-transform target)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean(filepath):
    """Load CSV and strip whitespace from column names."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    # Drop non-predictive columns if present
    for col in ['url', 'timedelta']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def drop_correlated(df, threshold=0.9, target='shares'):
    """Drop one feature from each highly-correlated pair (|r| > threshold)."""
    features = df.drop(columns=[target], errors='ignore')
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    df = df.drop(columns=to_drop)
    return df, to_drop


def prepare_classification(df, target='shares', test_size=0.2, seed=42):
    """
    Binary classification: Popular (shares >= median) vs Unpopular.
    Returns scaled train/test splits.
    """
    median_val = df[target].median()
    y = (df[target] >= median_val).astype(int)
    X = df.drop(columns=[target])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, median_val


def prepare_regression(df, target='shares', test_size=0.2, seed=42):
    """
    Regression: log1p-transform the target to reduce skewness.
    Returns scaled train/test splits.
    """
    y = np.log1p(df[target])
    X = df.drop(columns=[target])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test, scaler
