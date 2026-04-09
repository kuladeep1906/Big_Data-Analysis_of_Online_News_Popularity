
# Data Cleaning Module

import pandas as pd
import numpy as np


def _banner(title):
    """Print a section sub-banner."""
    print(f"\n  >> {title}")
    print(f"  {'─' * len(title)}")


def load_raw_data(filepath):
    """Load raw CSV and strip whitespace from column names."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(f"  Loaded raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def report_missing_values(df):
    """Report missing values per column."""
    _banner("Missing Values Check")
    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        print("  No missing values found in any column.")
    else:
        print(f"  Total missing values: {total_missing}")
        cols_with_missing = missing[missing > 0]
        for col, count in cols_with_missing.items():
            pct = count / len(df) * 100
            print(f"    {col}: {count} missing ({pct:.2f}%)")

    return total_missing


def report_duplicates(df):
    """Report and remove duplicate rows."""
    _banner("Duplicate Rows Check")
    n_duplicates = df.duplicated().sum()
    print(f"  Duplicate rows found: {n_duplicates}")

    if n_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  After removal: {df.shape[0]} rows remaining")

    return df, n_duplicates


def drop_non_predictive_columns(df):
    _banner("Dropping Non-Predictive Columns")
    dropped = []
    for col in ['url', 'timedelta']:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)
            print(f"  Dropped '{col}' — not a predictive feature")

    if not dropped:
        print("  No non-predictive columns to drop.")

    return df, dropped


def report_outliers(df, target='shares'):
    """Report outlier statistics for target variable using IQR"""
    _banner(f"Outlier Analysis on '{target}'")

    if target not in df.columns:
        print(f"  '{target}' column not found. Skipping outlier analysis.")
        return

    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = n_outliers / len(df) * 100

    print(f"  Q1: {q1:.2f}  |  Q3: {q3:.2f}  |  IQR: {iqr:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}  |  Upper bound: {upper_bound:.2f}")
    print(f"  Outliers found: {n_outliers} ({pct_outliers:.1f}% of data)")
    print(f"  Min: {df[target].min():.0f}  |  Max: {df[target].max():.0f}  |  Median: {df[target].median():.0f}")
    print(f"  NOTE: Outliers are NOT removed — shares are naturally right-skewed.")
    print(f"        We use log(1 + shares) transform for regression tasks instead.")


def report_data_types(df):
    """Report data types summary."""
    _banner("Data Types Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    binary_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1]).all()]

    print(f"  Total columns: {len(df.columns)}")
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Non-numeric columns: {len(non_numeric)}")
    print(f"  Binary indicator columns (0/1): {len(binary_cols)}")

    if len(non_numeric) > 0:
        print(f"  Non-numeric: {list(non_numeric)}")


def run_data_cleaning(filepath, target='shares'):
    
   # Full data cleaning pipeline with complete visibility.

    print("  PHASE 1, STEP 1: DATA CLEANING")

    # Load
    df = load_raw_data(filepath)
    shape_before = df.shape
    print(f"\n  Initial shape: {shape_before[0]} rows x {shape_before[1]} columns")

    # Missing values
    total_missing = report_missing_values(df)

    # Fill missing values if any
    if total_missing > 0:
        _banner("Imputing Missing Values")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(f"  Imputed {total_missing} missing values with column medians.")

    # Duplicates
    df, n_dupes = report_duplicates(df)

    # Drop non-predictive
    df, dropped_cols = drop_non_predictive_columns(df)

    # Outlier report (no removal)
    report_outliers(df, target)

    # Data types
    report_data_types(df)

    # Final summary
    shape_after = df.shape
    _banner("Data Cleaning Summary")
    print(f"  Shape BEFORE cleaning: {shape_before[0]} rows x {shape_before[1]} columns")
    print(f"  Shape AFTER  cleaning: {shape_after[0]} rows x {shape_after[1]} columns")
    print(f"  Missing values imputed: {total_missing}")
    print(f"  Duplicate rows removed: {n_dupes}")
    print(f"  Columns dropped: {dropped_cols}")
    print(f"  Outliers: reported but NOT removed (handled via log-transform)")
    print("=" * 70)

    return df
