#!/usr/bin/env python3
"""
Report — Unified Entry Script
Big Data Course Project — UCI Online News Popularity Dataset

Usage:
    python report_2_task.py <location_of_data_file>

This script runs the ENTIRE experimental pipeline in a single execution:
  0.  EDA — Statistical table (Range, Mean, Mode, Std, Median) + 5 raw plots
  1.  Preprocessing (correlation removal, scaling)
  1b. Post-Preprocessing EDA — 4 before/after comparison plots
  2.  Classification (Logistic Regression, Random Forest)
  3.  Regression (Linear, Ridge, Random Forest)
  4.  Clustering (K-Means, DBSCAN)
  5.  Association Rule Mining (Apriori)
  6.  Dimensionality Reduction (PCA, t-SNE)
  7.  Temporal Trend Analysis
  8.  Feature Importance (Gini, Permutation)
  9.  Spark Pipeline (PySpark Random Forest)
  10. ML Pipeline Comparison (Before vs After Preprocessing)
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.eda import run_eda, run_post_preprocessing_eda
from src.preprocessing import load_and_clean, drop_correlated, prepare_classification, prepare_regression
from src.classification import run_classification
from src.regression import run_regression
from src.clustering import run_clustering
from src.association_rules import run_association_rules
from src.dimensionality_reduction import run_dimensionality_reduction
from src.temporal_analysis import run_temporal_analysis
from src.feature_importance import run_feature_importance
from src.spark_pipeline import run_spark_pipeline
from src.ml_pipeline_comparison import run_comparison_experiment
from src.ml_visualizations import run_comparison_visualizations


def banner(title):
    """Print a clear section banner."""
    print("\n")
    print("=" * 60)
    print(f"  ===== {title} =====")
    print("=" * 60)


def main():
    # ------------------------------------------------------------------ #
    #  Argument handling
    # ------------------------------------------------------------------ #
    if len(sys.argv) != 2:
        print("Usage: python report_2_task.py <location_of_data_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Error: file '{filepath}' does not exist.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  STEP 0 — EDA (Preliminary Analysis, formerly report_1.py)
    # ------------------------------------------------------------------ #
    banner("EDA — PRELIMINARY ANALYSIS")

    # Load raw data (with url/timedelta stripped, no other changes)
    df_raw = load_and_clean(filepath)
    run_eda(df_raw)

    # ------------------------------------------------------------------ #
    #  STEP 1 & 2 — PREPROCESSING
    # ------------------------------------------------------------------ #
    banner("PREPROCESSING")

    df = load_and_clean(filepath)
    df, dropped = drop_correlated(df)
    print(f"Loaded : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Dropped {len(dropped)} collinear features: {dropped}")

    # Classification preparation
    X_train_c, X_test_c, y_train_c, y_test_c, scaler_c, median_val = \
        prepare_classification(df)
    print(f"Classification threshold (median shares): {median_val:.0f}")
    print(f"Train size: {len(X_train_c)} | Test size: {len(X_test_c)}")

    # Regression preparation
    X_train_r, X_test_r, y_train_r, y_test_r, scaler_r = \
        prepare_regression(df)
    print(f"Regression train size: {len(X_train_r)} | Test size: {len(X_test_r)}")

    # ------------------------------------------------------------------ #
    #  STEP 1b — POST-PREPROCESSING EDA (before/after comparison plots)
    # ------------------------------------------------------------------ #
    banner("POST-PREPROCESSING EDA — BEFORE vs AFTER")
    run_post_preprocessing_eda(df_raw, df)

    # ------------------------------------------------------------------ #
    #  STEP 3 — CLASSIFICATION
    # ------------------------------------------------------------------ #
    banner("CLASSIFICATION")
    clf_models = run_classification(X_train_c, X_test_c, y_train_c, y_test_c)

    # ------------------------------------------------------------------ #
    #  STEP 4 — REGRESSION
    # ------------------------------------------------------------------ #
    banner("REGRESSION")
    reg_models = run_regression(X_train_r, X_test_r, y_train_r, y_test_r)

    # ------------------------------------------------------------------ #
    #  STEP 5 — CLUSTERING
    # ------------------------------------------------------------------ #
    banner("CLUSTERING")
    km_labels, db_labels = run_clustering(df)

    # ------------------------------------------------------------------ #
    #  STEP 6 — ASSOCIATION RULES
    # ------------------------------------------------------------------ #
    banner("ASSOCIATION RULES")
    rules = run_association_rules(df)

    # ------------------------------------------------------------------ #
    #  STEP 7 — DIMENSIONALITY REDUCTION
    # ------------------------------------------------------------------ #
    banner("DIMENSIONALITY REDUCTION")
    X_pca = run_dimensionality_reduction(df)

    # ------------------------------------------------------------------ #
    #  STEP 8 — TEMPORAL ANALYSIS
    # ------------------------------------------------------------------ #
    banner("TEMPORAL ANALYSIS")
    run_temporal_analysis(df)

    # ------------------------------------------------------------------ #
    #  STEP 9 — FEATURE IMPORTANCE
    # ------------------------------------------------------------------ #
    banner("FEATURE IMPORTANCE")
    run_feature_importance(df)

    # ------------------------------------------------------------------ #
    #  STEP 10 — SPARK PIPELINE
    # ------------------------------------------------------------------ #
    banner("SPARK PIPELINE")
    spark_results = run_spark_pipeline(filepath)
    if spark_results:
        print(f"\nSpark Accuracy : {spark_results['accuracy']:.4f}")
        print(f"Spark ROC AUC  : {spark_results['roc_auc']:.4f}")

    # ------------------------------------------------------------------ #
    #  STEP 11 — ML PIPELINE COMPARISON (Before vs After Preprocessing)
    # ------------------------------------------------------------------ #
    banner("ML PIPELINE COMPARISON")

    # Reload clean data (without the earlier drop_correlated, because the
    # comparison module applies its own preprocessing pipeline internally)
    df_clean = load_and_clean(filepath)
    comparison_results = run_comparison_experiment(df_clean)

    # Generate all comparison visualizations
    run_comparison_visualizations(comparison_results)

    # ------------------------------------------------------------------ #
    #  DONE
    # ------------------------------------------------------------------ #
    banner("ALL TASKS COMPLETE")
    print("  All figures saved to figures/")
    print("  All CSV results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
