#!/usr/bin/env python3
"""
Report 2 - Unified Entry Script
Big Data Course Project - UCI Online News Popularity Dataset

Usage:
    python report_2_task.py <location_of_data_file>

Pipeline (4 Phases):
  PHASE 1: DATA PREPARATION
    Step 1: Data Cleaning (explicit, verbose)
    Step 2: EDA (exploratory data analysis)

  PHASE 2: FEATURE ENGINEERING
    Step 3: Preliminary Feature Analysis & Selection (CRITICAL)
    Step 4: Dimensionality Reduction (PCA, t-SNE on selected features)

  PHASE 3: TASK EXECUTION (6 Named Tasks)
    Task 1: Predicting Whether a News Article Will Be Popular (Classification)
    Task 2: Predicting the Number of Shares (Regression)
    Task 3: Discovering Natural Groupings of News Articles (Clustering)
    Task 4: Identifying Content Patterns for High Engagement (Association Rules)
    Task 5: Optimizing Article Formatting and Media Usage (Regression)
    Task 6: Recommending the Optimal Publication Window (Classification)

  PHASE 4: SCALABILITY
    Spark Pipeline (distributed ML demonstration)
"""

import sys
import os
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

from src.data_cleaning import run_data_cleaning
from src.eda import run_eda
from src.feature_importance import run_feature_selection
from src.dimensionality_reduction import run_dimensionality_reduction
from src.preprocessing import prepare_splits
from src.classification import run_task1_popularity_classification, run_task6_publication_window
from src.regression import run_task2_shares_regression, run_task5_formatting_optimization
from src.clustering import run_task3_clustering
from src.association_rules import run_task4_association_rules
from src.spark_pipeline import run_spark_pipeline


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

    print("\n" + "#" * 70)
    print("#  ONLINE NEWS POPULARITY ANALYSIS")
    print("#  Big Data Course Project - Report 2")
    print("#" * 70)

    # ================================================================== #
    #  PHASE 1: DATA PREPARATION
    # ================================================================== #
    print("\n\n" + "#" * 70)
    print("#  PHASE 1: DATA PREPARATION")
    print("#" * 70)

    # Step 1: Data Cleaning
    df = run_data_cleaning(filepath)

    # Step 2: EDA
    run_eda(df)

    # ================================================================== #
    #  PHASE 2: FEATURE ENGINEERING (BEFORE any model training)
    # ================================================================== #
    print("\n\n" + "#" * 70)
    print("#  PHASE 2: FEATURE ENGINEERING")
    print("#  (This happens BEFORE any model training)")
    print("#" * 70)

    # Step 3: Feature Selection (CRITICAL - Professor's Feedback #8)
    selected_features, df_reduced, n_selected, df_after_corr = run_feature_selection(df)

    # Step 4: Dimensionality Reduction (on selected features only)
    run_dimensionality_reduction(df_reduced, selected_features)

    # ================================================================== #
    #  PHASE 3: TASK EXECUTION (6 Named Tasks)
    # ================================================================== #
    print("\n\n" + "#" * 70)
    print("#  PHASE 3: TASK EXECUTION (6 Named Tasks)")
    print("#  Each task clearly states: Name, Type, Input, Output")
    print("#  Supervised tasks show: Before/After Preprocessing results")
    print("#" * 70)

    # Prepare data splits using selected features
    splits = prepare_splits(df_reduced, selected_features)

    # TASK 1: Predicting Whether a News Article Will Be Popular
    task1_results = run_task1_popularity_classification(splits)

    # TASK 2: Predicting the Number of Shares
    task2_results = run_task2_shares_regression(splits)

    # TASK 3: Discovering Natural Groupings of News Articles
    task3_km_labels, task3_db_labels = run_task3_clustering(
        df_reduced, selected_features
    )

    # TASK 4: Identifying Content Patterns for High Engagement
    task4_rules = run_task4_association_rules(df_after_corr)

    # TASK 5: Optimizing Article Formatting and Media Usage
    task5_results = run_task5_formatting_optimization(df_after_corr)

    # TASK 6: Recommending the Optimal Publication Window
    task6_results = run_task6_publication_window(df_after_corr)

    # ================================================================== #
    #  PHASE 4: SCALABILITY
    # ================================================================== #
    print("\n\n" + "#" * 70)
    print("#  PHASE 4: SCALABILITY DEMONSTRATION")
    print("#" * 70)

    spark_results = run_spark_pipeline(filepath)

    # ================================================================== #
    #  SUMMARY
    # ================================================================== #
    print("\n\n" + "#" * 70)
    print("#  ALL TASKS COMPLETE - SUMMARY")
    print("#" * 70)

    print(f"""
  Pipeline Summary:
  ─────────────────
  Phase 1: Data Preparation
    - Data cleaning: {df.shape[0]} rows, {df.shape[1]} columns
    - EDA: 6 visualizations saved

  Phase 2: Feature Engineering
    - Feature selection: {n_selected} features selected (data-driven)
    - Dimensionality reduction: PCA + t-SNE on selected features

  Phase 3: 6 Named Tasks
    - Task 1: Popularity Classification    (Best: {task1_results['best_model_name']})
    - Task 2: Shares Regression            (Best: {task2_results['best_model_name']})
    - Task 3: Article Clustering           (K-Means + DBSCAN)
    - Task 4: Engagement Patterns          ({len(task4_rules)} rules found)
    - Task 5: Formatting Optimization      (Ridge/Lasso/RF/GB analysis)
    - Task 6: Publication Window           (Best: {task6_results['best_model'] if task6_results else 'N/A'})

  Phase 4: Scalability
    - Spark Pipeline: {'Completed' if spark_results else 'Skipped (Java not available)'}

  All figures saved to: figures/
  All CSV results saved to: results/
""")


if __name__ == "__main__":
    main()
