# Big Data Course Project
## UCI Online News Popularity Dataset Analysis

This repository contains the complete analysis and machine learning pipeline for the Big Data course project using the UCI Online News Popularity dataset (ID: 332).

## Quick Start

```bash
# 1. Install Java 17 LTS (required for PySpark)
brew install --cask temurin@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # add to ~/.zshrc to persist

# 2. Set up Python virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the complete analysis pipeline
python report_2_task.py data/full_dataset.csv

# 4. Run on the ~500 sample subset (for quick testing / professor grading)
python report_2_task.py data/subset.csv
```

## How to Run

```bash
python report_2_task.py <location_of_data_file>
```

The script runs the entire task-oriented pipeline end-to-end and prints all results to the console. Figures are saved to `figures/` and CSV results to `results/`.

## Project Overview

The UCI Online News Popularity dataset contains 39,644 online news articles published by Mashable with 58 numeric features related to content, sentiment, topics, and publication timing. The target variable is `shares`, representing the number of times an article was shared on social media.

The analysis is organized around **six specific tasks**, each addressing a real-world question:

| Task | Name | Type |
|------|------|------|
| 1 | Predicting Whether a News Article Will Be Popular | Supervised - Binary Classification |
| 2 | Predicting the Number of Shares an Article Will Receive | Supervised - Regression |
| 3 | Discovering Natural Groupings of News Articles | Unsupervised - Clustering |
| 4 | Identifying Content Patterns Associated with High Engagement | Unsupervised - Association Rules |
| 5 | Optimizing Article Formatting and Media Usage | Supervised - Regression |
| 6 | Recommending the Optimal Publication Window | Supervised - Binary Classification |
| -- | Spark-Based Scalability Demonstration | PySpark Random Forest |

### Pipeline Execution Order

1. **Data Cleaning** - Missing values, duplicates, outlier analysis
2. **Exploratory Data Analysis** - Distributions, correlations, temporal patterns
3. **Feature Engineering** - Correlation filtering, RF importance, select top 29 features
4. **Dimensionality Reduction** - PCA and t-SNE on selected features
5. **Task 1** - Popularity classification (8 models, before/after preprocessing)
6. **Task 2** - Share count regression (7 models, before/after preprocessing)
7. **Task 3** - Clustering (K-Means + DBSCAN)
8. **Task 4** - Association rule mining (Apriori)
9. **Task 5** - Formatting analysis (4 regression models)
10. **Task 6** - Publication timing classification (4 models)
11. **Spark Pipeline** - PySpark Random Forest scalability demo

## Repository Structure

```
.
├── data/
│   ├── full_dataset.csv             # Full dataset (39,644 rows)
│   └── subset.csv                   # ~500-row subset for quick testing
├── figures/                         # Generated data visualizations
├── results/                         # Output CSVs for ML metric comparisons
├── report_2/                        # LaTeX report source files
│   ├── main.tex                     # Main document
│   ├── sections/                    # Report sections
│   └── figures/                     # Report-specific figures
├── src/
│   ├── data_cleaning.py             # Explicit data cleaning with full visibility
│   ├── preprocessing.py             # Train/test splits, before/after preprocessing
│   ├── eda.py                       # Exploratory data analysis and plots
│   ├── feature_importance.py        # RF importance, feature selection (90% threshold)
│   ├── dimensionality_reduction.py  # PCA + t-SNE on selected features
│   ├── classification.py            # Task 1 + Task 6 classification
│   ├── regression.py                # Task 2 + Task 5 regression
│   ├── clustering.py                # Task 3: K-Means + DBSCAN
│   ├── association_rules.py         # Task 4: Apriori rules
│   ├── temporal_analysis.py         # Weekday/weekend analysis
│   └── spark_pipeline.py            # PySpark MLlib pipeline
├── report_2_task.py                 # Main entry point
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

### System (install before Python packages)
| Requirement | Version | Install |
|---|---|---|
| **Java 17 LTS** | Temurin 17 | `brew install --cask temurin@17` |
| Python | 3.7+ | --- |

After installing Java 17, set `JAVA_HOME`:
```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
```

### Python packages (installed via pip)
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- mlxtend >= 0.23.0
- pyspark >= 3.4.0

## License
Created for University Big Data analysis coursework. Dataset is publicly available from the UCI Machine Learning Repository (CC BY 4.0).
