# Big Data Course Project
## UCI Online News Popularity Dataset Analysis

This repository contains the complete analysis and machine learning pipeline for the Big Data course project using the UCI Online News Popularity dataset (ID: 332).

## 🚀 Quick Start

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
```

## Project Overview

The UCI Online News Popularity dataset contains 39,644 online news articles published by Mashable with around 60 numeric features related to content, sentiment, topics, and publication timing. The target variable is `shares`, representing the number of times an article was shared on social media.

The core script is **`report_2_task.py`**, which executes a comprehensive 12-step data science pipeline:

1. **Exploratory Data Analysis (EDA)** — Baseline distributions and summary statistics
2. **Preprocessing Assessment** — Evaluating the impact of log-transformation and outlier removal
3. **Data Preprocessing** — Handling missing values, scaling, and feature engineering
4. **Classification** — Predicting high vs. low popularity using Logistic Regression and Random Forest
5. **Regression** — Predicting exact share counts using Linear, Ridge, and Random Forest regressors
6. **Clustering** — Unsupervised grouping using K-Means (Elbow method/Silhouette) and DBSCAN
7. **Association Rule Mining** — Discovering itemsets that lead to high shares using Apriori
8. **Dimensionality Reduction** — Visualizing the high-dimensional space with PCA and t-SNE
9. **Temporal Analysis** — Analyzing engagement patterns on weekends vs. weekdays
10. **Feature Importance** — Identifying the strongest drivers of virality using Gini and Permutation importance
11. **Spark Pipeline** — Demonstrating distributed data processing using PySpark
12. **ML Pipeline Comparison** — A robust evaluation of 8 Classification and 7 Regression algorithms, tracking metrics before and after advanced preprocessing

## Repository Structure

```
.
├── data/
│   ├── full_dataset.csv             # Full dataset (39,644 rows)
│   └── subset.csv                   # 500-row deterministic subset for quick testing
├── figures/                         # Over 25 generated data visualizations
├── results/                         # Output CSVs for ML metric comparisons
├── src/
│   ├── preprocessing.py             # Data loading, cleaning, log-transforms, scaling
│   ├── eda.py                       # Advanced plotting for Data Distributions
│   ├── classification.py            # Base Classification tasks
│   ├── regression.py                # Base Regression tasks
│   ├── clustering.py                # K-Means + DBSCAN with PCA 2D scatter
│   ├── association_rules.py         # Apriori (min_support adaptive sizing)
│   ├── dimensionality_reduction.py  # PCA + t-SNE with variance ratios
│   ├── temporal_analysis.py         # Weekday/weekend comparative analysis
│   ├── feature_importance.py        # Random Forest Gini + Permutation importance
│   ├── spark_pipeline.py            # PySpark MLlib distributed equivalent
│   ├── ml_pipeline_comparison.py    # Cross-algorithm evaluation (Before vs After preprocessing)
│   └── ml_visualizations.py         # Advanced ML charts (ROC, Learning Curves, Heatmaps)
├── report_2_task.py                 # Unified full pipeline runner (Entry point)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Key Findings

- **Ensemble Dominance:** Non-linear ensemble algorithms (Random Forest, Gradient Boosting) consistently outperformed linear models, achieving the highest Classification Accuracy (~66%) and ROC-AUC.
- **Predictive Ceiling:** Exact regression prediction of social media shares is inherently difficult; all regression models achieved low R² scores close to zero.
- **Top Features:** Keyword metrics (`kw_avg_avg`, `kw_max_avg`) and topic relevance (LDA features) are the strongest predictors of share counts.
- **Temporal Patterns:** Articles published on weekends exhibit significantly higher average share counts (~3,903) compared to weekday articles (~3,319).
- **Association Rules:** Apriori mining revealed that weekend publication, tech channel designation, and Friday publication are meaningful indicators of high virality.

*(For full technical analysis, review the generated charts in the `figures/` directory and metrics in `results/` following execution).*

## Requirements

### System (install before Python packages)
| Requirement | Version | Install |
|---|---|---|
| **Java 17 LTS** | Temurin 17 | `brew install --cask temurin@17` |
| Python | 3.7+ | — |

> ⚠️ **Java 17 is strictly required for PySpark.** Newer versions of Java (21, 23+) remove internal APIs that PySpark depends on, resulting in `PySparkRuntimeError: [JAVA_GATEWAY_EXITED]`. 

After installing Java 17, set `JAVA_HOME` and persist it (macOS ZSH):
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
- ucimlrepo >= 0.0.7

## Usage

**Activate Virtual Environment:**
Always activate the virtual environment before running the project:
```bash
source .venv/bin/activate
```

**Execute Full Analysis Suite:**
To run the full end-to-end Big Data pipeline on the comprehensive dataset:
```bash
python report_2_task.py data/full_dataset.csv
```

**Output Generated:**
Execution of the pipeline will populate the following:
1. Console: Printed metric readouts and step-by-step progress logging
2. `figures/`: Dozens of high-resolution `.png` charts characterizing the data and models
3. `results/`: `.csv` data dumps for ML model comparisons outperforming basic metrics

## License
Created for University Big Data analysis coursework. Dataset is publicly available from the UCI Machine Learning Repository (CC BY 4.0).
