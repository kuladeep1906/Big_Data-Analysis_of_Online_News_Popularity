# Big Data Course Project

## UCI Online News Popularity Dataset Analysis

End-to-end analysis and machine learning for the Big Data course project using the [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) dataset (ID 332).

## Quick start

```bash
# 1. Install Java 17 LTS (required for PySpark)
brew install --cask temurin@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # add to ~/.zshrc to persist

# 2. Set up Python virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) Download data from UCI if you do not have a CSV yet
python download_full_dataset.py

# 4. Run the full analysis pipeline
python report_2_task.py data/full_dataset.csv
```

For a smaller reproducible sample (500 rows), generate `data/subset.csv` with `python create_subset.py` (from UCI) or `python create_subset.py data/full_dataset.csv` (from a local CSV).

## Project overview

The dataset has tens of thousands of online news articles (Mashable) with roughly 60 numeric features (content, sentiment, keywords, channels, timing). The main target is `shares` (social shares).

The entry point is **`report_2_task.py`**, which runs a **four-phase** pipeline:

### Phase 1 — Data preparation

1. **Data cleaning** — Load CSV, report missing values/duplicates/outliers, drop non-predictive columns, verbose logging (`src/data_cleaning.py`).
2. **EDA** — Summary statistics and six figures (histogram, box plot, correlation heatmap, scatter, weekday bar chart, raw vs log target) (`src/eda.py`).

### Phase 2 — Feature engineering (before model training)

3. **Feature selection** — Correlation-based reduction, cumulative importance, permutation importance (`src/feature_importance.py`).
4. **Dimensionality reduction** — PCA (variance + 2D projection) and t-SNE on selected features (`src/dimensionality_reduction.py`).

### Phase 3 — Six course tasks

| Task | Question | Approach |
|------|-----------|----------|
| **1** | Will the article be popular? | Classification (multiple models, before/after preprocessing) |
| **2** | How many shares? | Regression on log-transformed target |
| **3** | Natural groupings of articles? | K-Means (elbow/silhouette) + DBSCAN |
| **4** | Content patterns for high engagement? | Apriori association rules |
| **5** | Formatting and media usage vs engagement? | Regression-style analysis on engineered signals |
| **6** | Best publication window? | Classification using weekday/weekend and timing features |

Train/test splits and shared preprocessing live in **`src/preprocessing.py`**. Task-specific logic is in **`src/classification.py`**, **`src/regression.py`**, **`src/clustering.py`**, and **`src/association_rules.py`**.

### Phase 4 — Scalability

- **PySpark pipeline** — Distributed-style workflow via **`src/spark_pipeline.py`** (requires Java 17). If Java is missing or misconfigured, this step is skipped with a clear message.

## Outputs

After a successful run:

- **Console** — Step banners, metrics, and a short summary (including best models per task where applicable).
- **`figures/`** — Task-specific charts (e.g. ROC, confusion matrices, heatmaps, clustering, PCA/t-SNE, association-rule support/confidence plots) plus EDA figures.
- **`results/`** — CSVs such as `task1_classification_before.csv` / `task1_classification_after.csv`, `task2_regression_*.csv`, `task4_association_rules.csv`, `task5_formatting_results.csv`, `task6_publication_window.csv`, and consolidated `classification_*.csv` / `regression_*.csv` where generated.

## Requirements

### System (before `pip install`)

| Requirement | Notes |
|-------------|--------|
| **Java 17 LTS** | e.g. Temurin 17: `brew install --cask temurin@17` |
| **Python** | 3.8+ recommended (matches `pandas` / `scikit-learn` stack) |

Java 17 is required for **PySpark**. Newer Java versions (21+) can break PySpark with gateway errors; use 17 and set `JAVA_HOME`:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
```

### Python packages

See **`requirements.txt`**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `mlxtend`, `pyspark`, `ucimlrepo`.

## Usage

```bash
source .venv/bin/activate
python report_2_task.py data/full_dataset.csv
```

### Run with 500-row subset (precise steps)

```bash
# 1) Activate environment
source .venv/bin/activate

# 2) Create deterministic 500-row subset in data/subset.csv
# Option A: from local full dataset
python create_subset.py data/full_dataset.csv
# Option B: fetch from UCI and sample
python create_subset.py


# 4) RECOMMENDED: we have already run that. So please use the below command directly
python report_2_task.py data/subset.csv
```

