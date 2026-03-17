# Big Data Course Project
## UCI Online News Popularity Dataset Analysis

This repository contains the complete analysis pipeline for the Big Data course project using the UCI Online News Popularity dataset (ID: 332).

## 🚀 Quick Start

```bash
# 1. Install Java 17 LTS (required for PySpark)
brew install --cask temurin@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # add to ~/.zshrc to persist

# 2. Set up Python virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run EDA on subset (Report 1)
python report_1.py data/subset.csv

# 4. Run full pipeline — single command
python report_2_task.py data/full_dataset.csv
# Or on the 500-row subset (faster, for professor testing):
python report_2_task.py data/subset.csv
```

## Project Overview

The UCI Online News Popularity dataset contains approximately 39,644 online news articles published by Mashable with around 60 numeric features related to content, sentiment, topics, and publication timing. The target variable is `shares`, representing the number of times an article was shared on social media.

The project is split into two entry-point scripts:
- **`report_1.py`** — EDA and visualization on the 500-row subset
- **`report_2_task.py`** — Full pipeline: Classification, Regression, Clustering, Association Rules, Dimensionality Reduction, Temporal Analysis, Feature Importance, and PySpark

## Dataset Information

- **Source**: UCI Machine Learning Repository
- **Dataset ID**: 332
- **Rows**: ~39,000 articles
- **Features**: ~60 numeric attributes
- **Target**: `shares` (number of social media shares)

## Repository Structure

```
.
├── data/
│   ├── full_dataset.csv        # Full dataset (~39,644 rows)
│   └── subset.csv              # 500-row deterministic subset (seed=42)
├── figures/                    # All generated visualizations
├── src/
│   ├── preprocessing.py        # Data loading, cleaning, train/test split
│   ├── classification.py       # Logistic Regression + Random Forest
│   ├── regression.py           # Linear, Ridge, Random Forest Regressor
│   ├── clustering.py           # K-Means + DBSCAN
│   ├── association_rules.py    # Apriori (mlxtend)
│   ├── dimensionality_reduction.py  # PCA + t-SNE
│   ├── temporal_analysis.py    # Weekday/weekend analysis
│   ├── feature_importance.py   # Gini + Permutation importance
│   └── spark_pipeline.py       # PySpark MLlib pipeline
├── report_1.py                 # EDA and visualization (subset)
├── report_2_task.py            # Unified full pipeline runner
├── create_subset.py            # Generates the 500-row subset
├── download_full_dataset.py    # Downloads full dataset from UCI
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

### System (install before Python packages)
| Requirement | Version | Install |
|---|---|---|
| **Java 17 LTS** | Temurin 17 | `brew install --cask temurin@17` |
| Python | 3.7+ | — |

> ⚠️ **Java 17 is required for PySpark.** Java 23+ removes internal APIs that PySpark depends on and will cause a runtime error. Do **not** use Java 21, 23, or 25.

After installing Java 17, set `JAVA_HOME` and persist it:
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

## Installation

1. **Install Java 17 LTS:**
```bash
brew install --cask temurin@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
```

2. **Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** Always activate the virtual environment before running scripts:
```bash
source .venv/bin/activate  # Run this each time you open a new terminal
```

## Usage

### Run Report 1 (EDA on subset)
```bash
source .venv/bin/activate
python report_1.py data/subset.csv
```

### Run the full pipeline (single command)
```bash
source .venv/bin/activate

# Full dataset:
python report_2_task.py data/full_dataset.csv

# 500-row subset (professor testing):
python report_2_task.py data/subset.csv
```

`report_2_task.py` runs all 10 tasks in order: Preprocessing → Classification → Regression → Clustering → Association Rules → Dimensionality Reduction → Temporal Analysis → Feature Importance → Spark Pipeline.

### Output

The script will:
1. Print an "ANALYSIS" section with computed metrics in a table format
2. **Display** 5 visualization windows (one at a time - close each to see the next)
3. **Save** 5 visualization figures to the `figures/` folder (PNG files with 200+ DPI)

**Metrics Computed:**
- **Range**: [min, max] for numeric columns
- **Mean**: Arithmetic mean (numeric only)
- **Mode**: Most frequent value, smallest in case of ties
- **M_a (Standard Deviation)**: Standard deviation (numeric only)
- **M_b (Median)**: Median value (numeric only)

**Visualizations Generated:**
1. `figures/fig1_hist_shares.png` - Histogram of shares
2. `figures/fig2_boxplot_shares.png` - Box plot of shares
3. `figures/fig3_correlation_heatmap.png` - Correlation heatmap of numeric features
4. `figures/fig4_scatter_plot.png` - Scatter plot: n_tokens_content vs shares
5. `figures/fig5_bar_chart.png` - Bar chart: average shares by weekday

### Downloading the Full Dataset

If you need to download the full dataset:

```bash
python download_full_dataset.py
```

This will download ~39,644 rows and save to `data/full_dataset.csv`.

### Creating a New Subset (Optional)

If you need to regenerate the subset for testing:

```bash
# Download from UCI and create subset
python create_subset.py

# Or create subset from a local CSV file
python create_subset.py path/to/full_dataset.csv
```

The subset uses a fixed random seed (42) for reproducibility.

## Script Details

### report_1.py

The script is organized into clear functions:

- **`load_data(filepath)`**: Loads the CSV file
- **`compute_metrics(df)`**: Computes all statistical metrics
- **`make_plots(df)`**: Generates all 5 visualizations
- **`main()`**: Orchestrates the analysis

**Robustness Features:**
- Handles missing columns gracefully
- Provides fallback column selection
- Handles constant columns (range = 0)
- Manages mode ties by selecting the smallest value
- No debug output (clean ANALYSIS section only)

### create_subset.py

Helper script to create the deterministic 500-row subset:

- Attempts to download from UCI using `ucimlrepo`
- Falls back to local CSV if provided
- Uses fixed random seed (42) for reproducibility
- Saves to `data/subset.csv`

```bash
python create_subset.py data/full_dataset.csv
```

## Example Output

```
ANALYSIS
                url  timedelta  n_tokens_title  ...  abs_title_sentiment_polarity  shares
Range           N/A  [17.0, 728.0]  [5.0, 15.0]  ...  [0.0, 1.0]                   [180, 29900]
Mean            N/A  360.0000       10.3200      ...  0.1750                       3175.3600
Mode            ...  48.0000        11.0000      ...  0.0000                       1200
M_a(std)        N/A  236.1286       2.0835       ...  0.2562                       4945.0266
M_b(median)     N/A  318.0000       11.0000      ...  0.0292                       1350.0000

Saved: figures/fig1_hist_shares.png
Saved: figures/fig2_boxplot_shares.png
Saved: figures/fig3_correlation_heatmap.png
Saved: figures/fig4_scatter_plot.png
Saved: figures/fig5_bar_chart.png
```

## Features

✅ Deterministic 500-row subset with fixed random seed  
✅ Unified pipeline — single command runs everything  
✅ Clean command-line interface  
✅ Robust error handling  
✅ Five different visualization techniques  
✅ High-resolution figures (200+ DPI) ready for reports  
✅ Clean metrics table output  
✅ PySpark distributed pipeline (requires Java 17 LTS)  

## Notes

- This is **Report 1** focusing on preliminary analysis only
- No predictive modeling (regression/classification) is included yet
- No clustering or association rule mining in this phase
- The script **displays** figures in windows AND saves them to the `figures/` folder
- All visualizations are saved as PNG files (200+ DPI) for inclusion in reports
- **For your report:** Use `data/full_dataset.csv` to represent the entire dataset
- **For testing:** Use `data/subset.csv` for faster iteration

## Contact & Attribution

- **Dataset**: UCI Machine Learning Repository (ID: 332)
- **Original Data**: Mashable news articles
- **Course**: Big Data
- **Phase**: Report 1 - Preliminary Analysis & Visualization

## License

This is a course project. The dataset is publicly available from the UCI Machine Learning Repository.
