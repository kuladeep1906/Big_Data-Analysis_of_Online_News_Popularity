# Big Data Course Project - Report 1
## UCI Online News Popularity Dataset Analysis

This repository contains the preliminary analysis and visualization code for Report 1 of the Big Data course project using the UCI Online News Popularity dataset (ID: 332).

## 🚀 Quick Start

```bash
# 1. Install dependencies (first time only)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run analysis on full dataset (for report)
source .venv/bin/activate  # Activate virtual environment
python report_1.py data/full_dataset.csv

# OR run on subset (for testing)
python report_1.py data/subset.csv
```

## Project Overview

The UCI Online News Popularity dataset contains approximately 39,000 online news articles published by Mashable with around 60 numeric features related to content, sentiment, topics, and publication timing. The target variable is `shares`, representing the number of times an article was shared on social media.

**Report 1 Focus**: Statistical analysis and visualization (no predictive modeling)

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
│   ├── full_dataset.csv        # Full dataset (~39,644 rows) - USE THIS FOR REPORT
│   └── subset.csv              # 100-row deterministic subset (for testing)
├── figures/                    # Generated visualizations folder
│   ├── fig1_hist_shares.png
│   ├── fig2_boxplot_shares.png
│   ├── fig3_correlation_heatmap.png
│   ├── fig4_scatter_plot.png
│   └── fig5_bar_chart.png
├── report_1.py                 # Main analysis script
├── create_subset.py            # Script to generate subset from full dataset
├── download_full_dataset.py    # Script to download full dataset from UCI
├── validate.py                 # Validation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.7+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- ucimlrepo >= 0.0.7

## Installation

1. Clone or download this repository
2. **Set up virtual environment (recommended):**

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

3. **Install all dependencies:**

```bash
pip install -r requirements.txt
```

This will install:
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- ucimlrepo >= 0.0.7

**Note:** Always activate the virtual environment before running scripts:
```bash
source .venv/bin/activate  # Run this each time you open a new terminal
```

## Usage

### Running the Analysis

**⚠️ IMPORTANT:** Always activate the virtual environment first:
```bash
source .venv/bin/activate
```

** (use full dataset - represents entire dataset):**
```bash
python report_1.py data/full_dataset.csv
```

** (use subset - faster):**
```bash
python report_1.py data/subset.csv
```

**One-line command** (full dataset):
```bash
source .venv/bin/activate && python report_1.py data/full_dataset.csv
```

**One-line command** (subset):
```bash
source .venv/bin/activate && python report_1.py data/subset.csv
```

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

Helper script to create the deterministic 100-row subset:

- Attempts to download from UCI using `ucimlrepo`
- Falls back to local CSV if provided
- Uses fixed random seed (42) for reproducibility
- Saves to `data/subset.csv`

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

✅ Deterministic 100-row subset with fixed random seed  
✅ Clean command-line interface  
✅ Robust error handling  
✅ Five different visualization techniques  
✅ High-resolution figures (200+ DPI) ready for reports  
✅ Clean metrics table output  
✅ No unnecessary debug messages  
✅ Graceful fallbacks for missing columns  

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
