# Code Rewrite Plan - Online News Popularity Analysis

> **Version 2** — Updated with friend's 2 new tasks + clarifications

## Professor's Core Feedback

The main problem is **structural/organizational**. Our current code and report treat "Classification", "Regression", "Clustering" as **tasks**. But the professor wants **real-world tasks** (questions we're trying to answer), with methods listed underneath. **"Regression" is a method, NOT a task.** "Predicting number of shares on weekends" IS a task.

### What the Professor Wants

```
TASK (a real question)  -->  Type (supervised/unsupervised, classification/regression)  -->  Methods (models)
```

---

## Professor's Feedback Summary

| # | Feedback | Action |
|---|----------|--------|
| 1 | Not clear if we used data cleaning or not | Add explicit data cleaning section with printed stats |
| 2 | Not clear what tasks are vs methods — "regression is not a task" | Restructure everything around 6 named real-world tasks |
| 3 | Needs to be more organized | New pipeline order + clear task-centric output |
| 4 | First identify which model is good, THEN apply dimensionality reduction | Change execution order |
| 5 | Can be a group of models that are good — identify that | Add model group comparison |
| 6 | Be clear under which task what was done | Each task prints its name, type, methods |
| 7 | Clear on input and output | Each task explicitly states input features and output target |
| 8 | **Preliminary analysis → identify important features → THEN PCA/t-SNE (VERY IMPORTANT)** | Feature count determined by analysis (elbow/threshold), then dim reduction |
| 9 | Divide day into 24hrs and classify which hour news is popular | **No hourly data in dataset — replaced with Task 6 (weekday vs weekend recommendation)** |
| 10 | Task naming like "Predicting number of shares on weekend" then specify type | All 6 tasks named properly |
| 11 | 4 tasks already present but not named | Now all named + 2 new ones added |
| 12 | Specify what is before model training and after model training | Show before/after preprocessing for every supervised task |

---

## All 6 Proposed Tasks

### Task 1: Predicting Whether a News Article Will Be Popular
- **Real-World Question**: "Will this article go viral or flop?"
- **Type**: Supervised — Binary Classification
- **Input**: Top N selected features (from preliminary analysis)
- **Output**: Popular (1) / Unpopular (0) — threshold at median shares
- **Models**: Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, Decision Tree, Gradient Boosting, XGBoost
- **Before/After**: Results shown with and without preprocessing (scaling, encoding)
- **Model Group Finding**: e.g., "Tree-based ensembles outperform linear models"

### Task 2: Predicting the Number of Shares an Article Will Receive
- **Real-World Question**: "How many shares can we expect from this article?"
- **Type**: Supervised — Regression
- **Input**: Top N selected features (from preliminary analysis)
- **Output**: Continuous value — log(1 + shares)
- **Models**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR
- **Before/After**: Results shown with and without preprocessing
- **Model Group Finding**: e.g., "Ensemble methods provide lowest RMSE"

### Task 3: Discovering Natural Groupings of News Articles
- **Real-World Question**: "Are there distinct types of articles in our dataset?"
- **Type**: Unsupervised — Clustering
- **Input**: Feature vectors (top N features, scaled)
- **Output**: Cluster labels
- **Models**: K-Means (with elbow method for optimal K), DBSCAN
- **Analysis**: Cluster profiling — what characterizes each group

### Task 4: Identifying Content Patterns Associated with High Engagement
- **Real-World Question**: "What combinations of features tend to appear together in high-share articles?"
- **Type**: Unsupervised — Association Rule Mining
- **Input**: Binary feature indicators (weekend, channel type, high/low polarity, etc.)
- **Output**: Association rules with support, confidence, lift
- **Models**: Apriori algorithm
- **Analysis**: Actionable rules like "IF published on weekend AND lifestyle channel THEN high shares"

### Task 5: Optimizing Article Formatting and Media Usage for Maximum Reach
- **Real-World Question**: "What is the optimal combination of text length, links, and media (images/videos) to drive shares?"
- **Type**: Supervised — Regression (with focus on feature coefficients / partial dependence)
- **Input**: ONLY structural/formatting features — `n_tokens_title`, `n_tokens_content`, `num_imgs`, `num_videos`, `num_hrefs`, `num_self_hrefs`
- **Output**: Continuous value — expected shares (or log(1 + shares))
- **Models**: Ridge, Lasso (for interpretable coefficients), Random Forest, Gradient Boosting (for partial dependence plots)
- **Key Deliverable**: Actionable insights like "Adding up to 3 videos increases shares, but after 4 engagement drops"
- **Why this task matters**: Instead of throwing all features at a model, we build a focused sub-model on formatting variables only — gives direct guidelines to content creators/editors

### Task 6: Recommending the Optimal Publication Window (Weekday vs. Weekend)
- **Real-World Question**: "Given the emotional tone and topic of a drafted article, should we publish it on a weekday or weekend?"
- **Type**: Supervised — Binary Classification
- **Input**: NLP features only — sentiment polarity, subjectivity, content channels (Tech, Lifestyle, Business, etc.)
- **Output**: Binary recommendation — Publish on Weekday (0) / Publish on Weekend (1)
- **Models**: Logistic Regression, Random Forest, SVM, Gradient Boosting
- **Why this task matters**: This turns a simple EDA observation ("weekends get more shares") into an actionable ML tool. We build a classifier that looks at a draft article and says "This article is highly subjective and positive — our model suggests publishing on Saturday." This also addresses the professor's temporal analysis concern (Point #9) without needing hourly data.

---

## Task Output Format (every task prints this)

```
╔══════════════════════════════════════════════════════════════════╗
║  TASK 3: Discovering Natural Groupings of News Articles         ║
║  Type: Unsupervised — Clustering                                ║
║  Input: Top N features (after feature selection + scaling)      ║
║  Output: Cluster labels (K-Means, DBSCAN)                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  [Results tables, charts, metrics]                               ║
║                                                                  ║
║  Best Model: K-Means (k=5, Silhouette=0.34)                    ║
╚══════════════════════════════════════════════════════════════════╝
```

For supervised tasks (1, 2, 5, 6), the box also includes:
```
  Before Preprocessing: [results table]
  After Preprocessing:  [results table]
  Best Model: Random Forest (Accuracy: 0.66)
  Best Model Group: Tree-based ensembles outperform linear models
```

---

## New Pipeline Execution Order

### OLD order (current code — method-centric, disorganized):
1. EDA → 2. Preprocess → 3. Classification → 4. Regression → 5. Clustering → 6. Association Rules → 7. Dimensionality Reduction → 8. Temporal Analysis → 9. Feature Importance → 10. Spark → 11. ML Comparison

### NEW order (task-centric, follows professor's requirements):

```
╔══════════════════════════════════════════════════════════════╗
║  PHASE 1: DATA PREPARATION                                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Step 1: DATA CLEANING (explicit, verbose, prints counts)    ║
║    → Missing values, duplicates, outliers, dropped columns   ║
║    → Shape before/after clearly printed                      ║
║                                                              ║
║  Step 2: EDA (exploratory data analysis)                     ║
║    → Distributions, correlations, target variable analysis   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  PHASE 2: FEATURE ENGINEERING (BEFORE any model training)    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Step 3: PRELIMINARY FEATURE ANALYSIS                        ║
║    → Correlation analysis (drop highly correlated pairs)     ║
║    → Random Forest importance ranking on ALL features        ║
║    → Select top N features using elbow/threshold method      ║
║    → N is DATA-DRIVEN (not hardcoded 20 or 30)              ║
║    → Print selected features clearly                         ║
║                                                              ║
║  Step 4: DIMENSIONALITY REDUCTION (PCA / t-SNE)             ║
║    → Applied ONLY on the N selected features from Step 3     ║
║    → Visualize clusters in 2D/3D                             ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  PHASE 3: TASK EXECUTION (AFTER feature selection)           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Step 5:  TASK 1 — Predicting Popularity (Classification)    ║
║  Step 6:  TASK 2 — Predicting Number of Shares (Regression)  ║
║  Step 7:  TASK 3 — Article Groupings (Clustering)            ║
║  Step 8:  TASK 4 — Engagement Patterns (Association Rules)   ║
║  Step 9:  TASK 5 — Formatting & Media Optimization (Regr.)   ║
║  Step 10: TASK 6 — Publication Window Recommender (Classif.) ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  PHASE 4: SCALABILITY                                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Step 11: SPARK PIPELINE (scalability demonstration)         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Detailed Code Changes

### 1. Data Cleaning — NEW module: `src/data_cleaning.py`

**Currently**: Data cleaning is buried inside preprocessing with no visibility.

**What to add**:
- Print missing value counts per column
- Print duplicate row count and removal
- Print outlier detection strategy (e.g., IQR or Z-score on `shares`)
- Print which columns are dropped and why (URL, timedelta)
- Print dataset shape before and after cleaning
- This must be the FIRST thing that runs

### 2. Feature Selection — MOVED TO BEGINNING

**Currently**: `feature_importance.py` runs near the end.

**Change**:
- Run feature importance FIRST (after EDA)
- Use a combination of correlation analysis + Random Forest feature importance
- The number of features (N) is determined by the data itself (e.g., cumulative importance threshold at 90%, or elbow method on sorted importances) — NOT hardcoded
- Print the selected features and the method used to select them
- All subsequent tasks use only these N features (except Task 5 and 6 which use specific subsets)

### 3. Dimensionality Reduction — AFTER Feature Selection

**Currently**: PCA/t-SNE runs on all 58 features.

**Change**: Apply PCA/t-SNE on the N features selected in Step 3. Professor specifically said this.

### 4. Task-Centric Output

**Currently**: Output is method-centric ("Running classification...", "Running regression...").

**Change**: Each task prints its own banner with name, type, input, output, before/after, best model, and best model group.

### 5. Before/After Preprocessing (for supervised tasks)

**Currently**: `ml_pipeline_comparison.py` runs all 15 models in one block.

**Change**:
- Split into individual tasks — Task 1 gets classification before/after, Task 2 gets regression before/after
- Tasks 5 and 6 also get their own before/after comparisons
- Each task identifies best MODEL and best MODEL GROUP

### 6. Task 5 — NEW: Formatting sub-model

**New module or section**: Trains models using ONLY structural features (`n_tokens_title`, `n_tokens_content`, `num_imgs`, `num_videos`, `num_hrefs`, `num_self_hrefs`). Focuses on feature coefficients (Ridge/Lasso) and partial dependence plots (tree models) to give actionable formatting guidelines.

### 7. Task 6 — NEW: Publication window recommender

**New module or section**: Uses only NLP and channel features as input. Target is `is_weekend`. Trains classifiers to recommend weekday vs weekend publication based on article content/tone.

---

## Feature Selection Strategy

Since the professor wants us to do preliminary analysis and FIND the right number of features (not pick an arbitrary number):

```
Method:
1. Correlation Analysis
   → Compute pairwise correlation matrix
   → Drop one feature from each pair with |correlation| > 0.85
   → This removes redundant features

2. Random Forest Feature Importance
   → Train RF on remaining features (target = shares)
   → Rank features by importance score
   → Plot cumulative importance curve

3. Select N features
   → N = number of features that collectively account for ~90% cumulative importance
   → OR use elbow method on the sorted importance curve
   → This makes N data-driven, not arbitrary

4. Print and save the selected features
   → This list is used by Tasks 1-4
   → Tasks 5 and 6 use their own specific feature subsets
```

---

## About the 24-Hour Issue (Professor's Point #9)

**The dataset does NOT have hour-level data.** Only day-of-week binary columns.

**Our solution**: Task 6 (Recommending the Optimal Publication Window) addresses the professor's temporal concern by turning day-of-week analysis into an actionable ML model. We will explain to the professor that the dataset lacks hourly granularity but we've built a recommender for weekday vs weekend publishing instead.

---

## What We Are NOT Changing

- The PPT (done, submitted)
- The LaTeX report (will modify LATER after professor approves the code)
- The dataset itself

---

## Files That Will Be Modified/Created

| File | Action |
|------|--------|
| `report_2_task.py` | **Rewrite** — new orchestrator with 4-phase pipeline |
| `src/data_cleaning.py` | **NEW** — explicit data cleaning module |
| `src/feature_importance.py` | **Rewrite** — runs first, returns data-driven top N features |
| `src/dimensionality_reduction.py` | **Modify** — takes selected features as input |
| `src/classification.py` | **Rewrite** — Task 1 (popularity) + Task 6 (publication window) |
| `src/regression.py` | **Rewrite** — Task 2 (shares) + Task 5 (formatting optimization) |
| `src/clustering.py` | **Modify** — Task 3, task-centric output |
| `src/association_rules.py` | **Modify** — Task 4, task-centric output |
| `src/temporal_analysis.py` | **Remove or merge** — replaced by Task 6 |
| `src/ml_pipeline_comparison.py` | **Remove** — before/after logic merged into each task |
| `src/eda.py` | **Minor changes** — cleaner output |
| `src/preprocessing.py` | **Modify** — uses selected features, cleaner interface |

---

## Summary of Task Coverage

| Professor's Concern | Covered By |
|---------------------|------------|
| Shares on particular days | Task 1 (popularity) + Task 6 (weekday/weekend) |
| Content-based engagement | Task 5 (formatting) + Task 4 (association rules) |
| What makes articles popular | Task 1 + Task 2 |
| Temporal patterns | Task 6 (publication window recommender) |
| Feature importance visible | Phase 2, Step 3 |
| Clear input/output | Every task banner |
| Before/After preprocessing | Tasks 1, 2, 5, 6 |
| Model groups identified | Tasks 1, 2, 5, 6 |
| Data cleaning visible | Phase 1, Step 1 |

---

## Next Steps

1. Everyone reviews this plan
2. Agree on any changes
3. Start coding (full rewrite)
4. Test and validate all 6 tasks run end-to-end
5. Show code to professor for approval
6. THEN modify the LaTeX report
