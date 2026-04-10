# Report Audit: LaTeX Report vs Professor's Guidelines vs Feedback

---

## Rubric Breakdown (from Report 2 Tasks_.pdf)

| Points | Category | Our Status |
|--------|----------|------------|
| 2 | LaTeX format | SEE ISSUES BELOW |
| 2 | Grammar | Generally good, minor issues |
| 6 | Tasks description/design | GOOD - 6 tasks well-defined |
| 20 | Methodologies and Results Analysis | NEEDS WORK - see details |
| 5 | Code | DONE - runs via `python report_2_task.py <file>` |

---

## CHECK 1: Professor's Guidelines (Report 2 Tasks_.pdf)

### Requirement: "must be written using provided latex style file and bibliography file"

**STATUS: POTENTIALLY MISSING**

The report uses `\documentclass[11pt]{article}` with custom packages. The professor said "provided latex style file" — we need to check if there was a specific `.sty` or `.cls` file the professor provided. If yes, we're not using it. The current `main.tex` uses a generic article class.

**ACTION NEEDED:** Confirm whether the professor provided a specific LaTeX template. If so, switch to it. This is worth 2 points.

---

### Requirement: "Explain in detail, how you designed your tasks/datasets. I should be able to follow your description and create the same setup"

**STATUS: PARTIALLY MET — NEEDS MORE DETAIL**

Current `task_definition.tex` defines tasks at a high level (2-3 sentences each) but doesn't give enough reproducibility detail. The professor wants to be able to recreate the EXACT same setup.

**What's missing per task:**
- Task 1: Doesn't specify the exact median threshold value (1400) in the task definition section (it appears later in results, but should be in the definition)
- Task 2: Doesn't mention the exact log transformation formula in the definition
- Task 3: Doesn't specify the feature scaling method used, range of k tested
- Task 4: Doesn't specify minimum support/confidence thresholds for Apriori
- Task 5: Doesn't list the exact 6 input features in the definition
- Task 6: Doesn't list the exact input features (NLP, channel, LDA)

**The methodology section (`task_methodology.tex`) covers more detail but still lacks:**
- Train/test split ratio (80/20)
- Random seed (42)
- Stratification mention
- Exact number of selected features (35)
- How the 35 features were chosen (cumulative importance threshold)

**ACTION NEEDED:** Add explicit setup parameters to each task's methodology description.

---

### Requirement: "Try to explore multiple approaches and compare their performances"

**STATUS: WELL MET**

- Task 1: 8 classifiers compared
- Task 2: 7 regressors compared
- Task 3: K-Means + DBSCAN compared
- Task 4: Apriori with rule metrics
- Task 5: 4 regressors compared
- Task 6: 4 classifiers compared
- Before/after preprocessing comparison across tasks

This is solid.

---

### Requirement: "Try to utilize various statistical analysis metrics"

**STATUS: WELL MET**

- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression: RMSE, MAE, R²
- Clustering: Silhouette, Inertia, Elbow
- Association: Support, Confidence, Lift

Good coverage.

---

### Requirement: "submit code with subset of data (~500 samples), run via python report_2_task.py <file>"

**STATUS: NEED TO CHECK**

Need to confirm:
1. `create_subset.py` creates a ~500 sample subset
2. The code runs end-to-end on the subset
3. A README explains how to run

---

### Requirement: "If you have tasks that focus on operations with Big Data tools like Hadoop and Spark, describe them properly"

**STATUS: MET**

The Spark section exists in `task_methodology.tex` and `results_by_task.tex`. It describes the PySpark Random Forest pipeline and compares with scikit-learn. Could add more detail about the Spark pipeline stages.

---

## CHECK 2: Professor's Verbal Feedback (from your first chat)

### Feedback #1: "Not clear if we used data cleaning or not"

**STATUS: ADDRESSED BUT COULD BE STRONGER**

`data_preparation.tex` has a "Data Cleaning" subsection that mentions: no missing values, no duplicates, all numeric columns, outlier handling via log-transform.

**Issue:** It's only 1 paragraph. The professor wants it to be VISIBLE. Consider:
- Adding a table showing the cleaning steps explicitly
- Mentioning the IQR outlier analysis numbers
- Being more explicit: "We checked for missing values (0 found), duplicate rows (0 found), non-numeric columns (0 found)"

---

### Feedback #2: "Not clear what tasks are, what methods are, regression is not a task"

**STATUS: FULLY ADDRESSED**

The report now has proper task names (e.g., "Predicting Whether a News Article Will Be Popular") with type specified (e.g., "Supervised - Binary Classification"). The task_definition.tex and task_methodology.tex sections clearly separate WHAT (task) from HOW (methods).

---

### Feedback #3: "Had to be more organized"

**STATUS: MUCH IMPROVED**

The report now follows a logical flow:
1. Dataset → 2. Task Definition → 3. Data Preparation → 4. Feature Engineering → 5. Methodology → 6. Results → 7. Discussion

This is well-organized.

---

### Feedback #4: "First identify which model is good, then apply dimensionality reduction"

**STATUS: ADDRESSED IN CODE, BUT NOT CLEAR IN REPORT**

The report's Feature Engineering section shows correlation filtering → feature importance → feature selection → PCA/t-SNE. This matches the feedback. However, the report doesn't explicitly say "we first identified important features using Random Forest, then applied PCA/t-SNE on the selected features."

**ACTION NEEDED:** Add a sentence explicitly stating this ordering and WHY.

---

### Feedback #5: "Can be a group of models which can be good — identify that"

**STATUS: PARTIALLY ADDRESSED**

The results section mentions "tree-based ensemble models" performing best, but doesn't have a clear **model group comparison summary**.

**ACTION NEEDED:** Add a summary table or paragraph like:
```
Model Group Comparison:
- Tree-based ensembles (RF, GB, AdaBoost): Best overall — Acc ~0.64, R² ~0.13
- Linear models (LogReg, Ridge, Lasso): Moderate — benefit from scaling
- Distance-based (KNN, SVM, SVR): Weakest — sensitive to scale and noise
- Probabilistic (Naive Bayes): Poor — independence assumption violated
```

---

### Feedback #6: "Be clear under which task what was done"

**STATUS: WELL ADDRESSED**

Each task has its own subsection in both methodology and results. Clear structure.

---

### Feedback #7: "Clear on what is input and what is output"

**STATUS: PARTIALLY ADDRESSED**

The task_definition.tex section mentions inputs and outputs conceptually but doesn't state them explicitly in a structured way.

**ACTION NEEDED:** For each task, add an explicit Input/Output box:
```
Input: 35 selected features (after correlation filtering and RF importance selection)
Output: Binary label (Popular=1 if shares ≥ 1400, Unpopular=0)
```

---

### Feedback #8: "Preliminary analysis → identify top 20/30 features → PCA/t-SNE (VERY IMPORTANT, bold in report)"

**STATUS: PRESENT BUT NOT BOLD/PROMINENT ENOUGH**

The Feature Engineering section describes this pipeline but doesn't make it **bold or visually prominent**. The professor specifically said this should be in bold.

**ACTION NEEDED:**
- Add a bold highlighted paragraph or box explicitly stating: "Based on preliminary analysis using Random Forest importance, we selected 35 features that capture 95% of cumulative importance. PCA and t-SNE were then applied ONLY to these selected features."
- Make this visually stand out (bold, maybe a framed box)

---

### Feedback #9: "Divide day into 24 hours"

**STATUS: SKIPPED (agreed to convince professor)**

Task 6 pivots to weekday vs weekend instead. This is reasonable given the dataset limitations. The report should explicitly state WHY hourly analysis is not possible.

**ACTION NEEDED:** Add a sentence in Task 6 methodology: "The dataset does not contain hour-level publication timestamps, so this task uses weekday vs. weekend as the temporal dimension."

---

### Feedback #10: "Task naming like 'predicting shares on weekend' then specify type"

**STATUS: FULLY ADDRESSED**

All 6 tasks have proper names and types specified.

---

### Feedback #11: "4 tasks already present but not named"

**STATUS: FULLY ADDRESSED**

All tasks are now named and defined in task_definition.tex.

---

### Feedback #12: "Specify what is before and after model training"

**STATUS: ADDRESSED**

Results show before/after preprocessing tables for Task 1, 2, and 6. The distinction between raw (imputed only) and preprocessed (imputed + scaled) is clear in the results tables.

**Could be improved:** Add a brief explanation of what "before" and "after" mean in the data preparation section (before = raw features with imputation only; after = imputed + StandardScaler).

---

## CHECK 3: Things MISSING from the Report

### Missing: Explicit feature list

The report says "35 selected features" but never lists them. The professor wants to be able to recreate the setup. At minimum, list the top 10-15 features by name.

### Missing: Number of features justification

The report doesn't explain HOW the number 35 was chosen. Was it a cumulative importance threshold (e.g., 95%)? The cumulative importance figure exists but isn't referenced with a specific threshold.

### Missing: Exact Apriori parameters

Task 4 doesn't specify min_support, min_confidence values used.

### Missing: DBSCAN parameters

Task 3 mentions DBSCAN but doesn't state eps and min_samples values used.

### Missing: Spark pipeline details

The Spark section is brief. It should mention the exact pipeline stages (VectorAssembler → RandomForest), the number of trees, and the train/test split used.

### Missing: README for code submission

Need to confirm a README exists that explains how to run the code.

---

## SUMMARY: Priority Fixes

### HIGH PRIORITY (affects major rubric points — Methodologies & Results: 20 pts)

| # | Issue | Where to Fix |
|---|-------|-------------|
| 1 | Add explicit Input/Output for each task | task_definition.tex or task_methodology.tex |
| 2 | Add model group comparison summary | discussion.tex or results_by_task.tex |
| 3 | Make feature selection pipeline BOLD and prominent | feature_engineering.tex |
| 4 | List selected features (at least top 15) | feature_engineering.tex |
| 5 | Add exact parameters: train/test split, seed, threshold, Apriori params, DBSCAN params | task_methodology.tex |
| 6 | Add explicit "before = imputed only, after = imputed + scaled" definition | data_preparation.tex |
| 7 | Explain why hourly analysis not possible | task_methodology.tex (Task 6) |

### MEDIUM PRIORITY (affects Task Description: 6 pts)

| # | Issue | Where to Fix |
|---|-------|-------------|
| 8 | Expand data cleaning section (more visible, table format) | data_preparation.tex |
| 9 | Explain how 35 features were chosen (cumulative importance threshold) | feature_engineering.tex |
| 10 | Add Spark pipeline details (stages, tree count) | task_methodology.tex |

### LOW PRIORITY (affects LaTeX format: 2 pts + Grammar: 2 pts)

| # | Issue | Where to Fix |
|---|-------|-------------|
| 11 | Verify using professor's provided LaTeX style | main.tex |
| 12 | Grammar pass on all sections | All .tex files |
| 13 | Ensure README exists | Project root |

---

## WHAT'S GOOD (No changes needed)

- 6 well-defined, properly named tasks
- Clear task-vs-method distinction (professor's main feedback)
- Before/after preprocessing comparison across tasks
- Multiple metrics per task type
- Good visualizations (figures for each task)
- References properly cited with BibTeX
- Spark scalability demonstration included
- Honest discussion of weak results (Task 6, clustering)
- Logical report structure and flow
