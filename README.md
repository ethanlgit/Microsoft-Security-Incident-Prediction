# Security Alert Verdict Classification with XGBoost and SHAP

This project predicts the final verdict of a security alert (`Malicious`, `Suspicious`, `NoThreatsFound`, or `Unknown`) from the Microsoft **GUIDE** dataset. It uses an XGBoost classifier inside a scikit-learn pipeline, and SHAP to explain each prediction.

## Dataset

[Microsoft Security Incident Prediction (GUIDE)](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction): about 9.5 million rows of alert and evidence records with 45 columns (detector, alert title, MITRE techniques, entity type, device/account/IP identifiers, OS, geography, and so on).

The notebook expects the CSV at `datasets/GUIDE_Train.csv`. If you only have the archive, unzip it first:

```bash
mkdir -p datasets
unzip GUIDE_Train.csv.zip -d datasets/
```

## Project structure

```
.
├── main.ipynb            # Full workflow: load → clean → train → tune → explain
├── helper.py             # Plotting helpers (confusion matrix, heatmap, decision contours)
├── shap_force_plot.html  # Exported SHAP force plot for one test sample
├── GUIDE_Train.csv.zip   # Compressed dataset
└── datasets/
    └── GUIDE_Train.csv   # Extracted dataset (read by the notebook)
```

## Workflow

1. **Loading.** The CSV is read in 1M-row chunks and then sampled down to 100,000 rows (`SEED = 66`).
2. **Cleaning**
   - Drops mostly-empty columns (`ActionGrouped`, `ActionGranular`, `EmailClusterId`, `ThreatFamily`, `ResourceType`, `Roles`, `AntispamDirection`) and ID columns (`Id`, `OrgId`, `IncidentId`, `AlertId`).
   - Removes one malformed `LastVerdict` value and fills missing verdicts with `Unknown`.
   - Splits `Timestamp` into second, minute, hour, day, day name, month, and year.
3. **Preprocessing** (`ColumnTransformer`)
   - Numerical columns: mean imputation followed by `MinMaxScaler`.
   - Categorical columns (`Category`, `MitreTechniques`, `IncidentGrade`, `EntityType`, `EvidenceRole`, `SuspicionLevel`, `DayName`): most-frequent imputation followed by `OneHotEncoder`.
4. **Model.** An `XGBClassifier` is trained on a stratified 80/20 train/test split.
5. **Tuning.** `GridSearchCV` (5-fold) searches over `n_estimators`, `max_depth`, and `learning_rate`.
6. **Explainability**
   - XGBoost gain-based feature importance, mapped back to readable feature names.
   - A SHAP summary plot and per-class force plots.
   - An interactive `ipywidgets` explorer for choosing a test sample and class and viewing its SHAP force plot.

## Results

| Model | Test accuracy |
|---|---|
| XGBoost (defaults) | 0.9448 |
| XGBoost (grid-search best) | 0.9444 |

The most important features by gain are `Category_CommandAndControl`, `EntityType_Process`, `Category_SuspiciousActivity`, `MitreTechniques_T1566.002` (spearphishing link), and `OSVersion`.

> Note: most rows in the source data have no `LastVerdict`, so after `fillna` the `Unknown` class dominates. Accuracy alone can therefore overstate performance, so look at per-class metrics as well.

## Getting started

### Requirements

- Python 3.10+
- Jupyter (Notebook, Lab, or VS Code)

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib ipywidgets
```

### Run

```bash
jupyter notebook main.ipynb
```

Run the cells from top to bottom. Loading the full CSV needs several GB of RAM, and the grid search takes a while (`n_jobs=-1` uses every core). On Windows, the notebook sets `OMP_NUM_THREADS=2` to avoid a known KMeans/OpenMP memory-leak warning.

## Helper functions (`helper.py`)

| Function | Purpose |
|---|---|
| `draw_confusion_matrix(y, yhat, classes)` | Plots an annotated confusion matrix |
| `heatmap(data, row_labels, col_labels, ...)` | Draws an annotated heatmap from a 2D array |
| `draw_contour(x, y, clf, class_labels)` | Plots a classifier's decision boundary over its first two features |
