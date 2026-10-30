# Airline-Satisfaction-Prediction
Predicting airline passenger satisfaction with XGBoost, CatBoost, and Random Forest on a Kaggle Playground dataset (700k+ rows). Includes feature engineering, hyperparameter tuning, and a blended submission. Best validation ROC-AUC: ~0.956.
# ✈️ Airline Passenger Satisfaction Prediction

Predicting whether an airline passenger was **satisfied** with their flight, using tabular survey and flight data from a Kaggle Playground Series competition. The project covers data cleaning, feature engineering, model comparison (Random Forest, XGBoost, CatBoost), hyperparameter tuning, and a blended submission.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Known Issues and Future Work](#known-issues-and-future-work)
- [Tech Stack](#tech-stack)

---

## Overview

The goal is to predict the binary target `satisfaction` (satisfied / not satisfied) for each passenger. Models output a probability-like score, and performance is evaluated mainly with **ROC-AUC**, alongside MSE and R².

## Dataset

- **Source:** Kaggle Playground Series, Airline Passenger Satisfaction (`train.csv`, `test.csv`)
- **Training size:** 699,635 rows × 23 columns
- **Target:** `satisfaction` (boolean, converted to 0/1)

| Type | Features |
|------|----------|
| Service ratings (0–5) | Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Cleanliness |
| Numeric | Age, Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes |
| Categorical | Gender, Customer Type, Type of Travel, Class (Eco / Eco Plus / Business) |

> The data files are **not** included in this repository. Download them from the Kaggle competition page and place them in a `data/` folder.

## Project Workflow

1. **Data cleaning**
   - Checked for missing values (~0.04% of rows) and dropped them (699,635 → 699,343 rows).
   - Confirmed there were no duplicate rows.
2. **Feature engineering**
   - `Flight Distance` binned into 20 equal-width intervals (`Flight Distance bins`).
   - `Age` binned into groups: Children (0–10), Young Adults (10–18), Adults (18–30), Mature (30–60), Seniors (60–100) (`Age_Group`).
   - Target converted from boolean to integer (`Satisfaction_target`).
   - The original `Flight Distance`, `Age`, and `id` columns were dropped from the features.
3. **Preprocessing pipeline** (scikit-learn `ColumnTransformer`)
   - Numeric columns: `StandardScaler`
   - Categorical columns: `OneHotEncoder(handle_unknown="ignore")`
4. **Data split**
   - 70% non-test / 30% test, then the non-test portion split 50/50 into train and validation (≈35% / 35% / 30% overall), `random_state=0`.
5. **Modeling**
   - Random Forest Regressor
   - XGBoost Regressor
   - CatBoost Regressor
6. **Blending:** simple 50/50 average of XGBoost and CatBoost predictions for the submission.
7. **Hyperparameter tuning:** `RandomizedSearchCV` (20 iterations, 5-fold `StratifiedKFold`, scoring = R²) on the XGBoost pipeline.

## Results

Metrics on the held-out validation and test splits:

| Model | Validation ROC-AUC | Test ROC-AUC | Test R² | Test MSE |
|-------|-------------------:|-------------:|--------:|---------:|
| Random Forest | 0.9520 | 0.9525 | 0.7346 | 0.0656 |
| XGBoost (baseline) | 0.9559 | 0.9563 | 0.7441 | 0.0632 |
| CatBoost | 0.9555 | 0.9561 | 0.7404 | 0.0641 |
| XGBoost (tuned) | 0.9565 | not evaluated | – | – |

(The tuned XGBoost validation R² is 0.7462 and MSE is 0.0627.)

**Observations**
- The Random Forest overfits heavily (training ROC-AUC ≈ 0.995 vs. ≈ 0.952 on validation).
- XGBoost and CatBoost generalize much better, with training and validation scores close together.
- Tuning gave a small but consistent improvement for XGBoost.

**Best tuned XGBoost parameters**

```python
{
    "subsample": 0.7,
    "reg_lambda": 10,
    "reg_alpha": 0,
    "n_estimators": 1500,
    "min_child_weight": 10,
    "max_depth": 10,
    "learning_rate": 0.01,
    "gamma": 0.3,
    "colsample_bytree": 0.7,
}
```

## Repository Structure

```
.
├── code.ipynb            # Full analysis, modeling, and submission code
├── data/                 # (not tracked) train.csv and test.csv from Kaggle
├── submissions/          # Generated submission CSVs
└── README.md
```

Adjust this to match your actual layout.

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost jupyter
```

## Usage

1. Download `train.csv` and `test.csv` from the Kaggle competition and place them in `data/`.
2. Update the file paths in the notebook. The original code uses an absolute local path, so change it to something like:
   ```python
   df = pd.read_csv("data/train.csv")
   test = pd.read_csv("data/test.csv")
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook code.ipynb
   ```
4. The notebook writes submission files (`submission_ver7_blend.csv`, `submission_ver8_search_xgb.csv`) with columns `id` and `satisfaction`.

## Known Issues and Future Work

- **Submission bug:** `submission_ver8_search_xgb` currently uses `blend_prob` instead of the tuned model's predictions (`test_pred_search_xgb`). Update the `satisfaction` column accordingly.
- **CatBoost tuning cell fails:** the categorical columns are passed as raw strings while `cat_features` is commented out. Pass `cat_features=cat_cols` on raw (non-encoded) data, or run the search inside the preprocessing pipeline.
- **Regression vs. classification:** the target is binary, so `XGBClassifier`, `CatBoostClassifier`, and `RandomForestClassifier` with log-loss/AUC scoring may perform better than regressors.
- **Overfitting in Random Forest:** limit `max_depth` or increase `min_samples_leaf`.
- Ideas: native categorical handling in CatBoost, tuning CatBoost/LightGBM, stacking instead of simple averaging, feature importance/SHAP analysis.

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · CatBoost

## License

Add a license of your choice (for example MIT) before publishing.
