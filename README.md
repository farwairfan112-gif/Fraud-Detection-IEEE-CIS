#  Fraud Detection Framework — IEEE-CIS Dataset

> **End-to-end ML pipeline for financial fraud detection** with missing-value flag engineering, SMOTE, hyperparameter tuning, threshold optimization, and temporal validation. Achieved **ROC-AUC of 0.9107** and **F1-Score of 0.5402** on 590K+ real-world transactions.

---

##  Results Summary

| Stage | Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline | LightGBM | 0.7312 | 0.3514 | 0.4747 | 0.9046 |
| + SMOTE | LightGBM | 0.7011 | 0.3573 | 0.4733 | 0.8950 |
| + Hyperparameter Tuning | LightGBM | 0.7698 | 0.3858 | 0.5140 | 0.9107 |
| **+ Threshold (0.20)** | **LightGBM** | **0.5980** | **0.4926** | **0.5402** | **0.9107** |

---

##  Key Contributions

- **Two-Version Framework** — compared pipelines with and without missing-value flags; flagged version consistently outperformed across recall and AUC
- **Leakage-Free Pipeline** — imputation fitted on training data only; SMOTE applied to training set only; frequency encoding learned from training set only
- **Temporal Validation** — chronological 80/20 split + quartile-based drift analysis across Q1–Q4
- **Threshold Optimization** — adjusted decision threshold from 0.5 → 0.20 to maximize F1-Score and improve fraud recall
- **Interactive Dashboard** — Streamlit app with real-time fraud prediction, risk scoring, and model comparison visualizations

---

##  Project Structure

```
fraud-detection/
├── DSTT_PROJECT.ipynb        # Main notebook — flagged version (final)
├── withoutflag.ipynb         # Comparison notebook — no missingness flags
├── dashboard.py              # Streamlit dashboard
├── data/
│   ├── df_before.csv         # Raw merged dataset
│   ├── df_clean.csv          # After preprocessing + flags
│   └── df_train.csv          # Training set only (for dashboard defaults)
└── models/
    ├── lgbm_tuned.pkl        # Final LightGBM model (tuned)
    ├── rf_model_sm.pkl       # Random Forest (post-SMOTE)
    ├── xgb_model_sm.pkl      # XGBoost (post-SMOTE)
    ├── lr_model_sm.pkl       # Logistic Regression (post-SMOTE)
    ├── scaler.pkl            # StandardScaler (fit on train only)
    ├── freq_encoders.pkl     # Frequency encoding maps (fit on train only)
    ├── num_imputer.pkl       # Numeric imputer (fit on train only)
    └── cat_imputer.pkl       # Categorical imputer (fit on train only)
```

---

##  Pipeline Steps

1. **Data Collection & Merging** — transaction + identity tables merged on `TransactionID` → 590,540 rows, 434 features
2. **Missing Value Analysis** — categorized features by missingness rate (0%, 1–50%, 51–80%, 81–99%, 100%)
3. **Missing-Value Flag Engineering** — binary flags added for features with 50–80% missingness
4. **Train/Test Split** — chronological split on `TransactionDT` (80/20) to prevent temporal leakage
5. **Imputation** — median for numeric, mode for categorical — fitted on training set only
6. **Frequency Encoding** — categorical features encoded by training-set frequency
7. **StandardScaler** — fitted on training set, applied to both
8. **Baseline Training** — Logistic Regression, Random Forest, XGBoost, LightGBM
9. **SMOTE** — applied to training set only to address 96.5% / 3.5% class imbalance
10. **Hyperparameter Tuning** — RandomizedSearchCV on LightGBM (learning rate, num_leaves, max_depth, feature_fraction, bagging_fraction, n_estimators)
11. **Threshold Tuning** — optimal threshold τ = 0.20 selected by maximizing F1-Score
12. **Temporal Evaluation** — test set divided into 4 chronological quartiles; Q3 drift detected, Q4 recovery observed

---

##  Temporal Evaluation

| Quartile | Period | Observation |
|---|---|---|
| Q1 | Early | Stable performance |
| Q2 | Mid-Early | Stable performance |
| Q3 | Mid-Late | Recall drop — temporal drift detected |
| Q4 | Late | Recovery after threshold adjustment |

---

##  Running the Dashboard

```bash
# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn lightgbm xgboost joblib

# Run dashboard
streamlit run dashboard.py
```

Make sure `data/` and `models/` folders are in the same directory as `dashboard.py`.

---

##  Limitations

- Test labels unavailable — real-world deployment accuracy unmeasured
- SMOTE generates synthetic minority points that may not reflect true fraud patterns
- Temporal drift in Q3 indicates model may degrade without retraining
- Only LightGBM was extensively tuned due to compute constraints (Kaggle/Colab T4)
- Lowering threshold to 0.20 increases false positives in production

---

##  Future Work

- Autoencoders for unsupervised anomaly detection
- Graph Neural Networks for detecting organized fraud rings
- Online learning for real-time adaptation to evolving fraud patterns
- SHAP explainability for per-transaction risk reasoning
- MLflow integration for experiment tracking and model versioning

---

##  Tech Stack

`Python` `LightGBM` `XGBoost` `Scikit-learn` `Pandas` `NumPy` `Streamlit` `Plotly` `imbalanced-learn`

---

##  Author

**Farwa Irfan** — MS Data Science, FAST-NUCES Islamabad  
[GitHub](https://github.com/farwairfan112-gif) · [LinkedIn](https://linkedin.com/in/farwairfan)
