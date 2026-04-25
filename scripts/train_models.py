import os
import json
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# 0. Global settings
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"

MODEL_DIR = OUTPUT_DIR / "models_baseline"
RESULT_DIR = OUTPUT_DIR / "baseline"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_processed.csv"

LABEL_COL = "label"
N_SPLITS = 5
THRESHOLD = 0.5


# ============================================================
# 1. Metric function
# ============================================================
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # confusion_matrix order: [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Recall": sensitivity,  # legacy alias, same as Sensitivity
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Youden": sensitivity + specificity - 1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def summarize_fold_metrics(fold_rows):
    df = pd.DataFrame(fold_rows)

    metric_cols = [
        "AUC",
        "PR_AUC",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Recall",
        "Specificity",
        "F1",
        "Youden",
    ]

    summary = {}
    for col in metric_cols:
        summary[col] = round(df[col].mean(), 4)
        summary[f"{col}_std"] = round(df[col].std(ddof=1), 4)

    return summary


# ============================================================
# 2. Load training data only
# ============================================================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {TRAIN_PATH}. Please run scripts/preprocess_features.py first."
    )

train = pd.read_csv(TRAIN_PATH)

if LABEL_COL not in train.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found in {TRAIN_PATH}")

feature_cols = [c for c in train.columns if c != LABEL_COL]

X_train = train[feature_cols].copy()
y_train = train[LABEL_COL].astype(int).copy()

unique_labels = sorted(y_train.unique())
if unique_labels != [0, 1]:
    raise ValueError(f"Label must be 0/1, but got {unique_labels}")

print("=" * 80)
print("BASELINE MODEL TRAINING — TRAIN CV ONLY")
print("=" * 80)
print(f"Train path : {TRAIN_PATH}")
print(f"Train X    : {X_train.shape}")
print(f"Train y    : {y_train.shape}")
print("\nLabel distribution:")
print(y_train.value_counts().sort_index())
print(y_train.value_counts(normalize=True).sort_index().round(4))

with open(RESULT_DIR / "feature_cols_baseline.json", "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2, ensure_ascii=False)


# ============================================================
# 3. Candidate baseline models
# ============================================================
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=5000,
        random_state=SEED,
    ),

    "ElasticNet": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=5000,
        random_state=SEED,
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        random_state=SEED,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        random_state=SEED,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    ),

    "SVM": SVC(
        probability=True,
        random_state=SEED,
    ),
}


# ============================================================
# 4. 5-fold stratified CV on training set only
# ============================================================
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

summary_rows = []
fold_rows_all = []

for model_name, base_model in models.items():
    print("\n" + "=" * 80)
    print(f"{model_name} — Baseline")
    print("=" * 80)

    fold_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]

        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        model = clone(base_model)
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_va)[:, 1]

        metrics = compute_metrics(y_va, y_prob, threshold=THRESHOLD)

        row = {
            "Model": model_name,
            "Stage": "Baseline",
            "Fold": fold_idx,
            "Threshold": THRESHOLD,
            "N_train_fold": len(y_tr),
            "N_valid_fold": len(y_va),
            "Positive_valid_fold": int(y_va.sum()),
            "Negative_valid_fold": int((y_va == 0).sum()),
        }
        row.update({k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()})

        fold_rows.append(row)
        fold_rows_all.append(row)

        print(
            f"Fold {fold_idx}: "
            f"AUC={metrics['AUC']:.4f}, "
            f"PR_AUC={metrics['PR_AUC']:.4f}, "
            f"Sensitivity={metrics['Sensitivity']:.4f}, "
            f"Specificity={metrics['Specificity']:.4f}, "
            f"F1={metrics['F1']:.4f}"
        )

    summary_metrics = summarize_fold_metrics(fold_rows)

    print("\nCV summary:")
    print(f"  AUC         = {summary_metrics['AUC']:.4f} ± {summary_metrics['AUC_std']:.4f}")
    print(f"  PR_AUC      = {summary_metrics['PR_AUC']:.4f} ± {summary_metrics['PR_AUC_std']:.4f}")
    print(f"  Sensitivity = {summary_metrics['Sensitivity']:.4f} ± {summary_metrics['Sensitivity_std']:.4f}")
    print(f"  Specificity = {summary_metrics['Specificity']:.4f} ± {summary_metrics['Specificity_std']:.4f}")
    print(f"  F1          = {summary_metrics['F1']:.4f} ± {summary_metrics['F1_std']:.4f}")

    summary_row = {
        "Model": model_name,
        "Stage": "Baseline",
        "CV": f"{N_SPLITS}-fold",
        "Threshold": THRESHOLD,
        "N_train_total": len(y_train),
        "Positive_total": int(y_train.sum()),
        "Negative_total": int((y_train == 0).sum()),
    }
    summary_row.update(summary_metrics)
    summary_rows.append(summary_row)

    # Fit final baseline model on entire training set only
    final_model = clone(base_model)
    final_model.fit(X_train, y_train)

    model_path = MODEL_DIR / f"{model_name}_baseline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    print(f"\nSaved final train-fitted baseline model to: {model_path}")


# ============================================================
# 5. Save results
# ============================================================
summary_df = pd.DataFrame(summary_rows)
fold_df = pd.DataFrame(fold_rows_all)

summary_path = RESULT_DIR / "cv_metrics_baseline.csv"
fold_path = RESULT_DIR / "cv_fold_metrics_baseline.csv"

summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"CV summary saved to : {summary_path}")
print(f"Fold metrics saved to: {fold_path}")
print(f"Models saved to      : {MODEL_DIR}")
print("\nImportant:")
print("Only train_processed.csv was used for CV and model fitting.")
print("internal_test_processed.csv and external_processed.csv were NOT used here.")
