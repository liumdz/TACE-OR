import os
import json
import random
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ============================================================
# 0. Global settings
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "output" / "tuned"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_processed.csv"

LABEL_COL = "label"
N_SPLITS = 5
THRESHOLD = 0.5

# 类别不平衡情况下，建议用 PR-AUC 作为主要调参目标
REFIT_METRIC = "CV_PR_AUC"
N_ITER = 20


# ============================================================
# 1. Metric function
# ============================================================
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Recall": sensitivity,
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

    out = {}
    for col in metric_cols:
        out[col] = round(df[col].mean(), 4)
        out[f"{col}_std"] = round(df[col].std(ddof=1), 4)

    return out


# ============================================================
# 2. Load train only
# ============================================================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {TRAIN_PATH}. Please run scripts/preprocess_features.py first."
    )

train = pd.read_csv(TRAIN_PATH)

if LABEL_COL not in train.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found.")

FEATURE_COLS = [c for c in train.columns if c != LABEL_COL]

X_train = train[FEATURE_COLS].copy()
y_train = train[LABEL_COL].astype(int).copy()

print("=" * 80)
print("HYPERPARAMETER TUNING — TRAIN CV ONLY")
print("=" * 80)
print(f"Train path: {TRAIN_PATH}")
print(f"Train X   : {X_train.shape}")
print(f"Train y   : {y_train.shape}")
print("\nLabel distribution:")
print(y_train.value_counts().sort_index())
print(y_train.value_counts(normalize=True).sort_index().round(4))
print(f"\nRefit metric: {REFIT_METRIC}")

with open(OUTPUT_DIR / "feature_cols_tuned.json", "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLS, f, indent=2, ensure_ascii=False)


# ============================================================
# 3. CV strategy and sample weights
# ============================================================
cv_strategy = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

sample_weights_full = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# ============================================================
# 4. Model search space
# ============================================================
def get_model_and_params(model_name):
    if model_name == "LogisticRegression":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        )
        param_grid = {
            "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            "solver": ["liblinear", "lbfgs"],
        }
        fit_params = {}

    elif model_name == "ElasticNet":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=8000,
            random_state=SEED,
        )
        param_grid = {
            "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            "l1_ratio": [0.1, 0.2, 0.5, 0.8, 0.9],
        }
        fit_params = {}

    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [2, 3, 4, 5, 8, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }
        fit_params = {}

    elif model_name == "SVM":
        model = SVC(
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        )
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["rbf", "linear"],
        }
        fit_params = {}

    elif model_name == "LightGBM":
        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=SEED,
            force_col_wise=True,
            verbose=-1,
            n_jobs=1,
        )
        param_grid = {
            "num_leaves": [3, 7, 15, 31],
            "max_depth": [2, 3, 4, 5, -1],
            "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "n_estimators": [50, 100, 200, 300],
            "min_child_samples": [5, 10, 20, 30],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        fit_params = {"sample_weight": sample_weights_full}

    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(
            random_state=SEED,
        )
        param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [1, 2, 3, 4],
            "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "min_samples_leaf": [1, 2, 4, 8],
        }
        fit_params = {"sample_weight": sample_weights_full}

    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
        )
        param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [1, 2, 3, 4],
            "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5, 10],
            "reg_lambda": [0.5, 1, 2, 5],
            "reg_alpha": [0, 0.1, 0.5, 1],
        }
        fit_params = {"sample_weight": sample_weights_full}

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, param_grid, fit_params


TUNE_MODELS = [
    "LogisticRegression",
    "ElasticNet",
    "RandomForest",
    "SVM",
    "LightGBM",
    "GradientBoosting",
    "XGBoost",
]

MODELS_NEED_SAMPLE_WEIGHT = {
    "LightGBM",
    "GradientBoosting",
    "XGBoost",
}


# ============================================================
# 5. Hyperparameter tuning + fold-level CV evaluation
# ============================================================
summary_rows = []
fold_rows_all = []
search_rows_all = []

for model_name in TUNE_MODELS:
    print("\n" + "=" * 80)
    print(f"Tuning: {model_name}")
    print("=" * 80)

    base_model, param_grid, fit_params = get_model_and_params(model_name)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring={
            "CV_AUC": "roc_auc",
            "CV_PR_AUC": "average_precision",
        },
        refit=REFIT_METRIC,
        cv=cv_strategy,
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    search.fit(X_train, y_train, **fit_params)

    best_idx = search.best_index_
    best_params = search.best_params_

    print(f"Best params: {best_params}")
    print(f"Best {REFIT_METRIC}: {search.best_score_:.4f}")
    print(
        f"Best CV_AUC={search.cv_results_['mean_test_CV_AUC'][best_idx]:.4f}, "
        f"Best CV_PR_AUC={search.cv_results_['mean_test_CV_PR_AUC'][best_idx]:.4f}"
    )

    # 保存每个模型的全部 search 结果
    cv_result = pd.DataFrame(search.cv_results_)
    cv_result["Model"] = model_name
    search_rows_all.append(cv_result)

    # 用最佳参数重新做 5-fold CV，收集完整指标
    fold_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv_strategy.split(X_train, y_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]

        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        model = clone(search.best_estimator_)

        if model_name in MODELS_NEED_SAMPLE_WEIGHT:
            sw_tr = compute_sample_weight(class_weight="balanced", y=y_tr)
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_va)[:, 1]
        metrics = compute_metrics(y_va, y_prob, threshold=THRESHOLD)

        row = {
            "Model": model_name,
            "Stage": "Tuned",
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

    row = {
        "Model": model_name,
        "Stage": "Tuned",
        "Refit_Metric": REFIT_METRIC,
        "Best_Params": json.dumps(best_params, ensure_ascii=False),
        "Search_Best_Score": round(float(search.best_score_), 6),
        "Search_Best_CV_AUC": round(float(search.cv_results_["mean_test_CV_AUC"][best_idx]), 6),
        "Search_Best_CV_PR_AUC": round(float(search.cv_results_["mean_test_CV_PR_AUC"][best_idx]), 6),
        "Threshold": THRESHOLD,
        "N_train_total": len(y_train),
        "Positive_total": int(y_train.sum()),
        "Negative_total": int((y_train == 0).sum()),
    }
    row.update(summary_metrics)
    summary_rows.append(row)

    print("\nCV summary:")
    print(f"  AUC         = {summary_metrics['AUC']:.4f} ± {summary_metrics['AUC_std']:.4f}")
    print(f"  PR_AUC      = {summary_metrics['PR_AUC']:.4f} ± {summary_metrics['PR_AUC_std']:.4f}")
    print(f"  Sensitivity = {summary_metrics['Sensitivity']:.4f} ± {summary_metrics['Sensitivity_std']:.4f}")
    print(f"  Specificity = {summary_metrics['Specificity']:.4f} ± {summary_metrics['Specificity_std']:.4f}")
    print(f"  F1          = {summary_metrics['F1']:.4f} ± {summary_metrics['F1_std']:.4f}")

    # 在整个 train 上重新拟合最终 tuned 模型
    final_model = clone(search.best_estimator_)

    if model_name in MODELS_NEED_SAMPLE_WEIGHT:
        final_model.fit(X_train, y_train, sample_weight=sample_weights_full)
    else:
        final_model.fit(X_train, y_train)

    model_path = MODEL_DIR / f"tuned_{model_name.lower()}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    print(f"Saved final train-fitted tuned model to: {model_path}")


# ============================================================
# 6. Save outputs
# ============================================================
summary_df = pd.DataFrame(summary_rows)
fold_df = pd.DataFrame(fold_rows_all)
search_df = pd.concat(search_rows_all, ignore_index=True)

summary_path = OUTPUT_DIR / "cv_metrics_tuned.csv"
fold_path = OUTPUT_DIR / "cv_fold_metrics_tuned.csv"
search_path = OUTPUT_DIR / "random_search_all_results.csv"

summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
search_df.to_csv(search_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Tuned CV summary saved to : {summary_path}")
print(f"Tuned fold metrics saved to: {fold_path}")
print(f"Search results saved to    : {search_path}")
print(f"Tuned models saved to      : {MODEL_DIR}")

print("\nImportant:")
print("Only train_processed.csv was used for hyperparameter tuning and model fitting.")
print("internal_test_processed.csv and external_processed.csv were NOT used here.")
