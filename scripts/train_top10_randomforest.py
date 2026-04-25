import os
import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

MODEL_NAME = "randomforest"

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
SHAP_DIR = ROOT / "output" / "shap_analysis" / MODEL_NAME
RESULT_DIR = ROOT / "output" / "top10_model" / MODEL_NAME
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_processed.csv"

TOP10_PATH = SHAP_DIR / "Top10_features_merged.csv"
MERGED_SHAP_PATH = SHAP_DIR / "merged_shap_importance.csv"

MODEL_CANDIDATES = [
    ROOT / "output" / "tuned" / "models" / "tuned_randomforest.pkl",
    ROOT / "output" / "tuned" / "tuned_randomforest.pkl",
]

TARGET_COL = "label"
N_SPLITS = 5
THRESHOLD = 0.5

print("=" * 80)
print("TRAIN TOP10 RANDOM FOREST MODEL")
print("Train data only. No internal test. No external validation.")
print("=" * 80)


# ============================================================
# 1. LOAD TRAIN DATA
# ============================================================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Cannot find train file: {TRAIN_PATH}")

train = pd.read_csv(TRAIN_PATH)

if TARGET_COL not in train.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in train_processed.csv")

# 重要：不要把空格替换成下划线
ALL_FEATURES = [c for c in train.columns if c != TARGET_COL]

X_train_full = train[ALL_FEATURES].copy()
y_train = train[TARGET_COL].astype(int).copy()

print(f"\n[1] Train data")
print(f"Samples: {len(train)}")
print(f"Full feature columns: {len(ALL_FEATURES)}")
print("Label distribution:")
print(y_train.value_counts().sort_index())
print(y_train.value_counts(normalize=True).sort_index().round(4))


# ============================================================
# 2. LOAD TOP10 FEATURES AND SHAP MERGE INFO
# ============================================================
if not TOP10_PATH.exists():
    raise FileNotFoundError(f"Cannot find Top10 file: {TOP10_PATH}")

if not MERGED_SHAP_PATH.exists():
    raise FileNotFoundError(f"Cannot find merged SHAP file: {MERGED_SHAP_PATH}")

top10_df = pd.read_csv(TOP10_PATH)
merged_df = pd.read_csv(MERGED_SHAP_PATH)

if "Feature" not in top10_df.columns:
    raise ValueError("Top10_features_merged.csv must contain a 'Feature' column.")

top10_features = top10_df["Feature"].tolist()

print("\n[2] Top 10 merged features from SHAP:")
for i, feat in enumerate(top10_features, start=1):
    print(f"  {i:2d}. {feat}")


# ============================================================
# 3. BUILD FEATURE GROUP MAP
# ============================================================
def split_merged_from(x):
    if pd.isna(x):
        return []
    return [c.strip() for c in str(x).split(",") if c.strip()]


def resolve_col(col, all_features):
    """
    将 merged feature 映射回 processed 数据中的实际列名。
    例如：
    AFP -> AFP_log
    diameter of tumor -> diameter of tumor
    PVTT -> PVTT_0, PVTT_1...
    """
    if col in all_features:
        return col

    if f"{col}_log" in all_features:
        return f"{col}_log"

    # 兼容极少数旧命名
    col_space = col.replace("_", " ")
    if col_space in all_features:
        return col_space

    if f"{col_space}_log" in all_features:
        return f"{col_space}_log"

    return None


group_map = {}

for _, row in merged_df.iterrows():
    feat = row["Feature"]
    merge_method = str(row.get("Merge_Method", "")).strip()
    merged_from = split_merged_from(row.get("Merged_From", ""))

    # onehot_sum: 用 Merged_From 里的所有 one-hot 列
    if merge_method in ["onehot_sum", "sum"]:
        source_cols = merged_from

    # original: 优先用 Merged_From，比如 AFP 对应 AFP_log
    else:
        source_cols = merged_from if merged_from else [feat]

    resolved_cols = []
    for c in source_cols:
        r = resolve_col(c, ALL_FEATURES)
        if r is not None:
            resolved_cols.append(r)

    group_map[feat] = resolved_cols


# 根据 Top10 展开成实际 processed 列
actual_top10_cols = []
seen = set()

for feat in top10_features:
    cols = group_map.get(feat, [])

    if not cols:
        # 兜底：尝试直接解析 feature 自身
        r = resolve_col(feat, ALL_FEATURES)
        if r is not None:
            cols = [r]

    if not cols:
        raise ValueError(
            f"Cannot map Top10 merged feature '{feat}' to processed columns.\n"
            f"Available train columns include:\n{ALL_FEATURES}"
        )

    for c in cols:
        if c not in seen:
            seen.add(c)
            actual_top10_cols.append(c)

print(f"\n[3] Top10 expanded to {len(actual_top10_cols)} processed columns:")
for c in actual_top10_cols:
    print(f"    - {c}")

X_train_top10 = train[actual_top10_cols].copy()


# ============================================================
# 4. LOAD TUNED RANDOM FOREST PARAMS
# ============================================================
model_path = None
for p in MODEL_CANDIDATES:
    if p.exists():
        model_path = p
        break

if model_path is None:
    raise FileNotFoundError(
        "Cannot find tuned RandomForest model. Checked:\n"
        + "\n".join(str(p) for p in MODEL_CANDIDATES)
    )

with open(model_path, "rb") as f:
    loaded_full_model = pickle.load(f)

if not hasattr(loaded_full_model, "get_params"):
    raise TypeError("Loaded model does not have get_params().")

params = loaded_full_model.get_params()
params["random_state"] = SEED

# 不重新 fit 已保存的 full model，这里只复制它的超参数
base_full_model = RandomForestClassifier(**params)
base_top10_model = RandomForestClassifier(**params)

print("\n[4] Loaded tuned RandomForest parameters")
print(f"Model path: {model_path}")
print("Important: loaded full model is NOT re-fitted here.")
print("Top10 model will be trained on train data only.")


# ============================================================
# 5. METRIC FUNCTION
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


METRIC_KEYS = [
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


# ============================================================
# 6. CV EVALUATION
# ============================================================
def cv_evaluate(model_template, X, y, model_label):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    fold_rows = []
    oof_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]

        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = clone(model_template)
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_va)[:, 1]
        metrics = compute_metrics(y_va, y_prob, threshold=THRESHOLD)

        row = {
            "Model": model_label,
            "Fold": fold_idx,
            "Threshold": THRESHOLD,
            "N_train_fold": len(y_tr),
            "N_valid_fold": len(y_va),
            "Positive_valid_fold": int(y_va.sum()),
            "Negative_valid_fold": int((y_va == 0).sum()),
        }
        row.update({
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        })
        fold_rows.append(row)

        for sample_idx, p, yt in zip(va_idx, y_prob, y_va):
            oof_rows.append({
                "Model": model_label,
                "Fold": fold_idx,
                "Sample_Index": int(sample_idx),
                "y_true": int(yt),
                "y_prob": float(p),
            })

    fold_df = pd.DataFrame(fold_rows)

    summary = {}
    for k in METRIC_KEYS:
        summary[k] = round(fold_df[k].mean(), 4)
        summary[f"{k}_std"] = round(fold_df[k].std(ddof=1), 4)

    return summary, fold_df, pd.DataFrame(oof_rows)


print("\n[5] 5-fold CV evaluation on train only...")

full_summary, full_fold_df, full_oof_df = cv_evaluate(
    base_full_model,
    X_train_full,
    y_train,
    "Full_RandomForest",
)

top10_summary, top10_fold_df, top10_oof_df = cv_evaluate(
    base_top10_model,
    X_train_top10,
    y_train,
    "Top10_RandomForest",
)


# ============================================================
# 7. PRINT COMPARISON
# ============================================================
print("\n[6] Full vs Top10 CV comparison")
print(f"{'Metric':<13} | {'Full RF':^18} {'Top10 RF':^18} {'Diff':^10}")
print("-" * 68)

comparison_rows = []

for k in METRIC_KEYS:
    diff = top10_summary[k] - full_summary[k]

    print(
        f"{k:<13} | "
        f"{full_summary[k]:.4f}±{full_summary[f'{k}_std']:.4f}   "
        f"{top10_summary[k]:.4f}±{top10_summary[f'{k}_std']:.4f}   "
        f"{diff:+.4f}"
    )

    comparison_rows.append({
        "Metric": k,
        "Full_CV": full_summary[k],
        "Full_CV_std": full_summary[f"{k}_std"],
        "Top10_CV": top10_summary[k],
        "Top10_CV_std": top10_summary[f"{k}_std"],
        "Diff_Top10_minus_Full": round(diff, 4),
    })

comparison_df = pd.DataFrame(comparison_rows)


# ============================================================
# 8. FIT FINAL TOP10 MODEL ON FULL TRAIN ONLY
# ============================================================
print("\n[7] Fit final Top10 RandomForest on full training set only...")

final_top10_model = clone(base_top10_model)
final_top10_model.fit(X_train_top10, y_train)

print("Top10 final model fitted.")


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================
top10_model_path = RESULT_DIR / "top10_model.pkl"
feature_info_path = RESULT_DIR / "feature_info.pkl"
feature_info_json_path = RESULT_DIR / "feature_info.json"

comparison_path = RESULT_DIR / "cv_performance_comparison.csv"
fold_path = RESULT_DIR / "cv_fold_metrics_full_vs_top10.csv"
oof_path = RESULT_DIR / "oof_predictions_full_vs_top10.csv"

with open(top10_model_path, "wb") as f:
    pickle.dump(final_top10_model, f)

feature_info = {
    "model_name": MODEL_NAME,
    "source_full_model_path": str(model_path),
    "top10_merged_features": top10_features,
    "top10_processed_columns": actual_top10_cols,
    "all_processed_features": ALL_FEATURES,
    "group_map": group_map,
    "note": (
        "Top10 model was trained using train_processed.csv only. "
        "No internal test or external validation data were used."
    ),
}

with open(feature_info_path, "wb") as f:
    pickle.dump(feature_info, f)

with open(feature_info_json_path, "w", encoding="utf-8") as f:
    json.dump(feature_info, f, indent=2, ensure_ascii=False)

comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

fold_df = pd.concat([full_fold_df, top10_fold_df], ignore_index=True)
fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")

oof_df = pd.concat([full_oof_df, top10_oof_df], ignore_index=True)
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")


print("\n[8] Saved outputs")
print(f"Top10 model: {top10_model_path}")
print(f"Feature info: {feature_info_path}")
print(f"Feature info JSON: {feature_info_json_path}")
print(f"CV comparison: {comparison_path}")
print(f"Fold metrics: {fold_path}")
print(f"OOF predictions: {oof_path}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Full RF CV AUC : {full_summary['AUC']:.4f}")
print(f"Top10 RF CV AUC: {top10_summary['AUC']:.4f}")
print("\nImportant:")
print("1. This script uses train_processed.csv only.")
print("2. It does NOT read internal_test_processed.csv.")
print("3. It does NOT read external_processed.csv.")
print("4. The loaded full RandomForest model is not re-fitted.")
print("5. The Top10 simplified model is trained on full train only after CV.")
