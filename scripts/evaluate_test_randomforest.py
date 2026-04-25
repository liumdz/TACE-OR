import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "randomforest"
MODEL_DISPLAY = "Random Forest"

DATA_DIR = ROOT / "data" / "processed"
TEST_PATH = DATA_DIR / "internal_test_processed.csv"

TUNED_MODEL_PATH = ROOT / "output" / "tuned" / "models" / "tuned_randomforest.pkl"
TUNED_FEATURE_JSON = ROOT / "output" / "tuned" / "feature_cols_tuned.json"

TOP10_MODEL_DIR = ROOT / "output" / "top10_model" / MODEL_NAME
TOP10_MODEL_PATH = TOP10_MODEL_DIR / "top10_model.pkl"
FEATURE_INFO_PATH = TOP10_MODEL_DIR / "feature_info.pkl"

THRESH_DIR = ROOT / "output" / "threshold" / MODEL_NAME
THRESH_PATH = THRESH_DIR / "best_threshold_full_vs_top10.csv"

RESULT_DIR = ROOT / "output" / "test_results" / MODEL_NAME
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"

FULL_COLOR =  "#BDBDBD"  
TOP10_COLOR = "#2E77BB"  

COMMON_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_unified",
    [FULL_COLOR, "#D6E4F0", TOP10_COLOR]
)

print("=" * 80)
print(f"INTERNAL TEST SET EVALUATION — {MODEL_DISPLAY.upper()}")
print("Full RandomForest vs Top10 RandomForest")
print("=" * 80)


# ============================================================
# 1. UTILS
# ============================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_features(model, fallback_json=None, model_key=None):
    """
    Prefer model.feature_names_in_.
    If unavailable, try JSON feature list.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if fallback_json is not None and Path(fallback_json).exists():
        with open(fallback_json, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            candidate_keys = [
                model_key,
                "RandomForest",
                "randomforest",
                "features",
                "feature_cols",
                "feature_columns",
            ]

            for key in candidate_keys:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]

    return None


def load_top10_features(top10_model, feature_info_path):
    """
    Current preferred source:
    1. top10_model.feature_names_in_
    2. feature_info.pkl['top10_processed_columns']
    3. feature_info.pkl['original_features_for_top10'] for old compatibility
    """
    if hasattr(top10_model, "feature_names_in_"):
        cols = list(top10_model.feature_names_in_)
        print("  ✓ Top10 feature columns loaded from model.feature_names_in_")
        return cols

    if not feature_info_path.exists():
        raise FileNotFoundError(f"Cannot find feature_info.pkl: {feature_info_path}")

    feature_info = load_pickle(feature_info_path)

    if not isinstance(feature_info, dict):
        raise TypeError(
            f"feature_info.pkl should be a dict, but got: {type(feature_info)}"
        )

    if "top10_processed_columns" in feature_info:
        cols = list(feature_info["top10_processed_columns"])
        print("  ✓ Top10 feature columns loaded from feature_info.pkl: top10_processed_columns")
        return cols

    if "original_features_for_top10" in feature_info:
        cols = list(feature_info["original_features_for_top10"])
        print("  ✓ Top10 feature columns loaded from feature_info.pkl: original_features_for_top10")
        return cols

    raise KeyError(
        "feature_info.pkl must contain 'top10_processed_columns' "
        "or 'original_features_for_top10'. "
        f"Available keys: {list(feature_info.keys())}"
    )


def prepare_X(df, feature_cols, model, model_label):
    """
    Strictly build X using the expected feature columns and preserve order.
    """
    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"\nMissing required columns for {model_label}:\n{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    X = df[feature_cols].copy()

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)

        if model_features != list(X.columns):
            raise ValueError(
                f"\nFeature names/order do not match the fitted {model_label} model.\n\n"
                f"Model features:\n{model_features}\n\n"
                f"Current features:\n{list(X.columns)}"
            )

    return X


def load_thresholds(thresh_path):
    """
    Load thresholds from:
    output/threshold/randomforest/best_threshold_full_vs_top10.csv

    Expected rows:
    Model = Full_RandomForest
    Model = Top10_RandomForest
    """
    if not thresh_path.exists():
        raise FileNotFoundError(
            f"Cannot find threshold file:\n{thresh_path}\n\n"
            "Please run scripts/threshold_selection_randomforest_full_vs_top10.py first."
        )

    thresh_df = pd.read_csv(thresh_path)

    required_cols = {"Model", "Best_Threshold"}
    missing_cols = required_cols - set(thresh_df.columns)

    if missing_cols:
        raise ValueError(
            f"Threshold file missing columns: {missing_cols}\n"
            f"Available columns: {thresh_df.columns.tolist()}"
        )

    threshold_map = {}

    for _, row in thresh_df.iterrows():
        threshold_map[row["Model"]] = float(row["Best_Threshold"])

    if "Full_RandomForest" not in threshold_map:
        raise ValueError(
            "Cannot find Full_RandomForest threshold in threshold file. "
            f"Available models: {list(threshold_map.keys())}"
        )

    if "Top10_RandomForest" not in threshold_map:
        raise ValueError(
            "Cannot find Top10_RandomForest threshold in threshold file. "
            f"Available models: {list(threshold_map.keys())}"
        )

    return threshold_map, thresh_df


def calc_metrics(y_true, y_probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "Threshold": round(float(threshold), 4),
        "AUC": round(roc_auc_score(y_true, y_probs), 4),
        "AUPRC": round(average_precision_score(y_true, y_probs), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision, 4),
        "Sensitivity": round(sensitivity, 4),
        "Recall": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F1": round(f1, 4),
        "Youden": round(sensitivity + specificity - 1, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# 2. LOAD INTERNAL TEST DATA
# ============================================================
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Cannot find internal test file: {TEST_PATH}")

test_df = pd.read_csv(TEST_PATH)

if TARGET_COL not in test_df.columns:
    raise KeyError(
        f"Cannot find target column '{TARGET_COL}' in {TEST_PATH}. "
        f"Available columns: {test_df.columns.tolist()}"
    )

y_test = test_df[TARGET_COL].astype(int).values

print(f"\n[1] Internal test data")
print(f"Path: {TEST_PATH}")
print(f"Samples: {len(test_df)}")
print(f"Positive: {int(y_test.sum())} ({np.mean(y_test) * 100:.1f}%)")
print("Label distribution:")
print(pd.Series(y_test).value_counts().sort_index())
print(pd.Series(y_test).value_counts(normalize=True).sort_index().round(4))


# ============================================================
# 3. LOAD MODELS
# ============================================================
for path in [TOP10_MODEL_PATH, FEATURE_INFO_PATH, TUNED_MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required file: {path}")

top10_model = load_pickle(TOP10_MODEL_PATH)
full_model = load_pickle(TUNED_MODEL_PATH)

print(f"\n[2] Loaded models")
print(f"Top10 model: {TOP10_MODEL_PATH}")
print(f"Full model : {TUNED_MODEL_PATH}")


# ============================================================
# 4. LOAD FEATURE COLUMNS
# ============================================================
top10_cols = load_top10_features(top10_model, FEATURE_INFO_PATH)

full_cols = load_model_features(
    full_model,
    fallback_json=TUNED_FEATURE_JSON,
    model_key="RandomForest",
)

if full_cols is None:
    raise RuntimeError(
        "Cannot determine full model feature columns. "
        "The full model has no feature_names_in_, and feature_cols_tuned.json was not usable."
    )

print(f"\n[3] Feature columns")
print(f"Top10 features: {len(top10_cols)}")
for c in top10_cols:
    print(f"  - {c}")
print(f"Full features: {len(full_cols)}")

X_test_top10 = prepare_X(
    test_df,
    top10_cols,
    top10_model,
    model_label="Top10 RandomForest",
)

X_test_full = prepare_X(
    test_df,
    full_cols,
    full_model,
    model_label="Full RandomForest",
)


# ============================================================
# 5. LOAD THRESHOLDS
# ============================================================
threshold_map, threshold_df = load_thresholds(THRESH_PATH)

best_thresh_full = threshold_map["Full_RandomForest"]
best_thresh_top10 = threshold_map["Top10_RandomForest"]

print(f"\n[4] Thresholds")
print(f"Threshold path: {THRESH_PATH}")
print(threshold_df)
print(f"Full RandomForest threshold : {best_thresh_full:.2f}")
print(f"Top10 RandomForest threshold: {best_thresh_top10:.2f}")


# ============================================================
# 6. PREDICTION
# ============================================================
y_probs_top10 = top10_model.predict_proba(X_test_top10)[:, 1]
y_probs_full = full_model.predict_proba(X_test_full)[:, 1]


# ============================================================
# 7. METRICS
# ============================================================
m_top10 = calc_metrics(y_test, y_probs_top10, best_thresh_top10)
m_full = calc_metrics(y_test, y_probs_full, best_thresh_full)

METRICS = [
    "AUC",
    "AUPRC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "Youden",
]

print(f"\n[5] Internal test metrics")
print(f"{'Metric':<13} | {'Top10 RF':^10} {'Full RF':^10} {'Diff':^10}")
print("-" * 52)

for k in METRICS:
    d = m_top10[k] - m_full[k]
    print(
        f"{k:<13} | "
        f"{m_top10[k]:<10.4f} "
        f"{m_full[k]:<10.4f} "
        f"{'+' if d >= 0 else ''}{d:.4f}"
    )

metrics_df = pd.DataFrame(
    [
        {"Model": "Top10_RandomForest", **m_top10},
        {"Model": "Full_RandomForest", **m_full},
    ]
)

metrics_path = RESULT_DIR / "internal_test_metrics_full_vs_top10.csv"
metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
print(f"\nSaved metrics: {metrics_path}")

pred_path = RESULT_DIR / "internal_test_predictions_full_vs_top10.csv"
pred_df = pd.DataFrame({
    "y_true": y_test.astype(int),
    "Top10_RandomForest_y_prob": y_probs_top10.astype(float),
    "Top10_RandomForest_threshold": best_thresh_top10,
    "Top10_RandomForest_y_pred": (y_probs_top10 >= best_thresh_top10).astype(int),
    "Full_RandomForest_y_prob": y_probs_full.astype(float),
    "Full_RandomForest_threshold": best_thresh_full,
    "Full_RandomForest_y_pred": (y_probs_full >= best_thresh_full).astype(int),
})

pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
print(f"Saved predictions: {pred_path}")


# ============================================================
# 8. BAR PLOT: BLUE AND GRAY
# ============================================================
x = np.arange(len(METRICS))
w = 0.35

fig, ax = plt.subplots(figsize=(13, 6))

b1 = ax.bar(
    x - w / 2,
    [m_top10[k] for k in METRICS],
    w,
    label=f"Top10 RandomForest (threshold={best_thresh_top10:.2f})",
    color=TOP10_COLOR,
    alpha=0.88,
    edgecolor="black",
    linewidth=0.8,
)

b2 = ax.bar(
    x + w / 2,
    [m_full[k] for k in METRICS],
    w,
    label=f"Full RandomForest (threshold={best_thresh_full:.2f})",
    color=FULL_COLOR,
    alpha=0.88,
    edgecolor="black",
    linewidth=0.8,
)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.012,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title(
    "Random Forest: Top10 vs Full — Internal Test Set",
    fontsize=14,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(fontsize=10, frameon=False)

plt.tight_layout()

bar_png = RESULT_DIR / "internal_test_comparison_bar_full_vs_top10.png"
bar_pdf = RESULT_DIR / "internal_test_comparison_bar_full_vs_top10.pdf"

plt.savefig(bar_png, dpi=300, bbox_inches="tight")
plt.savefig(bar_pdf, bbox_inches="tight")
plt.close()

print(f"Saved figure: {bar_png}")
print(f"Saved figure: {bar_pdf}")


# ============================================================
# 9. ROC AND PR CURVES: BLUE AND GRAY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
fpr_t, tpr_t, _ = roc_curve(y_test, y_probs_top10)
fpr_f, tpr_f, _ = roc_curve(y_test, y_probs_full)

axes[0].plot(
    fpr_t,
    tpr_t,
    color=TOP10_COLOR,
    lw=2.2,
    label=f"Top10 RF (AUC={m_top10['AUC']:.3f})",
)

axes[0].plot(
    fpr_f,
    tpr_f,
    color=FULL_COLOR,
    lw=2.2,
    linestyle="--",
    label=f"Full RF (AUC={m_full['AUC']:.3f})",
)

axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

axes[0].scatter(
    [1 - m_top10["Specificity"]],
    [m_top10["Sensitivity"]],
    s=120,
    color=TOP10_COLOR,
    edgecolor="black",
    linewidth=0.8,
    zorder=5,
    label=f"Top10 threshold={best_thresh_top10:.2f}",
)

axes[0].scatter(
    [1 - m_full["Specificity"]],
    [m_full["Sensitivity"]],
    s=120,
    color=FULL_COLOR,
    edgecolor="black",
    linewidth=0.8,
    marker="s",
    zorder=5,
    label=f"Full threshold={best_thresh_full:.2f}",
)

axes[0].set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
axes[0].set_ylabel("True Positive Rate / Sensitivity", fontsize=11, fontweight="bold")
axes[0].set_title("ROC Curve — Internal Test Set", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=8.5, frameon=False)
axes[0].grid(alpha=0.3, linestyle="--")


# PR
prec_t, rec_t, _ = precision_recall_curve(y_test, y_probs_top10)
prec_f, rec_f, _ = precision_recall_curve(y_test, y_probs_full)

axes[1].plot(
    rec_t,
    prec_t,
    color=TOP10_COLOR,
    lw=2.2,
    label=f"Top10 RF (AUPRC={m_top10['AUPRC']:.3f})",
)

axes[1].plot(
    rec_f,
    prec_f,
    color=FULL_COLOR,
    lw=2.2,
    linestyle="--",
    label=f"Full RF (AUPRC={m_full['AUPRC']:.3f})",
)

axes[1].axhline(
    y=np.mean(y_test),
    color="black",
    linestyle=":",
    lw=1.2,
    alpha=0.5,
    label=f"Prevalence={np.mean(y_test):.3f}",
)

axes[1].scatter(
    [m_top10["Sensitivity"]],
    [m_top10["Precision"]],
    s=120,
    color=TOP10_COLOR,
    edgecolor="black",
    linewidth=0.8,
    zorder=5,
    label=f"Top10 threshold={best_thresh_top10:.2f}",
)

axes[1].scatter(
    [m_full["Sensitivity"]],
    [m_full["Precision"]],
    s=120,
    color=FULL_COLOR,
    edgecolor="black",
    linewidth=0.8,
    marker="s",
    zorder=5,
    label=f"Full threshold={best_thresh_full:.2f}",
)

axes[1].set_xlabel("Recall / Sensitivity", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Precision", fontsize=11, fontweight="bold")
axes[1].set_title("Precision-Recall Curve — Internal Test Set", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=8.5, frameon=False)
axes[1].grid(alpha=0.3, linestyle="--")

fig.suptitle(
    "Random Forest: Top10 vs Full — Internal Test Set",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()

roc_pr_png = RESULT_DIR / "internal_test_roc_pr_full_vs_top10.png"
roc_pr_pdf = RESULT_DIR / "internal_test_roc_pr_full_vs_top10.pdf"

plt.savefig(roc_pr_png, dpi=300, bbox_inches="tight")
plt.savefig(roc_pr_pdf, bbox_inches="tight")
plt.close()

print(f"Saved figure: {roc_pr_png}")
print(f"Saved figure: {roc_pr_pdf}")


# ============================================================
# 10. DONE
# ============================================================
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Output directory: {RESULT_DIR}")
print("\nImportant:")
print("1. This script uses internal_test_processed.csv only.")
print("2. This script does NOT train or refit any model.")
print("3. Thresholds are loaded from best_threshold_full_vs_top10.csv.")
print("4. Top10 features are loaded from feature_info.pkl.")
print("5. Colors: Top10 = blue, Full = gray.")