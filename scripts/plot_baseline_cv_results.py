import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
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
RESULT_DIR = ROOT / "output" / "baseline"
FIG_DIR = ROOT / "output" / "figures" / "baseline_cv"

FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_processed.csv"
SUMMARY_PATH = RESULT_DIR / "cv_metrics_baseline.csv"

OOF_PATH = RESULT_DIR / "baseline_oof_predictions.csv"

LABEL_COL = "label"
N_SPLITS = 5


# ============================================================
# 1. Candidate baseline models
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


# 图中使用全称；SVM 按你的要求显示为 SVM
display_names = {
    "LogisticRegression": "Logistic Regression",
    "ElasticNet": "Elastic Net",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "SVM": "SVM",
}

model_order = [
    "LogisticRegression",
    "ElasticNet",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "SVM",
]


# ============================================================
# 2. Utility
# ============================================================
def save_fig(name):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# 3. Load data
# ============================================================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {TRAIN_PATH}. Please run scripts/preprocess_features.py first."
    )

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {SUMMARY_PATH}. Please run scripts/train_models.py first."
    )

train = pd.read_csv(TRAIN_PATH)
summary = pd.read_csv(SUMMARY_PATH)

feature_cols = [c for c in train.columns if c != LABEL_COL]
X = train[feature_cols].copy()
y = train[LABEL_COL].astype(int).copy()

summary = summary.set_index("Model").loc[model_order].reset_index()
summary["Display"] = summary["Model"].map(display_names)

print("=" * 80)
print("PLOT BASELINE CV RESULTS")
print("=" * 80)
print(f"Training data: {X.shape}")
print("Label distribution:")
print(y.value_counts().sort_index())
print(y.value_counts(normalize=True).sort_index().round(4))


# ============================================================
# 4. Generate out-of-fold predictions for ROC / PR curves
# ============================================================
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

oof_rows = []

for model_name in model_order:
    print(f"\nGenerating OOF predictions: {display_names[model_name]}")

    base_model = models[model_name]
    oof_prob = np.zeros(len(y), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]

        X_va = X.iloc[va_idx]

        model = clone(base_model)
        model.fit(X_tr, y_tr)

        prob = model.predict_proba(X_va)[:, 1]
        oof_prob[va_idx] = prob

        for idx, p in zip(va_idx, prob):
            oof_rows.append({
                "Model": model_name,
                "Model_Display": display_names[model_name],
                "Fold": fold_idx,
                "Sample_Index": int(idx),
                "y_true": int(y.iloc[idx]),
                "y_prob": float(p),
            })

oof_df = pd.DataFrame(oof_rows)
oof_df.to_csv(OOF_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved OOF predictions to: {OOF_PATH}")


# ============================================================
# 5. Figure 1: ROC curves
# ============================================================
plt.figure(figsize=(8.5, 7))

for model_name in model_order:
    sub = oof_df[oof_df["Model"] == model_name].sort_values("Sample_Index")

    y_true = sub["y_true"].values
    y_prob = sub["y_prob"].values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{display_names[model_name]} (AUC={auc_value:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Reference")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("ROC Curves of Baseline Models in 5-Fold Cross-Validation")
plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.legend(loc="lower right", frameon=False, fontsize=8)

save_fig("fig1_baseline_cv_roc_curves")


# ============================================================
# 6. Figure 2: Precision-Recall / AUPRC curves
# ============================================================
positive_rate = y.mean()

plt.figure(figsize=(8.5, 7))

for model_name in model_order:
    sub = oof_df[oof_df["Model"] == model_name].sort_values("Sample_Index")

    y_true = sub["y_true"].values
    y_prob = sub["y_prob"].values

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_value = average_precision_score(y_true, y_prob)

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{display_names[model_name]} (AUPRC={ap_value:.3f})"
    )

plt.hlines(
    positive_rate,
    xmin=0,
    xmax=1,
    linestyles="--",
    linewidth=1,
    label=f"Baseline prevalence ({positive_rate:.3f})"
)

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves of Baseline Models in 5-Fold Cross-Validation")
plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.legend(loc="upper right", frameon=False, fontsize=8)

save_fig("fig2_baseline_cv_pr_curves")


# ============================================================
# 7. Figure 3: All metrics grouped bar chart
#    Less colorful + value labels + short error bars
# ============================================================
metrics = [
    "AUC",
    "PR_AUC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
]

metric_labels = {
    "AUC": "AUC",
    "PR_AUC": "AUPRC",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Sensitivity": "Sensitivity",
    "Specificity": "Specificity",
    "F1": "F1 score",
}

# 灰色 + 蓝色系，避免颜色太杂
metric_colors = {
    "AUC": "#BDBDBD",          # light gray
    "PR_AUC": "#9ECAE1",      # light blue
    "Accuracy": "#D9D9D9",    # very light gray
    "Precision": "#6BAED6",   # medium blue
    "Sensitivity": "#4292C6", # blue
    "Specificity": "#7F7F7F", # dark gray
    "F1": "#2171B5",          # dark blue
}

x = np.arange(len(summary))
width = 0.105

fig, ax = plt.subplots(figsize=(15, 7.5))

all_tops = []

for i, metric in enumerate(metrics):
    offset = (i - (len(metrics) - 1) / 2) * width

    vals = summary[metric].astype(float).values

    std_col = f"{metric}_std"
    if std_col in summary.columns:
        errs = summary[std_col].astype(float).values
    else:
        errs = np.zeros(len(summary))

    bars = ax.bar(
        x + offset,
        vals,
        width,
        yerr=errs,
        capsize=3,
        color=metric_colors[metric],
        alpha=0.90,
        edgecolor="black",
        linewidth=0.45,
        ecolor="black",
        error_kw={
            "elinewidth": 0.9,
            "capthick": 0.9,
        },
        label=metric_labels[metric],
    )

    all_tops.extend(vals + errs)

    # 在柱子上方标注具体数值
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels(
    summary["Display"],
    rotation=25,
    ha="right",
    fontsize=10,
)

ax.set_ylabel("Score", fontsize=12, fontweight="bold")

# 自动给误差线和数值标注留空间
y_top = max(all_tops) if len(all_tops) > 0 else 1.0
ax.set_ylim(0, min(1.20, max(1.05, y_top + 0.12)))

ax.set_title(
    "Overall Performance Metrics of Baseline Models in 5-Fold Cross-Validation",
    fontsize=14,
    fontweight="bold",
)

ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False, ncol=4, fontsize=9)

plt.tight_layout()

save_fig("fig3_baseline_cv_all_metrics_bar")


# ============================================================
# 8. Save compact plotting table
# ============================================================
plot_table = summary[
    ["Model", "Display"]
    + metrics
    + [f"{m}_std" for m in metrics if f"{m}_std" in summary.columns]
].copy()

plot_table.to_csv(
    RESULT_DIR / "baseline_plot_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAll selected baseline figures saved to:")
print(FIG_DIR)

print("\nGenerated figures:")
print("1) fig1_baseline_cv_roc_curves.png / .pdf")
print("2) fig2_baseline_cv_pr_curves.png / .pdf")
print("3) fig3_baseline_cv_all_metrics_bar.png / .pdf")

print("\nNote:")
print("ROC and PR curves are based on out-of-fold predictions from the training-set 5-fold CV.")
print("No internal test or external validation data were used in these figures.")