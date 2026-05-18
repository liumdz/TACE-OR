import os
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


ACRONYMS = {
    "APTT", "AFP", "PVTT", "ALP", "GGT", "ALT", "AST",
    "PA", "PT", "TT", "ALB", "DBIL", "TBIL", "HBsAg",
    "BCLC", "TACE",
}


def prettify_feature_name(name: str) -> str:
    if not name:
        return name
    tokens = name.split()
    if not tokens:
        return name
    out = []
    for i, tok in enumerate(tokens):
        if tok in ACRONYMS:
            out.append(tok)
        elif i == 0:
            out.append(tok[0].upper() + tok[1:])
        else:
            out.append(tok)
    return " ".join(out)


# ============================================================
# 0. SETTINGS
# ============================================================
MODEL_NAME = "RandomForest"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "output" / "shap_analysis" / MODEL_NAME.lower()
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_processed.csv"

# 兼容两种模型保存位置
MODEL_CANDIDATES = [
    ROOT / "output" / "tuned" / "models" / "tuned_randomforest.pkl",
    ROOT / "output" / "tuned" / "tuned_randomforest.pkl",
]

TARGET_COL = "label"

print("=" * 80)
print(f"SHAP ANALYSIS — {MODEL_NAME}")
print("Use loaded tuned model directly. NO re-fit. Train data only.")
print("=" * 80)


# ============================================================
# 1. LOAD TRAIN DATA ONLY
# ============================================================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Cannot find train data: {TRAIN_PATH}")

train = pd.read_csv(TRAIN_PATH)

if TARGET_COL not in train.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

FEATURE_COLS = [c for c in train.columns if c != TARGET_COL]
X_train = train[FEATURE_COLS].copy()
y_train = train[TARGET_COL].astype(int).copy()

print(f"Train data: {X_train.shape}")
print("Label distribution:")
print(y_train.value_counts().sort_index())
print(y_train.value_counts(normalize=True).sort_index().round(4))


# ============================================================
# 2. LOAD TUNED MODEL — DO NOT FIT AGAIN
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
    model = pickle.load(f)

print(f"\nLoaded tuned model from: {model_path}")
print(f"Model type: {type(model).__name__}")
print("Important: model.fit() is NOT called in this script.")

# 检查模型是否已经训练
if not hasattr(model, "classes_"):
    raise RuntimeError(
        "Loaded model does not appear to be fitted. "
        "Please run hyperparameter_tuning.py first."
    )

# 如果模型记录了训练时特征名，检查是否一致
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if model_features != FEATURE_COLS:
        missing_in_train = sorted(set(model_features) - set(FEATURE_COLS))
        extra_in_train = sorted(set(FEATURE_COLS) - set(model_features))
        raise ValueError(
            "Feature names in train_processed.csv do not match the loaded model.\n"
            f"Missing in current train: {missing_in_train}\n"
            f"Extra in current train: {extra_in_train}\n"
            "Do not replace spaces with underscores before SHAP."
        )


# ============================================================
# 3. COMPUTE SHAP VALUES FOR POSITIVE CLASS label=1
# ============================================================
print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_train)

classes = list(getattr(model, "classes_", [0, 1]))
positive_class_index = classes.index(1) if 1 in classes else -1

if isinstance(sv, list):
    # 旧版 shap: [class0, class1]
    shap_values = sv[positive_class_index]
elif isinstance(sv, np.ndarray) and sv.ndim == 3:
    # 新版 shap: (n_samples, n_features, n_classes)
    shap_values = sv[:, :, positive_class_index]
else:
    # 二分类单数组
    shap_values = sv

shap_values = np.asarray(shap_values)

if shap_values.shape[1] != len(FEATURE_COLS):
    raise ValueError(
        f"SHAP shape mismatch: shap_values={shap_values.shape}, "
        f"n_features={len(FEATURE_COLS)}"
    )

print(f"SHAP values shape: {shap_values.shape}")


# ============================================================
# 4. RAW FEATURE IMPORTANCE
# ============================================================
importance = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "SHAP_Importance": importance,
}).sort_values("SHAP_Importance", ascending=False).reset_index(drop=True)

shap_importance["Rank"] = np.arange(1, len(shap_importance) + 1)
shap_importance["Importance_Percentage"] = (
    shap_importance["SHAP_Importance"] /
    shap_importance["SHAP_Importance"].sum() * 100
)

raw_importance_path = RESULT_DIR / "all_features_shap_importance.csv"
shap_importance.to_csv(raw_importance_path, index=False, encoding="utf-8-sig")
print(f"Saved: {raw_importance_path}")


# ============================================================
# 5. MERGE ONE-HOT FEATURES
# ============================================================
ONE_HOT_PREFIXES = {
    "PVTT": "PVTT_",
    "combined with other treatment": "combined with other treatment_",
    "tumor location": "tumor location_",
    "BCLC stage": "BCLC stage_",
}

col_index = {col: i for i, col in enumerate(FEATURE_COLS)}

onehot_groups = {}
for group_name, prefix in ONE_HOT_PREFIXES.items():
    cols = [c for c in FEATURE_COLS if c.startswith(prefix)]
    if cols:
        onehot_groups[group_name] = cols

all_onehot_cols = sorted({c for cols in onehot_groups.values() for c in cols})
non_onehot_cols = [c for c in FEATURE_COLS if c not in all_onehot_cols]

merged_X_dict = {}
merged_shap_list = []
merged_feat_names = []
merged_from = {}

# 合并 one-hot：SHAP 求和
for group_name, cols in onehot_groups.items():
    valid_cols = [c for c in cols if c in col_index]
    if not valid_cols:
        continue

    # 用 argmax 仅用于 beeswarm 的颜色展示
    merged_X_dict[group_name] = X_train[valid_cols].values.argmax(axis=1)
    merged_shap_list.append(
        np.sum([shap_values[:, col_index[c]] for c in valid_cols], axis=0)
    )
    merged_feat_names.append(group_name)
    merged_from[group_name] = valid_cols

# 非 one-hot 特征直接保留；log 特征显示时去掉 _log
for col in non_onehot_cols:
    display_name = col.replace("_log", "")
    merged_X_dict[display_name] = X_train[col].values
    merged_shap_list.append(shap_values[:, col_index[col]])
    merged_feat_names.append(display_name)
    merged_from[display_name] = [col]

merged_shap_matrix = np.column_stack(merged_shap_list)
merged_X_df = pd.DataFrame(merged_X_dict)[merged_feat_names]

merged_importance = np.abs(merged_shap_matrix).mean(axis=0)
sort_idx = np.argsort(merged_importance)[::-1]

merged_shap_sorted = merged_shap_matrix[:, sort_idx]
merged_X_sorted = merged_X_df.iloc[:, sort_idx]
merged_feat_sorted = [merged_feat_names[i] for i in sort_idx]
merged_imp_sorted = merged_importance[sort_idx]

print(f"\nMerged features: {len(FEATURE_COLS)} → {len(merged_feat_names)}")


def display_name(name: str) -> str:
    return prettify_feature_name(str(name).replace("_", " "))


merged_X_display = merged_X_sorted.copy()
merged_X_display.columns = [display_name(c) for c in merged_X_display.columns]


# ============================================================
# 6. SAVE MERGED IMPORTANCE
# ============================================================
merged_rows = []
total_imp = merged_imp_sorted.sum()

for rank, feat, imp in zip(
    range(1, len(merged_feat_sorted) + 1),
    merged_feat_sorted,
    merged_imp_sorted,
):
    source_cols = merged_from.get(feat, [feat])
    if feat in onehot_groups:
        merge_method = "onehot_sum"
    else:
        merge_method = "original"

    merged_rows.append({
        "Rank": rank,
        "Feature": feat,
        "Display_Feature": display_name(feat),
        "SHAP_Importance": imp,
        "Importance_Percentage": imp / total_imp * 100,
        "Merged_From": ", ".join(source_cols),
        "Num_Merged": len(source_cols),
        "Merge_Method": merge_method,
    })

merged_shap_df = pd.DataFrame(merged_rows)

merged_path = RESULT_DIR / "merged_shap_importance.csv"
merged_shap_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
print(f"Saved: {merged_path}")

top10 = merged_shap_df.head(10).copy()
top10_path = RESULT_DIR / "Top10_features_merged.csv"
top10.to_csv(top10_path, index=False, encoding="utf-8-sig")
print(f"Saved: {top10_path}")


# ============================================================
# 7. PLOT 1: SHAP BEESWARM
# ============================================================
plt.figure(figsize=(12, 10))
shap.summary_plot(
    merged_shap_sorted,
    merged_X_display,
    show=False,
    max_display=len(merged_feat_names),
)
ax = plt.gca()
ax.set_xlabel("SHAP value (impact on model output)", fontsize=18, fontweight="bold")
ax.tick_params(axis="both", labelsize=16)
# 调整 colorbar 字号
for child in plt.gcf().get_children():
    if hasattr(child, "yaxis") and child is not ax:
        child.tick_params(labelsize=14)
        if child.get_ylabel():
            child.set_ylabel(child.get_ylabel(), fontsize=16, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out = RESULT_DIR / "shap_summary_all_features.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ============================================================
# 8. PLOT 2: SHAP BAR ALL FEATURES
# ============================================================
plt.figure(figsize=(10, 12))

shap.summary_plot(
    merged_shap_sorted,
    merged_X_display,
    plot_type="bar",
    show=False,
    max_display=len(merged_feat_names),
)

ax = plt.gca()
ax.set_xlabel(
    "Mean |SHAP value|",
    fontsize=18,
    fontweight="bold"
)
ax.tick_params(axis="both", labelsize=16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out = RESULT_DIR / "shap_bar_all_features.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out}")


# ============================================================
# 9. PLOT 3: TOP10 HORIZONTAL BAR
# ============================================================
top10_display = top10["Display_Feature"].values
top10_imp = top10["SHAP_Importance"].values
top10_pct = top10["Importance_Percentage"].values

fig, ax = plt.subplots(figsize=(14, 8))

y_pos = np.arange(len(top10))
bars = ax.barh(
    y_pos,
    top10_imp[::-1],
    alpha=0.85,
    edgecolor="black",
    linewidth=1.2,
)

ax.set_yticks(y_pos)
ax.set_yticklabels(top10_display[::-1], fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=10)

for i, (bar, val, pct) in enumerate(zip(bars, top10_imp[::-1], top10_pct[::-1])):
    ax.text(
        val + max(top10_imp) * 0.01,
        i,
        f"{val:.4f} ({pct:.1f}%)",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
out = RESULT_DIR / "Top10_features_bar.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ============================================================
# 10. PLOT 4: TOP10 CUMULATIVE IMPORTANCE
# ============================================================
from matplotlib.colors import LinearSegmentedColormap

fig, ax1 = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(top10))

# 蓝灰渐变配色（高重要性=深蓝，低重要性=浅灰蓝）
blue_gray_cmap = LinearSegmentedColormap.from_list(
    "blue_gray",
    ["#2F5D8A", "#5D7FA3", "#8FA8C4", "#BCCCDC", "#D9E2EC"]
)

colors_bar = blue_gray_cmap(np.linspace(0.05, 0.95, len(top10)))

bars = ax1.bar(
    x_pos,
    top10_imp,
    color=colors_bar,
    alpha=0.9,
    edgecolor="black",
    linewidth=1.2,
)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(
    top10_display,
    fontsize=15,
    fontweight="bold",
    rotation=45,
    ha="right",
)
ax1.set_ylabel("Mean |SHAP value|", fontsize=17, fontweight="bold")
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis="y", labelsize=14)

# 柱子数值标注
for i, (bar, val) in enumerate(zip(bars, top10_imp)):
    ax1.text(
        i,
        val + max(top10_imp) * 0.015,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="#2F3E4E",
    )

# 右轴：累计百分比
ax2 = ax1.twinx()
cumulative_pct = top10_pct.cumsum()

ax2.plot(
    x_pos,
    cumulative_pct,
    color="#1F4E79",
    marker="o",
    linewidth=3,
    markersize=8,
    markerfacecolor="#1F4E79",
    markeredgecolor="#1F4E79",
    label="Cumulative %",
)

ax2.set_ylabel(
    "Cumulative Importance (%)",
    fontsize=17,
    fontweight="bold",
    color="#1F4E79",
)
ax2.tick_params(axis="y", labelcolor="#1F4E79", labelsize=14)
ax2.set_ylim(0, 105)
ax2.spines["top"].set_visible(False)

# 累计百分比标注
for i, pct in enumerate(cumulative_pct):
    ax2.text(
        i,
        pct + 2,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#1F4E79",
    )

# 80% 阈值线
ax2.axhline(
    y=80,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="80% threshold",
)

ax2.legend(loc="lower right", fontsize=14, frameon=True)

plt.tight_layout()
out = RESULT_DIR / "Top10_features_cumulative.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ============================================================
# 11. DONE
# ============================================================
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Output directory: {RESULT_DIR}")
print("Important:")
print("1. This script uses train_processed.csv only.")
print("2. This script loads the tuned RandomForest model.")
print("3. This script does NOT call model.fit().")
print("4. No internal test or external validation data are used.")
