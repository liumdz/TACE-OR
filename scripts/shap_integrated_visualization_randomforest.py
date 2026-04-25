import os
import pickle
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")


# ==========================================
# 1. 基础配置
# ==========================================
MODEL_NAME = "randomforest"
SEED = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train_processed.csv"

PREPROC_PATH = ROOT / "output" / "preprocessor.pkl"

TOP10_MODEL_DIR = ROOT / "output" / "top10_model" / MODEL_NAME
FEATURE_INFO_PATH = TOP10_MODEL_DIR / "feature_info.pkl"
TOP10_MODEL_PATH = TOP10_MODEL_DIR / "top10_model.pkl"

OUT_DIR = ROOT / "output" / "shap_analysis" / f"{MODEL_NAME}_integrated_two_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"

np.random.seed(SEED)


# ==========================================
# 2. 工具函数
# ==========================================
def normalize_name(name):
    return str(name).strip().replace(" ", "_")


def to_display_name(name):
    return str(name).replace("_log", "").replace("_", " ")


def build_col_lookup(cols):
    lookup = {}

    for c in cols:
        c = str(c)
        lookup[c] = c
        lookup[normalize_name(c)] = c
        lookup[to_display_name(c)] = c
        lookup[normalize_name(to_display_name(c))] = c

    return lookup


def build_X_for_model_features(df, expected_features):
    """
    按模型训练时的 feature_names_in_ 构造输入矩阵。
    兼容空格/下划线命名。
    """
    data = {}
    missing = []

    for feat in expected_features:
        candidates = [
            feat,
            feat.replace(" ", "_"),
            feat.replace("_", " "),
            normalize_name(feat),
            to_display_name(feat),
        ]

        found = False

        for c in candidates:
            if c in df.columns:
                data[feat] = df[c].values
                found = True
                break

        if not found:
            missing.append(feat)

    if missing:
        raise ValueError(
            f"Cannot find required model features:\n{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return pd.DataFrame(data, index=df.index)


def load_top10_feature_names(model, feature_info):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if "top10_processed_columns" in feature_info:
        return list(feature_info["top10_processed_columns"])

    if "original_features_for_top10" in feature_info:
        return list(feature_info["original_features_for_top10"])

    raise KeyError(
        "Cannot find Top10 feature columns. "
        "Expected model.feature_names_in_ or feature_info keys: "
        "top10_processed_columns / original_features_for_top10."
    )


def find_scaler_with_mean_scale(transformer):
    """
    递归寻找 StandardScaler 这类带 mean_ 和 scale_ 的对象。
    兼容：
    - StandardScaler
    - Pipeline([..., scaler])
    - ColumnTransformer 内部嵌套 Pipeline
    """
    if hasattr(transformer, "mean_") and hasattr(transformer, "scale_"):
        return transformer

    if hasattr(transformer, "named_steps"):
        for _, step in transformer.named_steps.items():
            found = find_scaler_with_mean_scale(step)
            if found is not None:
                return found

    if hasattr(transformer, "steps"):
        for _, step in transformer.steps:
            found = find_scaler_with_mean_scale(step)
            if found is not None:
                return found

    if hasattr(transformer, "transformers_"):
        for _, trans, _ in transformer.transformers_:
            found = find_scaler_with_mean_scale(trans)
            if found is not None:
                return found

    return None


def load_preprocessor_stats(preproc_path):
    """
    从 preprocessor.pkl 中读取数值特征 mean/std。
    这一版会递归查找 Pipeline 里的 StandardScaler。
    """
    mean_map = {}
    std_map = {}

    if not preproc_path.exists():
        print(f"[WARNING] Preprocessor not found: {preproc_path}")
        return mean_map, std_map

    preprocessor = joblib.load(preproc_path)

    if not hasattr(preprocessor, "transformers_"):
        print("[WARNING] Loaded preprocessor has no transformers_.")
        return mean_map, std_map

    numeric_features = None
    numeric_transformer = None

    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            numeric_features = list(cols)
            numeric_transformer = trans
            break

    if numeric_features is None or numeric_transformer is None:
        print("[WARNING] Cannot find numeric transformer named 'num'.")
        return mean_map, std_map

    scaler = find_scaler_with_mean_scale(numeric_transformer)

    if scaler is None:
        print("[WARNING] Cannot find StandardScaler-like step with mean_/scale_ inside numeric transformer.")
        return mean_map, std_map

    if len(scaler.mean_) != len(numeric_features):
        raise ValueError(
            "Length mismatch between numeric_features and scaler.mean_.\n"
            f"numeric_features: {len(numeric_features)}\n"
            f"scaler.mean_: {len(scaler.mean_)}"
        )

    for feat, mean, std in zip(numeric_features, scaler.mean_, scaler.scale_):
        keys = {
            feat,
            feat.replace(" ", "_"),
            feat.replace("_", " "),
            normalize_name(feat),
            to_display_name(feat),
            normalize_name(to_display_name(feat)),
        }

        for k in keys:
            mean_map[k] = float(mean)
            std_map[k] = float(std)

    print(f"[INFO] Loaded scaler statistics for {len(numeric_features)} numeric features.")
    return mean_map, std_map


def restore_numeric_values(col, vals, mean_map, std_map):
    """
    将标准化后的数值变量恢复到原始单位。
    如果找不到 mean/std，则返回原值。
    """
    col = str(col)
    vals = np.asarray(vals, dtype=float)

    base_col = col.replace("_log", "")

    candidates = [
        col,
        base_col,
        col.replace("_", " "),
        base_col.replace("_", " "),
        normalize_name(col),
        normalize_name(base_col),
        to_display_name(col),
        normalize_name(to_display_name(col)),
    ]

    matched_key = None

    for key in candidates:
        if key in mean_map and key in std_map:
            matched_key = key
            break

    if matched_key is None:
        return vals, False

    restored = vals * std_map[matched_key] + mean_map[matched_key]

    if col.endswith("_log"):
        restored = np.expm1(restored)

    return restored, True


def load_raw_display_values(train_processed):
    """
    如果存在与 processed 行数一致的原始训练文件，则优先使用原始值画图。
    如果没有，就使用 preprocessor 反标准化。
    """
    candidate_paths = [
        ROOT / "data" / "processed" / "train_full.csv",
        ROOT / "data" / "processed" / "train_raw.csv",
        ROOT / "data" / "raw" / "train_raw.csv",
        ROOT / "data" / "raw" / "train_all(8).csv",
        ROOT / "data" / "raw" / "train_all.csv",
    ]

    for p in candidate_paths:
        if not p.exists():
            continue

        try:
            raw_df = pd.read_csv(p)
        except Exception:
            continue

        if len(raw_df) == len(train_processed):
            print(f"[INFO] Raw display data loaded from: {p}")
            return raw_df.reset_index(drop=True), p

        print(
            f"[WARNING] Raw candidate found but row count mismatch: {p} "
            f"raw={len(raw_df)}, processed={len(train_processed)}"
        )

    print("[INFO] No matching raw train file found. Will try inverse transform using preprocessor.")
    return None, None


def get_raw_column(raw_df, candidates):
    if raw_df is None:
        return None

    lookup = {}

    for c in raw_df.columns:
        c = str(c)
        lookup[c] = c
        lookup[c.replace(" ", "_")] = c
        lookup[c.replace("_", " ")] = c
        lookup[normalize_name(c)] = c

    for cand in candidates:
        cand = str(cand)
        cand_list = [
            cand,
            cand.replace(" ", "_"),
            cand.replace("_", " "),
            normalize_name(cand),
        ]

        for cc in cand_list:
            if cc in lookup:
                return lookup[cc]

    return None


def resolve_feature_cols(feat_name, group_map, col_lookup):
    """
    将临床特征名映射到 Top10 模型实际输入列。
    """
    norm_feat = normalize_name(feat_name)

    if isinstance(group_map, dict):
        for k, cols in group_map.items():
            if normalize_name(k) == norm_feat:
                resolved = []

                for c in cols:
                    candidates = [
                        c,
                        normalize_name(c),
                        to_display_name(c),
                        normalize_name(to_display_name(c)),
                    ]

                    for cand in candidates:
                        if cand in col_lookup:
                            resolved.append(col_lookup[cand])
                            break

                out = []
                for c in resolved:
                    if c not in out:
                        out.append(c)

                if out:
                    return out

    direct_candidates = [
        feat_name,
        normalize_name(feat_name),
        to_display_name(feat_name),
        normalize_name(to_display_name(feat_name)),
        normalize_name(feat_name) + "_log",
        feat_name + "_log",
    ]

    resolved = []

    for cand in direct_candidates:
        if cand in col_lookup:
            resolved.append(col_lookup[cand])

    out = []
    for c in resolved:
        if c not in out:
            out.append(c)

    return out


def get_feature_values_and_shap(
    feat_name,
    X_model,
    shap_values,
    group_map,
    col_lookup,
    col_index,
    mean_map,
    std_map,
):
    """
    返回某个临床特征对应的：
    - display value
    - aggregated SHAP value
    - actual model columns
    """
    valid_cols = resolve_feature_cols(feat_name, group_map, col_lookup)

    if not valid_cols:
        return None, None, [], False

    shap_agg = np.sum(
        [shap_values[:, col_index[c]] for c in valid_cols],
        axis=0,
    )

    is_onehot = (
        len(valid_cols) > 1
        or any(str(c).split("_")[-1].isdigit() for c in valid_cols)
    )

    if is_onehot:
        x_display = X_model[valid_cols].values.argmax(axis=1).astype(float)
        restored_ok = True
    else:
        col = valid_cols[0]
        vals = X_model[col].values
        x_display, restored_ok = restore_numeric_values(col, vals, mean_map, std_map)

    return x_display, shap_agg, valid_cols, restored_ok


def validate_clinical_units(diam_vals, count_vals):
    """
    检查是否像真实临床单位。
    如果 diameter 或 number 仍然有大量负值，大概率还没反标准化。
    """
    diam_vals = np.asarray(diam_vals, dtype=float)
    count_vals = np.asarray(count_vals, dtype=float)

    diam_min = np.nanmin(diam_vals)
    count_min = np.nanmin(count_vals)

    if diam_min < 0:
        return False

    if count_min < 0:
        return False

    return True


# ==========================================
# 3. 加载训练数据和模型
# ==========================================
print("=" * 80)
print("INTEGRATED SHAP VISUALIZATION — RANDOM FOREST")
print("Only two panels: SHAP dependence + tumor burden heatmap")
print("=" * 80)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Cannot find train data: {TRAIN_PATH}")

if not FEATURE_INFO_PATH.exists():
    raise FileNotFoundError(f"Cannot find feature_info.pkl: {FEATURE_INFO_PATH}")

if not TOP10_MODEL_PATH.exists():
    raise FileNotFoundError(f"Cannot find Top10 model: {TOP10_MODEL_PATH}")

train = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

raw_train, raw_train_path = load_raw_display_values(train)

with open(FEATURE_INFO_PATH, "rb") as f:
    feature_info = pickle.load(f)

with open(TOP10_MODEL_PATH, "rb") as f:
    top10_model = pickle.load(f)

top10_expected_names = load_top10_feature_names(top10_model, feature_info)

X_train_top10 = build_X_for_model_features(train, top10_expected_names)

if TARGET_COL not in train.columns:
    raise ValueError(f"Cannot find target column: {TARGET_COL}")

y_train = train[TARGET_COL].astype(int)

print(f"\n[1] Loaded processed train data: {train.shape}")
print(f"[2] Loaded Top10 RandomForest model: {TOP10_MODEL_PATH}")
print(f"[3] Top10 model input shape: {X_train_top10.shape}")

print("\nTop10 model features:")
for c in X_train_top10.columns:
    print(f"  - {c}")


# ==========================================
# 4. 加载 preprocessor，用于恢复数值单位
# ==========================================
print("\n[4] Loading preprocessor statistics for inverse transform...")
mean_map, std_map = load_preprocessor_stats(PREPROC_PATH)


# ==========================================
# 5. 计算 Top10 RandomForest SHAP
# ==========================================
print("\n[5] Computing SHAP values from Top10 RandomForest model...")

explainer = shap.TreeExplainer(top10_model)
sv = explainer.shap_values(X_train_top10)

if isinstance(sv, list):
    shap_values = sv[1]
elif isinstance(sv, np.ndarray) and sv.ndim == 3:
    if sv.shape[2] == 2:
        shap_values = sv[:, :, 1]
    elif sv.shape[0] == 2:
        shap_values = sv[1, :, :]
    else:
        raise ValueError(f"Unexpected SHAP 3D shape: {sv.shape}")
else:
    shap_values = sv

shap_values = np.asarray(shap_values)

if shap_values.shape[1] != X_train_top10.shape[1]:
    raise ValueError(
        f"SHAP shape mismatch: shap_values={shap_values.shape}, "
        f"X={X_train_top10.shape}"
    )

print(f"  ✓ SHAP values shape: {shap_values.shape}")


# ==========================================
# 6. 提取 diameter 和 number 的 SHAP
# ==========================================
print("\n[6] Preparing diameter × number visualization data...")

group_map = feature_info.get("group_map", {})
col_lookup = build_col_lookup(X_train_top10.columns.tolist())
col_index = {col: i for i, col in enumerate(X_train_top10.columns)}

diam_candidates = [
    "diameter of tumor",
    "diameter_of_tumor",
]

count_candidates = [
    "number of tumor",
    "number_of_tumor",
]

diam_vals = None
shap_diam = None
diam_cols = []
diam_restored_ok = False

for cand in diam_candidates:
    diam_vals_tmp, shap_tmp, cols_tmp, restored_ok_tmp = get_feature_values_and_shap(
        cand,
        X_model=X_train_top10,
        shap_values=shap_values,
        group_map=group_map,
        col_lookup=col_lookup,
        col_index=col_index,
        mean_map=mean_map,
        std_map=std_map,
    )

    if cols_tmp:
        diam_vals = diam_vals_tmp
        shap_diam = shap_tmp
        diam_cols = cols_tmp
        diam_restored_ok = restored_ok_tmp
        break

count_vals = None
count_cols = []
count_restored_ok = False

for cand in count_candidates:
    count_vals_tmp, shap_tmp, cols_tmp, restored_ok_tmp = get_feature_values_and_shap(
        cand,
        X_model=X_train_top10,
        shap_values=shap_values,
        group_map=group_map,
        col_lookup=col_lookup,
        col_index=col_index,
        mean_map=mean_map,
        std_map=std_map,
    )

    if cols_tmp:
        count_vals = count_vals_tmp
        count_cols = cols_tmp
        count_restored_ok = restored_ok_tmp
        break

if diam_vals is None or count_vals is None:
    raise ValueError(
        "Cannot resolve diameter or number feature from Top10 model.\n"
        f"Available model columns: {list(X_train_top10.columns)}\n"
        f"group_map keys: {list(group_map.keys()) if isinstance(group_map, dict) else 'None'}"
    )

print(f"Diameter feature resolved from columns: {diam_cols}")
print(f"Number feature resolved from columns:   {count_cols}")


# ==========================================
# 7. 优先使用 raw 原始值；否则使用 preprocessor 反标准化值
# ==========================================
print("\n[7] Preparing clinical-unit display values...")

raw_units_available = False

diam_raw_col = get_raw_column(
    raw_train,
    ["diameter of tumor", "diameter_of_tumor"]
)

count_raw_col = get_raw_column(
    raw_train,
    ["number of tumor", "number_of_tumor"]
)

if raw_train is not None and diam_raw_col is not None and count_raw_col is not None:
    print(f"[INFO] Use raw diameter column for display: {diam_raw_col}")
    print(f"[INFO] Use raw number column for display:   {count_raw_col}")

    diam_vals = raw_train[diam_raw_col].astype(float).values
    count_vals = raw_train[count_raw_col].astype(float).values

    raw_units_available = True

else:
    print("[INFO] Raw clinical columns unavailable. Use inverse-transformed values from processed data.")

    if diam_restored_ok and count_restored_ok:
        raw_units_available = True
    else:
        raw_units_available = validate_clinical_units(diam_vals, count_vals)

print("\nDisplay value summary:")
display_summary = pd.DataFrame({
    "diameter_of_tumor_cm": diam_vals,
    "number_of_tumor": count_vals,
}).describe()
print(display_summary)

if not validate_clinical_units(diam_vals, count_vals):
    raise RuntimeError(
        "Diameter or number values still look like processed/standardized values.\n"
        "The script refuses to draw clinical-unit heatmap with wrong axes.\n"
        "Please check output/preprocessor.pkl or provide a raw train file with matching rows."
    )

# number of tumor 应为离散计数；反标准化后可能出现 1.0000002，做四舍五入
count_vals = np.rint(count_vals).astype(float)


# ==========================================
# 8. 预测概率，用于热图
# ==========================================
pred_proba_top10 = top10_model.predict_proba(X_train_top10)[:, 1]


# ==========================================
# 9. 保存绘图数据
# ==========================================
scatter_df = pd.DataFrame({
    "diameter_of_tumor_cm": diam_vals,
    "number_of_tumor": count_vals,
    "shap_value_for_diameter": shap_diam,
    "predicted_probability": pred_proba_top10,
    "label": y_train.values,
})

scatter_path = OUT_DIR / "diameter_number_shap_scatter_data.csv"
scatter_df.to_csv(scatter_path, index=False, encoding="utf-8-sig")

print(f"\nSaved scatter data: {scatter_path}")


# ==========================================
# 10. 生成两联图：SHAP dependence + heatmap
# ==========================================
print("\n[8] Generating two-panel figure...")

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, (ax_int, ax_heat) = plt.subplots(
    1,
    2,
    figsize=(20.5, 8.8),
    gridspec_kw={"width_ratios": [1.18, 1.12]},
)

fig.subplots_adjust(
    left=0.055,
    right=0.955,
    bottom=0.11,
    top=0.87,
    wspace=0.18,
)
# ------------------------------------------
# A. SHAP dependence scatter
# ------------------------------------------
sc = ax_int.scatter(
    diam_vals,
    shap_diam,
    c=count_vals,
    cmap="Blues",
    alpha=0.82,
    s=46,
    edgecolors="black",
    linewidths=0.35,
)

ax_int.axhline(
    0,
    color="gray",
    linestyle="--",
    linewidth=1.3,
    alpha=0.85,
)

ax_int.set_xlabel("Diameter of Tumor (cm)", fontsize=13, fontweight="bold")
ax_int.set_ylabel("SHAP Value for Diameter of Tumor", fontsize=13, fontweight="bold")
ax_int.set_title(
    "(A) SHAP Dependence: Diameter × Number",
    fontsize=15,
    fontweight="bold",
    pad=10,
)

ax_int.tick_params(axis="both", labelsize=11)
ax_int.grid(alpha=0.25, linestyle="--")
ax_int.spines["top"].set_visible(False)
ax_int.spines["right"].set_visible(False)

divider1 = make_axes_locatable(ax_int)
cax1 = divider1.append_axes("right", size="4.2%", pad=0.08)
cbar = fig.colorbar(sc, cax=cax1)
cbar.set_label("Number of Tumor", fontsize=11, fontweight="bold")
cbar.ax.tick_params(labelsize=10)

# ------------------------------------------
# B. Tumor burden heatmap
# ------------------------------------------
size_bins = [0, 3, 5, 8, 10, 15, np.inf]
size_labels = ["0–3", "3–5", "5–8", "8–10", "10–15", ">15"]

count_bins = [0, 1, 2, 3, 5, np.inf]
count_labels = ["1", "2", "3", "4–5", ">5"]

heatmap_df = pd.DataFrame({
    "size_cat": pd.cut(
        diam_vals,
        bins=size_bins,
        labels=size_labels,
        include_lowest=True,
    ),
    "count_cat": pd.cut(
        count_vals,
        bins=count_bins,
        labels=count_labels,
        include_lowest=True,
    ),
    "pred_proba": pred_proba_top10,
})

heatmap_pivot = heatmap_df.groupby(
    ["count_cat", "size_cat"],
    observed=True,
)["pred_proba"].mean().unstack("size_cat")

heatmap_pivot = heatmap_pivot.reindex(
    index=count_labels,
    columns=size_labels,
)

heatmap_data = heatmap_pivot.values.astype(float)

heatmap_path = OUT_DIR / "tumor_burden_heatmap_data.csv"
heatmap_pivot.to_csv(heatmap_path, encoding="utf-8-sig")

if np.all(np.isnan(heatmap_data)):
    raise ValueError(
        "All heatmap cells are NaN. "
        "Please check diameter/count values and bin settings."
    )

vmin = np.nanmin(heatmap_data)
vmax = np.nanmax(heatmap_data)

if np.isclose(vmin, vmax):
    vmin = vmin - 0.01
    vmax = vmax + 0.01

im = ax_heat.imshow(
    heatmap_data,
    cmap="Blues",   # 如果前面定义了统一色图，也可以改成 cmap=COMMON_CMAP
    aspect="auto",  # 关键：让热图填满右侧坐标轴区域
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)

ax_heat.set_xticks(range(len(size_labels)))
ax_heat.set_xticklabels(size_labels, fontsize=11)
ax_heat.set_yticks(range(len(count_labels)))
ax_heat.set_yticklabels(count_labels, fontsize=11)

ax_heat.set_xlabel("Diameter of Tumor (cm)", fontsize=13, fontweight="bold")
ax_heat.set_ylabel("Number of Tumor", fontsize=13, fontweight="bold")
ax_heat.set_title(
    "(B) Tumor Burden Risk Heatmap",
    fontsize=15,
    fontweight="bold",
    pad=10,
)

# 画网格线，让每个格子更清楚
ax_heat.set_xticks(np.arange(-0.5, len(size_labels), 1), minor=True)
ax_heat.set_yticks(np.arange(-0.5, len(count_labels), 1), minor=True)
ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
ax_heat.tick_params(which="minor", bottom=False, left=False)

threshold_for_text = np.nanmean(heatmap_data)

for r in range(heatmap_data.shape[0]):
    for c in range(heatmap_data.shape[1]):
        val = heatmap_data[r, c]

        if not np.isnan(val):
            text_color = "white" if val > threshold_for_text else "black"
            ax_heat.text(
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )
        else:
            ax_heat.text(
                c,
                r,
                "N/A",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

# 给右图单独挂 colorbar：这样右图更大，而且 colorbar 更贴近右图
divider2 = make_axes_locatable(ax_heat)
cax2 = divider2.append_axes("right", size="5.5%", pad=0.12)
cbar2 = fig.colorbar(im, cax=cax2)
cbar2.set_label("Mean Predicted Probability", fontsize=11, fontweight="bold")

fig.suptitle(
    "Random Forest Top10 Model: Tumor Burden Interaction Visualization",
    fontsize=17,
    fontweight="bold",
    y=0.98,
)

fig_path_png = OUT_DIR / "rf_top10_tumor_burden_two_panel.png"
fig_path_pdf = OUT_DIR / "rf_top10_tumor_burden_two_panel.pdf"

plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_pdf, bbox_inches="tight")
plt.close()

print("\n✅ DONE")
print(f"Two-panel figure PNG: {fig_path_png}")
print(f"Two-panel figure PDF: {fig_path_pdf}")
print(f"Scatter data: {scatter_path}")
print(f"Heatmap data: {heatmap_path}")
print("\n[INFO] Axes use clinical-unit bins:")
print("  Diameter: 0–3, 3–5, 5–8, 8–10, 10–15, >15 cm")
print("  Number: 1, 2, 3, 4–5, >5")