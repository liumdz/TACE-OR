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
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 13,
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


# 保留专业缩写大小写，其余首字母大写
ACRONYMS = {
    "APTT", "AFP", "PVTT", "ALP", "GGT", "ALT", "AST",
    "PA", "PT", "TT", "ALB", "DBIL", "TBIL", "HBsAg",
    "BCLC", "TACE",
}


def prettify_feature_name(name: str) -> str:
    """首字母大写但保留临床缩写大小写，例如:
       'diameter of tumor' -> 'Diameter of tumor'
       'BCLC stage' -> 'BCLC stage'
    """
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

OUT_DIR = ROOT / "output" / "shap_analysis" / f"{MODEL_NAME}_dependence_plots_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"

np.random.seed(SEED)

# 颜色设置
POINT_COLOR = "#5B9BD5"   # 蓝色点
LINE_COLOR = "#1F4E79"    # 深蓝色曲线


# ==========================================
# 2. 工具函数
# ==========================================
def normalize_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def to_display_name(name: str) -> str:
    return str(name).replace("_log", "").replace("_", " ")


def build_col_lookup(cols):
    lookup = {}
    for c in cols:
        c = str(c)
        lookup[c] = c
        lookup[normalize_name(c)] = c
        lookup[c.replace("_", " ")] = c
        lookup[c.replace(" ", "_")] = c
        lookup[to_display_name(c)] = c
        lookup[normalize_name(to_display_name(c))] = c
    return lookup


def build_X_for_model_features(df: pd.DataFrame, expected_features):
    """
    根据模型 feature_names_in_ 构造输入矩阵，兼容空格/下划线命名差异
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
            normalize_name(to_display_name(feat)),
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


def find_scaler_with_mean_scale(transformer):
    """
    递归寻找带 mean_ 和 scale_ 的 scaler
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


def build_mean_std_map(preproc_path: Path):
    if not preproc_path.exists():
        raise FileNotFoundError(f"Cannot find preprocessor: {preproc_path}")

    preprocessor = joblib.load(preproc_path)

    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("Loaded preprocessor has no transformers_ attribute.")

    numeric_features = None
    numeric_transformer = None

    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            numeric_features = list(cols)
            numeric_transformer = trans
            break

    if numeric_features is None or numeric_transformer is None:
        raise ValueError("Cannot find numeric transformer named 'num' in preprocessor.")

    scaler = find_scaler_with_mean_scale(numeric_transformer)

    if scaler is None:
        warnings.warn("Cannot find StandardScaler-like step with mean_/scale_. Use identity mapping.")
        mean_map = {}
        std_map = {}
        for feat in numeric_features:
            for key in [feat, feat.replace(" ", "_"), feat.replace("_", " ")]:
                mean_map[key] = 0.0
                std_map[key] = 1.0
        return mean_map, std_map

    mean_map = {}
    std_map = {}

    for feat, mean, std in zip(numeric_features, scaler.mean_, scaler.scale_):
        for key in [
            feat,
            feat.replace(" ", "_"),
            feat.replace("_", " "),
            normalize_name(feat),
        ]:
            mean_map[key] = float(mean)
            std_map[key] = float(std)

    return mean_map, std_map


def restore_numeric_values(col_name: str, values, mean_map, std_map):
    col_name = str(col_name)
    values = np.asarray(values, dtype=float)

    base_col = col_name.replace("_log", "")
    candidate_keys = [
        col_name,
        base_col,
        col_name.replace("_", " "),
        base_col.replace("_", " "),
        normalize_name(col_name),
        normalize_name(base_col),
    ]

    matched_key = None
    for key in candidate_keys:
        if key in mean_map and key in std_map:
            matched_key = key
            break

    if matched_key is None:
        return values

    restored = values * std_map[matched_key] + mean_map[matched_key]

    if col_name.endswith("_log"):
        restored = np.expm1(restored)

    return restored


def is_onehot_feat(cols):
    return len(cols) > 1 or any(str(c).split("_")[-1].isdigit() for c in cols)


def resolve_feature_cols(feat_name, group_map, col_lookup):
    """
    将原始特征名映射到模型实际输入列
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


def infer_plot_features_from_group_map(group_map, model_input_features):
    """
    从 group_map 中筛出真正属于当前 top10_model 输入的原始特征
    """
    if not isinstance(group_map, dict) or len(group_map) == 0:
        return model_input_features

    model_set = set(str(c) for c in model_input_features)
    plot_features = []

    for feat, cols in group_map.items():
        cols = [str(c) for c in cols]
        if any(c in model_set for c in cols):
            plot_features.append(feat)

    return plot_features


CATEGORY_LABELS = {
    "PVTT": {0: "None", 1: "Type I", 2: "Type II", 3: "Type III", 4: "Type IV"},
    "combined_with_other_treatment": {0: "Supportive", 1: "Targeted/PD-1", 2: "Ablation"},
    "tumor_location": {0: "Right lobe", 1: "Left lobe", 2: "Segment I", 3: "Segment IV", 4: "Others"},
    "BCLC_stage": {0: "A", 1: "B", 2: "C"},
}


# ==========================================
# 3. 加载数据和模型
# ==========================================
print("=" * 80)
print("SHAP DEPENDENCE PLOTS — RANDOMFOREST (combined figure, use saved Top10 model)")
print("=" * 80)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Cannot find train data: {TRAIN_PATH}")

if not PREPROC_PATH.exists():
    raise FileNotFoundError(f"Cannot find preprocessor: {PREPROC_PATH}")

if not FEATURE_INFO_PATH.exists():
    raise FileNotFoundError(f"Cannot find feature_info.pkl: {FEATURE_INFO_PATH}")

if not TOP10_MODEL_PATH.exists():
    raise FileNotFoundError(f"Cannot find top10 model: {TOP10_MODEL_PATH}")

train = pd.read_csv(TRAIN_PATH)
train.columns = [c.replace(" ", "_") for c in train.columns]

if TARGET_COL not in train.columns:
    raise ValueError(f"Cannot find target column: {TARGET_COL}")

with open(FEATURE_INFO_PATH, "rb") as f:
    fi = pickle.load(f)

with open(TOP10_MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if not hasattr(model, "feature_names_in_"):
    raise ValueError("Saved top10 model has no feature_names_in_. Cannot recover model input columns safely.")

model_input_features = list(model.feature_names_in_)

# 构造模型输入
X_train_top10 = build_X_for_model_features(train, model_input_features)

# group_map
group_map = fi.get("group_map", {})

# 正确推断 plot_features：只保留当前 top10_model 真正用到的原始特征
plot_features = infer_plot_features_from_group_map(group_map, model_input_features)

# 如果筛出来为空，就退回到模型输入列
if len(plot_features) == 0:
    plot_features = model_input_features
    print("[WARNING] No grouped raw features inferred. Fallback to model input features.")
else:
    print("[INFO] plot_features inferred from feature_info['group_map'] and model.feature_names_in_")

print(f"[INFO] Loaded train data: {train.shape}")
print(f"[INFO] Loaded top10 model: {TOP10_MODEL_PATH}")
print(f"[INFO] Top10 model input shape: {X_train_top10.shape}")
print(f"[INFO] Plot features ({len(plot_features)}):")
for feat in plot_features:
    print(f"  - {feat}")


# ==========================================
# 4. 构建反标准化映射
# ==========================================
mean_map, std_map = build_mean_std_map(PREPROC_PATH)


# ==========================================
# 5. 计算 SHAP
# ==========================================
print("\n[INFO] Computing SHAP values from saved top10 RandomForest model...")

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_train_top10)

if isinstance(sv, list):
    shap_values = sv[1] if len(sv) > 1 else sv[0]
elif isinstance(sv, np.ndarray):
    if sv.ndim == 3:
        shap_values = sv[:, :, 1]
    else:
        shap_values = sv
else:
    raise ValueError("Unsupported SHAP output format.")

shap_values = np.asarray(shap_values)

if shap_values.shape[1] != X_train_top10.shape[1]:
    raise ValueError(
        f"SHAP shape mismatch:\n"
        f"shap_values shape = {shap_values.shape}\n"
        f"X_train_top10 shape = {X_train_top10.shape}"
    )

print(f"[INFO] SHAP values shape: {shap_values.shape}")


# ==========================================
# 6. 合并 SHAP，并恢复显示值
# ==========================================
col_lookup = build_col_lookup(X_train_top10.columns.tolist())
col_index = {col: i for i, col in enumerate(X_train_top10.columns)}

merged_shap = np.zeros((len(X_train_top10), len(plot_features)))
x_display = {}

for j, feat in enumerate(plot_features):
    valid_cols = resolve_feature_cols(feat, group_map, col_lookup)

    if len(valid_cols) == 0:
        print(f"[WARNING] Skip feature with no valid columns: {feat}")
        continue

    merged_shap[:, j] = sum(shap_values[:, col_index[c]] for c in valid_cols)

    if is_onehot_feat(valid_cols):
        x_display[feat] = X_train_top10[valid_cols].values.argmax(axis=1)
    else:
        col = valid_cols[0]
        restored = restore_numeric_values(
            col_name=col,
            values=X_train_top10[col].values,
            mean_map=mean_map,
            std_map=std_map,
        )
        x_display[feat] = restored

# 只保留成功解析的特征
valid_plot_features = [feat for feat in plot_features if feat in x_display]
merged_shap = merged_shap[:, :len(valid_plot_features)]

if len(valid_plot_features) == 0:
    raise ValueError("No valid plot features remained after feature resolution.")

print(f"\n[INFO] Valid plot features finally used ({len(valid_plot_features)}):")
for feat in valid_plot_features:
    print(f"  - {feat}")


# ==========================================
# 7. 合成大图
# ==========================================
print("\n[INFO] Generating combined dependence figure...")

n_features = len(valid_plot_features)
ncols = 5
nrows = int(np.ceil(n_features / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(28, 11),
    constrained_layout=True
)

axes = np.array(axes).reshape(nrows, ncols)

for idx, feat in enumerate(valid_plot_features):
    row = idx // ncols
    col = idx % ncols
    ax = axes[row, col]

    disp_name = to_display_name(feat)
    x_vals = np.array(x_display[feat])
    y_vals = merged_shap[:, idx]

    valid_cols = resolve_feature_cols(feat, group_map, col_lookup)
    onehot = is_onehot_feat(valid_cols)

    if onehot:
        categories = sorted(np.unique(x_vals))

        for cat in categories:
            mask = x_vals == cat
            jitter = np.random.uniform(-0.12, 0.12, mask.sum())

            ax.scatter(
                np.full(mask.sum(), cat) + jitter,
                y_vals[mask],
                alpha=0.75,
                s=26,
                color=POINT_COLOR,
                edgecolors="white",
                linewidths=0.4,
            )

        ax.set_xticks(categories)

        label_map = CATEGORY_LABELS.get(feat, None)
        if label_map:
            ax.set_xticklabels(
                [label_map.get(int(c), str(int(c))) for c in categories],
                rotation=20,
                ha="right",
                fontsize=15
            )
        else:
            ax.set_xticklabels(
                [str(int(c)) for c in categories],
                rotation=20,
                ha="right",
                fontsize=15
            )

    else:
        ax.scatter(
            x_vals,
            y_vals,
            alpha=0.68,
            s=24,
            color=POINT_COLOR,
            edgecolors="white",
            linewidths=0.35,
        )

        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_fit = x_vals[valid_mask]
        y_fit = y_vals[valid_mask]

        if len(np.unique(x_fit)) >= 3:
            try:
                sort_idx = np.argsort(x_fit)
                x_sorted = x_fit[sort_idx]
                y_sorted = y_fit[sort_idx]

                coef = np.polyfit(x_sorted, y_sorted, deg=2)
                poly = np.poly1d(coef)

                x_curve = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                y_curve = poly(x_curve)

                ax.plot(
                    x_curve,
                    y_curve,
                    color=LINE_COLOR,
                    linewidth=2.0,
                )
            except Exception:
                pass

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=15)

    ax.set_title(
        prettify_feature_name(disp_name),
        fontsize=20,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel("Actual value", fontsize=17, fontweight="bold")
    ax.set_ylabel("SHAP value", fontsize=17, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# 关闭多余子图
for idx in range(n_features, nrows * ncols):
    row = idx // ncols
    col = idx % ncols
    axes[row, col].axis("off")

# ==========================================
# 8. 保存
# ==========================================
out_png = OUT_DIR / "all_dependence_plots_randomforest.png"
out_pdf = OUT_DIR / "all_dependence_plots_randomforest.pdf"

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print("\n✅ DONE!")
print(f"Saved PNG: {out_png}")
print(f"Saved PDF: {out_pdf}")