import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATH SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

TRAIN_CLEAN_PATH = ROOT / "data" / "interim" / "train_raw.csv"
PREPROC_PATH = ROOT / "output" / "preprocessor.pkl"

OUT_DIR = ROOT / "output" / "correlation_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"
ID_COLS = ["number", "source", "old_split"]

# 阈值可按需要调整
CORR_THRESHOLD = 0.75
VIF_WARNING_THRESHOLD = 5.0
VIF_HIGH_THRESHOLD = 10.0


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def normalize_name(name):
    return str(name).strip().replace(" ", "_")


def build_lookup(cols):
    lookup = {}
    for c in cols:
        c = str(c)
        lookup[c] = c
        lookup[normalize_name(c)] = c
        lookup[c.replace("_", " ")] = c
        lookup[c.replace(" ", "_")] = c
    return lookup


def resolve_columns(expected_cols, actual_cols):
    """
    根据空格/下划线差异，把 expected_cols 映射到 actual_cols 中真实存在的列
    """
    lookup = build_lookup(actual_cols)
    resolved = []
    missing = []

    for col in expected_cols:
        candidates = [
            col,
            normalize_name(col),
            str(col).replace("_", " "),
            str(col).replace(" ", "_"),
        ]

        found = None
        for cand in candidates:
            if cand in lookup:
                found = lookup[cand]
                break

        if found is not None:
            resolved.append(found)
        else:
            missing.append(col)

    return resolved, missing


def find_numeric_feature_names_from_preprocessor(preprocessor):
    """
    从 preprocessor.pkl 中找到 numeric transformer 对应的原始连续变量名
    """
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("Loaded preprocessor has no transformers_ attribute.")

    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            return list(cols)

    raise ValueError("Cannot find numeric transformer named 'num' in preprocessor.")


def compute_vif(df):
    """
    手动计算 VIF，避免依赖 statsmodels
    VIF_j = 1 / (1 - R^2_j)
    """
    vif_rows = []
    X_all = df.copy()

    for col in X_all.columns:
        y = X_all[col].values
        X = X_all.drop(columns=[col]).values

        # 如果只剩一个变量都没有，就无法算 VIF
        if X.shape[1] == 0:
            vif = np.nan
        else:
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)

            # 避免数值问题
            if r2 >= 0.999999:
                vif = np.inf
            else:
                vif = 1.0 / (1.0 - r2)

        vif_rows.append({"Feature": col, "VIF": vif})

    vif_df = pd.DataFrame(vif_rows)

    def classify_vif(v):
        if pd.isna(v):
            return "NA"
        if np.isinf(v):
            return "Infinite"
        if v >= VIF_HIGH_THRESHOLD:
            return "High"
        if v >= VIF_WARNING_THRESHOLD:
            return "Moderate"
        return "Acceptable"

    vif_df["Collinearity_Level"] = vif_df["VIF"].apply(classify_vif)
    return vif_df.sort_values(by="VIF", ascending=False, na_position="last").reset_index(drop=True)


def save_heatmap(corr_df, out_png, out_pdf):
    """
    用 matplotlib 画 Spearman 相关热图
    """
    fig_w = max(8, 0.65 * len(corr_df.columns))
    fig_h = max(7, 0.60 * len(corr_df.columns))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))

    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)

    ax.set_title("Spearman Correlation Heatmap (Train Set)", fontsize=14, fontweight="bold", pad=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman r", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. LOAD DATA
# ============================================================
print("=" * 80)
print("CORRELATION AND COLLINEARITY CHECK — TRAIN SET")
print("=" * 80)

if not TRAIN_CLEAN_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {TRAIN_CLEAN_PATH}")

if not PREPROC_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {PREPROC_PATH}")

train = pd.read_csv(TRAIN_CLEAN_PATH)
print(f"[1] Loaded train_clean_model_input: {train.shape}")

preprocessor = joblib.load(PREPROC_PATH)
numeric_features_expected = find_numeric_feature_names_from_preprocessor(preprocessor)

resolved_numeric_cols, missing_numeric_cols = resolve_columns(
    numeric_features_expected,
    train.columns.tolist()
)

if missing_numeric_cols:
    print(f"[WARNING] Some numeric features from preprocessor were not found in train_clean_model_input:")
    for x in missing_numeric_cols:
        print(f"  - {x}")

# 排除 label / ID
resolved_numeric_cols = [
    c for c in resolved_numeric_cols
    if c not in [TARGET_COL] + ID_COLS
]

if len(resolved_numeric_cols) == 0:
    raise ValueError("No numeric columns were resolved for correlation analysis.")

print(f"[2] Resolved numeric columns for analysis: {len(resolved_numeric_cols)}")
for c in resolved_numeric_cols:
    print(f"  - {c}")

# 提取连续变量数据
df_num = train[resolved_numeric_cols].copy()

# 转成数值
for col in df_num.columns:
    df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

# 用中位数填补少量缺失，保证相关/VIF可算
missing_before = df_num.isna().sum().sum()
if missing_before > 0:
    print(f"[WARNING] Missing numeric values detected: {missing_before}. Median imputation will be applied.")
    df_num = df_num.fillna(df_num.median(numeric_only=True))

# 删除零方差列（VIF 无法算）
std_series = df_num.std()
zero_var_cols = std_series[std_series == 0].index.tolist()

if zero_var_cols:
    print(f"[WARNING] Zero-variance columns removed before analysis:")
    for c in zero_var_cols:
        print(f"  - {c}")
    df_num = df_num.drop(columns=zero_var_cols)

print(f"[3] Final numeric matrix for analysis: {df_num.shape}")


# ============================================================
# 4. SPEARMAN CORRELATION
# ============================================================
print("\n[4] Computing Spearman correlation matrix...")
corr_df = df_num.corr(method="spearman")

corr_csv = OUT_DIR / "spearman_correlation_matrix.csv"
corr_df.to_csv(corr_csv, encoding="utf-8-sig")

# 提取高相关变量对
pairs = []
cols = corr_df.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr_df.iloc[i, j]
        pairs.append({
            "Feature_1": cols[i],
            "Feature_2": cols[j],
            "Spearman_r": r,
            "Abs_r": abs(r),
            "High_Correlation_Flag": abs(r) >= CORR_THRESHOLD
        })

pairs_df = pd.DataFrame(pairs).sort_values(by="Abs_r", ascending=False).reset_index(drop=True)
high_corr_df = pairs_df[pairs_df["Abs_r"] >= CORR_THRESHOLD].copy()

pairs_csv = OUT_DIR / "all_correlation_pairs.csv"
pairs_df.to_csv(pairs_csv, index=False, encoding="utf-8-sig")

high_corr_csv = OUT_DIR / "high_correlation_pairs.csv"
high_corr_df.to_csv(high_corr_csv, index=False, encoding="utf-8-sig")

# 热图
heatmap_png = OUT_DIR / "spearman_heatmap.png"
heatmap_pdf = OUT_DIR / "spearman_heatmap.pdf"
save_heatmap(corr_df, heatmap_png, heatmap_pdf)

print(f"  ✓ Saved full matrix: {corr_csv}")
print(f"  ✓ Saved all pairs:   {pairs_csv}")
print(f"  ✓ Saved high pairs:  {high_corr_csv}")
print(f"  ✓ Saved heatmap:     {heatmap_png}")
print(f"  ✓ Saved heatmap:     {heatmap_pdf}")


# ============================================================
# 5. VIF COLLINEARITY CHECK
# ============================================================
print("\n[5] Computing VIF...")
vif_df = compute_vif(df_num)

vif_csv = OUT_DIR / "vif_table.csv"
vif_df.to_csv(vif_csv, index=False, encoding="utf-8-sig")

print(f"  ✓ Saved VIF table: {vif_csv}")


# ============================================================
# 6. SUMMARY REPORT
# ============================================================
summary_txt = OUT_DIR / "correlation_summary.txt"

n_high_corr = int((pairs_df["Abs_r"] >= CORR_THRESHOLD).sum())
n_vif_warn = int(((vif_df["VIF"] >= VIF_WARNING_THRESHOLD) & (vif_df["VIF"] < VIF_HIGH_THRESHOLD)).sum())
n_vif_high = int((vif_df["VIF"] >= VIF_HIGH_THRESHOLD).sum())
n_vif_inf = int(np.isinf(vif_df["VIF"]).sum())

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("Correlation and Collinearity Check Summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"Input file: {TRAIN_CLEAN_PATH}\n")
    f.write(f"Number of analyzed numeric predictors: {df_num.shape[1]}\n")
    f.write(f"Correlation threshold (|r|): {CORR_THRESHOLD}\n")
    f.write(f"Number of high-correlation pairs: {n_high_corr}\n")
    f.write(f"VIF warning threshold: {VIF_WARNING_THRESHOLD}\n")
    f.write(f"VIF high threshold: {VIF_HIGH_THRESHOLD}\n")
    f.write(f"Number of moderate VIF features: {n_vif_warn}\n")
    f.write(f"Number of high VIF features: {n_vif_high}\n")
    f.write(f"Number of infinite VIF features: {n_vif_inf}\n\n")

    f.write("Top 10 strongest correlation pairs:\n")
    f.write("-" * 60 + "\n")
    for _, row in pairs_df.head(10).iterrows():
        f.write(f"{row['Feature_1']} vs {row['Feature_2']}: r = {row['Spearman_r']:.3f}\n")

    f.write("\nTop 10 highest VIF features:\n")
    f.write("-" * 60 + "\n")
    for _, row in vif_df.head(10).iterrows():
        vif_val = "inf" if np.isinf(row["VIF"]) else f"{row['VIF']:.3f}"
        f.write(f"{row['Feature']}: VIF = {vif_val} ({row['Collinearity_Level']})\n")

print(f"\n[6] Summary written to: {summary_txt}")


# ============================================================
# 7. DONE
# ============================================================
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Output directory:")
print(OUT_DIR)
print("\nGenerated files:")
print("  - spearman_correlation_matrix.csv")
print("  - all_correlation_pairs.csv")
print("  - high_correlation_pairs.csv")
print("  - vif_table.csv")
print("  - spearman_heatmap.png")
print("  - spearman_heatmap.pdf")
print("  - correlation_summary.txt")
print("\nNote:")
print("This script is a diagnostic step only. It does NOT change the dataset or retrain any model.")