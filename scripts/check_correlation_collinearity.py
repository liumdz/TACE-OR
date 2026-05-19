import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# PATH SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CLEAN_PATH = ROOT / "data" / "interim" / "train_raw.csv"
PREPROC_PATH = ROOT / "output" / "preprocessor.pkl"
OUT_DIR = ROOT / "output" / "correlation_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"
ID_COLS = ["number", "source", "old_split"]

CORR_THRESHOLD = 0.7  # 高相关阈值用于标注

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def normalize_name(name):
    return str(name).strip().replace(" ", "_")


def build_lookup(cols):
    lookup = {}
    for c in cols:
        lookup[c] = c
        lookup[normalize_name(c)] = c
        lookup[c.replace("_", " ")] = c
        lookup[c.replace(" ", "_")] = c
    return lookup


def resolve_columns(expected_cols, actual_cols):
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
        if found:
            resolved.append(found)
        else:
            missing.append(col)
    return resolved, missing


def find_numeric_feature_names_from_preprocessor(preprocessor):
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("Loaded preprocessor has no transformers_ attribute.")
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            return list(cols)
    raise ValueError("Cannot find numeric transformer named 'num'.")


def compute_vif(df):
    vif_rows = []
    X_all = df.copy()
    for col in X_all.columns:
        y = X_all[col].values
        X = X_all.drop(columns=[col]).values
        if X.shape[1] == 0:
            vif = np.nan
        else:
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            if r2 >= 0.999999:
                vif = np.inf
            else:
                vif = 1.0 / (1.0 - r2)
        vif_rows.append({"Feature": col, "VIF": vif})

    vif_df = pd.DataFrame(vif_rows)
    return vif_df.sort_values(by="VIF", ascending=False).reset_index(drop=True)


# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv(TRAIN_CLEAN_PATH)
preprocessor = joblib.load(PREPROC_PATH)

numeric_features_expected = find_numeric_feature_names_from_preprocessor(preprocessor)
resolved_numeric_cols, missing_numeric_cols = resolve_columns(
    numeric_features_expected, train.columns.tolist()
)
resolved_numeric_cols = [c for c in resolved_numeric_cols if c not in [TARGET_COL] + ID_COLS]

df_num = train[resolved_numeric_cols].copy()
df_num = df_num.fillna(df_num.median(numeric_only=True))

# ============================================================
# COMPUTE MATRICES
# ============================================================
corr_df = df_num.corr(method="spearman")
vif_df = compute_vif(df_num)

# ============================================================
# PLOT LEFT-RIGHT SUBPLOTS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

PANEL_LABEL_KW = dict(
    fontsize=20,
    fontweight="bold",
    va="bottom",
    ha="left",
)


def add_panel_label(ax, letter):
    ax.text(
        -0.08, 1.02, f"({letter})",
        transform=ax.transAxes,
        **PANEL_LABEL_KW,
    )


# ---------- Spearman heatmap ----------
ax = axes[0]
im1 = ax.imshow(corr_df.values, cmap="Blues", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(np.arange(len(corr_df.columns)))
ax.set_yticks(np.arange(len(corr_df.index)))
ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=14)
ax.set_yticklabels(corr_df.index, fontsize=14)

# 数值标注（对角线为深色填充，用白色字以保证可读性）
for i in range(len(corr_df.index)):
    for j in range(len(corr_df.columns)):
        val = corr_df.iloc[i, j]
        text_color = "white" if i == j else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label("Spearman r", fontsize=15, fontweight="bold")
cbar1.ax.tick_params(labelsize=13)

add_panel_label(ax, "A")

# ---------- VIF heatmap ----------
ax = axes[1]
vif_matrix = np.diag(vif_df["VIF"].values)
im2 = ax.imshow(vif_matrix, cmap="Blues", aspect="auto")
ax.set_xticks(np.arange(len(vif_df)))
ax.set_yticks(np.arange(len(vif_df)))
ax.set_xticklabels(vif_df["Feature"], rotation=45, ha="right", fontsize=14)
ax.set_yticklabels(vif_df["Feature"], fontsize=14)

# 数值标注（对角线带颜色填充，用白色字）
for i in range(len(vif_df)):
    ax.text(i, i, f"{vif_df['VIF'].iloc[i]:.2f}", ha="center", va="center", fontsize=9, color="white")

cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cbar2.set_label("VIF", fontsize=15, fontweight="bold")
cbar2.ax.tick_params(labelsize=13)

add_panel_label(ax, "B")

plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_vif_subplot_labeled.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "correlation_vif_subplot_labeled.pdf", bbox_inches="tight")
plt.close()

print(f"Saved combined figure with titles: {OUT_DIR / 'correlation_vif_subplot_labeled.png'}")