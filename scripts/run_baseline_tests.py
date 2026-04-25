import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATH SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "interim" / "train_raw.csv"
TEST_PATH = ROOT / "data" / "interim" / "internal_test_raw.csv"

OUT_DIR = ROOT / "output" / "baseline_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "table1_train_vs_internal_test.csv"
OUT_XLSX = OUT_DIR / "table1_train_vs_internal_test.xlsx"
OUT_VAR_TYPES = OUT_DIR / "variable_type_summary.csv"

TARGET_COL = "label"
ID_COLS = ["number", "source", "old_split"]

FORCE_CATEGORICAL = {
    "sex", "pvtt", "bclc stage", "bclc_stage",
    "tumor location", "tumor_location",
    "combined with other treatment", "combined_with_other_treatment",
    "label"
}

FORCE_CONTINUOUS = {
    "age", "aptt", "tt", "pt", "alb", "pa", "dbil", "ggt", "alp",
    "afp", "alt", "ast", "tbil", "time of tace", "time_of_tace",
    "number of tumor", "number_of_tumor",
    "diameter of tumor", "diameter_of_tumor",
    "hemoglobin", "hbsag"
}

# ============================================================
# 2. HELPERS
# ============================================================
def norm_name(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def is_missing_series(s: pd.Series) -> pd.Series:
    return s.isna()


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def summarize_continuous(train_s: pd.Series, test_s: pd.Series):
    train_s = pd.to_numeric(train_s, errors="coerce").dropna()
    test_s = pd.to_numeric(test_s, errors="coerce").dropna()

    if len(train_s) == 0 or len(test_s) == 0:
        return "", "", np.nan, "NA"

    # 正态性判断：Shapiro（样本量不大时可用）
    train_normal = len(train_s) >= 3 and stats.shapiro(train_s)[1] > 0.05
    test_normal = len(test_s) >= 3 and stats.shapiro(test_s)[1] > 0.05

    if train_normal and test_normal:
        # 用 Welch t-test 更稳
        p = stats.ttest_ind(train_s, test_s, equal_var=False, nan_policy="omit").pvalue
        train_txt = f"{train_s.mean():.2f} ± {train_s.std(ddof=1):.2f}"
        test_txt = f"{test_s.mean():.2f} ± {test_s.std(ddof=1):.2f}"
        test_name = "Welch t test"
    else:
        p = stats.mannwhitneyu(train_s, test_s, alternative="two-sided").pvalue
        train_txt = f"{train_s.median():.2f} ({train_s.quantile(0.25):.2f}, {train_s.quantile(0.75):.2f})"
        test_txt = f"{test_s.median():.2f} ({test_s.quantile(0.25):.2f}, {test_s.quantile(0.75):.2f})"
        test_name = "Mann–Whitney U"

    return train_txt, test_txt, p, test_name


def summarize_categorical(train_s: pd.Series, test_s: pd.Series):
    train_s = train_s.astype("object").fillna("Missing")
    test_s = test_s.astype("object").fillna("Missing")

    levels = sorted(set(train_s.unique()).union(set(test_s.unique())), key=lambda x: str(x))

    train_counts = train_s.value_counts(dropna=False)
    test_counts = test_s.value_counts(dropna=False)

    table = pd.DataFrame({
        "Train": train_counts,
        "Internal_test": test_counts
    }).fillna(0)

    table = table.reindex(levels).fillna(0)

    contingency = table.values

    p = np.nan
    test_name = ""

    if contingency.shape == (2, 2):
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
        if (expected < 5).any() or (contingency < 5).any():
            _, p = stats.fisher_exact(contingency)
            test_name = "Fisher exact"
        else:
            p = chi2_p
            test_name = "Chi-square"
    else:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        test_name = "Chi-square"

    train_total = len(train_s)
    test_total = len(test_s)

    rows = []
    for lvl in levels:
        n_train = int(train_counts.get(lvl, 0))
        n_test = int(test_counts.get(lvl, 0))
        rows.append({
            "Level": lvl,
            "Train": f"{n_train} ({100 * n_train / train_total:.1f}%)",
            "Internal_test": f"{n_test} ({100 * n_test / test_total:.1f}%)"
        })

    return rows, p, test_name


def infer_variable_type(series: pd.Series, col_name: str) -> str:
    name = norm_name(col_name)

    if name in FORCE_CATEGORICAL:
        return "categorical"
    if name in FORCE_CONTINUOUS:
        return "continuous"

    # object / category / bool 优先看作分类变量
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "categorical"

    # 数值型：如果唯一值很少，通常按分类变量处理
    non_missing = series.dropna()
    nunique = non_missing.nunique()

    if nunique <= 8:
        return "categorical"

    return "continuous"


# ============================================================
# 3. LOAD DATA
# ============================================================
print("=" * 80)
print("BASELINE TESTS: TRAIN vs INTERNAL TEST")
print("=" * 80)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {TRAIN_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {TEST_PATH}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"[1] Loaded train_raw: {train.shape}")
print(f"[2] Loaded internal_test_raw: {test.shape}")

common_cols = [c for c in train.columns if c in test.columns]
common_cols = [c for c in common_cols if c not in ID_COLS]

if len(common_cols) == 0:
    raise ValueError("No common columns found between train and internal test.")

# ============================================================
# 4. RUN TESTS
# ============================================================
results = []
var_types = []

for col in common_cols:
    var_type = infer_variable_type(train[col], col)
    var_types.append({"Variable": col, "Variable_Type": var_type})

    if var_type == "continuous":
        train_txt, test_txt, p, test_name = summarize_continuous(train[col], test[col])

        results.append({
            "Variable": col,
            "Level": "",
            "Variable_Type": "continuous",
            "Train": train_txt,
            "Internal_test": test_txt,
            "Test": test_name,
            "P_value": format_p(p)
        })

    else:
        level_rows, p, test_name = summarize_categorical(train[col], test[col])

        # 主变量行
        results.append({
            "Variable": col,
            "Level": "",
            "Variable_Type": "categorical",
            "Train": "",
            "Internal_test": "",
            "Test": test_name,
            "P_value": format_p(p)
        })

        # 各水平行
        for row in level_rows:
            results.append({
                "Variable": "",
                "Level": row["Level"],
                "Variable_Type": "",
                "Train": row["Train"],
                "Internal_test": row["Internal_test"],
                "Test": "",
                "P_value": ""
            })

results_df = pd.DataFrame(results)
var_types_df = pd.DataFrame(var_types)

# ============================================================
# 5. SAVE
# ============================================================
results_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
var_types_df.to_csv(OUT_VAR_TYPES, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="table1_tests", index=False)
    var_types_df.to_excel(writer, sheet_name="variable_types", index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved table: {OUT_CSV}")
print(f"Saved excel: {OUT_XLSX}")
print(f"Saved variable types: {OUT_VAR_TYPES}")
print("\nNote:")
print("1) Continuous variables: Welch t test if both groups pass Shapiro normality; otherwise Mann–Whitney U.")
print("2) Categorical variables: Chi-square test; Fisher exact test for small 2x2 tables.")
print("3) This script is for baseline comparison only and does NOT change any model results.")