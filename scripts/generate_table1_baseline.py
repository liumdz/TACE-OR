"""Generate Table 1: baseline characteristics for Train vs Internal Test.

Statistics (per the paper's methods section):
  - Continuous variables: median (Q1, Q3); Mann-Whitney U test.
  - Categorical variables: n (%); chi-square test, with Fisher's exact test
    substituted for 2x2 tables when any expected frequency < 5.
  - P < 0.05 considered statistically significant.
  - scipy 1.13.1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "interim" / "train_raw.csv"
TEST_PATH = ROOT / "data" / "interim" / "internal_test_raw.csv"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "table1_baseline.csv"

# ---------------------------------------------------------------------------
# Variable specification
# ---------------------------------------------------------------------------
CONTINUOUS_VARS = [
    "hemoglobin", "PT", "APTT", "TT",
    "ALT", "AST", "ALB", "TBIL", "DBIL", "GGT", "ALP", "PA",
    "AFP", "HBsAg",
    "number of tumor", "diameter of tumor",
]

CATEGORICAL_VARS = [
    "time of TACE",
    "PVTT",
    "combined with other treatment",
    "tumor location",
    "BCLC stage",
]

DISPLAY_NAMES = {
    "hemoglobin": "Hemoglobin (g/L)",
    "PT": "PT (s)",
    "APTT": "APTT (s)",
    "TT": "TT (s)",
    "ALT": "ALT (U/L)",
    "AST": "AST (U/L)",
    "ALB": "ALB (g/L)",
    "TBIL": "TBIL (umol/L)",
    "DBIL": "DBIL (umol/L)",
    "GGT": "GGT (U/L)",
    "ALP": "ALP (U/L)",
    "PA": "PA (mg/L)",
    "AFP": "AFP (ng/mL)",
    "HBsAg": "HBsAg (IU/mL)",
    "number of tumor": "Number of tumors",
    "diameter of tumor": "Diameter of tumor (cm)",
    "time of TACE": "Number of TACE sessions",
    "PVTT": "PVTT",
    "combined with other treatment": "Combined with other treatment",
    "tumor location": "Tumor location",
    "BCLC stage": "BCLC stage",
    "label": "Outcome (label)",
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_num(x: float, is_int_var: bool) -> str:
    if is_int_var:
        return f"{int(round(x))}"
    return f"{x:.2f}"


def median_iqr(series: pd.Series, is_int_var: bool) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    return f"{_fmt_num(med, is_int_var)} ({_fmt_num(q1, is_int_var)}, {_fmt_num(q3, is_int_var)})"


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def continuous_row(var: str, train: pd.DataFrame, test: pd.DataFrame, allset: pd.DataFrame) -> dict:
    tr = pd.to_numeric(train[var], errors="coerce").dropna()
    te = pd.to_numeric(test[var], errors="coerce").dropna()
    p = stats.mannwhitneyu(tr, te, alternative="two-sided").pvalue
    is_int_var = pd.api.types.is_integer_dtype(allset[var])
    return {
        "Variable": DISPLAY_NAMES.get(var, var),
        "ALL": median_iqr(allset[var], is_int_var),
        "Train": median_iqr(train[var], is_int_var),
        "Test": median_iqr(test[var], is_int_var),
        "P value": fmt_p(p),
    }


def categorical_rows(
    var: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    allset: pd.DataFrame,
    test_log: list[tuple[str, str, float]],
) -> list[dict]:
    categories = sorted(allset[var].dropna().unique().tolist())
    contingency = np.array(
        [[(train[var] == c).sum(), (test[var] == c).sum()] for c in categories]
    )

    # Chi-square first, to inspect expected frequencies; for 2x2 tables fall back
    # to Fisher's exact test when any expected frequency < 5.
    chi2_res = stats.chi2_contingency(contingency)
    expected = chi2_res.expected_freq
    use_fisher = contingency.shape == (2, 2) and expected.min() < 5
    if use_fisher:
        p = stats.fisher_exact(contingency)[1]
        test_used = "Fisher"
    else:
        p = chi2_res.pvalue
        test_used = "Chi-square"
    test_log.append((DISPLAY_NAMES.get(var, var), test_used, float(expected.min())))

    header = {
        "Variable": f"{DISPLAY_NAMES.get(var, var)}, n (%)",
        "ALL": "",
        "Train": "",
        "Test": "",
        "P value": fmt_p(p),
    }
    rows = [header]
    n_all, n_tr, n_te = len(allset), len(train), len(test)
    for c in categories:
        c_all = int((allset[var] == c).sum())
        c_tr = int((train[var] == c).sum())
        c_te = int((test[var] == c).sum())
        rows.append({
            "Variable": f"    {c}",
            "ALL": f"{c_all} ({c_all / n_all * 100:.1f})",
            "Train": f"{c_tr} ({c_tr / n_tr * 100:.1f})",
            "Test": f"{c_te} ({c_te / n_te * 100:.1f})",
            "P value": "",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_table1() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    allset = pd.concat([train, test], ignore_index=True)

    rows: list[dict] = [{
        "Variable": "N",
        "ALL": str(len(allset)),
        "Train": str(len(train)),
        "Test": str(len(test)),
        "P value": "",
    }]

    test_log: list[tuple[str, str, float]] = []
    for v in CONTINUOUS_VARS:
        rows.append(continuous_row(v, train, test, allset))
    for v in CATEGORICAL_VARS:
        rows.extend(categorical_rows(v, train, test, allset, test_log))

    table = pd.DataFrame(rows, columns=["Variable", "ALL", "Train", "Test", "P value"])
    return table, test_log


if __name__ == "__main__":
    table1, test_log = build_table1()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(table1.to_string(index=False))
    table1.to_csv(OUT_PATH, index=False)
    print(f"\nSaved Table 1 to: {OUT_PATH}")
    print("\nCategorical tests used (variable | test | min expected frequency):")
    for name, test_used, min_exp in test_log:
        print(f"  {name:40s} | {test_used:10s} | min E = {min_exp:.2f}")
