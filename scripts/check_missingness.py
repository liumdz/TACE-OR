import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "interim" / "train_raw.csv"
TEST_PATH = ROOT / "data" / "interim" / "internal_test_raw.csv"
EXTERNAL_PATH = ROOT / "data" / "interim" / "external_raw_aligned.csv"

OUT_DIR = ROOT / "output" / "distribution_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "hemoglobin",
    "PT",
    "APTT",
    "TT",
    "ALT",
    "AST",
    "ALB",
    "TBIL",
    "DBIL",
    "GGT",
    "ALP",
    "PA",
    "AFP",
    "time of TACE",
    "PVTT",
    "number of tumor",
    "diameter of tumor",
    "HBsAg",
    "combined with other treatment",
    "tumor location",
    "BCLC stage",
    "label",
]

MISSING_TOKENS = {"", "-", "—", "NA", "N/A", "nan", "None", "null", "NULL"}


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def is_missing_value(x):
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip() in MISSING_TOKENS
    return False


def missing_summary(df, dataset_name):
    rows = []
    n = len(df)

    for col in FEATURES:
        if col not in df.columns:
            rows.append({
                "dataset": dataset_name,
                "feature": col,
                "n_total": n,
                "n_missing": np.nan,
                "missing_rate": np.nan,
                "status": "COLUMN_NOT_FOUND"
            })
            continue

        miss = df[col].apply(is_missing_value)
        n_missing = int(miss.sum())

        rows.append({
            "dataset": dataset_name,
            "feature": col,
            "n_total": n,
            "n_missing": n_missing,
            "missing_rate": n_missing / n if n > 0 else np.nan,
            "status": "OK"
        })

    return pd.DataFrame(rows)


def main():
    train = read_csv(TRAIN_PATH)
    test = read_csv(TEST_PATH)
    external = read_csv(EXTERNAL_PATH)

    all_summary = pd.concat([
        missing_summary(train, "train"),
        missing_summary(test, "internal_test"),
        missing_summary(external, "external"),
    ], ignore_index=True)

    long_path = OUT_DIR / "missingness_long.csv"
    all_summary.to_csv(long_path, index=False, encoding="utf-8-sig")

    # 转成论文/补充表更容易看的宽表
    wide = all_summary.pivot(
        index="feature",
        columns="dataset",
        values=["n_missing", "missing_rate"]
    )

    # 扁平化列名
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    # 按 train 缺失率降序
    if "missing_rate_train" in wide.columns:
        wide = wide.sort_values("missing_rate_train", ascending=False)

    wide_path = OUT_DIR / "missingness_summary.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("Missingness summary")
    print("=" * 70)
    print(f"Saved long table to: {long_path}")
    print(f"Saved summary table to: {wide_path}")

    print("\nTop missing variables in train:")
    cols = ["feature", "n_missing_train", "missing_rate_train"]
    print(wide[cols].head(20))

    print("\nColumns with missingness > 10% in any dataset:")
    rate_cols = [c for c in wide.columns if c.startswith("missing_rate_")]
    mask = wide[rate_cols].gt(0.10).any(axis=1)
    print(wide.loc[mask, ["feature"] + rate_cols])


if __name__ == "__main__":
    main()
