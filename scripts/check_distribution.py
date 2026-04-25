import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "interim" / "train_raw.csv"
TEST_PATH = ROOT / "data" / "interim" / "internal_test_raw.csv"
EXTERNAL_PATH = ROOT / "data" / "interim" / "external_raw_aligned.csv"

OUT_DIR = ROOT / "output" / "distribution_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"

# 一些明显不参与比较的列名关键词
DROP_KEYWORDS = [
    "name", "姓名", "id", "ID", "编号", "登记号", "住院号",
    "source", "old_split", "date", "time", "初诊时间"
]

# 如果外部 label 是 pr/sd/pd，统一转成 response=1
LABEL_MAP = {
    "cr": 1, "CR": 1,
    "pr": 1, "PR": 1,
    "sd": 0, "SD": 0,
    "pd": 0, "PD": 0,
    "response": 1,
    "nonresponse": 0,
    "non-response": 0,
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def normalize_label(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()

    if LABEL_COL not in df.columns:
        print(f"⚠️  {name}: label column not found. Columns:")
        print(list(df.columns))
        return df

    if df[LABEL_COL].dtype == "object":
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
        df[LABEL_COL] = df[LABEL_COL].map(lambda x: LABEL_MAP.get(x, x))

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    if df[LABEL_COL].isna().any():
        raise ValueError(f"{name}: label contains unmapped or missing values.")

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def drop_non_predictive(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        for key in DROP_KEYWORDS:
            if key in col:
                drop_cols.append(col)
                break
    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def label_distribution(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    vc = df[LABEL_COL].value_counts(dropna=False).sort_index()
    vp = df[LABEL_COL].value_counts(normalize=True, dropna=False).sort_index()
    out = pd.DataFrame({
        "dataset": dataset_name,
        "label": vc.index,
        "count": vc.values,
        "proportion": vp.values
    })
    return out


def continuous_summary(group1, group2, feature):
    x = pd.to_numeric(group1[feature], errors="coerce").dropna()
    y = pd.to_numeric(group2[feature], errors="coerce").dropna()

    if len(x) == 0 or len(y) == 0:
        return None

    mean1, std1 = x.mean(), x.std()
    mean2, std2 = y.mean(), y.std()

    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    smd = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

    # 小样本/偏态下用 Mann-Whitney U 更稳一点
    try:
        p_value = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
    except Exception:
        p_value = np.nan

    return {
        "feature": feature,
        "type": "continuous",
        "group1_n": len(x),
        "group2_n": len(y),
        "group1_mean": mean1,
        "group2_mean": mean2,
        "group1_std": std1,
        "group2_std": std2,
        "group1_median": x.median(),
        "group2_median": y.median(),
        "group1_q1": x.quantile(0.25),
        "group2_q1": y.quantile(0.25),
        "group1_q3": x.quantile(0.75),
        "group2_q3": y.quantile(0.75),
        "SMD": abs(smd),
        "p_value": p_value
    }


def categorical_summary(group1, group2, feature):
    x = group1[feature].astype(str).fillna("Missing")
    y = group2[feature].astype(str).fillna("Missing")

    tab = pd.crosstab(
        pd.Series(["group1"] * len(x) + ["group2"] * len(y), name="group"),
        pd.Series(list(x) + list(y), name=feature)
    )

    if tab.shape[1] <= 1:
        return None

    try:
        if tab.shape == (2, 2):
            p_value = stats.fisher_exact(tab.values)[1]
        else:
            p_value = stats.chi2_contingency(tab.values)[1]
    except Exception:
        p_value = np.nan

    # 多分类变量这里用最大比例差作为简单差异指标
    prop1 = x.value_counts(normalize=True)
    prop2 = y.value_counts(normalize=True)
    cats = sorted(set(prop1.index) | set(prop2.index))
    max_abs_diff = max(abs(prop1.get(c, 0) - prop2.get(c, 0)) for c in cats)

    return {
        "feature": feature,
        "type": "categorical",
        "group1_n": len(x),
        "group2_n": len(y),
        "max_abs_prop_diff": max_abs_diff,
        "p_value": p_value
    }


def compare_groups(df1, df2, name1, name2, out_prefix):
    common_cols = sorted(set(df1.columns) & set(df2.columns))
    common_cols = [c for c in common_cols if c != LABEL_COL]

    rows = []
    cat_rows = []

    for col in common_cols:
        # 全是空就跳过
        if df1[col].isna().all() and df2[col].isna().all():
            continue

        # 判断连续变量：两边都能大部分转成数字
        x_num = pd.to_numeric(df1[col], errors="coerce")
        y_num = pd.to_numeric(df2[col], errors="coerce")

        numeric_ratio = (
            x_num.notna().mean() + y_num.notna().mean()
        ) / 2

        # 唯一值很多，且大部分能转数字 → 连续
        n_unique = pd.concat([x_num, y_num]).dropna().nunique()

        if numeric_ratio > 0.8 and n_unique > 5:
            res = continuous_summary(df1, df2, col)
            if res:
                res["group1"] = name1
                res["group2"] = name2
                rows.append(res)
        else:
            res = categorical_summary(df1, df2, col)
            if res:
                res["group1"] = name1
                res["group2"] = name2
                cat_rows.append(res)

    cont_df = pd.DataFrame(rows)
    cat_df = pd.DataFrame(cat_rows)

    if len(cont_df) > 0:
        cont_df = cont_df.sort_values("SMD", ascending=False)
        cont_df.to_csv(OUT_DIR / f"{out_prefix}_continuous.csv", index=False, encoding="utf-8-sig")

    if len(cat_df) > 0:
        cat_df = cat_df.sort_values("max_abs_prop_diff", ascending=False)
        cat_df.to_csv(OUT_DIR / f"{out_prefix}_categorical.csv", index=False, encoding="utf-8-sig")

    return cont_df, cat_df


def main():
    print("=" * 70)
    print("Distribution check")
    print("=" * 70)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Cannot find {TRAIN_PATH}. Run scripts/split_dataset.py first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Cannot find {TEST_PATH}. Run scripts/split_dataset.py first.")
    if not EXTERNAL_PATH.exists():
        raise FileNotFoundError(f"Cannot find {EXTERNAL_PATH}.")

    train = normalize_columns(pd.read_csv(TRAIN_PATH))
    internal_test = normalize_columns(pd.read_csv(TEST_PATH))
    external = normalize_columns(pd.read_csv(EXTERNAL_PATH))

    train = normalize_label(train, "train")
    internal_test = normalize_label(internal_test, "internal_test")
    external = normalize_label(external, "external")

    train = drop_non_predictive(train)
    internal_test = drop_non_predictive(internal_test)
    external = drop_non_predictive(external)

    print("\nShapes:")
    print("Train:", train.shape)
    print("Internal test:", internal_test.shape)
    print("External:", external.shape)

    # 保存 label 分布
    label_df = pd.concat([
        label_distribution(train, "train"),
        label_distribution(internal_test, "internal_test"),
        label_distribution(external, "external"),
    ], ignore_index=True)

    print("\nLabel distribution:")
    print(label_df)
    label_df.to_csv(OUT_DIR / "label_distribution.csv", index=False, encoding="utf-8-sig")

    # 列匹配情况
    col_report = pd.DataFrame({
        "column": sorted(set(train.columns) | set(internal_test.columns) | set(external.columns))
    })
    col_report["in_train"] = col_report["column"].isin(train.columns)
    col_report["in_internal_test"] = col_report["column"].isin(internal_test.columns)
    col_report["in_external"] = col_report["column"].isin(external.columns)
    col_report.to_csv(OUT_DIR / "column_overlap_report.csv", index=False, encoding="utf-8-sig")

    print("\nComparing train vs internal_test...")
    cont1, cat1 = compare_groups(
        train, internal_test,
        "train", "internal_test",
        "train_vs_internal_test"
    )

    print("\nComparing train vs external...")
    cont2, cat2 = compare_groups(
        train, external,
        "train", "external",
        "train_vs_external"
    )

    print("\nSaved files to:", OUT_DIR)

    if len(cont1) > 0:
        print("\nTop continuous differences: train vs internal_test")
        print(cont1[["feature", "SMD", "p_value"]].head(15))

    if len(cont2) > 0:
        print("\nTop continuous differences: train vs external")
        print(cont2[["feature", "SMD", "p_value"]].head(15))

    print("\n✅ DONE")


if __name__ == "__main__":
    main()
