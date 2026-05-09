import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ============================================================
# 1. PATH SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "interim" / "train_raw.csv"
TEST_PATH = ROOT / "data" / "interim" / "internal_test_raw.csv"
EXTERNAL_PATH = ROOT / "data" / "interim" / "external_raw_aligned.csv"

OUT_DIR = ROOT / "output" / "distribution_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"

# 明确删除的非预测性变量：病例编号、数据来源、旧划分标记
# 注意：不要删除 time_of_TACE
DROP_EXACT_COLS = {
    "number",
    "source",
    "old_split",
}

# 只删除明显的非预测性标识/日期变量
# 不要放 "time"，否则会误删 time_of_TACE
DROP_KEYWORDS = [
    "name",
    "姓名",
    "编号",
    "登记号",
    "住院号",
    "patient_id",
    "_id",
    "id_",
    "date",
    "初诊时间",
]

# 如果外部 label 是 cr/pr/sd/pd，统一转成 response=1
LABEL_MAP = {
    "cr": 1, "CR": 1,
    "pr": 1, "PR": 1,
    "sd": 0, "SD": 0,
    "pd": 0, "PD": 0,
    "response": 1,
    "nonresponse": 0,
    "non-response": 0,
}

# 强制指定变量类型，避免 time_of_TACE 被误判为分类变量
FORCE_CONTINUOUS = {
    "hemoglobin",
    "pt",
    "aptt",
    "tt",
    "alt",
    "ast",
    "alb",
    "tbil",
    "dbil",
    "ggt",
    "alp",
    "pa",
    "afp",
    "time_of_tace",
    "number_of_tumor",
    "diameter_of_tumor",
    "hbsag",
}

FORCE_CATEGORICAL = {
    "pvtt",
    "combined_with_other_treatment",
    "tumor_location",
    "bclc_stage",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一列名格式：
    - 去除前后空格
    - 空格替换为下划线
    例如：time of TACE -> time_of_TACE
    """
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def feature_key(col: str) -> str:
    """
    用于变量类型判断和删除判断的标准化名字。
    """
    return str(col).strip().replace(" ", "_").lower()


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
        bad_values = df.loc[df[LABEL_COL].isna(), LABEL_COL].unique()
        raise ValueError(f"{name}: label contains unmapped or missing values: {bad_values}")

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def drop_non_predictive(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除非预测性变量。
    重点：
    - 删除 number
    - 删除 source / old_split
    - 不删除 time_of_TACE
    """
    df = df.copy()
    drop_cols = []

    for col in df.columns:
        key = feature_key(col)

        # 1) 精确删除明确非预测变量
        if key in DROP_EXACT_COLS:
            drop_cols.append(col)
            continue

        # 2) 按关键词删除明显标识/日期变量
        for kw in DROP_KEYWORDS:
            if kw.lower() in key:
                drop_cols.append(col)
                break

    drop_cols = sorted(set(drop_cols))

    if drop_cols:
        print(f"Dropped non-predictive columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    return df


def label_distribution(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    vc = df[LABEL_COL].value_counts(dropna=False).sort_index()
    vp = df[LABEL_COL].value_counts(normalize=True, dropna=False).sort_index()

    out = pd.DataFrame({
        "dataset": dataset_name,
        "label": vc.index,
        "count": vc.values,
        "proportion": vp.values,
    })
    return out


def continuous_summary(group1: pd.DataFrame, group2: pd.DataFrame, feature: str):
    x = pd.to_numeric(group1[feature], errors="coerce").dropna()
    y = pd.to_numeric(group2[feature], errors="coerce").dropna()

    if len(x) == 0 or len(y) == 0:
        return None

    mean1, std1 = x.mean(), x.std(ddof=1)
    mean2, std2 = y.mean(), y.std(ddof=1)

    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    smd = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

    # 小样本/偏态下用 Mann-Whitney U 更稳
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
        "p_value": p_value,
    }


def categorical_summary(group1: pd.DataFrame, group2: pd.DataFrame, feature: str):
    # 先 fillna 再 astype(str)，避免 NaN 被转成字符串 "nan"
    x = group1[feature].fillna("Missing").astype(str)
    y = group2[feature].fillna("Missing").astype(str)

    tab = pd.crosstab(
        pd.Series(["group1"] * len(x) + ["group2"] * len(y), name="group"),
        pd.Series(list(x) + list(y), name=feature),
    )

    if tab.shape[1] <= 1:
        return None

    try:
        if tab.shape == (2, 2):
            # 2x2 表使用 Fisher 精确检验
            p_value = stats.fisher_exact(tab.values)[1]
            test_name = "Fisher exact"
        else:
            # 多分类变量使用 χ² 检验
            p_value = stats.chi2_contingency(tab.values)[1]
            test_name = "Chi-square"
    except Exception:
        p_value = np.nan
        test_name = "NA"

    # 多分类变量用最大比例差作为简单差异指标
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
        "p_value": p_value,
        "test": test_name,
    }


def infer_variable_type(df1: pd.DataFrame, df2: pd.DataFrame, col: str) -> str:
    """
    判断变量类型：
    1. 强制连续变量优先
    2. 强制分类变量其次
    3. 其他变量根据能否转数值和唯一值数量自动判断
    """
    key = feature_key(col)

    if key in FORCE_CONTINUOUS:
        return "continuous"

    if key in FORCE_CATEGORICAL:
        return "categorical"

    x_num = pd.to_numeric(df1[col], errors="coerce")
    y_num = pd.to_numeric(df2[col], errors="coerce")

    numeric_ratio = (x_num.notna().mean() + y_num.notna().mean()) / 2
    n_unique = pd.concat([x_num, y_num]).dropna().nunique()

    if numeric_ratio > 0.8 and n_unique > 5:
        return "continuous"

    return "categorical"


def compare_groups(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str, out_prefix: str):
    common_cols = sorted(set(df1.columns) & set(df2.columns))

    # label 只用于标签分布，不参与基线变量比较
    common_cols = [c for c in common_cols if c != LABEL_COL]

    rows = []
    cat_rows = []

    for col in common_cols:
        # 全是空就跳过
        if df1[col].isna().all() and df2[col].isna().all():
            continue

        var_type = infer_variable_type(df1, df2, col)

        if var_type == "continuous":
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
        cont_df.to_csv(
            OUT_DIR / f"{out_prefix}_continuous.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if len(cat_df) > 0:
        cat_df = cat_df.sort_values("max_abs_prop_diff", ascending=False)
        cat_df.to_csv(
            OUT_DIR / f"{out_prefix}_categorical.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return cont_df, cat_df


# ============================================================
# 3. MAIN
# ============================================================
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

    # 删除 number/source/old_split 等非预测性变量
    # 保留 time_of_TACE
    train = drop_non_predictive(train)
    internal_test = drop_non_predictive(internal_test)
    external = drop_non_predictive(external)

    print("\nShapes after dropping non-predictive columns:")
    print("Train:", train.shape)
    print("Internal test:", internal_test.shape)
    print("External:", external.shape)

    print("\nColumns kept in train:")
    print(list(train.columns))

    # 检查关键变量是否保留/删除正确
    if "number" in train.columns:
        print("⚠️ WARNING: number still exists in train columns.")
    else:
        print("✓ number removed.")

    if "time_of_TACE" in train.columns:
        print("✓ time_of_TACE kept.")
    else:
        print("⚠️ WARNING: time_of_TACE not found. Please check original column name.")

    # 保存 label 分布
    label_df = pd.concat(
        [
            label_distribution(train, "train"),
            label_distribution(internal_test, "internal_test"),
            label_distribution(external, "external"),
        ],
        ignore_index=True,
    )

    print("\nLabel distribution:")
    print(label_df)
    label_df.to_csv(
        OUT_DIR / "label_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 列匹配情况
    col_report = pd.DataFrame({
        "column": sorted(set(train.columns) | set(internal_test.columns) | set(external.columns))
    })
    col_report["in_train"] = col_report["column"].isin(train.columns)
    col_report["in_internal_test"] = col_report["column"].isin(internal_test.columns)
    col_report["in_external"] = col_report["column"].isin(external.columns)
    col_report.to_csv(
        OUT_DIR / "column_overlap_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nComparing train vs internal_test...")
    cont1, cat1 = compare_groups(
        train,
        internal_test,
        "train",
        "internal_test",
        "train_vs_internal_test",
    )

    print("\nComparing train vs external...")
    cont2, cat2 = compare_groups(
        train,
        external,
        "train",
        "external",
        "train_vs_external",
    )

    print("\nSaved files to:", OUT_DIR)

    if len(cont1) > 0:
        print("\nTop continuous differences: train vs internal_test")
        print(cont1[["feature", "SMD", "p_value"]].head(20))

    if len(cat1) > 0:
        print("\nTop categorical differences: train vs internal_test")
        print(cat1[["feature", "max_abs_prop_diff", "p_value", "test"]].head(20))

    if len(cont2) > 0:
        print("\nTop continuous differences: train vs external")
        print(cont2[["feature", "SMD", "p_value"]].head(20))

    if len(cat2) > 0:
        print("\nTop categorical differences: train vs external")
        print(cat2[["feature", "max_abs_prop_diff", "p_value", "test"]].head(20))

    print("\n✅ DONE")
    print("\nGenerated files:")
    print(f"  - {OUT_DIR / 'label_distribution.csv'}")
    print(f"  - {OUT_DIR / 'column_overlap_report.csv'}")
    print(f"  - {OUT_DIR / 'train_vs_internal_test_continuous.csv'}")
    print(f"  - {OUT_DIR / 'train_vs_internal_test_categorical.csv'}")
    print(f"  - {OUT_DIR / 'train_vs_external_continuous.csv'}")
    print(f"  - {OUT_DIR / 'train_vs_external_categorical.csv'}")


if __name__ == "__main__":
    main()