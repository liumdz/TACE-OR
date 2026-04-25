import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# 0. PATH SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = INTERIM_DIR / "train_raw.csv"
INTERNAL_TEST_PATH = INTERIM_DIR / "internal_test_raw.csv"
EXTERNAL_PATH = INTERIM_DIR / "external_raw_aligned.csv"

TRAIN_OUT = PROCESSED_DIR / "train_processed.csv"
INTERNAL_TEST_OUT = PROCESSED_DIR / "internal_test_processed.csv"
EXTERNAL_OUT = PROCESSED_DIR / "external_processed.csv"

# 为兼容旧脚本，也保存一个 test_processed.csv 指向 internal_test
LEGACY_TEST_OUT = PROCESSED_DIR / "test_processed.csv"

PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = OUTPUT_DIR / "feature_names.json"
PREPROCESSING_REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"

LABEL_COL = "label"


# ============================================================
# 1. FEATURE DEFINITIONS
# ============================================================
RAW_FEATURES = [
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
]

ID_COLS = ["number", "source", "old_split"]

LOG_FEATURES = ["AFP", "ALT", "AST", "TBIL"]

NUMERIC_FEATURES = [
    "hemoglobin",
    "PT",
    "APTT",
    "TT",
    "ALB",
    "DBIL",
    "GGT",
    "ALP",
    "PA",
    "AFP_log",
    "ALT_log",
    "AST_log",
    "TBIL_log",
    "time of TACE",
    "number of tumor",
    "diameter of tumor",
    "HBsAg",
]

CATEGORICAL_FEATURES = [
    "PVTT",
    "combined with other treatment",
    "tumor location",
    "BCLC stage",
]


# ============================================================
# 2. UTILS
# ============================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, na_values=["-", "", "NA", "N/A"])
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read file: {path}\nLast error: {last_error}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    保留内部数据原始风格：
    - 不把空格全部替换成下划线
    - 只去除首尾空格
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_extreme_value(x):
    """
    处理 >1000, <0.5, >=1000, <=0.5 这类半定量记录。
    原则：转换为边界数值本身。
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        s = x.strip()

        if s in ["", "-", "—", "NA", "N/A", "nan", "None"]:
            return np.nan

        # 去掉大于小于号
        s = re.sub(r"^[<>]=?", "", s).strip()

        try:
            return float(s)
        except ValueError:
            return x

    return x


def clean_numeric_column(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.apply(clean_extreme_value), errors="coerce")


def clean_categorical_value(x):
    """
    分类变量保留为字符串编码，保证 OneHot 后列名稳定：
    0 -> '0'
    1.0 -> '1'
    """
    if pd.isna(x):
        return np.nan

    v = clean_extreme_value(x)

    if pd.isna(v):
        return np.nan

    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    except Exception:
        return str(v).strip()


def check_required_columns(df: pd.DataFrame, name: str):
    required = RAW_FEATURES + [LABEL_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name}: missing required columns:\n{missing}\n\n"
            f"Current columns:\n{list(df.columns)}"
        )


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in ID_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def prepare_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    清洗单个数据集：
    1. 统一列名
    2. 检查必须字段
    3. 删除 number/source/old_split
    4. label 转 0/1
    5. 连续变量转数值
    6. 分类变量转字符串编码
    7. log 转换并删除原始偏态变量
    """
    print(f"\n--- Preparing {name} ---")

    df = normalize_column_names(df)
    check_required_columns(df, name)

    # 只保留建模所需字段 + label + 可选 ID 字段
    keep_cols = [c for c in ID_COLS if c in df.columns] + RAW_FEATURES + [LABEL_COL]
    df = df[keep_cols].copy()

    # label
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    if df[LABEL_COL].isna().any():
        raise ValueError(f"{name}: label contains missing or non-numeric values.")

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    bad_labels = sorted(set(df[LABEL_COL].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"{name}: label must be 0/1, got {bad_labels}")

    # 删除 ID / source
    df = drop_unused_columns(df)

    # 连续变量清洗
    raw_numeric_before_log = [
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
        "number of tumor",
        "diameter of tumor",
        "HBsAg",
    ]

    for col in raw_numeric_before_log:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    # 分类变量清洗
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(clean_categorical_value)

    # log transformation
    for col in LOG_FEATURES:
        if col not in df.columns:
            raise ValueError(f"{name}: log feature '{col}' not found.")

        # log1p 要求 x > -1
        if (df[col].dropna() < -1).any():
            bad_values = df.loc[df[col] < -1, col].head()
            raise ValueError(
                f"{name}: column {col} contains values < -1, cannot apply log1p.\n"
                f"Examples:\n{bad_values}"
            )

        df[f"{col}_log"] = np.log1p(df[col])
        df = df.drop(columns=[col])

    # 最终列检查
    final_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_final = [c for c in final_features if c not in df.columns]
    if missing_final:
        raise ValueError(f"{name}: missing final model features:\n{missing_final}")

    df = df[final_features + [LABEL_COL]].copy()

    print(f"{name} shape after cleaning: {df.shape}")
    print(f"{name} label distribution:")
    print(df[LABEL_COL].value_counts().sort_index())
    print(df[LABEL_COL].value_counts(normalize=True).sort_index().round(4))

    return df


def make_onehot_encoder():
    """
    兼容不同 sklearn 版本：
    - 新版本用 sparse_output=False
    - 老版本用 sparse=False
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def transform_to_dataframe(preprocessor, X: pd.DataFrame, y: pd.Series, dataset_name: str):
    X_processed = preprocessor.transform(X)

    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_feature_names = list(onehot.get_feature_names_out(CATEGORICAL_FEATURES))

    feature_names = NUMERIC_FEATURES + cat_feature_names

    out_df = pd.DataFrame(X_processed, columns=feature_names)
    out_df[LABEL_COL] = y.values.astype(int)

    if out_df.isnull().any().any():
        missing = out_df.isnull().sum()
        raise ValueError(
            f"{dataset_name}: missing values remain after preprocessing:\n"
            f"{missing[missing > 0]}"
        )

    return out_df, feature_names


# ============================================================
# 3. MAIN
# ============================================================
def main():
    print("=" * 80)
    print("PREPROCESS FEATURES — TRAIN FIT, INTERNAL/EXTERNAL TRANSFORM ONLY")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    train_raw = safe_read_csv(TRAIN_PATH)
    internal_test_raw = safe_read_csv(INTERNAL_TEST_PATH)
    external_raw = safe_read_csv(EXTERNAL_PATH)

    print("\nRaw shapes:")
    print(f"Train raw        : {train_raw.shape}")
    print(f"Internal test raw: {internal_test_raw.shape}")
    print(f"External raw     : {external_raw.shape}")

    # --------------------------------------------------------
    # Clean / log transform
    # --------------------------------------------------------
    train_clean = prepare_dataframe(train_raw, "train")
    internal_test_clean = prepare_dataframe(internal_test_raw, "internal_test")
    external_clean = prepare_dataframe(external_raw, "external")

    # 保存清洗后、标准化前的数据，方便审计
    train_clean.to_csv(INTERIM_DIR / "train_clean_model_input.csv", index=False, encoding="utf-8-sig")
    internal_test_clean.to_csv(INTERIM_DIR / "internal_test_clean_model_input.csv", index=False, encoding="utf-8-sig")
    external_clean.to_csv(INTERIM_DIR / "external_clean_model_input.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Split X/y
    # --------------------------------------------------------
    X_train = train_clean.drop(columns=[LABEL_COL])
    y_train = train_clean[LABEL_COL]

    X_internal_test = internal_test_clean.drop(columns=[LABEL_COL])
    y_internal_test = internal_test_clean[LABEL_COL]

    X_external = external_clean.drop(columns=[LABEL_COL])
    y_external = external_clean[LABEL_COL]

    # 确保三者列完全一致
    if list(X_train.columns) != list(X_internal_test.columns):
        raise ValueError("Train and internal_test feature columns are not identical.")
    if list(X_train.columns) != list(X_external.columns):
        raise ValueError("Train and external feature columns are not identical.")

    # --------------------------------------------------------
    # Fit preprocessor on TRAIN only
    # --------------------------------------------------------
    print("\nFitting preprocessor on train only...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    print("Transforming train/internal_test/external...")
    train_processed, feature_names = transform_to_dataframe(
        preprocessor, X_train, y_train, "train"
    )
    internal_test_processed, feature_names_2 = transform_to_dataframe(
        preprocessor, X_internal_test, y_internal_test, "internal_test"
    )
    external_processed, feature_names_3 = transform_to_dataframe(
        preprocessor, X_external, y_external, "external"
    )

    if feature_names != feature_names_2 or feature_names != feature_names_3:
        raise ValueError("Feature names are inconsistent after transformation.")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    train_processed.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
    internal_test_processed.to_csv(INTERNAL_TEST_OUT, index=False, encoding="utf-8-sig")
    external_processed.to_csv(EXTERNAL_OUT, index=False, encoding="utf-8-sig")

    # 兼容旧脚本命名
    internal_test_processed.to_csv(LEGACY_TEST_OUT, index=False, encoding="utf-8-sig")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)

    report = {
        "input_files": {
            "train": str(TRAIN_PATH),
            "internal_test": str(INTERNAL_TEST_PATH),
            "external": str(EXTERNAL_PATH),
        },
        "output_files": {
            "train_processed": str(TRAIN_OUT),
            "internal_test_processed": str(INTERNAL_TEST_OUT),
            "external_processed": str(EXTERNAL_OUT),
            "legacy_test_processed": str(LEGACY_TEST_OUT),
            "preprocessor": str(PREPROCESSOR_PATH),
            "feature_names": str(FEATURE_NAMES_PATH),
        },
        "n_samples": {
            "train": int(train_processed.shape[0]),
            "internal_test": int(internal_test_processed.shape[0]),
            "external": int(external_processed.shape[0]),
        },
        "n_features_without_label": int(len(feature_names)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "log_features": LOG_FEATURES,
        "fit_policy": "Preprocessor was fitted only on the training set; internal test and external cohorts were transformed only.",
    }

    with open(PREPROCESSING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"Train processed        : {TRAIN_OUT} {train_processed.shape}")
    print(f"Internal test processed: {INTERNAL_TEST_OUT} {internal_test_processed.shape}")
    print(f"External processed     : {EXTERNAL_OUT} {external_processed.shape}")
    print(f"Legacy test alias      : {LEGACY_TEST_OUT}")
    print(f"Preprocessor saved to  : {PREPROCESSOR_PATH}")
    print(f"Feature names saved to : {FEATURE_NAMES_PATH}")

    print("\nProcessed label distributions:")
    for name, df in [
        ("train", train_processed),
        ("internal_test", internal_test_processed),
        ("external", external_processed),
    ]:
        print(f"\n{name}")
        print(df[LABEL_COL].value_counts().sort_index())
        print(df[LABEL_COL].value_counts(normalize=True).sort_index().round(4))

    print("\nFirst 10 feature names:")
    print(feature_names[:10])


if __name__ == "__main__":
    main()
