import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "external_19_raw.csv"
OUT_DIR = ROOT / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "external_raw_aligned.csv"
META_PATH = OUT_DIR / "external_metadata.csv"

TARGET_COLUMNS = [
    "number",
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

DROP_COLS = [
    "病人姓名",
    "登记号",
    "初诊时间",
    "年龄",
    "sex",
    "性别",
    "腹水",
    "response_text",
]


def normalize_external_col(col: str) -> str:
    """
    将外部表复杂表头统一成内部数据字段名。
    特别注意：
    - 原始 label 列是 sd/pd/pr 文本，改成 response_text
    - Unnamed: 26 才是真正的 0/1 label
    """
    c = str(col).strip()

    if c == "编号":
        return "number"

    if c == "病人姓名":
        return "病人姓名"

    if c == "登记号":
        return "登记号"

    if c == "初诊时间":
        return "初诊时间"

    if c == "年龄":
        return "年龄"

    if c.startswith("性别"):
        return "sex"

    if c.startswith("number of tumor"):
        return "number of tumor"

    if c.startswith("combined with other treatment"):
        return "combined with other treatment"

    if c.startswith("tumor location"):
        return "tumor location"

    if c.startswith("BCLC stage"):
        return "BCLC stage"

    # 这个 label 是 sd/pd/pr 文本，不是真正 0/1 label
    if c == "label":
        return "response_text"

    # 这一列才是后面的 0/1 label
    if c.startswith("Unnamed"):
        return "label"

    return c


def parse_numeric_value(x):
    """
    处理外部数据中的 '-', '>1000.00' 等情况。
    '-' 作为缺失值；
    '>1000.00' 转成 1000.00，后续由统一清洗/预处理流程处理。
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s in ["", "-", "—", "NA", "N/A", "nan", "None"]:
        return np.nan

    if s.startswith(">") or s.startswith("<"):
        s = s[1:].strip()

    try:
        return float(s)
    except ValueError:
        return np.nan


print("=" * 70)
print("Prepare external validation cohort")
print("=" * 70)

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Cannot find external raw file: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
print(f"Raw external shape: {df.shape}")

print("\nOriginal columns:")
print(list(df.columns))

# 统一列名
df.columns = [normalize_external_col(c) for c in df.columns]

print("\nNormalized columns:")
print(list(df.columns))

# 保存 metadata
meta_cols = [c for c in ["number", "病人姓名", "登记号", "初诊时间", "年龄", "sex"] if c in df.columns]
if meta_cols:
    df[meta_cols].to_csv(META_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved metadata to: {META_PATH}")
    print(f"Metadata columns: {meta_cols}")

# 检查真正 label
if "label" not in df.columns:
    raise ValueError(
        "Cannot find numeric label column. Expected an unnamed column after response_text."
    )

# 原来的 sd/pd/pr 文本列必须存在，但后面会删除
if "response_text" in df.columns:
    print("\nResponse text distribution:")
    print(df["response_text"].value_counts(dropna=False))

# 删除非预测变量和泄露变量
drop_existing = [c for c in DROP_COLS if c in df.columns]
if drop_existing:
    print("\nDropping non-predictive/leakage columns:")
    print(drop_existing)
    df = df.drop(columns=drop_existing)

# 处理 label
df["label"] = pd.to_numeric(df["label"], errors="coerce")

if df["label"].isna().any():
    raise ValueError("Numeric label column contains missing/non-numeric values.")

df["label"] = df["label"].astype(int)

bad_labels = sorted(set(df["label"].unique()) - {0, 1})
if bad_labels:
    raise ValueError(f"Label must be 0/1, but got: {bad_labels}")

# 检查字段完整性
missing_cols = [c for c in TARGET_COLUMNS if c not in df.columns]
extra_cols = [c for c in df.columns if c not in TARGET_COLUMNS]

if missing_cols:
    print("\n❌ Missing required columns:")
    print(missing_cols)
    print("\nCurrent columns:")
    print(list(df.columns))
    raise ValueError("External data columns are not aligned with internal data.")

if extra_cols:
    print("\nExtra columns will be removed:")
    print(extra_cols)

aligned = df[TARGET_COLUMNS].copy()

# 将数值列统一转成数值；number 保留为病例编号
for col in TARGET_COLUMNS:
    if col == "number":
        continue
    aligned[col] = aligned[col].apply(parse_numeric_value)

# label 再转回 int
aligned["label"] = aligned["label"].astype(int)

aligned.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Aligned external shape: {aligned.shape}")
print(f"Saved to: {OUT_PATH}")

print("\nFinal columns:")
print(list(aligned.columns))

print("\nLabel distribution:")
print(aligned["label"].value_counts().sort_index())
print(aligned["label"].value_counts(normalize=True).sort_index().round(4))

print("\nMissing values per column:")
missing = aligned.isna().sum()
print(missing[missing > 0])
