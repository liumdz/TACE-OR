import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# 0. 路径设置
# ==========================================
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "interim"
CHECK_DIR = ROOT / "output" / "distribution_check"

CHECK_DIR.mkdir(parents=True, exist_ok=True)

OLD_TRAIN_PATH = RAW_DIR / "train_all(8).csv"
OLD_TEST_PATH  = RAW_DIR / "test_all(6).csv"
EXTERNAL_PATH  = RAW_DIR / "external_19_raw.csv"

INTERNAL_ALL_PATH = OUT_DIR / "internal_all_raw.csv"
TRAIN_RAW_PATH    = OUT_DIR / "train_raw.csv"
TEST_RAW_PATH     = OUT_DIR / "internal_test_raw.csv"

SEED = 42
TEST_SIZE = 0.20
LABEL_COL = "label"

# ==========================================
# 1. 读取旧 train/test，并合并为新的内部队列
# ==========================================
print("=" * 70)
print("STEP 1: Rebuild internal cohort from old train/test files")
print("=" * 70)

if not OLD_TRAIN_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {OLD_TRAIN_PATH}")

if not OLD_TEST_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {OLD_TEST_PATH}")

old_train = pd.read_csv(OLD_TRAIN_PATH)
old_test  = pd.read_csv(OLD_TEST_PATH)

old_train["old_split"] = "old_train"
old_test["old_split"]  = "old_test"

internal = pd.concat([old_train, old_test], axis=0, ignore_index=True)

print(f"Old train shape: {old_train.shape}")
print(f"Old test shape : {old_test.shape}")
print(f"Internal merged shape: {internal.shape}")

# ==========================================
# 2. 基础检查
# ==========================================
print("\n" + "=" * 70)
print("STEP 2: Basic checks")
print("=" * 70)

# 去掉可能导致泄露的 source 列
drop_cols = []
if "source" in internal.columns:
    drop_cols.append("source")

# old_split 只用于审计，不进入最终数据
if "old_split" in internal.columns:
    drop_cols.append("old_split")

if drop_cols:
    print(f"Dropping non-predictive/leakage columns: {drop_cols}")
    internal = internal.drop(columns=drop_cols)

if LABEL_COL not in internal.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found. Current columns:\n{list(internal.columns)}")

if internal[LABEL_COL].isna().any():
    raise ValueError("Missing values found in label column. Please check labels first.")

print("\nInternal label distribution:")
label_counts = internal[LABEL_COL].value_counts(dropna=False).sort_index()
label_props = internal[LABEL_COL].value_counts(normalize=True, dropna=False).sort_index()

dist_df = pd.DataFrame({
    "count": label_counts,
    "proportion": label_props.round(4)
})
print(dist_df)

# 完全重复行检查
dup_n = internal.duplicated().sum()
print(f"\nExact duplicated rows: {dup_n}")
if dup_n > 0:
    print("Warning: duplicated rows detected. They are kept for now. Please verify if this is expected.")

# 保存完整内部队列
internal.to_csv(INTERNAL_ALL_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved internal all raw cohort to: {INTERNAL_ALL_PATH}")

# ==========================================
# 3. 新的 8:2 分层划分
# ==========================================
print("\n" + "=" * 70)
print("STEP 3: Stratified 8:2 split for internal cohort")
print("=" * 70)

train, internal_test = train_test_split(
    internal,
    test_size=TEST_SIZE,
    stratify=internal[LABEL_COL],
    random_state=SEED,
    shuffle=True
)

train = train.reset_index(drop=True)
internal_test = internal_test.reset_index(drop=True)

print(f"Train size        : {len(train)}")
print(f"Internal test size: {len(internal_test)}")

print("\nTrain label distribution:")
print(pd.DataFrame({
    "count": train[LABEL_COL].value_counts().sort_index(),
    "proportion": train[LABEL_COL].value_counts(normalize=True).sort_index().round(4)
}))

print("\nInternal test label distribution:")
print(pd.DataFrame({
    "count": internal_test[LABEL_COL].value_counts().sort_index(),
    "proportion": internal_test[LABEL_COL].value_counts(normalize=True).sort_index().round(4)
}))

train.to_csv(TRAIN_RAW_PATH, index=False, encoding="utf-8-sig")
internal_test.to_csv(TEST_RAW_PATH, index=False, encoding="utf-8-sig")

print(f"\nSaved train raw to        : {TRAIN_RAW_PATH}")
print(f"Saved internal test raw to: {TEST_RAW_PATH}")

# ==========================================
# 4. 外部验证集检查：只检查，不参与划分
# ==========================================
print("\n" + "=" * 70)
print("STEP 4: External cohort check")
print("=" * 70)

if EXTERNAL_PATH.exists():
    external = pd.read_csv(EXTERNAL_PATH)
    print(f"External raw file found: {EXTERNAL_PATH}")
    print(f"External shape: {external.shape}")

    if LABEL_COL in external.columns:
        print("\nExternal label distribution:")
        print(pd.DataFrame({
            "count": external[LABEL_COL].value_counts(dropna=False).sort_index(),
            "proportion": external[LABEL_COL].value_counts(normalize=True, dropna=False).sort_index().round(4)
        }))
    else:
        print(f"Warning: label column '{LABEL_COL}' not found in external data.")
else:
    print(f"Warning: external file not found: {EXTERNAL_PATH}")

# ==========================================
# 5. 保存划分摘要
# ==========================================
summary = pd.DataFrame([
    {"dataset": "internal_all", "n": len(internal)},
    {"dataset": "train_raw", "n": len(train)},
    {"dataset": "internal_test_raw", "n": len(internal_test)},
])

if EXTERNAL_PATH.exists():
    summary.loc[len(summary)] = {"dataset": "external_19_raw", "n": len(external)}

summary_path = CHECK_DIR / "split_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Split summary saved to: {summary_path}")
print("\nNext step:")
print("1) Check train/internal_test/external distributions.")
print("2) Then clean/preprocess using train-fitted preprocessing only.")
