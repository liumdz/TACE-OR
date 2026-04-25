import os
import pickle
import numpy as np
import pandas as pd
import shap


# ==============================
# 1. 路径
# ==============================
MODEL_NAME = "randomforest"
SEED = 42

DATA_PATH = "data/processed/train_processed.csv"
FEATURE_INFO_PATH = f"output/top10_model/{MODEL_NAME}/feature_info.pkl"
TOP10_MODEL_PATH = f"output/top10_model/{MODEL_NAME}/top10_model.pkl"
OUT_DIR = f"output/shap_analysis/{MODEL_NAME}_interaction"

os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


# ==============================
# 2. one-hot 合并设置
# ==============================
ONE_HOT_PREFIXES = {
    "PVTT": "PVTT_",
    "combined with other treatment": "combined with other treatment_",
    "tumor location": "tumor location_",
    "BCLC stage": "BCLC stage_",
}


# ==============================
# 3. 工具函数
# ==============================
def normalize_name(x):
    """
    用于特征名匹配：
    diameter of tumor <-> diameter_of_tumor
    """
    return str(x).strip().replace(" ", "_")


def display_name(x):
    """
    用于输出展示：
    AFP_log -> AFP
    diameter_of_tumor -> diameter of tumor
    """
    x = str(x).strip()

    if x.endswith("_log"):
        x = x[:-4]

    return x.replace("_", " ")


def startswith_prefix(col, prefix):
    """
    同时兼容空格和下划线命名。
    """
    col = str(col)
    return col.startswith(prefix) or normalize_name(col).startswith(normalize_name(prefix))


def build_X_for_model_features(df, expected_features):
    """
    严格按照模型训练时的 feature_names_in_ 构建 X。
    同时兼容空格和下划线命名。
    """
    data = {}
    missing = []

    for feat in expected_features:
        candidates = [
            feat,
            feat.replace(" ", "_"),
            feat.replace("_", " "),
        ]

        found = False

        for c in candidates:
            if c in df.columns:
                data[feat] = df[c].values
                found = True
                break

        if not found:
            missing.append(feat)

    if missing:
        raise ValueError(
            f"Cannot find required model features:\n{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return pd.DataFrame(data, index=df.index)


def find_group_index(group_names, target_name):
    """
    在合并后的特征组中查找目标特征。
    """
    target_norm = normalize_name(target_name)

    for idx, name in enumerate(group_names):
        if normalize_name(name) == target_norm:
            return idx

    raise ValueError(
        f"Cannot find target merged feature: {target_name}\n"
        f"Available merged features: {group_names}"
    )


def build_merged_feature_groups(feature_names):
    """
    将 one-hot 输入列合并为临床变量组。

    返回：
    - merged_group_names: 合并后的特征名
    - merged_group_indices: 每个合并特征对应的原始模型输入列索引
    - group_info_df: 合并关系表
    """
    feature_names = list(feature_names)

    assigned = set()
    group_entries = []

    # 1) 先处理 one-hot 变量组
    for group_name, prefix in ONE_HOT_PREFIXES.items():
        idxs = [
            i for i, col in enumerate(feature_names)
            if startswith_prefix(col, prefix)
        ]

        if idxs:
            group_entries.append({
                "Merged_Feature": group_name,
                "Source_Columns": [feature_names[i] for i in idxs],
                "Source_Indices": idxs,
                "Merge_Method": "onehot_group",
                "First_Index": min(idxs),
            })
            assigned.update(idxs)

    # 2) 非 one-hot 特征直接保留
    for i, col in enumerate(feature_names):
        if i in assigned:
            continue

        merged_name = display_name(col)

        group_entries.append({
            "Merged_Feature": merged_name,
            "Source_Columns": [col],
            "Source_Indices": [i],
            "Merge_Method": "single_feature",
            "First_Index": i,
        })

    # 3) 按原始模型输入顺序排序，便于解释
    group_entries = sorted(group_entries, key=lambda x: x["First_Index"])

    merged_group_names = [g["Merged_Feature"] for g in group_entries]
    merged_group_indices = [g["Source_Indices"] for g in group_entries]

    group_info_rows = []
    for g in group_entries:
        group_info_rows.append({
            "Merged_Feature": g["Merged_Feature"],
            "Source_Columns": ", ".join(g["Source_Columns"]),
            "Num_Source_Columns": len(g["Source_Columns"]),
            "Merge_Method": g["Merge_Method"],
        })

    group_info_df = pd.DataFrame(group_info_rows)

    return merged_group_names, merged_group_indices, group_info_df


def aggregate_interaction_by_groups(siv, group_indices):
    """
    将原始 SHAP interaction values 按合并后的特征组聚合。

    对两个不同特征组 A 和 B：
    GroupInteraction(A, B) = sum interaction(i, j)
    for i in A, j in B

    输出：
    group_interactions: dict[(a, b)] -> 每个样本的组间交互值
    mean_abs_group_matrix: 合并后特征组之间的 mean |interaction|
    """
    n_samples = siv.shape[0]
    n_groups = len(group_indices)

    mean_abs_group_matrix = np.zeros((n_groups, n_groups))
    group_interactions = {}

    for a in range(n_groups):
        idx_a = group_indices[a]

        for b in range(a + 1, n_groups):
            idx_b = group_indices[b]

            # siv[:, idx_a][:, :, idx_b] shape:
            # (n_samples, len(idx_a), len(idx_b))
            vals = siv[:, idx_a][:, :, idx_b].sum(axis=(1, 2))

            group_interactions[(a, b)] = vals
            mean_abs_group_matrix[a, b] = np.mean(np.abs(vals))
            mean_abs_group_matrix[b, a] = mean_abs_group_matrix[a, b]

    return group_interactions, mean_abs_group_matrix


# ==============================
# 4. 读取数据和模型
# ==============================
train = pd.read_csv(DATA_PATH)

with open(FEATURE_INFO_PATH, "rb") as f:
    feature_info = pickle.load(f)

with open(TOP10_MODEL_PATH, "rb") as f:
    top10_model = pickle.load(f)

if hasattr(top10_model, "feature_names_in_"):
    top10_expected_names = list(top10_model.feature_names_in_)
else:
    if "top10_processed_columns" in feature_info:
        top10_expected_names = list(feature_info["top10_processed_columns"])
    elif "original_features_for_top10" in feature_info:
        top10_expected_names = list(feature_info["original_features_for_top10"])
    else:
        raise KeyError(
            "Cannot find Top10 feature columns. "
            "Expected model.feature_names_in_ or feature_info keys: "
            "top10_processed_columns / original_features_for_top10."
        )

X_train_top10 = build_X_for_model_features(train, top10_expected_names)
feature_names = list(X_train_top10.columns)

print("=" * 80)
print("SHAP INTERACTION ANALYSIS — RANDOM FOREST TOP10 MODEL")
print("One-hot features are merged before interaction ranking.")
print("=" * 80)

print(f"Training data used for interaction: {X_train_top10.shape}")
print("\nOriginal Top10 model input columns:")
for f in feature_names:
    print(f"  - {f}")


# ==============================
# 5. 构建合并后的特征组
# ==============================
merged_group_names, merged_group_indices, group_info_df = build_merged_feature_groups(feature_names)

print("\nMerged feature groups:")
for name, idxs in zip(merged_group_names, merged_group_indices):
    source_cols = [feature_names[i] for i in idxs]
    print(f"  - {name}: {source_cols}")

group_info_path = os.path.join(OUT_DIR, "merged_feature_groups.csv")
group_info_df.to_csv(group_info_path, index=False, encoding="utf-8-sig")
print(f"\nSaved merged feature groups to: {group_info_path}")


# ==============================
# 6. 计算原始 SHAP interaction values
# ==============================
print("\nComputing SHAP interaction values on original model input columns...")

explainer = shap.TreeExplainer(top10_model)
siv = explainer.shap_interaction_values(X_train_top10)

# 兼容不同 SHAP 输出格式
if isinstance(siv, list):
    # old shap for binary classifier: [class0, class1]
    siv = siv[1]
elif isinstance(siv, np.ndarray) and siv.ndim == 4:
    # new shap: (n_samples, n_features, n_features, n_classes)
    siv = siv[:, :, :, 1]

siv = np.asarray(siv)

if siv.ndim != 3:
    raise ValueError(
        f"Unexpected SHAP interaction values shape: {siv.shape}. "
        "Expected shape: (n_samples, n_features, n_features)."
    )

if siv.shape[1] != len(feature_names) or siv.shape[2] != len(feature_names):
    raise ValueError(
        f"SHAP interaction shape mismatch: {siv.shape}, "
        f"n_features={len(feature_names)}"
    )

print(f"SHAP interaction values shape: {siv.shape}")


# ==============================
# 7. 按合并后的特征组聚合 interaction
# ==============================
print("\nAggregating SHAP interactions to merged feature groups...")

group_interactions, mean_abs_group_matrix = aggregate_interaction_by_groups(
    siv=siv,
    group_indices=merged_group_indices,
)

print(f"Merged feature count: {len(feature_names)} original columns -> {len(merged_group_names)} merged features")


# ==============================
# 8. 目标特征名
# ==============================
# 这里可以写下划线形式，代码会自动匹配空格形式
feat_a = "diameter_of_tumor"
feat_b = "number_of_tumor"

i = find_group_index(merged_group_names, feat_a)
j = find_group_index(merged_group_names, feat_b)

if i == j:
    raise ValueError("Target features are mapped to the same merged group.")

a_idx, b_idx = sorted([i, j])

feat_a_actual = merged_group_names[i]
feat_b_actual = merged_group_names[j]

pair_interaction = group_interactions[(a_idx, b_idx)]
mean_abs_interaction = np.mean(np.abs(pair_interaction))

print(
    f"\nMean |merged SHAP interaction| for "
    f"{feat_a_actual} × {feat_b_actual}: {mean_abs_interaction:.6f}"
)


# ==============================
# 9. 所有合并后特征对交互强度排序
# ==============================
pairs = []

for a in range(len(merged_group_names)):
    for b in range(a + 1, len(merged_group_names)):
        vals = group_interactions[(a, b)]

        pairs.append({
            "Feature_A": merged_group_names[a],
            "Feature_B": merged_group_names[b],
            "Source_A": ", ".join([feature_names[i] for i in merged_group_indices[a]]),
            "Source_B": ", ".join([feature_names[i] for i in merged_group_indices[b]]),
            "MeanAbsInteraction": np.mean(np.abs(vals)),
        })

pairs_df = pd.DataFrame(pairs).sort_values(
    by="MeanAbsInteraction",
    ascending=False,
).reset_index(drop=True)

# 目标 pair 排名
mask = (
    (
        (pairs_df["Feature_A"] == feat_a_actual)
        & (pairs_df["Feature_B"] == feat_b_actual)
    )
    |
    (
        (pairs_df["Feature_A"] == feat_b_actual)
        & (pairs_df["Feature_B"] == feat_a_actual)
    )
)

if not mask.any():
    raise RuntimeError(
        f"Target pair not found in merged ranked pairs: "
        f"{feat_a_actual} × {feat_b_actual}"
    )

rank_position = pairs_df.index[mask][0] + 1

print(f"Rank among all merged feature pairs: {rank_position}/{len(pairs_df)}")


# ==============================
# 10. Bootstrap 95% CI
# ==============================
N_BOOTSTRAP = 5000
boot_vals = np.zeros(N_BOOTSTRAP)

for b in range(N_BOOTSTRAP):
    idx = rng.choice(
        len(pair_interaction),
        size=len(pair_interaction),
        replace=True,
    )
    boot_vals[b] = np.mean(np.abs(pair_interaction[idx]))

ci_lower, ci_upper = np.percentile(boot_vals, [2.5, 97.5])

print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")


# ==============================
# 11. 保存结果
# ==============================
summary_df = pd.DataFrame([{
    "Model": "Top10_RandomForest",
    "Feature_A": feat_a_actual,
    "Feature_B": feat_b_actual,
    "MeanAbsInteraction": mean_abs_interaction,
    "CI_Lower": ci_lower,
    "CI_Upper": ci_upper,
    "RankAmongMergedPairs": rank_position,
    "TotalMergedPairs": len(pairs_df),
    "OriginalInputFeatureCount": len(feature_names),
    "MergedFeatureCount": len(merged_group_names),
    "N_Bootstrap": N_BOOTSTRAP,
    "Seed": SEED,
    "Note": (
        "SHAP interaction values were computed on original model input columns, "
        "then one-hot encoded columns were aggregated into merged clinical feature groups."
    ),
}])

summary_path = os.path.join(OUT_DIR, "target_interaction_summary_merged_onehot.csv")
pairs_path = os.path.join(OUT_DIR, "all_pair_interactions_ranked_merged_onehot.csv")
matrix_path = os.path.join(OUT_DIR, "merged_interaction_mean_abs_matrix.csv")

summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
pairs_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")

matrix_df = pd.DataFrame(
    mean_abs_group_matrix,
    index=merged_group_names,
    columns=merged_group_names,
)

matrix_df.to_csv(matrix_path, encoding="utf-8-sig")

print(f"\nSaved target summary to: {summary_path}")
print(f"Saved ranked merged pairs to: {pairs_path}")
print(f"Saved merged interaction matrix to: {matrix_path}")
print("\nDONE")