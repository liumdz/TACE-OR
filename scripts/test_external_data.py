import warnings
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# PATH SETTINGS
# ============================================================
ROOT = Path("/home/mumulinux/liver_cancer_external_valid")

EXTERNAL_TEST_PATH = ROOT / "data" / "processed" / "external_processed.csv"
FULL_MODEL_PATH = ROOT / "output" / "tuned" / "models" / "tuned_randomforest.pkl"
TOP10_MODEL_PATH = ROOT / "output" / "top10_model" / "randomforest" / "top10_model.pkl"

RESULT_DIR = ROOT / "output" / "external_test_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"

# ============================================================
# FIXED THRESHOLDS (从训练数据中优化得出)
# ============================================================
best_thresh_full = 0.44
best_thresh_top10 = 0.46

print(f"Using thresholds:")
print(f"  Full RandomForest: {best_thresh_full}")
print(f"  Top10 RandomForest: {best_thresh_top10}")

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
print(f"\nLoading external test data from: {EXTERNAL_TEST_PATH}")
test_df = pd.read_csv(EXTERNAL_TEST_PATH)
y_test = test_df[TARGET_COL].values.astype(int)

print(f"Test set size: {len(test_df)}")
print(f"Positive samples: {np.sum(y_test)}")
print(f"Negative samples: {len(y_test) - np.sum(y_test)}")

print(f"\nLoading models...")
with open(FULL_MODEL_PATH, "rb") as f:
    full_model = pickle.load(f)
    print(f"Loaded: {FULL_MODEL_PATH}")

with open(TOP10_MODEL_PATH, "rb") as f:
    top10_model = pickle.load(f)
    print(f"Loaded: {TOP10_MODEL_PATH}")

# 使用模型内的 feature_names_in_ 构建 X
X_full = test_df[full_model.feature_names_in_].copy()
X_top10 = test_df[top10_model.feature_names_in_].copy()

print(f"\nFull model features: {len(X_full.columns)}")
print(f"Top10 model features: {len(X_top10.columns)}")

# ============================================================
# PREDICTIONS
# ============================================================
print(f"\nGenerating predictions...")
y_prob_full = full_model.predict_proba(X_full)[:, 1]
y_prob_top10 = top10_model.predict_proba(X_top10)[:, 1]

y_pred_full = (y_prob_full >= best_thresh_full).astype(int)
y_pred_top10 = (y_prob_top10 >= best_thresh_top10).astype(int)

print(f"Full model positive predictions: {np.sum(y_pred_full)}")
print(f"Top10 model positive predictions: {np.sum(y_pred_top10)}")

# ============================================================
# METRICS FUNCTION
# ============================================================
def calc_metrics(y_true, y_probs, y_pred):
    """计算性能指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    auc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    accuracy = accuracy_score(y_true, y_pred)
    youden = sensitivity + specificity - 1
    
    return {
        "AUC": auc,
        "AUPRC": auprc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "Youden": youden,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


# ============================================================
# CALCULATE METRICS
# ============================================================
print(f"\nCalculating metrics...")
metrics_full = calc_metrics(y_test, y_prob_full, y_pred_full)
metrics_top10 = calc_metrics(y_test, y_prob_top10, y_pred_top10)

metrics_df = pd.DataFrame([
    {
        "Model": "Full_RandomForest",
        "Threshold": best_thresh_full,
        **metrics_full
    },
    {
        "Model": "Top10_RandomForest",
        "Threshold": best_thresh_top10,
        **metrics_top10
    }
])

metrics_csv = RESULT_DIR / "external_test_metrics_full_vs_top10.csv"
metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
print(f"✓ Saved metrics: {metrics_csv}")

# ============================================================
# SAVE PREDICTIONS
# ============================================================
pred_df = pd.DataFrame({
    "y_true": y_test,
    "Full_RF_y_prob": y_prob_full,
    "Full_RF_threshold": best_thresh_full,
    "Full_RF_y_pred": y_pred_full,
    "Top10_RF_y_prob": y_prob_top10,
    "Top10_RF_threshold": best_thresh_top10,
    "Top10_RF_y_pred": y_pred_top10,
})

pred_csv = RESULT_DIR / "external_test_predictions_full_vs_top10.csv"
pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
print(f"✓ Saved predictions: {pred_csv}")

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("EXTERNAL VALIDATION RESULTS - Full vs Top10 RandomForest")
print("=" * 80)

for row in metrics_df.itertuples():
    print(f"\nModel: {row.Model}")
    print(f"  Threshold: {row.Threshold:.4f}")
    print(f"  AUC: {row.AUC:.4f}, AUPRC: {row.AUPRC:.4f}, Accuracy: {row.Accuracy:.4f}")
    print(f"  Precision: {row.Precision:.4f}, Sensitivity: {row.Sensitivity:.4f}")
    print(f"  Specificity: {row.Specificity:.4f}, F1: {row.F1:.4f}, Youden: {row.Youden:.4f}")
    print(f"  Confusion Matrix: TP={row.TP}, FP={row.FP}, TN={row.TN}, FN={row.FN}")

print("\n" + "=" * 80)
print("✓ DONE")
print("=" * 80)