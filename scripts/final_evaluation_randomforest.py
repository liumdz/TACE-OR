import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix

warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "randomforest"

DATA_PATH = ROOT / "data" / "processed" / "internal_test_processed.csv"
TARGET_COL = "label"

FULL_MODEL_PATH = ROOT / "output" / "tuned" / "models" / "tuned_randomforest.pkl"
FULL_FEATURE_JSON = ROOT / "output" / "tuned" / "feature_cols_tuned.json"

TOP10_DIR = ROOT / "output" / "top10_model" / MODEL_NAME
TOP10_MODEL_PATH = TOP10_DIR / "top10_model.pkl"
TOP10_FEATURE_INFO_PATH = TOP10_DIR / "feature_info.pkl"

THRESHOLD_PATH = (
    ROOT / "output" / "threshold" / MODEL_NAME / "best_threshold_full_vs_top10.csv"
)

OUT_DIR = ROOT / "output" / "final_evaluation" / MODEL_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_COLOR = "#BDBDBD"
TOP10_COLOR = "#2E77BB"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_cols(model, fallback_path=None, fallback_key=None):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if fallback_path is None or not Path(fallback_path).exists():
        return None

    if str(fallback_path).endswith(".json"):
        with open(fallback_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = load_pickle(fallback_path)

    if isinstance(obj, list):
        return list(obj)

    if isinstance(obj, dict):
        for key in [
            fallback_key,
            "RandomForest",
            "randomforest",
            "top10_processed_columns",
            "feature_cols",
            "feature_columns",
            "features",
            "original_features_for_top10",
        ]:
            if key in obj and isinstance(obj[key], list):
                return list(obj[key])

    return None


def prepare_features(df, feature_cols, model_name):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {model_name}: {missing}")
    return df[feature_cols].copy()


def load_thresholds(path):
    df = pd.read_csv(path)
    required = {"Model", "Best_Threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Threshold file missing columns: {sorted(missing)}")
    return {row["Model"]: float(row["Best_Threshold"]) for _, row in df.iterrows()}


def net_benefit(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    values = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n = len(y_true)
        values.append((tp / n) - (fp / n) * (threshold / (1 - threshold)))

    return np.asarray(values)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]
        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)

        if not np.any(mask):
            continue

        weight = np.mean(mask)
        ece += weight * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))

    return float(ece)


def get_calibration_rows(model_name, y_true, y_prob):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "Model": model_name,
            "Mean_Predicted_Probability": prob_pred,
            "Observed_Fraction": prob_true,
        }
    )


def main():
    for path in [
        DATA_PATH,
        FULL_MODEL_PATH,
        FULL_FEATURE_JSON,
        TOP10_MODEL_PATH,
        TOP10_FEATURE_INFO_PATH,
        THRESHOLD_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    test_df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in test_df.columns:
        raise KeyError(f"Cannot find target column: {TARGET_COL}")

    y_true = test_df[TARGET_COL].astype(int).to_numpy()

    full_model = load_pickle(FULL_MODEL_PATH)
    top10_model = load_pickle(TOP10_MODEL_PATH)

    full_cols = load_feature_cols(
        full_model,
        fallback_path=FULL_FEATURE_JSON,
        fallback_key="RandomForest",
    )
    top10_cols = load_feature_cols(
        top10_model,
        fallback_path=TOP10_FEATURE_INFO_PATH,
        fallback_key="top10_processed_columns",
    )

    if full_cols is None:
        raise RuntimeError("Could not determine full RandomForest feature columns.")
    if top10_cols is None:
        raise RuntimeError("Could not determine Top10 RandomForest feature columns.")

    thresholds = load_thresholds(THRESHOLD_PATH)
    full_threshold = thresholds["Full_RandomForest"]
    top10_threshold = thresholds["Top10_RandomForest"]

    x_full = prepare_features(test_df, full_cols, "Full_RandomForest")
    x_top10 = prepare_features(test_df, top10_cols, "Top10_RandomForest")

    full_prob = full_model.predict_proba(x_full)[:, 1]
    top10_prob = top10_model.predict_proba(x_top10)[:, 1]

    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "Top10_RandomForest_y_prob": top10_prob,
            "Top10_RandomForest_threshold": top10_threshold,
            "Full_RandomForest_y_prob": full_prob,
            "Full_RandomForest_threshold": full_threshold,
        }
    )
    pred_path = OUT_DIR / "internal_test_predictions_for_decision_calibration.csv"
    pred_df.to_csv(pred_path, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "Model": "Top10_RandomForest",
                "Threshold": top10_threshold,
                "Brier_Score": brier_score_loss(y_true, top10_prob),
                "ECE": expected_calibration_error(y_true, top10_prob),
                "N": len(y_true),
                "N_positive": int(np.sum(y_true)),
                "N_negative": int(len(y_true) - np.sum(y_true)),
            },
            {
                "Model": "Full_RandomForest",
                "Threshold": full_threshold,
                "Brier_Score": brier_score_loss(y_true, full_prob),
                "ECE": expected_calibration_error(y_true, full_prob),
                "N": len(y_true),
                "N_positive": int(np.sum(y_true)),
                "N_negative": int(len(y_true) - np.sum(y_true)),
            },
        ]
    )
    summary_path = OUT_DIR / "internal_test_decision_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    calibration_df = pd.concat(
        [
            get_calibration_rows("Top10_RandomForest", y_true, top10_prob),
            get_calibration_rows("Full_RandomForest", y_true, full_prob),
        ],
        ignore_index=True,
    )
    calibration_path = OUT_DIR / "internal_test_calibration_curve_data.csv"
    calibration_df.to_csv(calibration_path, index=False)

    threshold_grid = np.linspace(0.01, 0.99, 99)
    prevalence = float(np.mean(y_true))
    decision_df = pd.DataFrame(
        {
            "Threshold": threshold_grid,
            "Top10_RandomForest_Net_Benefit": net_benefit(
                y_true, top10_prob, threshold_grid
            ),
            "Full_RandomForest_Net_Benefit": net_benefit(
                y_true, full_prob, threshold_grid
            ),
            "Treat_All_Net_Benefit": prevalence
            - (1 - prevalence) * (threshold_grid / (1 - threshold_grid)),
            "Treat_None_Net_Benefit": np.zeros_like(threshold_grid),
        }
    )
    decision_path = OUT_DIR / "internal_test_decision_curve_data.csv"
    decision_df.to_csv(decision_path, index=False)

    top10_cal = calibration_df[calibration_df["Model"] == "Top10_RandomForest"]
    full_cal = calibration_df[calibration_df["Model"] == "Full_RandomForest"]

    top10_summary = summary_df[summary_df["Model"] == "Top10_RandomForest"].iloc[0]
    full_summary = summary_df[summary_df["Model"] == "Full_RandomForest"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(
        top10_cal["Mean_Predicted_Probability"],
        top10_cal["Observed_Fraction"],
        marker="o",
        color=TOP10_COLOR,
        markersize=4,
        linewidth=2.0,
        label=(
            f"Top10 model, Brier={top10_summary['Brier_Score']:.3f}, "
            f"ECE={top10_summary['ECE']:.3f}"
        ),
    )
    axes[0].plot(
        full_cal["Mean_Predicted_Probability"],
        full_cal["Observed_Fraction"],
        marker="s",
        color=FULL_COLOR,
        linestyle="--",
        markersize=4,
        linewidth=2.0,
        label=(
            f"Full model, Brier={full_summary['Brier_Score']:.3f}, "
            f"ECE={full_summary['ECE']:.3f}"
        ),
    )
    axes[0].plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Perfect calibration",
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Predicted probability", fontweight="bold", fontsize=17)
    axes[0].set_ylabel("Observed fraction", fontweight="bold", fontsize=17)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(axis="both", labelsize=14)
    axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    axes[0].text(
        -0.10, 1.02, "(A)",
        transform=axes[0].transAxes,
        fontsize=18, fontweight="bold",
        va="bottom", ha="left",
    )

    axes[1].plot(
        decision_df["Threshold"],
        decision_df["Top10_RandomForest_Net_Benefit"],
        color=TOP10_COLOR,
        linewidth=2.0,
        label="Top10 model",
    )
    axes[1].plot(
        decision_df["Threshold"],
        decision_df["Full_RandomForest_Net_Benefit"],
        color=FULL_COLOR,
        linestyle="--",
        linewidth=2.0,
        label="Full model",
    )
    axes[1].plot(
        decision_df["Threshold"],
        decision_df["Treat_All_Net_Benefit"],
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="Treat All",
    )
    axes[1].plot(
        decision_df["Threshold"],
        decision_df["Treat_None_Net_Benefit"],
        color="black",
        linestyle="-.",
        linewidth=1.0,
        label="Treat None",
    )
    axes[1].axvline(
        top10_threshold,
        color=TOP10_COLOR,
        linestyle=":",
        linewidth=0.8,
        alpha=0.65,
    )
    axes[1].axvline(
        full_threshold,
        color=FULL_COLOR,
        linestyle=":",
        linewidth=0.8,
        alpha=0.65,
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.15, 0.25)
    axes[1].set_xlabel("Threshold probability", fontweight="bold", fontsize=17)
    axes[1].set_ylabel("Net benefit", fontweight="bold", fontsize=17)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(axis="both", labelsize=14)
    axes[1].legend(frameon=False, fontsize=12, loc="upper right")
    axes[1].text(
        -0.10, 1.02, "(B)",
        transform=axes[1].transAxes,
        fontsize=18, fontweight="bold",
        va="bottom", ha="left",
    )

    plt.tight_layout()

    png_path = OUT_DIR / "internal_test_decision_calibration_curves.png"
    pdf_path = OUT_DIR / "internal_test_decision_calibration_curves.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("Final evaluation files saved to:")
    print(OUT_DIR)
    for path in [
        pred_path,
        summary_path,
        calibration_path,
        decision_path,
        png_path,
        pdf_path,
    ]:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
