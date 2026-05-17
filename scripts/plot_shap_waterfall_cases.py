"""SHAP waterfall plots for two representative test-set cases.

Case selection (no manual fabrication):
  - Panel A — Representative responder:    true label == 1, predicted prob is the maximum
                                            among positives in the test set.
  - Panel B — Representative non-responder: true label == 0, predicted prob is the minimum
                                            among negatives in the test set.

Everything else is computed end-to-end from the real artefacts:
  - Model:  output/tuned/models/tuned_randomforest.pkl
  - Test:   data/processed/test_processed.csv (model-input space, one-hot encoded,
            standardised; aligned with data/interim/internal_test_raw.csv for
            human-readable raw values).
  - SHAP:   TreeExplainer on the loaded model, positive-class column.
  - Baseline E[f(x)] = explainer.expected_value for the positive class.
  - One-hot groups (PVTT, BCLC stage) are merged by summing SHAP values,
    matching the convention used in merged_shap_importance.csv.
  - Top-10 feature ordering is taken from merged_shap_importance.csv.

The waterfall rendering keeps the custom matplotlib arrow-bar style; we do not
call shap.plots.waterfall.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Polygon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "output" / "tuned" / "models" / "tuned_randomforest.pkl"
TEST_PROC_PATH = ROOT / "data" / "processed" / "test_processed.csv"
TEST_RAW_PATH = ROOT / "data" / "interim" / "internal_test_raw.csv"
MERGED_IMP_PATH = ROOT / "output" / "shap_analysis" / "randomforest" / "merged_shap_importance.csv"
OUT_DIR = ROOT / "output" / "shap_analysis" / "randomforest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "shap_waterfall_two_cases.png"

POSITIVE_COLOR = "#ff0051"
NEGATIVE_COLOR = "#008bfb"

# One-hot groups (sum SHAP values across the group's processed columns); matches
# the merge logic in shap_analysis_randomforest.py.
ONE_HOT_GROUPS = {
    "PVTT": ["PVTT_0", "PVTT_1", "PVTT_2", "PVTT_3"],
    "BCLC stage": ["BCLC stage_1", "BCLC stage_2"],
    "combined with other treatment": [
        "combined with other treatment_0",
        "combined with other treatment_1",
        "combined with other treatment_2",
    ],
    "tumor location": ["tumor location_1", "tumor location_2", "tumor location_3", "tumor location_4"],
}

# Map from merged feature name -> raw column name in internal_test_raw.csv.
RAW_COL = {
    "diameter of tumor": "diameter of tumor",
    "APTT": "APTT",
    "AFP": "AFP",
    "number of tumor": "number of tumor",
    "PVTT": "PVTT",
    "ALP": "ALP",
    "HBsAg": "HBsAg",
    "PA": "PA",
    "GGT": "GGT",
    "BCLC stage": "BCLC stage",
}

UNITS = {
    "diameter of tumor": "cm",
    "APTT": "s",
    "AFP": "ng/mL",
    "number of tumor": "",
    "PVTT": "",
    "ALP": "U/L",
    "HBsAg": "IU/mL",
    "PA": "mg/L",
    "GGT": "U/L",
    "BCLC stage": "",
}

DISPLAY_LABEL = {
    "diameter of tumor": "Diameter of tumor",
    "APTT": "APTT",
    "AFP": "AFP",
    "number of tumor": "Number of tumors",
    "PVTT": "PVTT",
    "ALP": "ALP",
    "HBsAg": "HBsAg",
    "PA": "PA",
    "GGT": "GGT",
    "BCLC stage": "BCLC stage",
}


# ---------------------------------------------------------------------------
# SHAP utilities
# ---------------------------------------------------------------------------
def positive_class_shap(explainer: shap.TreeExplainer, X: pd.DataFrame, model) -> tuple[np.ndarray, float]:
    sv = explainer.shap_values(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else -1

    if isinstance(sv, list):
        shap_values = sv[pos_idx]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_values = sv[:, :, pos_idx]
    else:
        shap_values = sv

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0:
        baseline = float(np.asarray(ev)[pos_idx])
    else:
        baseline = float(ev)
    return np.asarray(shap_values), baseline


def merge_onehot_shap(shap_row: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    """Return {merged feature name -> SHAP value} for one sample."""
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    merged: dict[str, float] = {}
    consumed: set[str] = set()
    for group, cols in ONE_HOT_GROUPS.items():
        present = [c for c in cols if c in col_idx]
        if not present:
            continue
        merged[group] = float(sum(shap_row[col_idx[c]] for c in present))
        consumed.update(present)
    for c in feature_cols:
        if c in consumed:
            continue
        display = c.replace("_log", "")
        merged[display] = float(shap_row[col_idx[c]])
    return merged


def format_value(name: str, raw_val) -> str:
    unit = UNITS.get(name, "")
    label = DISPLAY_LABEL.get(name, name)
    if name in ("number of tumor", "PVTT", "BCLC stage"):
        return f"{label} = {int(raw_val)}"
    return f"{label} = {raw_val:g}" + (f" {unit}" if unit else "")


# ---------------------------------------------------------------------------
# Waterfall renderer (unchanged custom matplotlib style)
# ---------------------------------------------------------------------------
def _arrow_polygon(start: float, end: float, y: float, half_h: float, head_w: float):
    direction = 1 if end >= start else -1
    head_w_signed = head_w * direction
    shaft_end = end - head_w_signed
    if direction > 0:
        shaft_end = max(start, shaft_end)
    else:
        shaft_end = min(start, shaft_end)
    return [
        (start, y - half_h),
        (shaft_end, y - half_h),
        (end, y),
        (shaft_end, y + half_h),
        (start, y + half_h),
    ]


def draw_waterfall(ax: plt.Axes, case: dict, baseline: float) -> None:
    fx = case["fx"]
    feats = case["features"]  # list of (label, shap_value)
    n = len(feats)

    # |SHAP| ascending -> smallest at bottom, largest at top.
    feats_sorted = sorted(feats, key=lambda kv: abs(kv[1]))

    cumulative = baseline
    bars = []
    for i, (label, val) in enumerate(feats_sorted):
        bars.append((i, cumulative, cumulative + val, val, label))
        cumulative += val
    cum_end = cumulative  # cumulative after applying the displayed 10 features

    xs = [baseline, fx, cum_end] + [b[1] for b in bars] + [b[2] for b in bars]
    xmin, xmax = min(xs), max(xs)
    span = max(xmax - xmin, 1e-6)
    ax.set_xlim(xmin - 0.20 * span, xmax + 0.22 * span)
    head_w = span * 0.014
    half_h = 0.32

    ax.axvline(baseline, color="#999999", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.axvline(fx, color="#444444", lw=0.8, ls=":", alpha=0.7, zorder=1)

    for y, start, end, val, _ in bars:
        color = POSITIVE_COLOR if val > 0 else NEGATIVE_COLOR
        verts = _arrow_polygon(start, end, y, half_h, head_w)
        ax.add_patch(Polygon(verts, closed=True, facecolor=color,
                              edgecolor="white", linewidth=0.6, zorder=3))
        sign = "+" if val > 0 else "−"
        text = f"{sign}{abs(val):.3f}"
        off = span * 0.012
        if val > 0:
            ax.text(end + off, y, text, va="center", ha="left",
                    fontsize=9, color=color, fontweight="bold", zorder=4)
        else:
            ax.text(end - off, y, text, va="center", ha="right",
                    fontsize=9, color=color, fontweight="bold", zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels([b[4] for b in bars], fontsize=10)
    ax.set_ylim(-1.0, n - 1 + 1.05)
    ax.tick_params(axis="y", length=0)

    ax.annotate(rf"$E[f(x)] = {baseline:.3f}$",
                xy=(baseline, -0.70), ha="center", va="top",
                fontsize=10, color="#555555")
    ax.annotate(rf"$f(x) = {fx:.3f}$",
                xy=(fx, n - 1 + 0.75), ha="center", va="bottom",
                fontsize=11, color="#222222", fontweight="bold")

    # Footnote: residual contribution from the features NOT among Top 10.
    residual = fx - cum_end
    ax.text(
        0.99, -0.06,
        rf"$\sum$ remaining features $= {residual:+.3f}$",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#888888", style="italic",
    )

    ax.set_title(case["title"], loc="left", fontsize=12, pad=10)
    ax.set_xlabel(r"Model output  $f(x)$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Load data + model
    test_proc = pd.read_csv(TEST_PROC_PATH)
    test_raw = pd.read_csv(TEST_RAW_PATH)
    assert (test_proc["label"].values == test_raw["label"].values).all(), \
        "test_processed.csv and internal_test_raw.csv are not row-aligned."

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    feature_cols = [c for c in test_proc.columns if c != "label"]
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    X_test = test_proc[feature_cols].copy()
    y_test = test_proc["label"].astype(int).values

    # 2. Predict probabilities, pick representative cases
    proba_pos = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    pos_mask = y_test == 1
    neg_mask = y_test == 0
    idx_resp = int(np.where(pos_mask)[0][np.argmax(proba_pos[pos_mask])])
    idx_non_resp = int(np.where(neg_mask)[0][np.argmin(proba_pos[neg_mask])])

    print("Selected cases from test set:")
    print(f"  Responder      | row index = {idx_resp:>3d} | "
          f"true label = {int(y_test[idx_resp])} | "
          f"predicted prob = {proba_pos[idx_resp]:.4f} | "
          f"id = {test_raw['number'].iloc[idx_resp]}")
    print(f"  Non-responder  | row index = {idx_non_resp:>3d} | "
          f"true label = {int(y_test[idx_non_resp])} | "
          f"predicted prob = {proba_pos[idx_non_resp]:.4f} | "
          f"id = {test_raw['number'].iloc[idx_non_resp]}")

    # 3. SHAP values for these two rows
    explainer = shap.TreeExplainer(model)
    rows = X_test.iloc[[idx_resp, idx_non_resp]]
    shap_vals, baseline = positive_class_shap(explainer, rows, model)
    print(f"\nExplainer expected value (positive class) = {baseline:.4f}")

    # 4. Merge one-hot SHAP values per case
    merged_a = merge_onehot_shap(shap_vals[0], feature_cols)
    merged_b = merge_onehot_shap(shap_vals[1], feature_cols)

    # 5. Top-10 merged features (global ordering from merged_shap_importance.csv)
    top10 = pd.read_csv(MERGED_IMP_PATH).head(10)["Feature"].tolist()
    missing = [t for t in top10 if t not in merged_a]
    if missing:
        raise RuntimeError(f"Missing merged features for cases: {missing}")

    def build_feature_rows(merged: dict[str, float], raw_row: pd.Series) -> list[tuple[str, float]]:
        out = []
        for feat in top10:
            raw_val = raw_row[RAW_COL[feat]]
            out.append((format_value(feat, raw_val), merged[feat]))
        return out

    case_a = {
        "title": (f"Case A — Representative responder "
                  f"(test row {idx_resp}, true label = 1)"),
        "fx": float(proba_pos[idx_resp]),
        "features": build_feature_rows(merged_a, test_raw.iloc[idx_resp]),
    }
    case_b = {
        "title": (f"Case B — Representative non-responder "
                  f"(test row {idx_non_resp}, true label = 0)"),
        "fx": float(proba_pos[idx_non_resp]),
        "features": build_feature_rows(merged_b, test_raw.iloc[idx_non_resp]),
    }

    # 6. Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 13), dpi=300)
    draw_waterfall(axes[0], case_a, baseline)
    draw_waterfall(axes[1], case_b, baseline)
    for ax, lbl in zip(axes, ["(A)", "(B)"]):
        ax.text(-0.30, 1.04, lbl, transform=ax.transAxes,
                fontsize=14, fontweight="normal", va="bottom", ha="left")

    plt.tight_layout(h_pad=3.5)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved waterfall figure to: {OUT_PATH}")


if __name__ == "__main__":
    main()
