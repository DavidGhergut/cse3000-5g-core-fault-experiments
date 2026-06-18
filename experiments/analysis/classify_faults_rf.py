#!/usr/bin/env python3
"""
classify_faults_rf.py  —  Random Forest fault classifier on Δ|r| features + SHAP.

Usage:
    python3 classify_faults_rf.py \
        --corr correlations_o5g/boyan/correlations.csv \
               correlations_o5g/boyan-2/correlations.csv \
        --out  output/classifier_delta

Outputs:
    feature_matrix.csv          Δ|r| feature matrix (rows=faults, cols=signal pairs)
    classification_report.txt   LOOCV accuracy + per-class precision/recall/F1
    confusion_matrix.png        LOOCV confusion matrix
    shap_bar.png                Mean |SHAP| per feature (top-20, averaged across classes)
    shap_perclass.png           Top-15 SHAP features per fault category
    shap_importance.csv         Full SHAP importance table
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import shap

# ── Constants ─────────────────────────────────────────────────────────────────

CROSS_MODAL = {"metrics↔traces", "metrics↔logs", "traces↔logs"}

# Modality pair sets for ablation study (SQ4)
MODAL_SETS = {
    "cross":   {"metrics↔traces", "metrics↔logs", "traces↔logs"},   # default
    "metrics": {"metrics↔metrics"},
    "logs":    {"logs↔logs"},
    "traces":  {"traces↔traces"},
    "all":     {"metrics↔traces", "metrics↔logs", "traces↔logs",
                "metrics↔metrics", "logs↔logs", "traces↔traces"},
}

RELABEL = {"cascade": "pod_crash"}   # NRF cascade = NRF pod failure

CATS = ["infrastructure", "pod_crash", "network", "pfcp"]

CAT_COLORS = {
    "infrastructure":  "#e07b39",
    "pod_crash":       "#c0392b",
    "network":         "#2980b9",
    "pfcp":            "#8e44ad",
    "security_nas":    "#2ecc71",
    "security_flood":  "#e74c3c",
    "security_auth":   "#9b59b6",
}

MODE_CONFIGS = {
    "operational": {
        "cats":    ["infrastructure", "pod_crash", "network", "pfcp"],
        "relabel": {"cascade": "pod_crash"},
    },
    "security": {
        "cats":    ["security_nas", "security_flood", "security_auth"],
        "relabel": {},
    },
    "combined": {
        "cats":    ["infrastructure", "pod_crash", "network", "pfcp",
                    "security_nas", "security_flood", "security_auth"],
        "relabel": {"cascade": "pod_crash"},
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def pair_label(sig_a, sig_b):
    """'cpu_amf', 'jaeger_amf'  →  'AMF: cpu → traces'"""
    def parts(sig):
        for prefix, mod in [
            ("net_rx_", "net rx"), ("net_tx_", "net tx"),
            ("loki_vol_", "log vol"), ("loki_", "log errors"),
            ("jaeger_", "traces"), ("cpu_", "cpu"), ("mem_", "memory"),
        ]:
            if sig.startswith(prefix):
                return sig[len(prefix):].upper(), mod
        return sig.upper(), "?"
    nf_a, mod_a = parts(sig_a)
    nf_b, mod_b = parts(sig_b)
    return f"{nf_a}: {mod_a} → {mod_b}" if nf_a == nf_b else f"{nf_a} {mod_a} → {nf_b} {mod_b}"


# ── Feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(corr_paths, cats=None, relabel=None, modal_set=None, min_frac=0.20):
    """
    For each trial's correlations.csv, compute Δ|r| = |r_during| − |r_pre|
    per (fault_name, sig_a, sig_b).  Pivot to wide format:
        rows  = one fault instance (fault × trial)
        cols  = signal pairs (filtered by modal_set)
        values = Δ|r|  (0 where pair was unobservable)
    """
    if cats is None:
        cats = CATS
    if relabel is None:
        relabel = RELABEL
    if modal_set is None:
        modal_set = CROSS_MODAL

    frames = []
    for path in corr_paths:
        df = pd.read_csv(path)
        df = df[df["modality_pair"].isin(modal_set)]

        # Include trial in index when the CSV has per-trial rows (security faults);
        # fall back to path-based label for single-trial datasets (boyan, david-3, etc.)
        has_trial_col = "trial" in df.columns
        idx = ["fault_name", "sig_a", "sig_b", "fault_category"]
        if has_trial_col:
            idx = ["fault_name", "trial", "sig_a", "sig_b", "fault_category"]

        # Use Spearman rank correlation if available (robust to non-linear
        # relationships and outlier bins); fall back to Pearson for older CSVs.
        corr_col = "spearman_r" if "spearman_r" in df.columns else "correlation"
        pre  = df[df["window"] == "pre"   ].set_index(idx)[corr_col].abs()
        dur  = df[df["window"] == "during"].set_index(idx)[corr_col].abs()
        # Align on common index before subtraction (pre may be absent for some pairs)
        common = dur.index.intersection(pre.index)
        delta_vals = dur.loc[common] - pre.loc[common]
        # For pairs where pre is missing (flat baseline), use |r_during| directly
        only_dur = dur.index.difference(pre.index)
        if len(only_dur) > 0:
            delta_vals = pd.concat([delta_vals, dur.loc[only_dur]])
        delta = delta_vals.dropna().reset_index()
        delta.columns = idx + ["delta_r"]

        if has_trial_col:
            delta["trial"] = Path(path).parent.name + "_t" + delta["trial"].astype(str)
        else:
            delta["trial"] = Path(path).parent.name
        frames.append(delta)

    data = pd.concat(frames, ignore_index=True)
    data["fault_category"] = data["fault_category"].replace(relabel)
    data = data[data["fault_category"].isin(cats)]
    data["pair_key"] = data["sig_a"] + "|||" + data["sig_b"]

    # One row per (fault_name, trial); one column per pair
    pivot = (
        data.groupby(["fault_name", "trial", "fault_category", "pair_key"])["delta_r"]
        .mean()
        .unstack("pair_key")
        .reset_index()
    )

    labels    = pivot["fault_category"].values
    fault_ids = (pivot["fault_name"] + " [" + pivot["trial"] + "]").values
    pair_keys = [c for c in pivot.columns if "|||" in str(c)]

    X = pivot[pair_keys].fillna(0)

    # Drop columns that are non-zero in fewer than 20% of rows (very sparse)
    min_present = max(2, int(min_frac * len(X)))
    X = X.loc[:, (X != 0).sum() >= min_present]
    pair_keys = X.columns.tolist()

    feat_labels = [pair_label(*k.split("|||")) for k in pair_keys]
    return X.values.astype(float), labels, fault_ids, feat_labels


# ── Model training + LOOCV ────────────────────────────────────────────────────

def train_and_evaluate(X, y, out_dir):
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    y_pred = cross_val_predict(rf, X, y, cv=LeaveOneOut(), n_jobs=-1)

    report = classification_report(y, y_pred, target_names=sorted(set(y)), zero_division=0)
    print(report)
    (out_dir / "classification_report.txt").write_text(report)
    print(f"  [saved] {out_dir / 'classification_report.txt'}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=sorted(set(y)),
        ax=ax, colorbar=False, cmap="Blues",
        xticks_rotation=30,
    )
    ax.set_title("LOOCV Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_dir / 'confusion_matrix.png'}")

    # Train final model on all data for SHAP
    rf.fit(X, y)
    return rf


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def run_shap(rf, X, feat_labels, out_dir):
    explainer   = shap.TreeExplainer(rf)
    raw         = explainer.shap_values(X)
    classes     = rf.classes_

    # Normalise to (n_classes, n_samples, n_features) regardless of SHAP version
    if isinstance(raw, list):
        sv_array = np.array(raw)                      # (n_classes, n_samples, n_features)
    elif raw.ndim == 3:
        sv_array = raw.transpose(2, 0, 1)             # (n_samples, n_features, n_classes) → reorder
    else:
        sv_array = raw[np.newaxis]                     # binary edge case

    shap_values = [sv_array[i] for i in range(len(classes))]
    mean_abs_per_feat = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    # Global bar chart (top-20)
    top20 = np.argsort(mean_abs_per_feat)[-20:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top20)), mean_abs_per_feat[top20], color="#4878d0", alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([feat_labels[i] for i in top20], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (averaged across fault categories)", fontsize=10)
    ax.set_title("Signal pairs ranked by SHAP importance\n(RF fault classifier, LOOCV)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_dir / 'shap_bar.png'}")

    # Per-class bar charts
    top_n = 15
    n_cls = len(classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(4.5 * n_cls, max(5, top_n * 0.42 + 1.5)),
                             sharey=False)
    if n_cls == 1:
        axes = [axes]

    for ax, cls, sv in zip(axes, classes, shap_values):
        mean_abs = np.abs(sv).mean(axis=0)
        order    = np.argsort(mean_abs)[-top_n:]
        color    = CAT_COLORS.get(cls, "#888888")
        ax.barh(range(len(order)), mean_abs[order], color=color, alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feat_labels[i] for i in order], fontsize=8)
        ax.set_title(cls, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Mean |SHAP|", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Per-class SHAP feature importance  (top-15 signal pairs per fault category)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_perclass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_dir / 'shap_perclass.png'}")

    # Importance CSV
    imp = pd.DataFrame({"pair": feat_labels, "mean_abs_shap": mean_abs_per_feat})
    for cls, sv in zip(classes, shap_values):
        imp[f"shap_{cls}"] = np.abs(sv).mean(axis=0)
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(
        out_dir / "shap_importance.csv", index=False)
    print(f"  [saved] {out_dir / 'shap_importance.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr", nargs="+", required=True,
                        help="correlations.csv file(s), one per trial")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["operational", "security", "combined"],
                        default="operational",
                        help="Dataset mode: operational (default), security, or combined")
    parser.add_argument("--modal", choices=["cross", "metrics", "logs", "traces", "all"],
                        default="cross",
                        help="Modality filter for SQ4 ablation: cross (default), metrics, logs, traces, all")
    args = parser.parse_args()

    cfg     = MODE_CONFIGS[args.mode]
    cats    = cfg["cats"]
    relabel = cfg["relabel"]
    modal_set = MODAL_SETS[args.modal]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[mode: {args.mode}]  [modal: {args.modal}]  categories: {cats}")
    print("\n[1/3] Building Δ|r| feature matrix...")
    X, y, fault_ids, feat_labels = build_feature_matrix(
        [Path(p) for p in args.corr], cats=cats, relabel=relabel, modal_set=modal_set)

    print(f"  {X.shape[0]} fault instances  ×  {X.shape[1]} cross-modal pairs")
    for cat in cats:
        print(f"  {cat}: {(y == cat).sum()} instances")

    feat_df = pd.DataFrame(X, columns=feat_labels)
    feat_df.insert(0, "label", y)
    feat_df.insert(0, "fault_id", fault_ids)
    feat_df.to_csv(out_dir / "feature_matrix.csv", index=False)
    print(f"  [saved] {out_dir / 'feature_matrix.csv'}")

    print("\n[2/3] RF training + LOOCV evaluation...")
    rf = train_and_evaluate(X, y, out_dir)

    print("\n[3/3] SHAP analysis...")
    run_shap(rf, X, feat_labels, out_dir)

    print(f"\n✓ All outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
