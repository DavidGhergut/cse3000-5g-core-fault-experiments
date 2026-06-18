#!/usr/bin/env python3
"""
experiments/david/raw_intershap.py

Alternative to the Δ|ρ| approach: train RF on raw z-scored signal means
and apply SHAP interaction values to discover cross-modal interactions.

For each fault instance:
  1. Compute z-scored mean of each signal during fault window (relative to pre)
  2. Build a feature vector of raw z-scores (one per signal per NF)
  3. Train RF with LOOCV
  4. Apply SHAP TreeExplainer.shap_interaction_values()
  5. Plot top cross-modal interactions per fault category

Usage:
    python3 experiments/david/raw_intershap.py \
        --out data/5GCore/classifier/intershap
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import _paths

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import shap

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

BIN_SEC   = 10
N_BINS    = 30
PRE_BINS  = 60   # 600s pre-fault / 10s

CORE_NFS  = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf", "mongodb", "upf"]
TRACE_NFS = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf"]

TRIAL_PATHS = [_paths.trial_dir(t) for t in ["boyan", "boyan-2", "trial4", "trial5"]]

FAULT_CATEGORIES = {
    "01-cpu-stress-amf":                       "infrastructure",
    "02-memory-pressure-upf":                  "infrastructure",
    "03-pod-crash-amf":                        "pod_crash",
    "04-network-delay-gnb-amf":                "network",
    "05-network-partition-amf-scp":            "network",
    "06-packet-loss-upf":                      "network",
    "07-pod-crash-smf":                        "pod_crash",
    "08-cpu-stress-scp":                       "infrastructure",
    "09-network-delay-nrf":                    "network",
    "10-pfcp-session-establishment-flood-upf": "pfcp",
    "11-pfcp-session-deletion-upf":            "pfcp",
    "12-pfcp-session-modification-drop-upf":   "pfcp",
    "13-pfcp-session-modification-dupl-upf":   "pfcp",
    "14-upf-infrastructure-packet-loss":       "network",
    "15-nrf-cascade":                          "pod_crash",   # merged
    "16-cpu-stress-ausf":                      "infrastructure",
    "17-network-delay-scp":                    "network",
    "18-cpu-stress-nrf":                       "infrastructure",
    "19-udm-pod-crash":                        "pod_crash",
    "20-mongodb-pod-kill":                     "pod_crash",
    "21-n2-partition-amf-gnb":                 "network",
    "22-memory-pressure-amf":                  "infrastructure",
}

# ── Data loaders (same logic as cross_correlation.py) ────────────────────────

def pod_to_nf(pod):
    m = re.match(r'open5gs-([a-z]+)-', str(pod))
    return m.group(1) if m else None


def load_prom_nf(fault_dir, phase, fname, nf):
    path = fault_dir / "prometheus" / phase / fname
    if not path.exists() or path.stat().st_size == 0:
        return None, None
    df = pd.read_csv(path)
    if "pod" not in df.columns:
        return None, None
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = df["pod"].apply(lambda p: pod_to_nf(p) == nf)
    sub = df[mask].dropna(subset=["value"])
    if len(sub) == 0:
        return None, None
    grouped = sub.groupby("timestamp")["value"].mean()
    return grouped.index.values, grouped.values


def bin_series(ts, vals, t_start, n, bin_sec=BIN_SEC):
    result = np.full(n, np.nan)
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        mask = (ts >= t0) & (ts < t1)
        if mask.any():
            result[i] = np.nanmean(vals[mask])
    return result


def load_jaeger_p99_bins(fault_dir, phase, nf, t_start, n, bin_sec=BIN_SEC):
    path = fault_dir / "jaeger" / phase / "spans_flat.csv"
    result = np.full(n, np.nan)
    if not path.exists() or path.stat().st_size == 0:
        return result
    df = pd.read_csv(path)
    df = df[df["service"] == nf].copy()
    if len(df) == 0:
        return result
    df["ts_sec"] = df["start_us"] / 1e6
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        mask = (df["ts_sec"] >= t0) & (df["ts_sec"] < t1)
        spans = df.loc[mask, "duration_us"]
        if len(spans) >= 3:
            result[i] = np.percentile(spans, 99)
    s = pd.Series(result).interpolate(method="linear", limit_direction="both", limit=3)
    return s.values


def load_loki_errors_bins(fault_dir, phase, nf, t_start, n, bin_sec=BIN_SEC):
    path = fault_dir / "loki" / phase / "errors.csv"
    result = np.zeros(n)
    if not path.exists() or path.stat().st_size == 0:
        return result
    df = pd.read_csv(path)
    if "app" not in df.columns or len(df) == 0:
        return result
    df = df[df["app"] == nf].copy()
    if len(df) == 0:
        return result
    df["ts_sec"] = df["timestamp_ns"] / 1e9
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        result[i] = ((df["ts_sec"] >= t0) & (df["ts_sec"] < t1)).sum()
    return result


def load_loki_vol_bins(fault_dir, phase, nf, t_start, n, bin_sec=BIN_SEC):
    path = fault_dir / "loki" / phase / "all.csv"
    result = np.zeros(n)
    if not path.exists() or path.stat().st_size == 0:
        return result
    df = pd.read_csv(path)
    if "app" not in df.columns or len(df) == 0:
        return result
    df = df[df["app"] == nf].copy()
    if len(df) == 0:
        return result
    df["ts_sec"] = df["timestamp_ns"] / 1e9
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        result[i] = ((df["ts_sec"] >= t0) & (df["ts_sec"] < t1)).sum()
    return result


def zscore_mean(during_bins, pre_bins):
    """Compute z-scored mean: (mean_during - mean_pre) / std_pre"""
    pre_mean = np.nanmean(pre_bins)
    pre_std  = np.nanstd(pre_bins)
    dur_mean = np.nanmean(during_bins)
    if pre_std < 1e-10:
        return dur_mean - pre_mean
    return (dur_mean - pre_mean) / pre_std


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(fault_dir):
    """
    Extract raw z-scored mean features for one fault instance.
    Returns (feature_dict, modality_dict) where modality_dict maps
    feature_name -> modality ('metric', 'trace', 'log')
    """
    tl_path = fault_dir / "timeline.json"
    if not tl_path.exists():
        return None, None
    with open(tl_path) as f:
        tl = json.load(f)

    pre_start    = tl["pre"]["start"]
    fault_start  = tl["fault"]["start"]

    features  = {}
    modality  = {}

    for nf in CORE_NFS:
        # CPU
        ts, vals = load_prom_nf(fault_dir, "pre",    "container_cpu_usage_rate.csv", nf)
        ts2, v2  = load_prom_nf(fault_dir, "during", "container_cpu_usage_rate.csv", nf)
        if ts is not None and ts2 is not None:
            pre_b = bin_series(ts,  vals, pre_start,   PRE_BINS)
            dur_b = bin_series(ts2, v2,   fault_start, N_BINS)
            fname = f"cpu_{nf}"
            features[fname] = zscore_mean(dur_b, pre_b)
            modality[fname] = "metric"

        # Memory
        ts, vals = load_prom_nf(fault_dir, "pre",    "container_memory_working_set_bytes.csv", nf)
        ts2, v2  = load_prom_nf(fault_dir, "during", "container_memory_working_set_bytes.csv", nf)
        if ts is not None and ts2 is not None:
            pre_b = bin_series(ts,  vals, pre_start,   PRE_BINS)
            dur_b = bin_series(ts2, v2,   fault_start, N_BINS)
            fname = f"mem_{nf}"
            features[fname] = zscore_mean(dur_b, pre_b)
            modality[fname] = "metric"

        # Net RX
        ts, vals = load_prom_nf(fault_dir, "pre",    "network_rx_bytes_rate.csv", nf)
        ts2, v2  = load_prom_nf(fault_dir, "during", "network_rx_bytes_rate.csv", nf)
        if ts is not None and ts2 is not None:
            pre_b = bin_series(ts,  vals, pre_start,   PRE_BINS)
            dur_b = bin_series(ts2, v2,   fault_start, N_BINS)
            fname = f"net_rx_{nf}"
            features[fname] = zscore_mean(dur_b, pre_b)
            modality[fname] = "metric"

        # Net TX
        ts, vals = load_prom_nf(fault_dir, "pre",    "network_tx_bytes_rate.csv", nf)
        ts2, v2  = load_prom_nf(fault_dir, "during", "network_tx_bytes_rate.csv", nf)
        if ts is not None and ts2 is not None:
            pre_b = bin_series(ts,  vals, pre_start,   PRE_BINS)
            dur_b = bin_series(ts2, v2,   fault_start, N_BINS)
            fname = f"net_tx_{nf}"
            features[fname] = zscore_mean(dur_b, pre_b)
            modality[fname] = "metric"

        # Traces (P99)
        if nf in TRACE_NFS:
            pre_b  = load_jaeger_p99_bins(fault_dir, "pre",    nf, pre_start,   PRE_BINS)
            dur_b  = load_jaeger_p99_bins(fault_dir, "during", nf, fault_start, N_BINS)
            fname  = f"jaeger_{nf}"
            features[fname] = zscore_mean(dur_b, pre_b)
            modality[fname] = "trace"

        # Log errors
        pre_b  = load_loki_errors_bins(fault_dir, "pre",    nf, pre_start,   PRE_BINS)
        dur_b  = load_loki_errors_bins(fault_dir, "during", nf, fault_start, N_BINS)
        fname  = f"loki_{nf}"
        features[fname] = zscore_mean(dur_b, pre_b)
        modality[fname] = "log"

        # Log volume
        pre_b  = load_loki_vol_bins(fault_dir, "pre",    nf, pre_start,   PRE_BINS)
        dur_b  = load_loki_vol_bins(fault_dir, "during", nf, fault_start, N_BINS)
        fname  = f"loki_vol_{nf}"
        features[fname] = zscore_mean(dur_b, pre_b)
        modality[fname] = "log"

    return features, modality


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_paths.OUTPUT / "intershap"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all instances ─────────────────────────────────────────────────
    print("Loading raw features from all trials...")
    all_features  = []
    all_labels    = []
    modality_map  = None

    for trial_path in TRIAL_PATHS:
        if not trial_path.exists():
            print(f"  Skipping {trial_path} (not found)")
            continue
        for fault_name, category in FAULT_CATEGORIES.items():
            fault_dir = trial_path / fault_name
            if not fault_dir.exists():
                continue
            feats, mods = extract_features(fault_dir)
            if feats is None or len(feats) == 0:
                continue
            all_features.append(feats)
            all_labels.append(category)
            if modality_map is None:
                modality_map = mods
            print(f"  {trial_path.name}/{fault_name}: {len(feats)} features")

    print(f"\nTotal instances: {len(all_labels)}")
    print(f"Label counts: {pd.Series(all_labels).value_counts().to_dict()}")

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    # Get union of all feature names, fill missing with 0
    all_feat_names = sorted(set(k for f in all_features for k in f.keys()))
    X = np.array([[f.get(fn, 0.0) for fn in all_feat_names] for f in all_features])
    X = np.nan_to_num(X, nan=0.0)

    le = LabelEncoder()
    y  = le.fit_transform(all_labels)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Classes: {le.classes_}")

    # ── 3. LOOCV accuracy ─────────────────────────────────────────────────────
    print("\nRunning LOOCV...")
    loo   = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(X):
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_features="sqrt", random_state=42
        )
        clf.fit(X[train_idx], y[train_idx])
        preds.append(clf.predict(X[test_idx])[0])

    acc = accuracy_score(y, preds)
    print(f"LOOCV accuracy (raw z-score features): {acc:.1%}")

    # ── 4. Train final model on all data for SHAP ────────────────────────────
    print("\nTraining final RF for SHAP interaction analysis...")
    clf_full = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_features="sqrt", random_state=42
    )
    clf_full.fit(X, y)

    # ── 5. SHAP interaction values ────────────────────────────────────────────
    print("Computing SHAP interaction values (this may take a minute)...")
    explainer    = shap.TreeExplainer(clf_full)
    # shap_interaction_values returns shape (n_classes, n_samples, n_features, n_features)
    # or (n_samples, n_features, n_features) for binary. For multiclass it's a list.
    shap_iv = explainer.shap_interaction_values(X)

    # shap_iv is a list of length n_classes, each (n_samples, n_features, n_features)
    n_classes  = len(le.classes_)
    n_features = len(all_feat_names)

    # ── 6. Extract cross-modal interaction pairs per class ───────────────────
    print("Extracting cross-modal interactions...")

    def get_modality(feat_name):
        return modality_map.get(feat_name, "metric")

    # Build modality array aligned with all_feat_names
    feat_mods = [get_modality(fn) for fn in all_feat_names]

    categories = list(le.classes_)
    fig, axes  = plt.subplots(1, len(categories), figsize=(5 * len(categories), 7))
    if len(categories) == 1:
        axes = [axes]

    # Inspect actual shape of shap_iv to handle SHAP version differences
    print(f"\nshap_iv type: {type(shap_iv)}")
    if isinstance(shap_iv, list):
        print(f"shap_iv length: {len(shap_iv)}, element shape: {np.array(shap_iv[0]).shape}")
    else:
        print(f"shap_iv shape: {np.array(shap_iv).shape}")

    # shap_iv shape: (n_samples, n_features, n_features, n_classes)
    shap_iv_arr = np.array(shap_iv)
    print(f"shap_iv shape: {shap_iv_arr.shape}")

    for cls_idx, cat in enumerate(categories):
        # Extract interaction matrix for this class: (n_samples, n_features, n_features)
        iv_cls  = shap_iv_arr[:, :, :, cls_idx]
        mean_iv = np.abs(iv_cls).mean(axis=0)  # (n_features, n_features)

        # Find top cross-modal pairs (off-diagonal where modalities differ)
        cross_modal_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if feat_mods[i] != feat_mods[j]:
                    # symmetrise: take average of (i,j) and (j,i)
                    val = (mean_iv[i, j] + mean_iv[j, i]) / 2
                    cross_modal_pairs.append((val, all_feat_names[i], all_feat_names[j],
                                              feat_mods[i], feat_mods[j]))

        cross_modal_pairs.sort(reverse=True)
        top15 = cross_modal_pairs[:15]

        ax = axes[cls_idx]
        labels = [f"{a} ↔ {b}" for _, a, b, _, _ in top15]
        vals   = [v for v, *_ in top15]
        colors = []
        for _, a, b, ma, mb in top15:
            pair = tuple(sorted([ma, mb]))
            if pair == ("metric", "trace"):
                colors.append("#2196F3")
            elif pair == ("log", "metric"):
                colors.append("#FF9800")
            else:
                colors.append("#9C27B0")

        ax.barh(range(len(top15)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(labels[::-1], fontsize=7)
        ax.set_title(cat, fontsize=10, color={
            "infrastructure":"#FF9800","network":"#2196F3",
            "pfcp":"#9C27B0","pod_crash":"#F44336"
        }.get(cat, "black"), fontweight="bold")
        ax.set_xlabel("Mean |SHAP interaction|", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="metrics↔traces"),
        Patch(facecolor="#FF9800", label="metrics↔logs"),
        Patch(facecolor="#9C27B0", label="traces↔logs"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Top-15 cross-modal InterSHAP pairs per fault category\n"
        f"(Raw z-score features, LOOCV accuracy: {acc:.1%})",
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = out_dir / "intershap_crossmodal.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # ── 7. Save summary CSV ───────────────────────────────────────────────────
    rows = []
    for cls_idx, cat in enumerate(categories):
        iv_cls  = shap_iv_arr[:, :, :, cls_idx]
        mean_iv = np.abs(iv_cls).mean(axis=0)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if feat_mods[i] != feat_mods[j]:
                    val = (mean_iv[i, j] + mean_iv[j, i]) / 2
                    rows.append({
                        "category":  cat,
                        "feat_a":    all_feat_names[i],
                        "feat_b":    all_feat_names[j],
                        "mod_a":     feat_mods[i],
                        "mod_b":     feat_mods[j],
                        "mean_abs_interaction": val,
                    })
    df_out = pd.DataFrame(rows).sort_values(["category","mean_abs_interaction"], ascending=[True,False])
    df_out.to_csv(out_dir / "intershap_crossmodal.csv", index=False)
    print(f"Saved: {out_dir / 'intershap_crossmodal.csv'}")
    print(f"\nLOOCV accuracy summary:")
    print(f"  Raw z-score + InterSHAP: {acc:.1%}")
    print(f"  Δ|ρ| cross-modal (paper): 40%")


if __name__ == "__main__":
    main()
