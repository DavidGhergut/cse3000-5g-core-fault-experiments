#!/usr/bin/env python3
"""
classify_extended_features.py
─────────────────────────────
Extended-feature RF classifier. Builds on classify_raw_signals_intershap.py by:

  TIER 2 — new signal types (previously unused):
    mem            container_memory_working_set_bytes   (memory-pressure faults)
    net_tx         network_tx_bytes_rate                (flood / egress faults)
    cpu_throttled  container_cpu_throttled_rate         (cpu-stress faults)
    rtt            ue_rtt.csv                            (end-to-end service KPI, global)

  TIER 1 — richer statistics per signal (was: mean only):
    mean, std, max_abs, slope   (captures temporal SHAPE, not just level)

Each per-NF signal contributes up to 4 stat-features; RTT is a single global
signal (not per-NF). A column is kept only if it is non-NaN/non-constant in
>=20% of fault instances (sparsity filter), so absent signals are dropped.

Compares LOOCV accuracy against the 21-feature Δmean baseline.

Usage (same --data layout as classify_raw_signals_intershap.py):
    python3 experiments/david/classify_extended_features.py \\
        --data data/5GCore/final_dataset/boyan \\
               data/5GCore/final_dataset/boyan-2 \\
               data/5GCore/final_dataset/trial4 \\
               data/5GCore/final_dataset/trial5 \\
               data/5GCore/final_dataset/trial6/C-fault-detection \\
               data/5GCore/final_dataset/trial7/C-fault-detection \\
        --out  data/5GCore/classifier/extended_all6
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, ConfusionMatrixDisplay)

# ── Import loaders from ccf_profile ──────────────────────────────────────────

def _import_ccf():
    script = Path(__file__).parent / "ccf_profile.py"
    spec   = importlib.util.spec_from_file_location("ccf_profile", script)
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ccf = _import_ccf()
load_prom_nf           = _ccf.load_prom_nf
bin_prom               = _ccf.bin_prom
load_jaeger_p99        = _ccf.load_jaeger_p99
load_loki_errors       = _ccf.load_loki_errors
load_loki_volume       = _ccf.load_loki_volume
load_o5g_counter       = _ccf.load_o5g_counter
zscore                 = _ccf.zscore
OPERATIONAL_CATEGORIES = _ccf.OPERATIONAL_CATEGORIES
SECURITY_CATEGORIES    = _ccf.SECURITY_CATEGORIES
N_BINS                 = _ccf.N_BINS
BIN_SEC                = _ccf.BIN_SEC

# ── Config ────────────────────────────────────────────────────────────────────

RELABEL = {"cascade": "pod_crash"}
CATS_OPERATIONAL = ["infrastructure", "pod_crash", "network", "pfcp"]
CATS_SECURITY    = ["security_nas", "security_flood", "security_auth"]
# Aligned with cross_correlation.py: all 10 core NFs + RAN log sources, so both the
# raw and the Delta|rho| models draw from an identical underlying signal pool.
TARGET_NFS  = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf", "mongodb", "upf"]
TRACE_NFS   = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf"]
RAN_NFS     = ["ueransim-gnb", "ueransim-ues"]  # RAN-side log sources (logs only)


def modality_of(sig_key):
    base = sig_key.split("__")[0]   # strip stat suffix
    if base.startswith("jaeger_"):
        return "traces"
    if base.startswith("loki_"):
        return "logs"
    if base.startswith("rtt"):
        return "rtt"
    return "metrics"


# ── New signal loaders ────────────────────────────────────────────────────────

def load_jaeger_summary_stats(fault_dir, phase, nf):
    """
    Load pre-computed trace statistics from summary.json.
    Returns dict with single scalar values (not time series):
      {span_count, error_rate, mean_latency, p50_latency, p95_latency}
    These are phase-level aggregates, so they cannot be used for per-bin correlation,
    but can be used as direct features in the extended classifier.
    """
    path = fault_dir / "jaeger" / phase / "summary.json"
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text())
        if nf not in summary:
            return None
        nf_stats = summary[nf]
        return {
            "span_count":   nf_stats.get("span_count", 0),
            "error_rate":   nf_stats.get("error_rate", 0.0),
            "mean_latency": nf_stats.get("duration_us_mean", 0.0),
            "p50_latency":  nf_stats.get("duration_us_p50", 0.0),
            "p95_latency":  nf_stats.get("duration_us_p95", 0.0),
        }
    except:
        return None


def load_rtt(fault_dir, phase, t_start, n=N_BINS, bin_sec=BIN_SEC):
    """Bin UE round-trip-time (ms) into the standard grid (global, not per-NF)."""
    path = fault_dir / "rtt" / phase / "ue_rtt.csv"
    result = np.full(n, np.nan)
    if not path.exists() or path.stat().st_size == 0:
        return result
    try:
        df = pd.read_csv(path)
    except Exception:
        return result
    if "timestamp_ms" not in df.columns or "rtt_ms" not in df.columns or len(df) == 0:
        return result
    ts = df["timestamp_ms"].values / 1000.0   # → seconds
    vals = pd.to_numeric(df["rtt_ms"], errors="coerce").values
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        mask = (ts >= t0) & (ts < t1)
        if mask.any():
            result[i] = np.nanmean(vals[mask])
    return result


def _pod_to_nf(pod):
    import re
    m = re.match(r'open5gs-([a-z]+)-', str(pod))
    return m.group(1) if m else None


def load_prom_all_nfs(fault_dir, phase, fname, t_start, interface=None,
                      n=N_BINS, bin_sec=BIN_SEC):
    """
    Read a prometheus CSV ONCE and return {nf: binned_array} for all target NFs.
    Much faster than re-reading per NF. Only keeps open5gs-<nf>-* pods.
    """
    path = fault_dir / "prometheus" / phase / fname
    out = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    try:
        df = pd.read_csv(path)   # plain read = fast C parser (callable usecols is slow path)
    except Exception:
        return out
    if "pod" not in df.columns or "value" not in df.columns or "timestamp" not in df.columns:
        return out
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if interface is not None and "interface" in df.columns:
        df = df[df["interface"] == interface]
    if len(df) == 0:
        return out
    df["nf"] = df["pod"].map(_pod_to_nf)
    df = df[df["nf"].isin(TARGET_NFS)]
    if len(df) == 0:
        return out
    for nf, sub in df.groupby("nf"):
        grouped = sub.groupby("timestamp")["value"].mean()
        out[nf] = bin_prom(grouped.index.values, grouped.values, t_start, n, bin_sec)
    return out


def build_signals_ext(fault_dir, phase, t_start):
    """Load ALL signal time-series for a phase window (raw, un-zscored)."""
    raw = {}
    # Read each prometheus CSV ONCE (not per-NF)
    prom_specs = [
        ("cpu",    "container_cpu_usage_rate.csv",            None),
        ("net_rx", "network_rx_bytes_rate.csv",               "eth0"),
        ("mem",    "container_memory_working_set_bytes.csv",  None),
        ("net_tx", "network_tx_bytes_rate.csv",               "eth0"),
        ("cputhr", "container_cpu_throttled_rate.csv",        None),
    ]
    for prefix, fname, iface in prom_specs:
        per_nf = load_prom_all_nfs(fault_dir, phase, fname, t_start, interface=iface)
        for nf, arr in per_nf.items():
            raw[f"{prefix}_{nf}"] = arr

    # Open5GS application-layer KPI counters (functional metric sub-layer)
    for fname in _ccf.O5G_COUNTERS:
        arr = load_o5g_counter(fault_dir, phase, fname, t_start)
        if arr is not None:
            raw["o5g_" + fname.replace("open5gs_", "").replace(".csv", "")] = arr

    # logs (error count + volume) + traces (per-NF filtered)
    for nf in TARGET_NFS:
        raw[f"loki_{nf}"]     = load_loki_errors(fault_dir, phase, nf, t_start)
        raw[f"loki_vol_{nf}"] = load_loki_volume(fault_dir, phase, nf, t_start)
    for nf in RAN_NFS:                       # RAN-side logs (ueransim gNB / UEs)
        key = nf.replace("-", "_")
        raw[f"loki_{key}"]     = load_loki_errors(fault_dir, phase, nf, t_start)
        raw[f"loki_vol_{key}"] = load_loki_volume(fault_dir, phase, nf, t_start)
    for nf in TRACE_NFS:
        arr = load_jaeger_p99(fault_dir, phase, nf, t_start)
        if not np.all(np.isnan(arr)):
            raw[f"jaeger_{nf}"] = arr

    # Global RTT
    raw["rtt"] = load_rtt(fault_dir, phase, t_start)
    return raw


# ── Stat features ─────────────────────────────────────────────────────────────

def stat_features(zarr):
    """4 shape-stats from a z-scored during-window array (len N_BINS, may have NaN)."""
    valid = zarr[~np.isnan(zarr)]
    if len(valid) == 0:
        return dict(mean=0.0, std=0.0, maxabs=0.0, slope=0.0)
    mean = float(np.mean(valid))
    std  = float(np.std(valid))
    maxabs = float(np.max(np.abs(valid)))
    # slope: linear fit over bin index (only non-NaN points)
    idx = np.where(~np.isnan(zarr))[0]
    if len(idx) >= 2:
        slope = float(np.polyfit(idx, zarr[idx], 1)[0])
    else:
        slope = 0.0
    return dict(mean=mean, std=std, maxabs=maxabs, slope=slope)


def load_fault_features(fault_dir, cat_map, stats):
    tl_path = fault_dir / "timeline.json"
    if not tl_path.exists():
        return None
    fault_name = fault_dir.name
    category   = cat_map.get(fault_name)
    if category is None:
        return None

    tl = json.loads(tl_path.read_text())
    pre_start = tl["pre"]["start"]
    dur_start = tl["fault"]["start"]

    try:
        pre_raw = build_signals_ext(fault_dir, "pre", pre_start)
        dur_raw = build_signals_ext(fault_dir, "during", dur_start)
    except Exception:
        return None

    # z-score during vs pre baseline
    feats = {"fault_name": fault_name, "fault_category": category}
    for key, arr in dur_raw.items():
        if key not in pre_raw:
            continue
        pm = np.nanmean(pre_raw[key])
        ps = np.nanstd(pre_raw[key])
        if np.isnan(pm):
            continue
        zarr = zscore(arr, pm, ps if not np.isnan(ps) else 0.0)
        s = stat_features(zarr)
        for stat in stats:
            feats[f"{key}__{stat}"] = s[stat]

    # Add summary statistics delta (during - pre) for traces
    for nf in TRACE_NFS:
        pre_summary = load_jaeger_summary_stats(fault_dir, "pre", nf)
        dur_summary = load_jaeger_summary_stats(fault_dir, "during", nf)
        if pre_summary is not None and dur_summary is not None:
            for stat_name in ["span_count", "error_rate", "mean_latency", "p50_latency", "p95_latency"]:
                delta = dur_summary[stat_name] - pre_summary[stat_name]
                feats[f"jaeger_{stat_name}_{nf}"] = delta

    return feats


def load_dataset(data_roots, cat_map, stats):
    rows = []
    for root in data_roots:
        for fault_dir in sorted(root.iterdir()):
            if not fault_dir.is_dir():
                continue
            rec = load_fault_features(fault_dir, cat_map, stats)
            if rec is not None:
                rec["dataset"] = root.parent.name if root.name == "C-fault-detection" else root.name
                rows.append(rec)
    return pd.DataFrame(rows)


def build_matrix(df, cats, relabel, min_frac=0.20):
    df = df.copy()
    df["fault_category"] = df["fault_category"].replace(relabel)
    df = df[df["fault_category"].isin(cats)]
    feat_cols = [c for c in df.columns
                 if c not in ("fault_name", "fault_category", "dataset")]
    # sparsity filter: keep cols non-zero in >=min_frac of rows
    min_present = max(2, int(min_frac * len(df)))
    feat_cols = [c for c in feat_cols if (df[c].fillna(0) != 0).sum() >= min_present]
    X = df[feat_cols].fillna(0).values.astype(float)
    y = df["fault_category"].values
    ids = (df["fault_name"] + " [" + df["dataset"] + "]").values
    return X, y, ids, feat_cols


def evaluate(X, y, label, out_dir):
    # FULLY single-threaded — no joblib fork anywhere (avoids macOS deadlock).
    # 200 trees: accuracy is insensitive to tree count; 2.5x faster than 500.
    print(f"  [{label}] fitting {X.shape[0]} LOO folds x {X.shape[1]} feats ...", flush=True)
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                class_weight="balanced", random_state=42, n_jobs=1)
    y_pred = cross_val_predict(rf, X, y, cv=LeaveOneOut(), n_jobs=1)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="macro", zero_division=0)
    rep = classification_report(y, y_pred, target_names=sorted(set(y)), zero_division=0)
    print(f"\n=== {label} ===")
    print(f"LOOCV accuracy = {acc:.3f}   macro-F1 = {f1:.3f}   ({X.shape[1]} features)")
    print(rep)
    (out_dir / f"report_{label}.txt").write_text(
        f"{label}: acc={acc:.3f} f1={f1:.3f} features={X.shape[1]}\n\n{rep}")
    return acc, f1, X.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["operational", "security", "combined"],
                    default="operational")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_map = (OPERATIONAL_CATEGORIES if args.mode == "operational"
               else {**OPERATIONAL_CATEGORIES, **SECURITY_CATEGORIES})
    cats    = (CATS_OPERATIONAL if args.mode == "operational"
               else CATS_SECURITY if args.mode == "security"
               else CATS_OPERATIONAL + CATS_SECURITY)
    relabel = RELABEL if args.mode != "security" else {}

    data_roots = [Path(p) for p in args.data]

    # Load once with all 4 stats
    ALL_STATS = ["mean", "std", "maxabs", "slope"]
    print("Loading datasets with extended signals + stats ...", flush=True)
    df_parts = []
    for root in data_roots:
        import time as _t
        _s = _t.time()
        part = load_dataset([root], cat_map, ALL_STATS)
        print(f"  loaded {root.parent.name if root.name=='C-fault-detection' else root.name}: "
              f"{len(part)} faults in {_t.time()-_s:.1f}s", flush=True)
        df_parts.append(part)
    df = pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()
    print(f"Loaded {len(df)} fault instances from {len(data_roots)} dataset(s)", flush=True)
    if df.empty:
        sys.exit("No data.")

    results = []

    # 1) Baseline: existing 4 signals, mean only (≈ the 21-feature model)
    base_sigs = ("cpu_", "net_rx_", "loki_", "jaeger_")
    df_base = df[["fault_name", "fault_category", "dataset"]
                 + [c for c in df.columns
                    if c.endswith("__mean") and c.startswith(base_sigs)]].copy()
    Xb, yb, _, fb = build_matrix(df_base, cats, relabel)
    results.append(("baseline_mean_4sig", *evaluate(Xb, yb, "baseline_mean_4sig", out_dir)))

    # 2) Tier-2 signals added, mean only
    df_t2 = df[["fault_name", "fault_category", "dataset"]
               + [c for c in df.columns if c.endswith("__mean")]].copy()
    X2, y2, _, f2 = build_matrix(df_t2, cats, relabel)
    results.append(("tier2_mean_allsig", *evaluate(X2, y2, "tier2_mean_allsig", out_dir)))

    # 3) Tier-1+2: all signals, all 4 stats
    Xf, yf, idf, ff = build_matrix(df, cats, relabel)
    results.append(("tier1+2_allstats", *evaluate(Xf, yf, "tier1+2_allstats", out_dir)))

    # 4) Tier-1+2 but mean+std+max (drop slope, often noisy)
    df_ms = df[["fault_name", "fault_category", "dataset"]
               + [c for c in df.columns
                  if c.endswith(("__mean", "__std", "__maxabs"))]].copy()
    Xm, ym, _, fm = build_matrix(df_ms, cats, relabel)
    results.append(("tier1+2_no_slope", *evaluate(Xm, ym, "tier1+2_no_slope", out_dir)))

    # Save full feature matrix (all stats) for downstream SHAP if wanted
    feat_df = pd.DataFrame(Xf, columns=ff)
    feat_df.insert(0, "label", yf)
    feat_df.insert(0, "fault_id", idf)
    feat_df.to_csv(out_dir / "feature_matrix_extended.csv", index=False)

    # Summary table
    print("\n\n================ SUMMARY ================")
    print(f"{'variant':<22} {'acc':>6} {'macroF1':>8} {'#feat':>6}")
    for name, acc, f1, nf in results:
        print(f"{name:<22} {acc:>6.3f} {f1:>8.3f} {nf:>6}")
    summ = pd.DataFrame(results, columns=["variant", "accuracy", "macro_f1", "n_features"])
    summ.to_csv(out_dir / "variant_comparison.csv", index=False)

    # bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r[0] for r in results]
    accs  = [r[1] for r in results]
    bars = ax.barh(names[::-1], accs[::-1], color="#4C72B0", alpha=0.85)
    for b, a in zip(bars, accs[::-1]):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2, f"{a:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("LOOCV accuracy")
    ax.set_title("Feature-set ablation (extended signals + stats)", fontweight="bold")
    ax.set_xlim(0, max(accs) * 1.2)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "variant_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
