#!/usr/bin/env python3
"""
experiments/david/ccf_profile.py
─────────────────────────────────
Full cross-correlation function (CCF) profile analysis + coherence.

Unlike cross_correlation.py which keeps only the peak (max |r|, lag_at_max),
this script keeps the *entire* CCF(τ) curve — correlation as a function of
time-lag — and adds Welch coherence in the frequency domain.

Works with the C-fault-detection data format (same layout as cross_correlation.py):
  <data_root>/<fault_name>/<trial>/
      timeline.json
      prometheus/{pre,during,post}/*.csv
      jaeger/{pre,during,post}/spans_flat.csv
      loki/{pre,during,post}/{errors.csv,all.csv}
      rtt/{pre,during,post}/ue_rtt.csv

For each fault × trial × signal pair:
  1. Load the full 30-bin time series (10s bins, 300s window)
  2. Z-score against the pre-fault baseline (same as existing pipeline)
  3. Compute CCF(τ) for τ ∈ [-MAX_LAG, +MAX_LAG] bins for both PRE and DURING windows
  4. Extract features from the CCF curve:
        peak_corr_pre, peak_lag_pre          (existing Δ|r| approach captures this)
        peak_corr_during, peak_lag_during
        Δpeak_corr = |peak_corr_during| - |peak_corr_pre|     (= existing Δ|r|)
        Δpeak_lag  = peak_lag_during - peak_lag_pre           (NEW)
        ccf_auc_during                                        (NEW: area under |CCF|)
        ccf_spread_during                                     (NEW: std of CCF over lags)
  5. Compute Welch coherence (frequency domain):
        mean_coh_pre, mean_coh_during, Δmean_coh             (NEW)

Outputs:
  <out>/ccf_features.csv            feature matrix (one row per fault×trial×pair)
  <out>/ccf_curves_<pair>.png       CCF(τ) curves: pre vs during, per fault category
  <out>/coherence_<pair>.png        Coherence spectra: pre vs during, per category
  <out>/ccf_delta_lag_heatmap.png   Δpeak_lag heatmap across pairs × fault categories

Usage:
    python3 experiments/david/ccf_profile.py \\
        --data data/5GCore/final_dataset/boyan \\
        --out  data/5GCore/correlations/boyan/ccf \\
        --mode operational

    python3 experiments/david/ccf_profile.py \\
        --data /path/to/security-faults \\
        --out  data/5GCore/correlations/security/ccf \\
        --mode security
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal as scipy_signal
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

BIN_SEC = 10      # bin size in seconds
N_BINS  = 30      # 300s fault window / 10s
MAX_LAG = 8       # max lag in bins (= ±80 seconds)

CORE_NFS  = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf", "upf"]
TRACE_NFS = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf"]

# Key signal pairs to analyse — chosen because they capture cross-modal and
# cross-NF propagation paths most relevant to 5G core faults.
FOCUS_PAIRS = [
    ("cpu_amf",      "loki_amf",      "metrics↔logs",    "AMF cpu ↔ AMF errors"),
    ("net_rx_amf",   "loki_amf",      "metrics↔logs",    "AMF net_rx ↔ AMF errors"),
    ("cpu_nrf",      "jaeger_nrf",    "metrics↔traces",  "NRF cpu ↔ NRF trace latency"),
    ("cpu_amf",      "jaeger_amf",    "metrics↔traces",  "AMF cpu ↔ AMF trace latency"),
    ("net_rx_amf",   "jaeger_amf",    "metrics↔traces",  "AMF net_rx ↔ AMF trace latency"),
    ("jaeger_amf",   "jaeger_scp",    "traces↔traces",   "AMF latency ↔ SCP latency"),
    ("jaeger_nrf",   "jaeger_amf",    "traces↔traces",   "NRF latency ↔ AMF latency"),
    ("cpu_amf",      "cpu_nrf",       "metrics↔metrics", "AMF cpu ↔ NRF cpu"),
    ("net_rx_nrf",   "loki_scp",      "metrics↔logs",    "NRF net_rx ↔ SCP errors"),
]

OPERATIONAL_CATEGORIES = {
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
    "15-nrf-cascade":                          "cascade",
    "16-cpu-stress-ausf":                      "infrastructure",
    "17-network-delay-scp":                    "network",
    "18-cpu-stress-nrf":                       "infrastructure",
    "19-udm-pod-crash":                        "pod_crash",
    "20-mongodb-pod-kill":                     "pod_crash",
    "21-n2-partition-amf-gnb":                 "network",
    "22-memory-pressure-amf":                  "infrastructure",
}

SECURITY_CATEGORIES = {
    "nas-registration-storm":  "security_nas",
    "sbi-http2-flood-amf":    "security_flood",
    "sbi-http2-flood-nrf":    "security_flood",
    "sbi-http2-flood-scp":    "security_flood",
    "sbi-http2-flood-smf":    "security_flood",
    "authentication-exhaustion": "security_auth",
}

# ── Signal loaders (shared logic from cross_correlation.py) ──────────────────

def pod_to_nf(pod):
    m = re.match(r'open5gs-([a-z]+)-', str(pod))
    return m.group(1) if m else None


def make_grid(t_start, n=N_BINS, bin_sec=BIN_SEC):
    return np.array([t_start + i * bin_sec for i in range(n)])


def load_prom_nf(fault_dir, phase, fname, nf, interface=None):
    path = fault_dir / "prometheus" / phase / fname
    if not path.exists() or path.stat().st_size == 0:
        return None, None
    df = pd.read_csv(path)
    if "pod" not in df.columns:
        return None, None
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = df["pod"].apply(lambda p: pod_to_nf(p) == nf)
    sub = df[mask].dropna(subset=["value"])
    if interface is not None and "interface" in sub.columns:
        sub = sub[sub["interface"] == interface]
    if len(sub) == 0:
        return None, None
    grouped = sub.groupby("timestamp")["value"].mean()
    return grouped.index.values, grouped.values


def bin_prom(ts, vals, t_start, n=N_BINS, bin_sec=BIN_SEC):
    result = np.full(n, np.nan)
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        mask = (ts >= t0) & (ts < t1)
        if mask.any():
            result[i] = np.nanmean(vals[mask])
    return result


def load_jaeger_p99(fault_dir, phase, nf, t_start, n=N_BINS, bin_sec=BIN_SEC):
    path = fault_dir / "jaeger" / phase / "spans_flat.csv"
    if not path.exists() or path.stat().st_size == 0:
        return np.full(n, np.nan)
    df = pd.read_csv(path)
    df = df[df["service"] == nf].copy()
    if len(df) == 0:
        return np.full(n, np.nan)
    df["ts_sec"] = df["start_us"] / 1e6
    result = np.full(n, np.nan)
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        mask = (df["ts_sec"] >= t0) & (df["ts_sec"] < t1)
        spans = df.loc[mask, "duration_us"]
        if len(spans) >= 3:
            result[i] = np.percentile(spans, 99)
    s = pd.Series(result).interpolate(method="linear", limit_direction="both", limit=3)
    return s.values


def load_loki_errors(fault_dir, phase, nf, t_start, n=N_BINS, bin_sec=BIN_SEC):
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


def zscore(arr, mean, std):
    if std < 1e-10:
        return arr - mean
    return (arr - mean) / std


def build_signals(fault_dir, phase, t_start, pre_means=None, pre_stds=None):
    """Load and optionally z-score all signals for a phase window."""
    raw = {}
    for nf in CORE_NFS:
        ts, v = load_prom_nf(fault_dir, phase, "container_cpu_usage_rate.csv", nf)
        if ts is not None:
            raw[f"cpu_{nf}"] = bin_prom(ts, v, t_start)
        ts, v = load_prom_nf(fault_dir, phase, "network_rx_bytes_rate.csv", nf, interface="eth0")
        if ts is not None:
            raw[f"net_rx_{nf}"] = bin_prom(ts, v, t_start)
        raw[f"loki_{nf}"] = load_loki_errors(fault_dir, phase, nf, t_start)

    for nf in TRACE_NFS:
        arr = load_jaeger_p99(fault_dir, phase, nf, t_start)
        if not np.all(np.isnan(arr)):
            raw[f"jaeger_{nf}"] = arr

    if pre_means is None:
        means = {k: np.nanmean(v) for k, v in raw.items()}
        stds  = {k: np.nanstd(v)  for k, v in raw.items()}
        return raw, means, stds

    signals = {}
    for k, arr in raw.items():
        if k in pre_means:
            signals[k] = zscore(arr, pre_means[k], pre_stds[k])
    return signals


# ── Core CCF and coherence functions ─────────────────────────────────────────

def full_ccf(a, b, max_lag=MAX_LAG):
    """
    Return the full normalized cross-correlation curve.

    CCF(τ) = (1/N) * Σ_t  a(t) * b(t+τ)  / (σ_a * σ_b)

    Returns:
        lags   : int array of lag values in bins  (len = 2*max_lag+1)
        ccf    : float array of correlation values
        Returns (None, None) if insufficient data.

    Sign convention: positive τ means b is shifted forward — i.e., a *leads* b.
    Peak at τ=+k means signal A changes k bins *before* signal B responds.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 6:
        return None, None
    # Spearman cross-correlation: rank-transform the overlapping (non-NaN) samples,
    # then the Pearson normalization below is computed on ranks (= Spearman at each lag),
    # consistent with the rank-based Delta|rho| coupling metric.
    a_c = rankdata(a[mask]) - np.mean(rankdata(a[mask]))
    b_c = rankdata(b[mask]) - np.mean(rankdata(b[mask]))
    std_a, std_b = np.std(a_c), np.std(b_c)
    if std_a < 1e-10 or std_b < 1e-10:
        return None, None
    n = len(a_c)
    full = np.correlate(a_c, b_c, mode="full") / (n * std_a * std_b)
    all_lags = np.arange(-(n - 1), n)
    mask_lag = np.abs(all_lags) <= max_lag
    return all_lags[mask_lag], full[mask_lag]


def ccf_features(lags, ccf_vals):
    """Extract scalar features from a CCF curve."""
    if lags is None:
        return dict(peak_corr=np.nan, peak_lag=np.nan, auc=np.nan, spread=np.nan)
    abs_ccf = np.abs(ccf_vals)
    idx     = np.argmax(abs_ccf)
    return {
        "peak_corr": float(ccf_vals[idx]),        # signed peak
        "peak_lag":  int(lags[idx]),               # lag at peak (bins)
        "auc":       float(np.trapz(abs_ccf, lags) / (2 * MAX_LAG)),  # normalized AUC
        "spread":    float(np.std(abs_ccf)),       # how "peaked" the CCF is
    }


def compute_coherence(a, b, fs=1.0 / BIN_SEC):
    """
    Welch cross-spectral coherence between two signals.
    C(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))  ∈ [0, 1]

    Returns (freqs, coherence) or (None, None) if insufficient data.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 8:
        return None, None
    a_c = a[mask] - np.mean(a[mask])
    b_c = b[mask] - np.mean(b[mask])
    nperseg = min(len(a_c), 8)   # short window given only 30 bins
    try:
        f, Cxy = scipy_signal.coherence(a_c, b_c, fs=fs, nperseg=nperseg)
        return f, Cxy
    except Exception:
        return None, None


# ── Per-trial analysis ────────────────────────────────────────────────────────

def analyze_trial(fault_dir):
    """
    Load one fault trial directory and compute CCF + coherence for all focus pairs.
    Returns a list of feature dicts.
    """
    tl_path = fault_dir / "timeline.json"
    if not tl_path.exists():
        return []
    tl          = json.loads(tl_path.read_text())
    pre_start   = tl["pre"]["start"]
    fault_start = tl["fault"]["start"]

    pre_raw, pre_means, pre_stds = build_signals(fault_dir, "pre", pre_start)
    pre_signals    = {k: zscore(v, pre_means[k], pre_stds[k]) for k, v in pre_raw.items()}
    during_signals = build_signals(fault_dir, "during", fault_start, pre_means, pre_stds)

    rows = []
    for sig_a, sig_b, modality, label in FOCUS_PAIRS:
        # ── PRE window
        pre_lags, pre_ccf = (None, None)
        pre_coh_f, pre_coh = (None, None)
        if sig_a in pre_signals and sig_b in pre_signals:
            pre_lags, pre_ccf = full_ccf(pre_signals[sig_a], pre_signals[sig_b])
            pre_coh_f, pre_coh = compute_coherence(pre_signals[sig_a], pre_signals[sig_b])

        # ── DURING (fault) window
        dur_lags, dur_ccf = (None, None)
        dur_coh_f, dur_coh = (None, None)
        if sig_a in during_signals and sig_b in during_signals:
            dur_lags, dur_ccf = full_ccf(during_signals[sig_a], during_signals[sig_b])
            dur_coh_f, dur_coh = compute_coherence(during_signals[sig_a], during_signals[sig_b])

        pre_feat  = ccf_features(pre_lags,  pre_ccf)
        dur_feat  = ccf_features(dur_lags,  dur_ccf)

        mean_coh_pre  = float(np.nanmean(pre_coh))  if pre_coh  is not None else np.nan
        mean_coh_dur  = float(np.nanmean(dur_coh))  if dur_coh  is not None else np.nan

        rows.append({
            "sig_a":         sig_a,
            "sig_b":         sig_b,
            "modality":      modality,
            "pair_label":    label,
            # PRE features
            "pre_peak_corr": pre_feat["peak_corr"],
            "pre_peak_lag":  pre_feat["peak_lag"],
            "pre_auc":       pre_feat["auc"],
            "pre_spread":    pre_feat["spread"],
            "pre_mean_coh":  mean_coh_pre,
            # DURING features
            "dur_peak_corr": dur_feat["peak_corr"],
            "dur_peak_lag":  dur_feat["peak_lag"],
            "dur_auc":       dur_feat["auc"],
            "dur_spread":    dur_feat["spread"],
            "dur_mean_coh":  mean_coh_dur,
            # DELTA features (the interesting ones)
            "delta_peak_corr": (abs(dur_feat["peak_corr"]) - abs(pre_feat["peak_corr"]))
                                if not (np.isnan(dur_feat["peak_corr"]) or np.isnan(pre_feat["peak_corr"]))
                                else np.nan,
            "delta_peak_lag":  (dur_feat["peak_lag"] - pre_feat["peak_lag"])
                                if not (np.isnan(dur_feat["peak_lag"]) or np.isnan(pre_feat["peak_lag"]))
                                else np.nan,
            "delta_auc":       (dur_feat["auc"] - pre_feat["auc"])
                                if not (np.isnan(dur_feat["auc"]) or np.isnan(pre_feat["auc"]))
                                else np.nan,
            "delta_mean_coh":  (mean_coh_dur - mean_coh_pre)
                                if not (np.isnan(mean_coh_dur) or np.isnan(mean_coh_pre))
                                else np.nan,
            # Store the raw CCF arrays for plotting (named _pre_* / _during_*)
            "_pre_lags":     pre_lags.tolist()  if pre_lags  is not None else None,
            "_pre_ccf":      pre_ccf.tolist()   if pre_ccf   is not None else None,
            "_during_lags":  dur_lags.tolist()  if dur_lags  is not None else None,
            "_during_ccf":   dur_ccf.tolist()   if dur_ccf   is not None else None,
            "_pre_coh_f":    pre_coh_f.tolist() if pre_coh_f is not None else None,
            "_pre_coh":      pre_coh.tolist()   if pre_coh   is not None else None,
            "_during_coh_f": dur_coh_f.tolist() if dur_coh_f is not None else None,
            "_during_coh":   dur_coh.tolist()   if dur_coh   is not None else None,
        })
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────────

CAT_COLORS = {
    "infrastructure": "#4C72B0",
    "pod_crash":      "#DD8452",
    "network":        "#55A868",
    "pfcp":           "#C44E52",
    "cascade":        "#8172B2",
    "security_nas":   "#937860",
    "security_flood": "#DA8BC3",
    "security_auth":  "#CCB974",
}


def plot_ccf_curves(df, out_dir, categories):
    """
    For each focus pair, plot the full CCF(τ) curve averaged per fault category.
    One figure per pair: 2 rows (PRE / DURING), one line per category.
    """
    pairs = [(r["sig_a"], r["sig_b"], r["pair_label"]) for _, r in
             df.drop_duplicates(["sig_a", "sig_b"]).iterrows()]

    for sig_a, sig_b, label in pairs:
        sub = df[(df["sig_a"] == sig_a) & (df["sig_b"] == sig_b)]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        fig.suptitle(f"Cross-Correlation Profile: {label}", fontsize=12, fontweight="bold")

        for window, ax, title in [("pre", axes[0], "PRE window (baseline)"),
                                   ("during", axes[1], "DURING fault")]:
            lags_col = f"_{window}_lags"
            ccf_col  = f"_{window}_ccf"
            ax.set_title(title, fontsize=10)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
            ax.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.set_xlabel("Lag (bins, 1 bin = 10s)", fontsize=9)
            ax.set_ylabel("CCF(τ)", fontsize=9)

            for cat in categories:
                cat_rows = sub[sub["fault_category"] == cat]
                ccf_list = [r for r in cat_rows[ccf_col] if r is not None]
                lag_list = [r for r in cat_rows[lags_col] if r is not None]
                if not ccf_list:
                    continue
                # Average across instances with same lag grid
                max_len = min(len(c) for c in ccf_list)
                mat = np.array([c[:max_len] for c in ccf_list])
                lags = np.array(lag_list[0][:max_len])
                mean_ccf = np.nanmean(mat, axis=0)
                std_ccf  = np.nanstd(mat, axis=0)
                color = CAT_COLORS.get(cat, "#333333")
                ax.plot(lags * BIN_SEC, mean_ccf, label=cat, color=color, linewidth=2)
                ax.fill_between(lags * BIN_SEC,
                                mean_ccf - std_ccf, mean_ccf + std_ccf,
                                alpha=0.15, color=color)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlim(-MAX_LAG * BIN_SEC, MAX_LAG * BIN_SEC)
            ax.set_xticks(np.arange(-MAX_LAG, MAX_LAG + 1, 2) * BIN_SEC)
            ax.set_xticklabels([f"{int(x)}s" for x in np.arange(-MAX_LAG, MAX_LAG + 1, 2) * BIN_SEC],
                                fontsize=8)

        # Annotate interpretation
        axes[1].annotate("← B leads A      A leads B →",
                         xy=(0.5, 0.02), xycoords="axes fraction",
                         ha="center", fontsize=7, color="grey")

        fig.tight_layout()
        slug = f"{sig_a}_vs_{sig_b}".replace("/", "_")
        out  = out_dir / f"ccf_curve_{slug}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved: {out.name}")


def plot_coherence(df, out_dir, categories):
    """
    For each focus pair, plot Welch coherence spectrum averaged per fault category.
    PRE vs DURING overlaid.
    """
    pairs = [(r["sig_a"], r["sig_b"], r["pair_label"]) for _, r in
             df.drop_duplicates(["sig_a", "sig_b"]).iterrows()]

    for sig_a, sig_b, label in pairs:
        sub = df[(df["sig_a"] == sig_a) & (df["sig_b"] == sig_b)]
        if sub.empty:
            continue

        has_data = any(r is not None for r in sub["_pre_coh"])
        if not has_data:
            continue

        fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4), sharey=True)
        if len(categories) == 1:
            axes = [axes]
        fig.suptitle(f"Coherence Spectrum: {label}", fontsize=12, fontweight="bold")

        for ax, cat in zip(axes, categories):
            cat_rows = sub[sub["fault_category"] == cat]
            color = CAT_COLORS.get(cat, "#333333")
            ax.set_title(cat, fontsize=9)
            ax.set_xlabel("Frequency (Hz)", fontsize=8)
            ax.set_ylabel("Coherence C(f)", fontsize=8)
            ax.set_ylim(0, 1)

            for window, linestyle, alpha, wlabel in [
                ("pre",    "--", 0.6, "PRE"),
                ("during", "-",  1.0, "DURING"),
            ]:
                f_col   = f"_{window}_coh_f"    # _pre_coh_f or _during_coh_f
                coh_col = f"_{window}_coh"       # _pre_coh   or _during_coh
                coh_list = [r for r in cat_rows[coh_col] if r is not None]
                f_list   = [r for r in cat_rows[f_col]   if r is not None]
                if not coh_list:
                    continue
                max_len = min(len(c) for c in coh_list)
                mat = np.array([c[:max_len] for c in coh_list])
                freqs = np.array(f_list[0][:max_len])
                mean_coh = np.nanmean(mat, axis=0)
                ax.plot(freqs, mean_coh, color=color, linestyle=linestyle,
                        linewidth=1.8, label=wlabel, alpha=alpha)
            ax.axhline(0.5, color="red", linewidth=0.7, linestyle=":", alpha=0.5,
                       label="C=0.5")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        slug = f"{sig_a}_vs_{sig_b}".replace("/", "_")
        out  = out_dir / f"coherence_{slug}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved: {out.name}")


def plot_delta_lag_heatmap(feat_df, out_dir, categories):
    """
    Heatmap: Δpeak_lag (during − pre, in seconds) across pairs × fault categories.
    Positive = the lag at peak correlation INCREASES during the fault (slower response).
    """
    pairs  = feat_df["pair_label"].unique()
    matrix = np.full((len(pairs), len(categories)), np.nan)

    for i, pair in enumerate(pairs):
        for j, cat in enumerate(categories):
            sub = feat_df[(feat_df["pair_label"] == pair) & (feat_df["fault_category"] == cat)]
            vals = sub["delta_peak_lag"].dropna()
            if len(vals) > 0:
                matrix[i, j] = vals.mean() * BIN_SEC   # convert bins → seconds

    if np.all(np.isnan(matrix)):
        print("  [skip] delta_lag heatmap — no valid data")
        return

    vmax = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.8), max(4, len(pairs) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    for i in range(len(pairs)):
        for j in range(len(categories)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.0f}s", ha="center", va="center",
                        fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, label="Δ peak lag (s)  during − pre\n+ = peak shifts to longer lag during fault")
    ax.set_title("Propagation Lag Shift During Fault  (Δ peak lag)", fontsize=12, pad=10)
    fig.tight_layout()
    out = out_dir / "ccf_delta_lag_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")


def plot_delta_auc_heatmap(feat_df, out_dir, categories):
    """
    Heatmap: Δ AUC (during − pre) — how much the area under |CCF| grows during fault.
    Captures whether the entire CCF shifts up (not just peak).
    """
    pairs  = feat_df["pair_label"].unique()
    matrix = np.full((len(pairs), len(categories)), np.nan)

    for i, pair in enumerate(pairs):
        for j, cat in enumerate(categories):
            sub = feat_df[(feat_df["pair_label"] == pair) & (feat_df["fault_category"] == cat)]
            vals = sub["delta_auc"].dropna()
            if len(vals) > 0:
                matrix[i, j] = vals.mean()

    if np.all(np.isnan(matrix)):
        print("  [skip] delta_auc heatmap — no valid data")
        return

    vmax = max(np.nanmax(np.abs(matrix)), 0.01)
    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.8), max(4, len(pairs) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    for i in range(len(pairs)):
        for j in range(len(categories)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center",
                        fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, label="Δ AUC of |CCF|  (during − pre)\n+ = coupling strengthens across all lags")
    ax.set_title("CCF Area-Under-Curve Increase During Fault  (Δ AUC)", fontsize=12, pad=10)
    fig.tight_layout()
    out = out_dir / "ccf_delta_auc_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")


def plot_coherence_delta_heatmap(feat_df, out_dir, categories):
    """Heatmap: Δmean_coherence across pairs × categories."""
    pairs  = feat_df["pair_label"].unique()
    matrix = np.full((len(pairs), len(categories)), np.nan)

    for i, pair in enumerate(pairs):
        for j, cat in enumerate(categories):
            sub = feat_df[(feat_df["pair_label"] == pair) & (feat_df["fault_category"] == cat)]
            vals = sub["delta_mean_coh"].dropna()
            if len(vals) > 0:
                matrix[i, j] = vals.mean()

    if np.all(np.isnan(matrix)):
        print("  [skip] coherence delta heatmap — no valid data")
        return

    vmax = max(np.nanmax(np.abs(matrix)), 0.01)
    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.8), max(4, len(pairs) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    for i in range(len(pairs)):
        for j in range(len(categories)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center",
                        fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, label="Δ mean coherence  (during − pre)\n+ = more frequency-domain coupling during fault")
    ax.set_title("Coherence Increase During Fault  (Δ mean coherence)", fontsize=12, pad=10)
    fig.tight_layout()
    out = out_dir / "ccf_delta_coherence_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True,
                        help="Root of fault data (contains fault-name/ sub-dirs)")
    parser.add_argument("--out",   required=True,
                        help="Output directory for plots and CSV")
    parser.add_argument("--mode",  choices=["operational", "security"], default="operational",
                        help="Dataset type — controls fault category mapping")
    parser.add_argument("--trials", type=int, default=None,
                        help="Only analyse first N trials per fault (default: all)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_map = OPERATIONAL_CATEGORIES if args.mode == "operational" else SECURITY_CATEGORIES

    all_rows = []

    for fault_dir in sorted(data_dir.iterdir()):
        if not fault_dir.is_dir():
            continue

        # Check if this is a flat fault dir (contains timeline.json directly)
        # or a fault dir containing trial sub-dirs (1/, 2/, 3/)
        has_timeline = (fault_dir / "timeline.json").exists()
        trial_dirs = sorted([d for d in fault_dir.iterdir()
                              if d.is_dir() and d.name.isdigit()],
                             key=lambda p: int(p.name))

        if has_timeline:
            # Single-trial flat layout (original C-fault format: boyan/trial4)
            fault_name = fault_dir.name
            if fault_name not in cat_map:
                continue
            print(f"[ccf] {fault_name} (single trial)...", end=" ", flush=True)
            rows = analyze_trial(fault_dir)
            for r in rows:
                r["fault_name"]     = fault_name
                r["fault_category"] = cat_map[fault_name]
                r["trial"]          = 1
            all_rows.extend(rows)
            print(f"{len(rows)} pairs")

        elif trial_dirs:
            # Multi-trial layout (security faults: fault-name/1/, /2/, /3/)
            fault_name = fault_dir.name
            if fault_name not in cat_map:
                continue
            if args.trials:
                trial_dirs = trial_dirs[:args.trials]
            print(f"[ccf] {fault_name} ({len(trial_dirs)} trials)...", end=" ", flush=True)
            total = 0
            for td in trial_dirs:
                rows = analyze_trial(td)
                for r in rows:
                    r["fault_name"]     = fault_name
                    r["fault_category"] = cat_map[fault_name]
                    r["trial"]          = int(td.name)
                all_rows.extend(rows)
                total += len(rows)
            print(f"{total} pairs")

    if not all_rows:
        print("[warn] No data found — check --data path and --mode flag.")
        return

    df = pd.DataFrame(all_rows)

    # ── Save feature CSV (drop raw CCF arrays — keep only scalars)
    feat_cols = [c for c in df.columns if not c.startswith("_")]
    feat_df   = df[feat_cols].copy()
    feat_df.to_csv(out_dir / "ccf_features.csv", index=False)
    print(f"\n[done] {len(feat_df)} rows → ccf_features.csv")

    categories = sorted(df["fault_category"].unique())
    print(f"[info] Fault categories: {categories}")

    # ── Plots
    print("\n[plots — CCF curves]")
    plot_ccf_curves(df, out_dir, categories)

    print("\n[plots — Coherence spectra]")
    plot_coherence(df, out_dir, categories)

    print("\n[plots — Delta heatmaps]")
    plot_delta_lag_heatmap(feat_df, out_dir, categories)
    plot_delta_auc_heatmap(feat_df, out_dir, categories)
    plot_coherence_delta_heatmap(feat_df, out_dir, categories)

    # ── Print summary table
    print("\n=== CCF Feature Summary (mean across trials) ===")
    print(f"{'Pair':<35} {'Cat':<20} {'Δpeak_corr':>12} {'Δpeak_lag(s)':>13} {'ΔAUC':>8} {'ΔCoh':>8}")
    print("-" * 100)
    for pair in feat_df["pair_label"].unique():
        for cat in categories:
            sub = feat_df[(feat_df["pair_label"] == pair) & (feat_df["fault_category"] == cat)]
            if sub.empty:
                continue
            dc  = sub["delta_peak_corr"].mean()
            dl  = sub["delta_peak_lag"].mean() * BIN_SEC
            da  = sub["delta_auc"].mean()
            dco = sub["delta_mean_coh"].mean()
            if any(not np.isnan(v) for v in [dc, dl, da, dco]):
                print(f"  {pair:<33} {cat:<20} {dc:>+12.3f} {dl:>+13.0f}s {da:>+8.3f} {dco:>+8.3f}")


if __name__ == "__main__":
    main()
