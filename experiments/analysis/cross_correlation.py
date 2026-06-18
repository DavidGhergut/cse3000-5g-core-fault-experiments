#!/usr/bin/env python3
"""
cross_correlation.py

Compute pairwise cross-correlation between telemetry signals across modalities
for each fault. Reports correlation strength and propagation lag.

For each fault:
  1. Load raw time series for metrics, traces, logs, RTT
  2. Resample to a common 10s bin grid (30 bins over 300s fault window)
  3. Normalize against pre-baseline (z-score)
  4. Compute cross-correlation for cross-modal signal pairs
  5. Report max correlation and lag at max

Output:
  <out>/correlations.csv        one row per (fault, signal_pair)
  <out>/correlation_heatmap.png average correlation per fault category
  <out>/lag_plot.png            average lag per fault category

Usage:
    python3 cross_correlation.py \\
        --data final_dataset/boyan \\
        --out  correlations_o5g/boyan
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
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

BIN_SEC   = 10    # bin size in seconds
N_BINS    = 30    # 300s / 10s
MAX_LAG   = 8     # max lag to consider in bins (= 80s)

CORE_NFS  = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf", "mongodb", "upf"]
TRACE_NFS = ["amf", "smf", "nrf", "scp", "ausf", "udm", "udr", "pcf"]
RAN_NFS   = ["ueransim-gnb", "ueransim-ues"]  # RAN-side log sources

# Open5GS application-layer KPI counters (functional 5G-procedure metrics).
# Each file is one counter for one NF; home NF used for cross-modal pairing.
O5G_COUNTERS = (
    [f"open5gs_amf_{x}.csv" for x in
     ["reg_init_req", "reg_init_succ", "reg_init_fail", "auth_fail", "auth_reject",
      "sessions", "registered_subscribers", "ran_ue_count", "gnb_count", "paging_req"]]
    + [f"open5gs_smf_{x}.csv" for x in
       ["pdu_session_req", "pdu_session_succ", "session_nbr", "n4_session_estab",
        "n4_session_report", "n4_session_report_succ", "bearers_active", "qos_flow_nbr", "ues_active"]]
    + [f"open5gs_upf_{x}.csv" for x in ["n4_session_estab", "qos_flows", "session_nbr"]]
    + ["open5gs_pfcp_sessions_active.csv", "open5gs_pfcp_peers_active.csv", "open5gs_gtp_node_failed.csv"]
)
O5G_HOME = {"amf": "amf", "smf": "smf", "upf": "upf", "pfcp": "smf", "gtp": "upf"}

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
    "15-nrf-cascade":                          "cascade",
    "16-cpu-stress-ausf":                      "infrastructure",
    "17-network-delay-scp":                    "network",
    "18-cpu-stress-nrf":                       "infrastructure",
    "19-udm-pod-crash":                        "pod_crash",
    "20-mongodb-pod-kill":                     "pod_crash",
    "21-n2-partition-amf-gnb":                 "network",
    "22-memory-pressure-amf":                  "infrastructure",
}

# ── Time series loaders ───────────────────────────────────────────────────────

import re

def pod_to_nf(pod):
    m = re.match(r'open5gs-([a-z]+)-', str(pod))
    return m.group(1) if m else None


def make_bins(t_start, n=N_BINS, bin_sec=BIN_SEC):
    """Return bin edges and bin centers as Unix timestamps."""
    edges   = np.array([t_start + i * bin_sec for i in range(n + 1)])
    centers = edges[:-1] + bin_sec / 2
    return edges, centers


def normalize(series, pre_mean, pre_std):
    """Z-score normalize using pre-baseline statistics."""
    if pre_std < 1e-10:
        return series - pre_mean
    return (series - pre_mean) / pre_std


def load_prom_nf(fault_dir, phase, fname, nf, interface=None):
    """Load a Prometheus metric for a specific NF, return (timestamps, values)."""
    path = fault_dir / "prometheus" / phase / fname
    if not path.exists() or path.stat().st_size == 0:
        return None, None
    df = pd.read_csv(path)
    if "pod" not in df.columns:
        return None, None
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = df["pod"].apply(lambda p: pod_to_nf(p) == nf)
    sub  = df[mask].dropna(subset=["value"])
    if interface is not None and "interface" in sub.columns:
        sub = sub[sub["interface"] == interface]
    if len(sub) == 0:
        return None, None
    # Average across containers/instances per timestamp
    grouped = sub.groupby("timestamp")["value"].mean()
    return grouped.index.values, grouped.values


def bin_prom(ts, vals, t_start, n=N_BINS, bin_sec=BIN_SEC):
    """Bin Prometheus time series (already 5s) into bin_sec bins by averaging."""
    result = np.full(n, np.nan)
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        mask = (ts >= t0) & (ts < t1)
        if mask.any():
            result[i] = np.nanmean(vals[mask])
    return result


def load_o5g_counter(fault_dir, phase, fname, t_start):
    """Load one Open5GS application-KPI counter file (timestamp,value) and bin it.
    Each file is a single counter for one NF; aggregate any duplicate series by mean."""
    path = fault_dir / "prometheus" / phase / fname
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if len(df) == 0:
        return None
    g = df.groupby("timestamp")["value"].mean()
    return bin_prom(g.index.values, g.values, t_start)


def load_jaeger_p99(fault_dir, phase, nf, t_start, n=N_BINS, bin_sec=BIN_SEC):
    """Bin Jaeger span durations into bins and compute p99 per bin."""
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
    # Forward-fill sparse bins
    df_r = pd.Series(result)
    df_r = df_r.interpolate(method="linear", limit_direction="both", limit=3)
    return df_r.values


def load_jaeger_summary(fault_dir, phase, nf, t_start, n=N_BINS, bin_sec=BIN_SEC):
    """
    Load pre-computed trace statistics from summary.json and spread uniformly
    across bins (phase-level aggregates, not per-bin).
    Returns dict: {span_count, error_rate, mean_latency, p50_latency, p95_latency}
    Each as a constant array of length n.
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
            "span_count":   np.full(n, nf_stats.get("span_count", 0)),
            "error_rate":   np.full(n, nf_stats.get("error_rate", 0.0)),
            "mean_latency": np.full(n, nf_stats.get("duration_us_mean", 0.0)),
            "p50_latency":  np.full(n, nf_stats.get("duration_us_p50", 0.0)),
            "p95_latency":  np.full(n, nf_stats.get("duration_us_p95", 0.0)),
        }
    except:
        return None


def load_loki_errors(fault_dir, phase, nf, t_start, n=N_BINS, bin_sec=BIN_SEC):
    """Count error log lines per bin for a given NF."""
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


def load_loki_volume(fault_dir, phase, nf, t_start, n=N_BINS, bin_sec=BIN_SEC):
    """Count all log lines per bin for a given NF."""
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


def load_rtt(fault_dir, phase, t_start, n=N_BINS, bin_sec=BIN_SEC):
    """Bin RTT measurements into bins by averaging."""
    path = fault_dir / "rtt" / phase / "ue_rtt.csv"
    result = np.full(n, np.nan)
    if not path.exists() or path.stat().st_size == 0:
        return result
    df = pd.read_csv(path)
    if "rtt_ms" not in df.columns or len(df) == 0:
        return result
    df["ts_sec"] = df["timestamp_ms"] / 1e3
    df["rtt_ms"] = pd.to_numeric(df["rtt_ms"], errors="coerce")
    for i in range(n):
        t0 = t_start + i * bin_sec
        t1 = t0 + bin_sec
        mask = (df["ts_sec"] >= t0) & (df["ts_sec"] < t1)
        vals = df.loc[mask, "rtt_ms"].dropna()
        if len(vals) > 0:
            result[i] = vals.mean()
    return result


# ── Cross-correlation ─────────────────────────────────────────────────────────

def xcorr(a, b, max_lag=MAX_LAG):
    """
    Compute normalized cross-correlation between two signals.
    Returns (max_corr, lag_bins) where lag_bins > 0 means b lags behind a.
    """
    # Drop NaN positions present in either signal
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 6:
        return np.nan, np.nan
    # Spearman cross-correlation: rank-transform before correlating, so the peak
    # lag is rank-based, consistent with the lag-zero spearman_r coupling metric.
    a_clean = rankdata(a[mask]) - np.mean(rankdata(a[mask]))
    b_clean = rankdata(b[mask]) - np.mean(rankdata(b[mask]))
    std_a = np.std(a_clean)
    std_b = np.std(b_clean)
    if std_a < 1e-10 or std_b < 1e-10:
        return np.nan, np.nan  # flat signal, correlation undefined
    n = len(a_clean)
    corr = np.correlate(a_clean, b_clean, mode="full") / (n * std_a * std_b)
    lags = np.arange(-(n - 1), n)
    # Restrict to max_lag
    mask_lag = np.abs(lags) <= max_lag
    corr_restricted = corr[mask_lag]
    lags_restricted = lags[mask_lag]
    idx = np.argmax(np.abs(corr_restricted))
    return float(corr_restricted[idx]), int(lags_restricted[idx])


def spearman_corr(a, b):
    """
    Spearman rank correlation at lag=0.
    Used as the Δ|r| feature for the RF classifier (robust to non-linear
    relationships and outlier bins).  The rank cross-correlation in xcorr()
    provides the (rank-based) lag profile for the SQ3 temporal analysis.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 6:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


# ── Per-fault analysis ────────────────────────────────────────────────────────

def build_signals(fault_dir, phase, phase_start, pre_means, pre_stds):
    """
    Load and z-score all signals for a given phase window.
    pre_means / pre_stds are dicts keyed by signal name, computed from the
    pre-fault window so all phases share the same normalization baseline.
    Pass pre_means=None to compute and return the baseline stats instead.
    """
    raw = {}

    for nf in CORE_NFS:
        ts, v = load_prom_nf(fault_dir, phase, "container_cpu_usage_rate.csv", nf)
        if ts is not None:
            raw[f"cpu_{nf}"] = bin_prom(ts, v, phase_start)
        ts, v = load_prom_nf(fault_dir, phase, "container_memory_working_set_bytes.csv", nf)
        if ts is not None:
            raw[f"mem_{nf}"] = bin_prom(ts, v, phase_start)
        ts, v = load_prom_nf(fault_dir, phase, "network_rx_bytes_rate.csv", nf, interface="eth0")
        if ts is not None:
            raw[f"net_rx_{nf}"] = bin_prom(ts, v, phase_start)
        ts, v = load_prom_nf(fault_dir, phase, "network_tx_bytes_rate.csv", nf, interface="eth0")
        if ts is not None:
            raw[f"net_tx_{nf}"] = bin_prom(ts, v, phase_start)
        ts, v = load_prom_nf(fault_dir, phase, "container_cpu_throttled_rate.csv", nf)
        if ts is not None:
            raw[f"cputhr_{nf}"] = bin_prom(ts, v, phase_start)

    # Open5GS application-layer KPI counters (functional metric sub-layer)
    for fname in O5G_COUNTERS:
        arr = load_o5g_counter(fault_dir, phase, fname, phase_start)
        if arr is not None:
            raw["o5g_" + fname.replace("open5gs_", "").replace(".csv", "")] = arr

    for nf in TRACE_NFS:
        # p99 latency (existing per-bin feature)
        arr = load_jaeger_p99(fault_dir, phase, nf, phase_start)
        if not np.all(np.isnan(arr)):
            raw[f"jaeger_{nf}"] = arr

        # Additional trace statistics from summary.json
        summary_stats = load_jaeger_summary(fault_dir, phase, nf, phase_start)
        if summary_stats is not None:
            raw[f"jaeger_count_{nf}"]  = summary_stats["span_count"]
            raw[f"jaeger_err_{nf}"]    = summary_stats["error_rate"]
            raw[f"jaeger_mean_{nf}"]   = summary_stats["mean_latency"]
            raw[f"jaeger_p50_{nf}"]    = summary_stats["p50_latency"]
            raw[f"jaeger_p95_{nf}"]    = summary_stats["p95_latency"]

    for nf in CORE_NFS:
        raw[f"loki_{nf}"]     = load_loki_errors(fault_dir, phase, nf, phase_start)
        raw[f"loki_vol_{nf}"] = load_loki_volume(fault_dir, phase, nf, phase_start)

    for nf in RAN_NFS:
        key = nf.replace("-", "_")
        raw[f"loki_{key}"]     = load_loki_errors(fault_dir, phase, nf, phase_start)
        raw[f"loki_vol_{key}"] = load_loki_volume(fault_dir, phase, nf, phase_start)

    # RTT (global end-to-end UE round-trip time) as a metric-modality signal,
    # paired cross-modally in build_pairs() for parity with the raw model.
    rtt_arr = load_rtt(fault_dir, phase, phase_start)
    if rtt_arr is not None and not np.all(np.isnan(rtt_arr)):
        raw["rtt"] = rtt_arr

    if pre_means is None:
        # Compute baseline stats from this (pre) window and return raw arrays too
        means = {k: np.nanmean(v) for k, v in raw.items()}
        stds  = {k: np.nanstd(v)  for k, v in raw.items()}
        return raw, means, stds

    # Normalize against pre-fault baseline
    signals = {}
    for k, arr in raw.items():
        if k in pre_means:
            signals[k] = normalize(arr, pre_means[k], pre_stds[k])
    return signals


def build_pairs(signals):
    pairs = []

    for nf in CORE_NFS:
        # ── cross-modal (same NF) ──────────────────────────────────────────────
        # Existing p99 latency pairs
        if f"cpu_{nf}" in signals and f"jaeger_{nf}" in signals:
            pairs.append((f"cpu_{nf}", f"jaeger_{nf}", "metrics↔traces"))
        if f"mem_{nf}" in signals and f"jaeger_{nf}" in signals:
            pairs.append((f"mem_{nf}", f"jaeger_{nf}", "metrics↔traces"))

        # New summary statistic pairs (span_count, error_rate, mean/p50/p95 latency)
        for trace_stat in ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]:
            trace_sig = f"{trace_stat}_{nf}"
            if trace_sig not in signals:
                continue
            if f"cpu_{nf}" in signals:
                pairs.append((f"cpu_{nf}", trace_sig, "metrics↔traces"))
            if f"mem_{nf}" in signals:
                pairs.append((f"mem_{nf}", trace_sig, "metrics↔traces"))
            for net_sig in (f"net_rx_{nf}", f"net_tx_{nf}"):
                if net_sig in signals:
                    pairs.append((net_sig, trace_sig, "metrics↔traces"))

        # Logs cross-modal
        for log_sig in (f"loki_{nf}", f"loki_vol_{nf}"):
            if f"cpu_{nf}" in signals and log_sig in signals:
                pairs.append((f"cpu_{nf}", log_sig, "metrics↔logs"))
            if f"mem_{nf}" in signals and log_sig in signals:
                pairs.append((f"mem_{nf}", log_sig, "metrics↔logs"))
            # p99 latency vs logs
            if f"jaeger_{nf}" in signals and log_sig in signals:
                pairs.append((f"jaeger_{nf}", log_sig, "traces↔logs"))
            # New trace stats vs logs
            for trace_stat in ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]:
                trace_sig = f"{trace_stat}_{nf}"
                if trace_sig in signals and log_sig in signals:
                    pairs.append((trace_sig, log_sig, "traces↔logs"))

        # ── network I/O cross-modal (same NF) ────────────────────────────────
        for net_sig in (f"net_rx_{nf}", f"net_tx_{nf}"):
            if net_sig not in signals:
                continue
            if f"jaeger_{nf}" in signals:
                pairs.append((net_sig, f"jaeger_{nf}", "metrics↔traces"))
            for log_sig in (f"loki_{nf}", f"loki_vol_{nf}"):
                if log_sig in signals:
                    pairs.append((net_sig, log_sig, "metrics↔logs"))

        # ── cpu_throttled cross-modal (same NF) ───────────────────────────────
        if f"cputhr_{nf}" in signals:
            if f"jaeger_{nf}" in signals:
                pairs.append((f"cputhr_{nf}", f"jaeger_{nf}", "metrics↔traces"))
            for log_sig in (f"loki_{nf}", f"loki_vol_{nf}"):
                if log_sig in signals:
                    pairs.append((f"cputhr_{nf}", log_sig, "metrics↔logs"))

        # ── single-modal (same NF, same modality) ─────────────────────────────
        if f"cpu_{nf}" in signals and f"mem_{nf}" in signals:
            pairs.append((f"cpu_{nf}", f"mem_{nf}", "metrics↔metrics"))
        if f"net_rx_{nf}" in signals and f"cpu_{nf}" in signals:
            pairs.append((f"net_rx_{nf}", f"cpu_{nf}", "metrics↔metrics"))
        if f"net_rx_{nf}" in signals and f"mem_{nf}" in signals:
            pairs.append((f"net_rx_{nf}", f"mem_{nf}", "metrics↔metrics"))
        if f"loki_{nf}" in signals and f"loki_vol_{nf}" in signals:
            pairs.append((f"loki_{nf}", f"loki_vol_{nf}", "logs↔logs"))

    # ── Open5GS application-metric cross-modal pairs (same home NF) ────────────
    for sig in [s for s in signals if s.startswith("o5g_")]:
        nf = O5G_HOME.get(sig.split("_")[1])
        if nf is None:
            continue
        for log_sig in (f"loki_{nf}", f"loki_vol_{nf}"):
            if log_sig in signals:
                pairs.append((sig, log_sig, "metrics↔logs"))
        if f"jaeger_{nf}" in signals:
            pairs.append((sig, f"jaeger_{nf}", "metrics↔traces"))
        for trace_stat in ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]:
            trace_sig = f"{trace_stat}_{nf}"
            if trace_sig in signals:
                pairs.append((sig, trace_sig, "metrics↔traces"))

    # ── RTT (global metric) cross-modal pairs ─────────────────────────────────
    # RTT is end-to-end (not per-NF); pair it with every NF's logs and traces and
    # let the sparsity filter prune channels that are absent for most faults.
    if "rtt" in signals:
        for nf in CORE_NFS:
            for log_sig in (f"loki_{nf}", f"loki_vol_{nf}"):
                if log_sig in signals:
                    pairs.append(("rtt", log_sig, "metrics↔logs"))
        for nf in TRACE_NFS:
            if f"jaeger_{nf}" in signals:
                pairs.append(("rtt", f"jaeger_{nf}", "metrics↔traces"))
            for trace_stat in ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]:
                trace_sig = f"{trace_stat}_{nf}"
                if trace_sig in signals:
                    pairs.append(("rtt", trace_sig, "metrics↔traces"))

    # ── cross-NF metric→trace correlations (existing p99 + new stats) ─────────
    cross_nf_traces_base = [
        ("cpu_nrf", "scp"), ("cpu_nrf", "amf"), ("cpu_amf", "scp"), ("cpu_scp", "amf"),
        ("mem_nrf", "scp"), ("mem_amf", "scp"),
    ]
    for metric_sig, trace_nf in cross_nf_traces_base:
        if metric_sig not in signals:
            continue
        # p99 latency
        if f"jaeger_{trace_nf}" in signals:
            pairs.append((metric_sig, f"jaeger_{trace_nf}", "metrics↔traces"))
        # New trace statistics
        for trace_stat in ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]:
            trace_sig = f"{trace_stat}_{trace_nf}"
            if trace_sig in signals:
                pairs.append((metric_sig, trace_sig, "metrics↔traces"))

    cross_nf_net = [
        ("net_rx_amf", "loki_ueransim_gnb",     "metrics↔logs"),
        ("net_rx_amf", "loki_vol_ueransim_gnb",  "metrics↔logs"),
        ("net_rx_upf", "loki_smf",               "metrics↔logs"),
        ("net_rx_nrf", "loki_scp",               "metrics↔logs"),
        ("net_rx_amf", "jaeger_scp",             "metrics↔traces"),
        ("net_rx_nrf", "jaeger_amf",             "metrics↔traces"),
    ]
    for a, b, cat in cross_nf_net:
        if a in signals and b in signals:
            pairs.append((a, b, cat))

        # Add new trace stats to existing net cross-NF pairs
        if cat == "metrics↔traces":
            trace_nf = b.split("_")[1]  # extract NF from jaeger_XXX
            for trace_stat in ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]:
                trace_sig = f"{trace_stat}_{trace_nf}"
                if a in signals and trace_sig in signals:
                    pairs.append((a, trace_sig, "metrics↔traces"))

    cross_nf_logs = [
        ("cpu_nrf",  "loki_scp",                "metrics↔logs"),
        ("mem_nrf",  "loki_scp",                "metrics↔logs"),
        ("cpu_nrf",  "loki_vol_scp",            "metrics↔logs"),
        ("cpu_amf",  "loki_ueransim_gnb",       "metrics↔logs"),
        ("mem_amf",  "loki_ueransim_gnb",       "metrics↔logs"),
        ("cpu_amf",  "loki_vol_ueransim_gnb",   "metrics↔logs"),
        ("cpu_smf",  "loki_ueransim_ues",       "metrics↔logs"),
        ("cpu_upf",  "loki_smf",                "metrics↔logs"),
        ("jaeger_nrf", "loki_scp",              "traces↔logs"),
        ("jaeger_nrf", "loki_vol_scp",          "traces↔logs"),
        ("jaeger_amf", "loki_ueransim_gnb",     "traces↔logs"),
        ("jaeger_smf", "loki_ueransim_ues",     "traces↔logs"),
        ("loki_nrf", "loki_scp",                "logs↔logs"),
        ("loki_amf", "loki_ueransim_gnb",       "logs↔logs"),
    ]
    for a, b, cat in cross_nf_logs:
        if a in signals and b in signals:
            pairs.append((a, b, cat))

        # Add new trace stats to existing trace→log cross-NF pairs
        if cat == "traces↔logs" and a.startswith("jaeger_") and not any(x in a for x in ["count", "err", "mean", "p50", "p95"]):
            trace_nf = a.split("_")[1]  # extract NF from jaeger_XXX
            for trace_stat in ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]:
                trace_sig = f"{trace_stat}_{trace_nf}"
                if trace_sig in signals and b in signals:
                    pairs.append((trace_sig, b, "traces↔logs"))

    # ── single-modal traces↔traces (cross-NF) — p99 + new statistics ──────────
    cross_nf_traces_single_base = [
        ("amf", "smf"), ("amf", "nrf"), ("smf", "nrf"),
        ("amf", "ausf"), ("amf", "scp"), ("smf", "scp"), ("nrf", "ausf"),
    ]
    for nf_a, nf_b in cross_nf_traces_single_base:
        # p99 latency pairs
        if f"jaeger_{nf_a}" in signals and f"jaeger_{nf_b}" in signals:
            pairs.append((f"jaeger_{nf_a}", f"jaeger_{nf_b}", "traces↔traces"))

        # New trace stat pairs (all combinations)
        all_trace_types = ["jaeger"] + ["jaeger_count", "jaeger_err", "jaeger_mean", "jaeger_p50", "jaeger_p95"]
        for stat_a in all_trace_types:
            for stat_b in all_trace_types:
                sig_a = f"{stat_a}_{nf_a}"
                sig_b = f"{stat_b}_{nf_b}"
                if sig_a in signals and sig_b in signals and sig_a != sig_b:
                    pairs.append((sig_a, sig_b, "traces↔traces"))

    return pairs


def analyze_fault(fault_dir):
    tl_path = fault_dir / "timeline.json"
    if not tl_path.exists():
        return []
    tl = json.loads(tl_path.read_text())
    pre_start   = tl["pre"]["start"]
    fault_start = tl["fault"]["start"]
    post_start  = tl.get("post", {}).get("start")

    # Compute pre-fault baseline stats and normalized pre signals
    pre_raw, pre_means, pre_stds = build_signals(fault_dir, "pre", pre_start, None, None)
    # Normalize pre against itself to get baseline correlation level
    pre_signals    = {k: normalize(v, pre_means[k], pre_stds[k]) for k, v in pre_raw.items()}
    during_signals = build_signals(fault_dir, "during", fault_start, pre_means, pre_stds)

    windows = {"pre": pre_signals, "during": during_signals}
    if post_start is not None:
        windows["post"] = build_signals(fault_dir, "post", post_start, pre_means, pre_stds)

    rows = []
    pairs = build_pairs(during_signals)  # pair list driven by during-window signal availability

    for window_name, signals in windows.items():
        for sig_a, sig_b, category in pairs:
            if sig_a not in signals or sig_b not in signals:
                continue
            corr, lag = xcorr(signals[sig_a], signals[sig_b])
            rho = spearman_corr(signals[sig_a], signals[sig_b])
            if not np.isnan(corr):
                rows.append({
                    "sig_a":         sig_a,
                    "sig_b":         sig_b,
                    "modality_pair": category,
                    "correlation":   corr,       # rank cross-correlation peak (used for SQ3 lag analysis)
                    "spearman_r":    rho,         # Spearman at lag 0 (used for the Δ|r| classifier feature)
                    "lag_bins":      lag,
                    "lag_sec":       lag * BIN_SEC if lag is not None else np.nan,
                    "window":        window_name,
                })

    return rows


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_heatmap(df, out_dir):
    """Average |correlation| per modality pair per fault category."""
    df = df.copy()
    df["fault_category"] = df["fault_category"].replace({"cascade": "pod_crash"})
    cats = ["infrastructure", "pod_crash", "network", "pfcp"]
    mod_pairs = ["metrics↔traces", "metrics↔logs", "traces↔logs"]
    corr_col = "spearman_r" if "spearman_r" in df.columns else "correlation"

    matrix = np.full((len(mod_pairs), len(cats)), np.nan)
    for i, mp in enumerate(mod_pairs):
        for j, cat in enumerate(cats):
            sub = df[(df["modality_pair"] == mp) & (df["fault_category"] == cat)]
            if len(sub) > 0:
                matrix[i, j] = sub[corr_col].abs().mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=11)
    ax.set_yticks(range(len(mod_pairs)))
    ax.set_yticklabels(mod_pairs, fontsize=11)
    for i in range(len(mod_pairs)):
        for j in range(len(cats)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if matrix[i,j] < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="|correlation|")
    ax.set_title("Average Cross-Modal Correlation by Fault Category", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  saved: correlation_heatmap.png")


def plot_lag(df, out_dir):
    """Average lag (seconds) per modality pair per fault category — metrics→traces only."""
    cats = ["infrastructure", "pod_crash", "network", "pfcp", "cascade"]
    mod_pairs = ["metrics↔traces", "metrics↔logs", "traces↔logs"]

    fig, axes = plt.subplots(1, len(cats), figsize=(14, 4), sharey=True)
    for ax, cat in zip(axes, cats):
        sub = df[df["fault_category"] == cat]
        means, labels = [], []
        for mp in mod_pairs:
            vals = sub[sub["modality_pair"] == mp]["lag_sec"].dropna()
            if len(vals) > 0:
                means.append(vals.mean())
                labels.append(mp)
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        bars = ax.barh(labels, means, color=colors[:len(labels)])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(cat, fontsize=10)
        ax.set_xlabel("lag (s)", fontsize=9)
        for bar, val in zip(bars, means):
            ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                    f"{val:.0f}s", va="center", fontsize=8)
    axes[0].set_ylabel("Modality pair")
    fig.suptitle("Average Propagation Lag Between Modality Pairs\n(positive = A leads B)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "lag_plot.png", dpi=150)
    plt.close(fig)
    print(f"  saved: lag_plot.png")


def plot_per_fault(df, out_dir):
    """Heatmap of |correlation| per fault for metrics↔traces pairs."""
    faults = sorted(df["fault_name"].unique())
    mod_pairs = ["metrics↔traces", "metrics↔logs", "traces↔logs"]
    corr_col = "spearman_r" if "spearman_r" in df.columns else "correlation"

    matrix = np.full((len(mod_pairs), len(faults)), np.nan)
    for i, mp in enumerate(mod_pairs):
        for j, fault in enumerate(faults):
            sub = df[(df["modality_pair"] == mp) & (df["fault_name"] == fault)]
            if len(sub) > 0:
                matrix[i, j] = sub[corr_col].abs().mean()

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(faults)))
    short_names = [f.split("-", 1)[1][:22] for f in faults]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mod_pairs)))
    ax.set_yticklabels(mod_pairs, fontsize=10)
    plt.colorbar(im, ax=ax, label="|correlation|")
    ax.set_title("Cross-Modal Correlation per Fault", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_per_fault.png", dpi=150)
    plt.close(fig)
    print(f"  saved: correlation_per_fault.png")


def plot_delta_heatmap(df, out_dir):
    """
    Heatmap of correlation delta: |corr_during| - |corr_pre|.
    Positive = cross-modal coupling increases when fault is active.
    Cascade is merged into pod_crash (NRF cascade = NRF pod failure).
    """
    df = df.copy()
    df["fault_category"] = df["fault_category"].replace({"cascade": "pod_crash"})
    cats      = ["infrastructure", "pod_crash", "network", "pfcp"]
    mod_pairs = ["metrics↔traces", "metrics↔logs", "traces↔logs"]
    corr_col  = "spearman_r" if "spearman_r" in df.columns else "correlation"

    pre    = df[df["window"] == "pre"]
    during = df[df["window"] == "during"]

    def avg(subset, mp, cat):
        s = subset[(subset["modality_pair"] == mp) & (subset["fault_category"] == cat)]
        return s[corr_col].abs().mean() if len(s) > 0 else np.nan

    matrix = np.full((len(mod_pairs), len(cats)), np.nan)
    for i, mp in enumerate(mod_pairs):
        for j, cat in enumerate(cats):
            pre_val    = avg(pre,    mp, cat)
            during_val = avg(during, mp, cat)
            if not np.isnan(pre_val) and not np.isnan(during_val):
                matrix[i, j] = during_val - pre_val

    fig, ax = plt.subplots(figsize=(9, 4))
    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 0.3
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=11)
    ax.set_yticks(range(len(mod_pairs)))
    ax.set_yticklabels(mod_pairs, fontsize=11)
    for i in range(len(mod_pairs)):
        for j in range(len(cats)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="Δ|correlation| (during − pre)")
    ax.set_title("Cross-Modal Correlation Increase During Fault (during − pre baseline)", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_delta.png", dpi=150)
    plt.close(fig)
    print(f"  saved: correlation_delta.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out",  required=True)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for fault_dir in sorted(data_dir.iterdir()):
        if not fault_dir.is_dir() or fault_dir.name not in FAULT_CATEGORIES:
            continue
        print(f"[xcorr] {fault_dir.name} ...", end=" ", flush=True)
        rows = analyze_fault(fault_dir)
        for r in rows:
            r["fault_name"]     = fault_dir.name
            r["fault_category"] = FAULT_CATEGORIES[fault_dir.name]
        all_rows.extend(rows)
        print(f"{len(rows)} pairs")

    df = pd.DataFrame(all_rows)
    out_csv = out_dir / "correlations.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[done] {len(df)} correlation pairs → {out_csv}")

    print("\n[plots]")
    plot_heatmap(df, out_dir)
    plot_lag(df, out_dir)
    plot_per_fault(df, out_dir)
    plot_delta_heatmap(df, out_dir)

    # Summary: top correlated pairs per fault category
    print("\n=== Top cross-modal correlations per category ===")
    for cat in ["infrastructure", "pod_crash", "network", "pfcp", "cascade"]:
        sub = df[df["fault_category"] == cat]
        if len(sub) == 0:
            continue
        top = sub.groupby(["sig_a", "sig_b"])["correlation"].apply(
            lambda x: x.abs().mean()
        ).sort_values(ascending=False).head(3)
        print(f"\n{cat}:")
        for (sa, sb), corr in top.items():
            lag = sub[(sub["sig_a"] == sa) & (sub["sig_b"] == sb)]["lag_sec"].mean()
            print(f"  {sa} ↔ {sb}: |corr|={corr:.3f}  lag={lag:.0f}s")


if __name__ == "__main__":
    main()
