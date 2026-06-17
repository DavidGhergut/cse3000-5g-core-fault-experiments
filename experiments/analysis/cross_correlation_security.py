"""
cross_correlation_security.py
──────────────────────────────
Cross-modal correlation analysis for security fault experiments.
Works with the scenarios/ CSV format produced by run_security_experiment.sh.

For each fault type × trial:
  1. Load metrics.csv + logs.csv, align on inject_time
  2. Interpolate 30s-sampled data to 10s bins
  3. Z-score normalise against pre-fault baseline
  4. Compute cross-correlation (max |r| and lag) for pre and fault windows
  5. Compute Δ|r| = |r_fault| − |r_pre|

Outputs:
  <out>/xcorr_summary.csv          one row per (fault, trial, signal_pair, window)
  <out>/xcorr_delta_bar.png        Δ|r| bar chart: storm vs sbi_flood per signal pair
  <out>/xcorr_lag_bar.png          lag bar chart: storm vs sbi_flood
  <out>/xcorr_timeseries_<fault>.png  normalised signals overlay for each fault type

Usage:
    python3 cross_correlation_security.py \\
        --data  /path/to/scenarios/security_faults \\
        --out   /path/to/output \\
        --faults open5gs-amf_storm open5gs-amf_sbi_flood
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
NF          = "open5gs-amf"
BIN_SEC     = 10          # interpolate to 10s bins
PRE_DUR     = 600         # seconds of pre-fault baseline
FAULT_DUR   = 300         # seconds of fault window
MAX_LAG     = 5           # max lag in bins (= 50s)

SIGNAL_PAIRS = [
    ("cpu_rate",    "log_total",  "metrics↔logs"),
    ("net_rx_rate", "log_total",  "metrics↔logs"),
    ("cpu_rate",    "log_error",  "metrics↔logs"),
    ("cpu_rate",    "log_warn",   "metrics↔logs"),
    ("cpu_rate",    "net_rx_rate","metrics↔metrics"),
]

FAULT_LABELS = {
    "open5gs-amf_storm":     "Registration Storm",
    "open5gs-amf_sbi_flood": "SBI HTTP/2 Flood (AMF)",
    "open5gs-amf_nrf_flood": "SBI HTTP/2 Flood (NRF)",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_trial(trial_dir: Path, nf: str = NF):
    """Load and return aligned metrics + logs as a single DataFrame on 10s bins."""
    inject_time = int((trial_dir / "inject_time.txt").read_text().strip())

    m = pd.read_csv(trial_dir / "metrics.csv")
    l = pd.read_csv(trial_dir / "logs.csv")

    m["t"] = m["time"] - inject_time
    l["t"] = l["time"] - inject_time

    # Rename columns to short names
    col_map_m = {
        f"{nf}_cpu_rate":    "cpu_rate",
        f"{nf}_net_rx_rate": "net_rx_rate",
        f"{nf}_net_tx_rate": "net_tx_rate",
        f"{nf}_mem_bytes":   "mem_bytes",
    }
    col_map_l = {
        f"{nf}_log_total": "log_total",
        f"{nf}_log_info":  "log_info",
        f"{nf}_log_error": "log_error",
        f"{nf}_log_warn":  "log_warn",
    }

    m = m.rename(columns=col_map_m)[["t"] + [v for v in col_map_m.values() if v in m.rename(columns=col_map_m).columns]]
    l = l.rename(columns=col_map_l)[["t"] + [v for v in col_map_l.values() if v in l.rename(columns=col_map_l).columns]]

    merged = pd.merge(m, l, on="t", how="outer").sort_values("t").reset_index(drop=True)

    # Interpolate to uniform 10s grid
    t_min = -PRE_DUR
    t_max = FAULT_DUR + 300  # include post-fault recovery
    t_grid = np.arange(t_min, t_max + BIN_SEC, BIN_SEC, dtype=float)

    result = pd.DataFrame({"t": t_grid})
    for col in merged.columns:
        if col == "t":
            continue
        result[col] = np.interp(
            t_grid,
            merged["t"].values,
            merged[col].fillna(method="ffill").fillna(method="bfill").values,
            left=np.nan, right=np.nan
        )

    return result


# ── Cross-correlation ─────────────────────────────────────────────────────────

def xcorr(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG):
    """
    Normalised cross-correlation. Returns (max_abs_corr, lag_bins).
    Positive lag = b lags behind a (a leads).
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return np.nan, np.nan
    a_c = a[mask] - np.mean(a[mask])
    b_c = b[mask] - np.mean(b[mask])
    if np.std(a_c) < 1e-10 or np.std(b_c) < 1e-10:
        return np.nan, np.nan
    n = len(a_c)
    corr = np.correlate(a_c, b_c, mode="full") / (n * np.std(a_c) * np.std(b_c))
    lags = np.arange(-(n - 1), n)
    mask_lag = np.abs(lags) <= max_lag
    corr_r = corr[mask_lag]
    lags_r = lags[mask_lag]
    idx = np.argmax(np.abs(corr_r))
    return float(corr_r[idx]), int(lags_r[idx])


def normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std < 1e-10:
        return arr - mean
    return (arr - mean) / std


# ── Per-trial analysis ────────────────────────────────────────────────────────

def analyze_trial(df: pd.DataFrame):
    """Return list of correlation rows for pre and fault windows."""
    pre   = df[df["t"] <  0]
    fault = df[(df["t"] >= 0) & (df["t"] <= FAULT_DUR)]

    rows = []
    for sig_a, sig_b, modality in SIGNAL_PAIRS:
        if sig_a not in df.columns or sig_b not in df.columns:
            continue

        # Pre-fault baseline stats
        pre_mean_a = np.nanmean(pre[sig_a].values)
        pre_std_a  = np.nanstd(pre[sig_a].values)
        pre_mean_b = np.nanmean(pre[sig_b].values)
        pre_std_b  = np.nanstd(pre[sig_b].values)

        for window_name, window_df in [("pre", pre), ("fault", fault)]:
            a_norm = normalize(window_df[sig_a].values, pre_mean_a, pre_std_a)
            b_norm = normalize(window_df[sig_b].values, pre_mean_b, pre_std_b)
            corr, lag = xcorr(a_norm, b_norm)
            rows.append({
                "sig_a":    sig_a,
                "sig_b":    sig_b,
                "modality": modality,
                "window":   window_name,
                "corr":     corr,
                "abs_corr": abs(corr) if not np.isnan(corr) else np.nan,
                "lag_bins": lag,
                "lag_sec":  lag * BIN_SEC if lag is not None and not np.isnan(lag) else np.nan,
            })
    return rows


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_delta_bar(summary: pd.DataFrame, out_dir: Path, fault_types: list):
    """Bar chart: Δ|r| per signal pair per fault type."""
    pairs     = summary["pair_label"].unique()
    x         = np.arange(len(pairs))
    width     = 0.35
    colors    = ["#E07B54", "#4C8BB5"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (fault, color) in enumerate(zip(fault_types, colors)):
        label  = FAULT_LABELS.get(fault, fault)
        deltas = []
        errs   = []
        for pair in pairs:
            sub = summary[(summary["fault"] == fault) & (summary["pair_label"] == pair)]
            deltas.append(sub["delta_r"].mean() if len(sub) > 0 else 0)
            errs.append(sub["delta_r"].std()   if len(sub) > 1 else 0)
        bars = ax.bar(x + i * width, deltas, width, label=label, color=color,
                      yerr=errs, capsize=4, alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([p.replace("↔", " ↔\n") for p in pairs], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Δ|r|  (fault − pre baseline)")
    ax.set_title("Cross-Modal Correlation Increase During Fault  (Δ|r|)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "xcorr_delta_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_lag_bar(summary: pd.DataFrame, out_dir: Path, fault_types: list):
    """Bar chart: lag (seconds) at peak correlation per signal pair per fault type."""
    pairs  = summary["pair_label"].unique()
    x      = np.arange(len(pairs))
    width  = 0.35
    colors = ["#E07B54", "#4C8BB5"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (fault, color) in enumerate(zip(fault_types, colors)):
        label = FAULT_LABELS.get(fault, fault)
        lags  = []
        errs  = []
        for pair in pairs:
            sub = summary[(summary["fault"] == fault) & (summary["pair_label"] == pair)]
            lags.append(sub["lag_sec_fault"].mean() if len(sub) > 0 else 0)
            errs.append(sub["lag_sec_fault"].std()  if len(sub) > 1 else 0)
        ax.bar(x + i * width, lags, width, label=label, color=color,
               yerr=errs, capsize=4, alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([p.replace("↔", " ↔\n") for p in pairs], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Lag at peak |r| (seconds)\npositive = signal A leads signal B")
    ax.set_title("Propagation Lag Between Modality Pairs During Fault", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "xcorr_lag_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_timeseries(fault_dir: Path, fault_name: str, out_dir: Path, n_trials: int = 3):
    """Normalised time-series overlay for the main signal pair (cpu_rate vs log_total)."""
    inject_time_label = FAULT_LABELS.get(fault_name, fault_name)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{inject_time_label} — Normalised Signals (all trials)", fontsize=13, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for trial_idx in range(1, n_trials + 1):
        trial_dir = fault_dir / str(trial_idx)
        if not trial_dir.exists():
            continue
        df = load_trial(trial_dir)
        pre_mean_cpu = np.nanmean(df[df["t"] < 0]["cpu_rate"])
        pre_std_cpu  = np.nanstd(df[df["t"] < 0]["cpu_rate"])
        pre_mean_log = np.nanmean(df[df["t"] < 0]["log_total"])
        pre_std_log  = np.nanstd(df[df["t"] < 0]["log_total"])

        cpu_norm = normalize(df["cpu_rate"].values,   pre_mean_cpu, pre_std_cpu)
        log_norm = normalize(df["log_total"].values, pre_mean_log, pre_std_log)

        color = colors[trial_idx - 1]
        axes[0].plot(df["t"], cpu_norm, color=color, linewidth=1.5, label=f"Trial {trial_idx}")
        axes[1].plot(df["t"], log_norm, color=color, linewidth=1.5, label=f"Trial {trial_idx}")

    for ax in axes:
        ax.axvspan(0, FAULT_DUR, alpha=0.08, color="red")
        ax.axvline(0,          color="red",   linewidth=1.5, linestyle="--", label="Fault start")
        ax.axvline(FAULT_DUR,  color="green", linewidth=1.0, linestyle="--", label="Fault end")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("CPU rate  (z-score)")
    axes[1].set_ylabel("Log total  (z-score)")
    axes[1].set_xlabel("Time relative to fault injection (seconds)")
    axes[0].set_title("AMF CPU rate")
    axes[1].set_title("AMF Log volume")

    fig.tight_layout()
    slug = fault_name.replace("open5gs-amf_", "")
    out  = out_dir / f"xcorr_timeseries_{slug}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to security_faults/ folder")
    parser.add_argument("--out",    required=True, help="Output directory for plots/CSV")
    parser.add_argument("--faults", nargs="+",
                        default=["open5gs-amf_storm", "open5gs-amf_sbi_flood"])
    parser.add_argument("--nf",     default=None,
                        help="Override target NF (e.g. open5gs-nrf). "
                             "If omitted, inferred per fault as first path component.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []

    for fault_name in args.faults:
        fault_dir = data_dir / fault_name
        if not fault_dir.exists():
            print(f"[skip] {fault_name} — directory not found")
            continue
        # Infer target NF from fault folder name (e.g. open5gs-amf_nrf_flood → open5gs-nrf)
        # unless overridden by --nf
        if args.nf:
            target_nf = args.nf
        elif "nrf_flood" in fault_name:
            target_nf = "open5gs-nrf"
        else:
            target_nf = NF  # default: open5gs-amf
        print(f"\n[xcorr] {fault_name}  (target NF: {target_nf})")

        trial_results = []  # one entry per trial: dict of pair→(pre_r, fault_r, lag)
        trials = sorted([d for d in fault_dir.iterdir() if d.is_dir()], key=lambda p: int(p.name))

        for trial_dir in trials:
            trial_num = int(trial_dir.name)
            print(f"  trial {trial_num} ...", end=" ")
            try:
                df   = load_trial(trial_dir, nf=target_nf)
                rows = analyze_trial(df)
                for r in rows:
                    r["fault"] = fault_name
                    r["trial"] = trial_num
                all_rows.extend(rows)
                print(f"{len(rows)//2} pairs")
            except Exception as e:
                print(f"ERROR: {e}")

        # Build per-trial summary (Δ|r| and lag)
        trial_df = pd.DataFrame([r for r in all_rows if r["fault"] == fault_name])
        if len(trial_df) == 0:
            continue

        for (sig_a, sig_b, modality), grp in trial_df.groupby(["sig_a", "sig_b", "modality"]):
            pair_label = f"{sig_a}\n↔ {sig_b}"
            for trial_num, t_grp in grp.groupby("trial"):
                pre_r   = t_grp[t_grp["window"] == "pre"]["abs_corr"].values
                fault_r = t_grp[t_grp["window"] == "fault"]["abs_corr"].values
                lag_f   = t_grp[t_grp["window"] == "fault"]["lag_sec"].values
                if len(pre_r) == 0 or len(fault_r) == 0:
                    continue
                # Treat NaN pre_r as 0: flat/zero pre-fault signal = no correlation baseline.
                # This happens when log counts are all-zero in baseline (e.g. SBI flood).
                pre_val = 0.0 if np.isnan(pre_r[0]) else float(pre_r[0])
                delta = float(fault_r[0]) - pre_val if not np.isnan(fault_r[0]) else np.nan
                summary_rows.append({
                    "fault":          fault_name,
                    "fault_label":    FAULT_LABELS.get(fault_name, fault_name),
                    "trial":          trial_num,
                    "sig_a":          sig_a,
                    "sig_b":          sig_b,
                    "modality":       modality,
                    "pair_label":     f"{sig_a} ↔ {sig_b}",
                    "pre_r":          float(pre_r[0])   if len(pre_r)   else np.nan,
                    "fault_r":        float(fault_r[0]) if len(fault_r) else np.nan,
                    "delta_r":        delta,
                    "lag_sec_fault":  float(lag_f[0])   if len(lag_f)   else np.nan,
                })

    # ── Save raw correlation CSV
    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(out_dir / "xcorr_raw.csv", index=False)
    print(f"\n[done] {len(raw_df)} correlation entries → xcorr_raw.csv")

    # ── Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "xcorr_summary.csv", index=False)
    print(f"       {len(summary_df)} summary rows  → xcorr_summary.csv")

    # ── Print Δ|r| table
    print("\n=== Δ|r| Summary (mean across trials) ===")
    print(f"{'Pair':<35} {'Storm Δ|r|':>12} {'SBI Flood Δ|r|':>15}")
    print("-" * 65)
    for pair_label in summary_df["pair_label"].unique():
        for fault in args.faults:
            sub = summary_df[(summary_df["pair_label"] == pair_label) & (summary_df["fault"] == fault)]
            mean_d = sub["delta_r"].mean()
            print(f"  {fault.split('_',2)[-1]:<10}  {pair_label:<30}  Δ|r|={mean_d:+.3f}  (n={len(sub)})")
        print()

    # ── Plots
    print("\n[plots]")
    plot_delta_bar(summary_df, out_dir, args.faults)
    plot_lag_bar(summary_df, out_dir, args.faults)
    for fault_name in args.faults:
        fault_dir = data_dir / fault_name
        if fault_dir.exists():
            plot_timeseries(fault_dir, fault_name, out_dir)


if __name__ == "__main__":
    main()
