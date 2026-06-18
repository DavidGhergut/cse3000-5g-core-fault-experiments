"""Render two views of the per-category cross-modal Delta|rho| heatmap:

  (1) combined  : the metrics modality = resource + Open5GS counters averaged into one cell
  (2) sublayers : resource-metrics and Open5GS application-metrics shown as separate cells

Writes both PNGs to the output directory and prints the numeric tables.
"""

import _paths
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CROSS_PAIRS = ["metrics↔logs", "metrics↔traces", "traces↔logs"]
RELABEL = {"cascade": "pod_crash"}
CATEGORIES = ["infrastructure", "network", "pfcp", "pod_crash"]
CATEGORY_LABELS = ["Infrastructure", "Network", "PFCP", "Pod crash"]
PAIR_INDEX = ["fault_name", "trial", "sig_a", "sig_b", "modality_pair", "fault_category"]

OUTPUT = _paths.OUTPUT
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load every trial's correlations.
frames = []
for trial in _paths.TRIALS:
    path = _paths.corr_csv(trial)
    if path.exists():
        df_trial = pd.read_csv(path)
        df_trial["trial"] = trial
        frames.append(df_trial)
corr = pd.concat(frames, ignore_index=True)
print("trials:", sorted(corr.trial.unique()))
corr["abs_rho"] = pd.to_numeric(corr["spearman_r"], errors="coerce").abs()

# Delta|rho| = |rho_during| - |rho_pre| per signal pair.
pre = corr[corr.window == "pre"].set_index(PAIR_INDEX)["abs_rho"]
during = corr[corr.window == "during"].set_index(PAIR_INDEX)["abs_rho"]
common = during.index.intersection(pre.index)
delta = (during.loc[common] - pre.loc[common]).reset_index()
delta.columns = PAIR_INDEX + ["delta_rho"]
delta["cat"] = delta["fault_category"].replace(RELABEL)
# Whether either signal in the pair is an Open5GS application counter.
delta["is_o5g"] = (delta.sig_a.astype(str).str.startswith("o5g_")
                   | delta.sig_b.astype(str).str.startswith("o5g_"))


def cell(category, modality_pair, sublayer=None):
    """Mean Delta|rho| for one (category, modality pair), optionally split by metric sublayer."""
    subset = delta[(delta.cat == category) & (delta.modality_pair == modality_pair)]
    if sublayer == "res":       # resource (infrastructure) metrics only
        subset = subset[~subset.is_o5g]
    elif sublayer == "o5g":     # Open5GS application metrics only
        subset = subset[subset.is_o5g]
    return subset.delta_rho.mean() if len(subset) else np.nan


def render(matrix, col_labels, fname, title, rot=20):
    mat = np.array(matrix)
    fig, ax = plt.subplots(figsize=(1.15 * len(col_labels) + 1.5, 3.8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.13, vmax=0.13, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=rot, ha="right" if rot else "center")
    ax.set_yticks(range(4))
    ax.set_yticklabels(CATEGORY_LABELS, fontsize=9)
    ax.set_xlabel("Cross-modal signal pair", fontsize=9)
    ax.set_ylabel("Fault category", fontsize=9)
    for i in range(4):
        for j in range(len(col_labels)):
            val = mat[i, j]
            ax.text(j, i, "—" if np.isnan(val) else f"{val:+.3f}", ha="center", va="center",
                    color="white" if (not np.isnan(val) and abs(val) > 0.07) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r"mean $\Delta|\rho|$", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT / fname, dpi=150)
    print("wrote", fname)


# (1) Combined view: one metrics cell per pair.
combined = [[cell(c, "metrics↔logs"), cell(c, "metrics↔traces"), cell(c, "traces↔logs")]
            for c in CATEGORIES]
render(combined, ["Metrics–Logs", "Metrics–Traces", "Traces–Logs"],
       "heatmap_combined.png", "Combined: resource + Open5GS averaged")

# (2) Sub-layer view: resource vs Open5GS application metrics shown separately.
sublayers = [[cell(c, "metrics↔logs", "res"), cell(c, "metrics↔logs", "o5g"),
              cell(c, "metrics↔traces", "res"), cell(c, "metrics↔traces", "o5g"),
              cell(c, "traces↔logs")]
             for c in CATEGORIES]
render(sublayers, ["Infrastructure\n↔ Logs", "Application\n↔ Logs", "Infrastructure\n↔ Traces",
                   "Application\n↔ Traces", "Traces\n↔ Logs"],
       "heatmap_sublayers.png", "Metric sub-layers: infrastructure vs application (Open5GS)", rot=0)

# Numeric tables.
print("\n=== COMBINED ===")
print(f"{'cat':15}{'M-L':>9}{'M-T':>9}{'T-L':>9}")
for label, row in zip(CATEGORY_LABELS, combined):
    print(f"{label:15}" + "".join(f"{v:>+9.3f}" for v in row))
print("\n=== SUB-LAYERS ===")
print(f"{'cat':15}{'Res-L':>8}{'Func-L':>8}{'Res-T':>8}{'Func-T':>8}{'T-L':>8}")
for label, row in zip(CATEGORY_LABELS, sublayers):
    print(f"{label:15}" + "".join(f"{v:>+8.3f}" for v in row))
