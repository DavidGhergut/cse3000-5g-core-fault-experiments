"""Render BOTH heatmap options from correlations_o5g/ so we can compare:
  (1) combined  : metrics modality = resource + Open5GS averaged into one M-L / M-T cell
  (2) sublayers : resource-metrics and functional(Open5GS)-metrics shown as separate cells
Writes to /tmp/o5g_heatmaps/ and prints the numeric tables. Read-only."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]
BASE = "data/5GCore/correlations_o5g"
CROSS = ["metrics↔logs", "metrics↔traces", "traces↔logs"]
RELABEL = {"cascade": "pod_crash"}
ROWS = ["infrastructure", "network", "pfcp", "pod_crash"]
RLAB = ["Infrastructure", "Network", "PFCP", "Pod crash"]
OUT = "/tmp/o5g_heatmaps"; os.makedirs(OUT, exist_ok=True)

dfs = []
for t in TRIALS:
    p = Path(BASE) / t / "correlations.csv"
    if p.exists():
        d = pd.read_csv(p); d["trial"] = t; dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
print("trials:", sorted(df.trial.unique()))
df["_c"] = pd.to_numeric(df["spearman_r"], errors="coerce").abs()
idx = ["fault_name", "trial", "sig_a", "sig_b", "modality_pair", "fault_category"]
pre = df[df.window == "pre"].set_index(idx)["_c"]
dur = df[df.window == "during"].set_index(idx)["_c"]
common = dur.index.intersection(pre.index)
d = (dur.loc[common] - pre.loc[common]).reset_index()
d.columns = idx + ["dr"]
d["cat"] = d["fault_category"].replace(RELABEL)
d["is_o5g"] = d.sig_a.astype(str).str.startswith("o5g_") | d.sig_b.astype(str).str.startswith("o5g_")


def cell(cat, mp, sub=None):
    s = d[(d.cat == cat) & (d.modality_pair == mp)]
    if sub == "res":
        s = s[~s.is_o5g]
    elif sub == "o5g":
        s = s[s.is_o5g]
    return s.dr.mean() if len(s) else np.nan


def render(matrix, cols, fname, title, rot=20):
    M = np.array(matrix)
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 1.5, 3.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.13, vmax=0.13, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8, rotation=rot, ha="right" if rot else "center")
    ax.set_yticks(range(4)); ax.set_yticklabels(RLAB, fontsize=9)
    ax.set_xlabel("Cross-modal signal pair", fontsize=9)
    ax.set_ylabel("Fault category", fontsize=9)
    for i in range(4):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:+.3f}", ha="center", va="center",
                    color="white" if (not np.isnan(v) and abs(v) > 0.07) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r"mean $\Delta|\rho|$", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150); print("wrote", fname)


# (1) combined
comb = [[cell(c, "metrics↔logs"), cell(c, "metrics↔traces"), cell(c, "traces↔logs")] for c in ROWS]
render(comb, ["Metrics–Logs", "Metrics–Traces", "Traces–Logs"],
       "heatmap_combined.png", "Combined: resource + Open5GS averaged")

# (2) sublayers
sub = [[cell(c, "metrics↔logs", "res"), cell(c, "metrics↔logs", "o5g"),
        cell(c, "metrics↔traces", "res"), cell(c, "metrics↔traces", "o5g"),
        cell(c, "traces↔logs")] for c in ROWS]
render(sub, ["Infrastructure\n↔ Logs", "Application\n↔ Logs", "Infrastructure\n↔ Traces",
             "Application\n↔ Traces", "Traces\n↔ Logs"],
       "heatmap_sublayers.png", "Metric sub-layers: infrastructure vs application (Open5GS)", rot=0)

print("\n=== COMBINED ===")
print(f"{'cat':15}{'M-L':>9}{'M-T':>9}{'T-L':>9}")
for c, r in zip(RLAB, comb):
    print(f"{c:15}" + "".join(f"{v:>+9.3f}" for v in r))
print("\n=== SUB-LAYERS ===")
print(f"{'cat':15}{'Res-L':>8}{'Func-L':>8}{'Res-T':>8}{'Func-T':>8}{'T-L':>8}")
for c, r in zip(RLAB, sub):
    print(f"{c:15}" + "".join(f"{v:>+8.3f}" for v in r))
