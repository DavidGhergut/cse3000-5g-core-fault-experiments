#!/usr/bin/env python3
"""Figure for the partition-decoupling finding (per-trial paired view).

For each of the 7 trials, compare the mean traces<->logs coupling change Delta|rho|
of the interface-partition faults (AMF-SCP, N2 AMF-gNB) against the mean of all
non-partition faults. In every trial the partition group decouples more strongly
(more negative), which directly visualises the 7/7 sign test (p=0.016) the paper
reports, rather than a per-fault ranking where one non-partition fault could
misleadingly outrank a partition fault.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths

OUTPUT = _paths.OUTPUT
OUTPUT.mkdir(parents=True, exist_ok=True)

PARTITION_FAULTS = {"05-network-partition-amf-scp", "21-n2-partition-amf-gnb"}
TRIAL_LABELS = {"boyan": "Trial 1", "boyan-2": "Trial 2", "trial4": "Trial 3",
                "trial5": "Trial 4", "trial6": "Trial 5", "trial7": "Trial 6",
                "trial8": "Trial 7"}
PAIR_INDEX = ["trial", "fault_name", "sig_a", "sig_b", "modality_pair"]

# Load all trials and compute traces<->logs Delta|rho| per (trial, fault).
frames = []
for trial in _paths.TRIALS:
    df = pd.read_csv(_paths.corr_csv(trial))
    df["trial"] = trial
    frames.append(df)
corr = pd.concat(frames, ignore_index=True)
corr["abs_rho"] = pd.to_numeric(corr["spearman_r"], errors="coerce").abs()

pre = corr[corr.window == "pre"].set_index(PAIR_INDEX)["abs_rho"]
during = corr[corr.window == "during"].set_index(PAIR_INDEX)["abs_rho"]
common = pre.index.intersection(during.index)
delta = (during.loc[common] - pre.loc[common]).rename("d").reset_index()
tl_delta = delta[delta.modality_pair == "traces↔logs"]
per_fault = tl_delta.groupby(["trial", "fault_name"]).d.mean().reset_index()

# Per trial: mean Delta|rho| of partition faults vs non-partition faults.
rows = []
for trial in _paths.TRIALS:
    in_trial = per_fault[per_fault.trial == trial]
    partition = in_trial[in_trial.fault_name.isin(PARTITION_FAULTS)].d
    other = in_trial[~in_trial.fault_name.isin(PARTITION_FAULTS)].d
    if len(partition) == 0 or len(other) == 0:
        continue
    rows.append((trial, partition.mean(), other.mean()))
summary = pd.DataFrame(rows, columns=["trial", "part", "rest"])
summary["diff"] = summary["part"] - summary["rest"]

n_trials = len(summary)
n_partition_lower = int((summary["diff"] < 0).sum())  # trials where partition decoupled more
mean_diff = summary["diff"].mean()
se = summary["diff"].std(ddof=1) / np.sqrt(n_trials)
print(summary.to_string(index=False))
print(f"\nn_trials={n_trials}  partition-more-decoupled={n_partition_lower}/{n_trials}")
print(f"mean difference = {mean_diff:.3f}   SE = {se:.3f}   (sign test {n_partition_lower}/{n_trials})")

# Paired dumbbell plot: partition vs non-partition mean per trial.
fig, ax = plt.subplots(figsize=(7, 4.6))
y_pos = np.arange(n_trials)[::-1]
for y, (_, row) in zip(y_pos, summary.iterrows()):
    ax.plot([row["rest"], row["part"]], [y, y], color="#bbb", lw=1.4, zorder=1)
ax.scatter(summary["rest"], y_pos, s=70, color="#5b8db8", zorder=3, label="Non-partition faults (mean)")
ax.scatter(summary["part"], y_pos, s=70, color="#c0392b", zorder=3, label="Partition faults (mean)")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([TRIAL_LABELS[t] for t in summary["trial"]], fontsize=9)
ax.set_xlabel(r"$\leftarrow$ stronger decoupling$\qquad$Traces$\leftrightarrow$Logs  $\Delta|\rho|$$\qquad$coupling $\rightarrow$", fontsize=9)
ax.set_title("Partition faults decouple trace–log coupling more than\nnon-partition faults in every trial",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper center", frameon=True)
ax.tick_params(axis="x", labelsize=8)
ax.margins(y=0.08)
ax.text(0.02, 0.04, f"mean difference {mean_diff:.2f}, sign test {n_partition_lower}/{n_trials} (p=0.016)",
        transform=ax.transAxes, fontsize=8, color="#444", style="italic")
plt.tight_layout()
out = OUTPUT / "fig_partition_decoupling.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
