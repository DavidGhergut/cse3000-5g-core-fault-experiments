#!/usr/bin/env python3
"""Significance tests for the two headline correlation findings:

  (1) interface-partition faults decouple traces<->logs more than other faults
  (2) metrics<->logs Delta|rho| is positive across the operational fault set

Uses the same Delta|rho| = |rho_during| - |rho_pre| construction as the paper figures.
"""

import numpy as np
import pandas as pd
from scipy import stats

import _paths

# The two interface-partition faults, which we expect to decouple traces from logs.
PARTITION_FAULTS = {"05-network-partition-amf-scp", "21-n2-partition-amf-gnb"}

# Columns that uniquely identify one signal pair within one fault and trial.
PAIR_INDEX = ["trial", "fault_name", "sig_a", "sig_b", "modality_pair"]

# Load every trial's correlations into one table and pre-compute |rho|.
frames = []
for trial in _paths.TRIALS:
    df = pd.read_csv(_paths.corr_csv(trial))
    df["trial"] = trial
    frames.append(df)
corr = pd.concat(frames, ignore_index=True)
corr["abs_rho"] = pd.to_numeric(corr["spearman_r"], errors="coerce").abs()

n_trials = len(_paths.TRIALS)


def delta_by_fault_trial(modality_pair):
    """Mean Delta|rho| (= |rho_during| - |rho_pre|) per (fault, trial) for one modality pair."""
    subset = corr[corr.modality_pair == modality_pair]
    pre = subset[subset.window == "pre"].set_index(PAIR_INDEX)["abs_rho"]
    during = subset[subset.window == "during"].set_index(PAIR_INDEX)["abs_rho"]
    common = pre.index.intersection(during.index)
    delta = (during.loc[common] - pre.loc[common]).rename("d").reset_index()
    return delta.groupby(["fault_name", "trial"]).d.mean().reset_index()


# ── Finding 1: partition faults collapse traces<->logs coupling ──────────────
print("=" * 64)
print("FINDING 1: partition faults collapse traces<->logs coupling")
print("=" * 64)

tl_delta = delta_by_fault_trial("traces↔logs")

# Per trial: mean Delta|rho| of the partition faults vs all other faults.
rows = []
for trial in _paths.TRIALS:
    in_trial = tl_delta[tl_delta.trial == trial]
    partition_mean = in_trial[in_trial.fault_name.isin(PARTITION_FAULTS)].d.mean()
    other_mean = in_trial[~in_trial.fault_name.isin(PARTITION_FAULTS)].d.mean()
    rows.append((trial, partition_mean, other_mean, partition_mean - other_mean))
per_trial = pd.DataFrame(rows, columns=["trial", "partition", "non_partition", "diff"])
print(per_trial.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

n_partition_lower = int((per_trial["diff"] < 0).sum())  # trials where partition decoupled more
se = per_trial["diff"].std(ddof=1) / np.sqrt(n_trials)
print(f"\n  trials where partition < non-partition: {n_partition_lower}/{n_trials}")
print(f"  mean difference: {per_trial['diff'].mean():+.4f}  (SE {se:.4f})")

wilcoxon = stats.wilcoxon(per_trial.partition, per_trial.non_partition)
print(f"  paired Wilcoxon signed-rank (n={n_trials}): W={wilcoxon.statistic:.1f}, p={wilcoxon.pvalue:.4f}")
sign_p = stats.binomtest(n_partition_lower, n_trials, 0.5, alternative="two-sided").pvalue
print(f"  sign test ({n_partition_lower}/{n_trials}): p={sign_p:.4f}")

# Fault-level: average each fault across trials, then test the 2 partition faults vs the rest.
fault_level = tl_delta.groupby("fault_name").d.mean()
partition_vals = fault_level[fault_level.index.isin(PARTITION_FAULTS)].values
other_vals = fault_level[~fault_level.index.isin(PARTITION_FAULTS)].values
print(f"\n  fault-level traces<->logs Delta|rho| (avg over {n_trials} trials):")
for fault_name, value in fault_level[fault_level.index.isin(PARTITION_FAULTS)].items():
    print(f"    {fault_name:32s} {value:+.4f}")
print(f"    most-negative non-partition fault: {other_vals.min():+.4f}")

mann_whitney = stats.mannwhitneyu(partition_vals, other_vals, alternative="less")
print(f"  Mann-Whitney U (partition < others): U={mann_whitney.statistic:.1f}, p={mann_whitney.pvalue:.4f}")

# Permutation test: how often does a random pick of 2 faults separate as strongly?
observed_gap = other_vals.mean() - partition_vals.mean()
all_vals = fault_level.values
n_partition = len(partition_vals)
rng = np.random.default_rng(0)
n_permutations = 50000
count = 0
for _ in range(n_permutations):
    picked = rng.choice(len(all_vals), n_partition, replace=False)
    if (np.delete(all_vals, picked).mean() - all_vals[picked].mean()) >= observed_gap:
        count += 1
perm_p = (count + 1) / (n_permutations + 1)
print(f"  permutation test (50k, one-sided): p={perm_p:.5f}  (obs gap {observed_gap:+.4f})")

# ── Finding 2: metrics<->logs Delta|rho| is positive across faults ───────────
print("\n" + "=" * 64)
print("FINDING 2: metrics<->logs Delta|rho| is positive across faults")
print("=" * 64)

ml_delta = delta_by_fault_trial("metrics↔logs").groupby("fault_name").d.mean()
n_positive = int((ml_delta > 0).sum())
print(f"  faults with positive metrics<->logs Delta|rho|: {n_positive}/{len(ml_delta)}")
print(f"  mean over faults: {ml_delta.mean():+.4f}  (range {ml_delta.min():+.4f} .. {ml_delta.max():+.4f})")
wilcoxon_ml = stats.wilcoxon(ml_delta.values)
print(f"  Wilcoxon signed-rank vs 0: W={wilcoxon_ml.statistic:.1f}, p={wilcoxon_ml.pvalue:.2e}")
sign_p_ml = stats.binomtest(n_positive, len(ml_delta), 0.5, alternative="greater").pvalue
print(f"  binomial sign test ({n_positive}/{len(ml_delta)} positive): p={sign_p_ml:.2e}")

# ── Context: metrics<->traces coupling is weak everywhere (the trace blindspot) ──
print("\n" + "=" * 64)
print("CONTEXT: metrics<->traces weakness")
print("=" * 64)

mt_delta = delta_by_fault_trial("metrics↔traces").groupby("fault_name").d.mean()
print(f"  |metrics<->traces Delta|rho|| < 0.05 for {int((mt_delta.abs() < 0.05).sum())}/{len(mt_delta)} faults")
print(f"  cross-fault mean |Delta|rho||: {mt_delta.abs().mean():.4f}")
