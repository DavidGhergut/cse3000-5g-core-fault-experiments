#!/usr/bin/env python3
"""Significance tests for the two headline correlation findings:
  (1) interface-partition faults decouple traces<->logs more than other faults
  (2) metrics<->logs Delta|rho| is positive across the operational fault set
Uses the same Delta|rho| = |r_during| - |r_pre| construction as the paper figures."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]
BASE = Path("data/5GCore/correlations")
PART = {"05-network-partition-amf-scp", "21-n2-partition-amf-gnb"}

fr = []
for t in TRIALS:
    d = pd.read_csv(BASE / t / "correlations.csv"); d["trial"] = t; fr.append(d)
a = pd.concat(fr, ignore_index=True)
a["_c"] = pd.to_numeric(a["spearman_r"], errors="coerce").abs()
IDX = ["trial", "fault_name", "sig_a", "sig_b", "modality_pair"]


def delta_by_fault_trial(pair):
    sub = a[a.modality_pair == pair]
    p = sub[sub.window == "pre"].set_index(IDX)["_c"]
    q = sub[sub.window == "during"].set_index(IDX)["_c"]
    c = p.index.intersection(q.index)
    D = (q.loc[c] - p.loc[c]).rename("d").reset_index()
    return D.groupby(["fault_name", "trial"]).d.mean().reset_index()


print("=" * 64)
print("FINDING 1: partition faults collapse traces<->logs coupling")
print("=" * 64)
tl = delta_by_fault_trial("traces↔logs")
rows = []
for t in TRIALS:
    s = tl[tl.trial == t]
    pm = s[s.fault_name.isin(PART)].d.mean()
    nm = s[~s.fault_name.isin(PART)].d.mean()
    rows.append((t, pm, nm, pm - nm))
pt = pd.DataFrame(rows, columns=["trial", "partition", "non_partition", "diff"])
print(pt.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
n_neg = int((pt["diff"] < 0).sum())
print(f"\n  trials where partition < non-partition: {n_neg}/7")
print(f"  mean difference: {pt['diff'].mean():+.4f}  (SE {pt['diff'].std(ddof=1)/np.sqrt(7):.4f})")
w = stats.wilcoxon(pt.partition, pt.non_partition)
print(f"  paired Wilcoxon signed-rank (n=7): W={w.statistic:.1f}, p={w.pvalue:.4f}")
sgn = stats.binomtest(n_neg, 7, 0.5, alternative="two-sided").pvalue
print(f"  sign test (7/7): p={sgn:.4f}")

# fault-level: average each fault across trials, then test the 2 partition vs 20 others
fl = tl.groupby("fault_name").d.mean()
pv = fl[fl.index.isin(PART)].values
nv = fl[~fl.index.isin(PART)].values
print("\n  fault-level traces<->logs Delta|rho| (avg over 7 trials):")
for k, v in fl[fl.index.isin(PART)].items():
    print(f"    {k:32s} {v:+.4f}")
print(f"    most-negative non-partition fault: {fl[~fl.index.isin(PART)].min():+.4f}")
mw = stats.mannwhitneyu(pv, nv, alternative="less")
print(f"  Mann-Whitney U (partition < others): U={mw.statistic:.1f}, p={mw.pvalue:.4f}")
obs = nv.mean() - pv.mean()
allv = fl.values; k = len(pv); rng = np.random.default_rng(0); N = 50000; cnt = 0
for _ in range(N):
    ip = rng.choice(len(allv), k, replace=False)
    if (np.delete(allv, ip).mean() - allv[ip].mean()) >= obs:
        cnt += 1
print(f"  permutation test (50k, one-sided): p={(cnt+1)/(N+1):.5f}  (obs gap {obs:+.4f})")

print("\n" + "=" * 64)
print("FINDING 2: metrics<->logs Delta|rho| is positive across faults")
print("=" * 64)
ml = delta_by_fault_trial("metrics↔logs").groupby("fault_name").d.mean()
npos = int((ml > 0).sum())
print(f"  faults with positive metrics<->logs Delta|rho|: {npos}/{len(ml)}")
print(f"  mean over faults: {ml.mean():+.4f}  (range {ml.min():+.4f} .. {ml.max():+.4f})")
ws = stats.wilcoxon(ml.values)
print(f"  Wilcoxon signed-rank vs 0: W={ws.statistic:.1f}, p={ws.pvalue:.2e}")
bt = stats.binomtest(npos, len(ml), 0.5, alternative="greater").pvalue
print(f"  binomial sign test ({npos}/{len(ml)} positive): p={bt:.2e}")

print("\n" + "=" * 64)
print("CONTEXT: metrics<->traces weakness")
print("=" * 64)
mt = delta_by_fault_trial("metrics↔traces").groupby("fault_name").d.mean()
print(f"  |metrics<->traces Delta|rho|| < 0.05 for {int((mt.abs()<0.05).sum())}/{len(mt)} faults")
print(f"  cross-fault mean |Delta|rho||: {mt.abs().mean():.4f}")
