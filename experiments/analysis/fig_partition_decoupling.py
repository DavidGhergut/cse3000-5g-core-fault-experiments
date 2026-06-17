#!/usr/bin/env python3
"""Figure for the validated SQ3 finding (per-trial paired view).

For each of the 7 independent-deployment trials, compare the mean traces<->logs coupling
change Delta|rho| of the interface-partition faults (AMF-SCP, N2 AMF-gNB) against the mean of
all non-partition faults. In EVERY trial the partition group decouples more strongly (more
negative) -> this directly visualises the sign test (7/7, p=0.016) that the text reports,
instead of a per-fault ranking where one non-partition fault (memory pressure AMF) misleadingly
outranks a partition fault. Partition-group mean is always to the LEFT of (more negative than)
the non-partition-group mean."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]
BASE = Path("data/5GCore/correlations"); FIG = "/Users/david/Desktop/Research paper_david/figures"
PART = {"05-network-partition-amf-scp", "21-n2-partition-amf-gnb"}
PRETTY = {"boyan": "Trial 1", "boyan-2": "Trial 2", "trial4": "Trial 3", "trial5": "Trial 4",
          "trial6": "Trial 5", "trial7": "Trial 6", "trial8": "Trial 7"}

# per-fault traces<->logs Delta|rho| per trial
fr = []
for t in TRIALS:
    d = pd.read_csv(BASE / t / "correlations.csv"); d["trial"] = t; fr.append(d)
a = pd.concat(fr, ignore_index=True); a["_c"] = pd.to_numeric(a["spearman_r"], errors="coerce")
idx = ["trial", "fault_name", "sig_a", "sig_b", "modality_pair"]
p = a[a.window == "pre"].set_index(idx)["_c"].abs()
q = a[a.window == "during"].set_index(idx)["_c"].abs()
c = p.index.intersection(q.index)
D = (q.loc[c] - p.loc[c]).rename("d").reset_index()
tl = D[D.modality_pair == "traces↔logs"]
ft = tl.groupby(["trial", "fault_name"]).d.mean().reset_index()  # signed mean per (trial, fault)

# per-trial group means: partition vs non-partition
rows = []
for t in TRIALS:
    sub = ft[ft.trial == t]
    part = sub[sub.fault_name.isin(PART)].d
    rest = sub[~sub.fault_name.isin(PART)].d
    if len(part) == 0 or len(rest) == 0:
        continue
    rows.append((t, part.mean(), rest.mean()))
res = pd.DataFrame(rows, columns=["trial", "part", "rest"])
res["diff"] = res["part"] - res["rest"]

n = len(res); wins = int((res["diff"] < 0).sum())  # partition more negative
md = res["diff"].mean(); se = res["diff"].std(ddof=1) / np.sqrt(n)
print(res.to_string(index=False))
print(f"\nn_trials={n}  partition-more-decoupled={wins}/{n}")
print(f"mean difference = {md:.3f}   SE = {se:.3f}   (sign test {wins}/{n})")

# ---- paired dumbbell plot ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.6))
y = np.arange(n)[::-1]
for yi, (_, r) in zip(y, res.iterrows()):
    ax.plot([r["rest"], r["part"]], [yi, yi], color="#bbb", lw=1.4, zorder=1)
ax.scatter(res["rest"], y, s=70, color="#5b8db8", zorder=3, label="Non-partition faults (mean)")
ax.scatter(res["part"], y, s=70, color="#c0392b", zorder=3, label="Partition faults (mean)")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([PRETTY[t] for t in res["trial"]], fontsize=9)
ax.set_xlabel(r"$\leftarrow$ stronger decoupling$\qquad$Traces$\leftrightarrow$Logs  $\Delta|\rho|$$\qquad$coupling $\rightarrow$", fontsize=9)
ax.set_title("Partition faults decouple trace–log coupling more than\nnon-partition faults in every trial", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper center", frameon=True)
ax.tick_params(axis="x", labelsize=8); ax.margins(y=0.08)
ax.text(0.02, 0.04, f"mean difference {md:.2f}, sign test {wins}/{n} (p=0.016)",
        transform=ax.transAxes, fontsize=8, color="#444", style="italic")
plt.tight_layout(); out = f"{FIG}/fig_partition_decoupling.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
