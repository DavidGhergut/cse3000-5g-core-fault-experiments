"""Learning curve: does adding more trials (deployments) raise held-out-trial
accuracy, or does it plateau?

For each held-out test trial, train on every subset of size k of the remaining
trials (exhaustive), and test on the held-out trial. Plots mean test accuracy vs
k, for both the raw shape-stat features and the Delta|rho| coupling features.
"""

import os
import re
import itertools
import collections

import _paths
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import classify_extended_features as ext
import classify_faults_rf as cf
from sklearn.ensemble import RandomForestClassifier

RAW_CACHE = "/tmp/lc_raw.npz"


def trial_of(fault_id):
    match = re.search(r"\[([^\]]+)\]", fault_id)
    return match.group(1) if match else fault_id


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


# Raw shape-stat features (cached, with correct per-trial labels).
if os.path.exists(RAW_CACHE):
    cached = np.load(RAW_CACHE, allow_pickle=True)
    X_raw, y_raw, groups_raw = cached["X"], cached["y"], cached["g"]
    print("raw cache loaded", flush=True)
else:
    frames = []
    for t in _paths.TRIALS:
        print(f"  loading {t}", flush=True)
        df = ext.load_dataset([_paths.trial_dir(t)], ext.OPERATIONAL_CATEGORIES,
                              ["mean", "std", "maxabs", "slope"])
        df["dataset"] = t
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    X_raw, y_raw, ids_raw, _ = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
    X_raw, y_raw = np.asarray(X_raw), np.asarray(y_raw)
    groups_raw = np.array([trial_of(i) for i in ids_raw])
    np.savez(RAW_CACHE, X=X_raw, y=y_raw, g=groups_raw)
print("RAW trials:", pd.Series(groups_raw).value_counts().to_dict(), flush=True)

# Delta|rho| coupling features (read directly from the correlation CSVs).
corr_paths = [str(_paths.corr_csv(t)) for t in _paths.TRIALS if _paths.corr_csv(t).exists()]
cfg = cf.MODE_CONFIGS["operational"]
X_delta, y_delta, ids_delta, _ = cf.build_feature_matrix(
    corr_paths, cfg["cats"], cfg["relabel"], cf.MODAL_SETS["cross"])
X_delta, y_delta = np.asarray(X_delta), np.asarray(y_delta)
groups_delta = np.array([trial_of(i) for i in ids_delta])
print("Δ|ρ| trials:", pd.Series(groups_delta).value_counts().to_dict(), flush=True)


def compute_curve(X, y, groups):
    """Mean/std held-out-trial accuracy for each training-set size k."""
    trials = sorted(set(groups))
    curve = collections.defaultdict(list)
    for held_out in trials:
        test_mask = groups == held_out
        others = [t for t in trials if t != held_out]
        for k in range(1, len(others) + 1):
            for combo in itertools.combinations(others, k):
                train_mask = np.isin(groups, combo)
                clf = make_rf().fit(X[train_mask], y[train_mask])
                curve[k].append((clf.predict(X[test_mask]) == y[test_mask]).mean())
    ks = sorted(curve)
    return ks, [np.mean(curve[k]) * 100 for k in ks], [np.std(curve[k]) * 100 for k in ks]


print("computing raw learning curve...", flush=True)
ks_raw, mean_raw, std_raw = compute_curve(X_raw, y_raw, groups_raw)
print("computing Δ|ρ| learning curve...", flush=True)
ks_delta, mean_delta, std_delta = compute_curve(X_delta, y_delta, groups_delta)

print("\n# train trials | RAW acc | Δ|ρ| acc")
for i, k in enumerate(ks_raw):
    print(f"  {k}  |  {mean_raw[i]:.1f}±{std_raw[i]:.1f}  |  {mean_delta[i]:.1f}±{std_delta[i]:.1f}")

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.errorbar(ks_raw, mean_raw, yerr=std_raw, marker="o", capsize=3, color="#2c6fbb",
            label="Raw shape-stat features")
ax.errorbar(ks_delta, mean_delta, yerr=std_delta, marker="s", capsize=3, color="#c0392b",
            label="Δ|ρ| coupling features")
ax.set_xlabel("Number of training deployments (trials)", fontsize=10)
ax.set_ylabel("Held-out-trial test accuracy (%)", fontsize=10)
ax.set_title("Test accuracy vs training deployments: raw still rising, Δ|ρ| plateauing",
             fontsize=10.5, fontweight="bold")
ax.axhline(32, ls=":", color="grey", lw=1)
ax.text(1, 33, "majority-class baseline (32%)", fontsize=7, color="grey")
ax.set_xticks(ks_raw)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="center right")
plt.tight_layout()
_paths.OUTPUT.mkdir(parents=True, exist_ok=True)
out_path = _paths.OUTPUT / "fig_learning_curve.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("\nwrote", out_path)
print("[done]")
