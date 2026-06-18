"""Leave-one-FAULT-out: the unseen-fault generalization test.

Holds out all instances of one fault scenario (across all trials), trains on the
remaining scenarios, and checks whether the held-out scenario's instances are
classified into the correct category. Tests whether the model can categorize a
fault scenario it has never seen from its category-neighbours' signatures.
"""

import os
import re
from collections import Counter

import _paths
import numpy as np
import pandas as pd
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier

IDCACHE = "/tmp/op_lofo_ids.npz"

# Build (or load) the feature matrix with per-instance fault and trial labels.
if os.path.exists(IDCACHE):
    cached = np.load(IDCACHE, allow_pickle=True)
    X, y = cached["X"], cached["y"]
    fault, trial, feature_names = cached["fault"], cached["trial"], list(cached["feat"])
else:
    frames = []
    for t in _paths.TRIALS:
        print(f"  loading {t}", flush=True)
        df = ext.load_dataset([_paths.trial_dir(t)], ext.OPERATIONAL_CATEGORIES,
                              ["mean", "std", "maxabs", "slope"])
        df["dataset"] = t
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    X, y, ids, feature_names = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
    X, y = np.asarray(X), np.asarray(y)
    fault = np.array([i.split(" [")[0] for i in ids])
    trial = np.array([re.search(r"\[([^\]]+)\]", i).group(1) for i in ids])
    np.savez(IDCACHE, X=X, y=y, fault=fault, trial=trial, feat=np.array(feature_names, dtype=object))

print(f"n={len(y)} instances, p={X.shape[1]} features, {len(set(fault))} fault scenarios\n")


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


# Hold out each fault scenario in turn, predict its category from the rest.
all_preds = np.empty(len(y), dtype=object)
rows = []
for fault_name in sorted(set(fault)):
    test = fault == fault_name
    train = ~test
    clf = make_rf().fit(X[train], y[train])
    pred = clf.predict(X[test])
    all_preds[test] = pred
    true_cat = y[test][0]
    pred_counts = ", ".join(f"{cat}:{count}" for cat, count in Counter(pred).most_common())
    rows.append((fault_name, true_cat, test.sum(), (pred == true_cat).mean(), pred_counts))

print(f"{'held-out fault':<42}{'true cat':<14}{'n':>3}{'correct':>9}   predicted")
for fault_name, cat, count, acc, pred_counts in rows:
    print(f"{fault_name:<42}{cat:<14}{count:>3}{acc * 100:>8.0f}%   {pred_counts}")

overall = (all_preds == y).mean()
print(f"\nOverall leave-one-FAULT-out CATEGORY accuracy: {overall * 100:.1f}%")
print("per-category recall under LOFO:")
for cat in sorted(set(y)):
    mask = y == cat
    print(f"  {cat:<16}{(all_preds[mask] == y[mask]).mean() * 100:>5.0f}%   (n={mask.sum()})")
