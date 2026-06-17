"""Leave-one-FAULT-out: the novel-fault generalization test. Hold out ALL
instances of one fault scenario (across all 7 trials), train on the other 21
scenarios, and check whether the held-out scenario's instances are classified
into the correct CATEGORY. Tests whether the model can categorize a fault
scenario it has never seen, from the signatures of its category-neighbours."""
import sys, os, re
sys.path.insert(0, "experiments/david")
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier

IDCACHE = "/tmp/op_lofo_ids.npz"
if os.path.exists(IDCACHE):
    d = np.load(IDCACHE, allow_pickle=True)
    X, y, fault, trial, feat = d["X"], d["y"], d["fault"], d["trial"], list(d["feat"])
else:
    TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]
    frames = []
    for t in TRIALS:
        root = None
        for c in [f"data/5GCore/final_dataset/{t}/C-fault-detection", f"data/5GCore/final_dataset/{t}"]:
            if os.path.isdir(c):
                root = Path(c); break
        print(f"  loading {t}", flush=True)
        df = ext.load_dataset([root], ext.OPERATIONAL_CATEGORIES, ["mean", "std", "maxabs", "slope"])
        df["dataset"] = t; frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    X, y, ids, feat = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
    X = np.asarray(X); y = np.asarray(y)
    fault = np.array([s.split(" [")[0] for s in ids])
    trial = np.array([re.search(r"\[([^\]]+)\]", s).group(1) for s in ids])
    np.savez(IDCACHE, X=X, y=y, fault=fault, trial=trial, feat=np.array(feat, dtype=object))

print(f"n={len(y)} instances, p={X.shape[1]} features, {len(set(fault))} fault scenarios\n")


def mkrf():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


allpred = np.empty(len(y), dtype=object)
rows = []
for f in sorted(set(fault)):
    te = fault == f; tr = ~te
    clf = mkrf().fit(X[tr], y[tr])
    pred = clf.predict(X[te]); allpred[te] = pred
    true = y[te][0]
    pc = ", ".join(f"{k}:{v}" for k, v in Counter(pred).most_common())
    rows.append((f, true, te.sum(), (pred == true).mean(), pc))

print(f"{'held-out fault':<42}{'true cat':<14}{'n':>3}{'correct':>9}   predicted")
for f, cat, n, acc, pc in rows:
    print(f"{f:<42}{cat:<14}{n:>3}{acc*100:>8.0f}%   {pc}")

overall = (allpred == y).mean()
print(f"\nOverall leave-one-FAULT-out CATEGORY accuracy: {overall*100:.1f}%")
print("per-category recall under LOFO:")
for c in sorted(set(y)):
    m = y == c
    print(f"  {c:<16}{(allpred[m]==y[m]).mean()*100:>5.0f}%   (n={m.sum()})")
