"""tab:cv at the full 30 seeds (matches the paper's stated protocol): Delta|rho|, raw, combined,
LOOCV + LOTO, mean +/- std. Matrices built once; foreground."""
import sys, re
sys.path.insert(0, "experiments/david")
import numpy as np, pandas as pd
import classify_faults_rf as cf
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import f1_score

TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]
SEEDS = 10
def rf(s): return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                         class_weight="balanced", random_state=s, n_jobs=-1)
def report(name, X, y, g):
    la, lf, ta, tf = [], [], [], []
    for s in range(SEEDS):
        p = cross_val_predict(rf(s), X, y, cv=LeaveOneOut()); la.append((p == y).mean()); lf.append(f1_score(y, p, average="macro"))
        q = cross_val_predict(rf(s), X, y, cv=LeaveOneGroupOut(), groups=g); ta.append((q == y).mean()); tf.append(f1_score(y, q, average="macro"))
    print(f"[{name}] {X.shape[1]}f  LOOCV {np.mean(la)*100:.1f}±{np.std(la)*100:.1f} F1 {np.mean(lf):.2f} "
          f"| LOTO {np.mean(ta)*100:.1f}±{np.std(ta)*100:.1f} F1 {np.mean(tf):.2f}", flush=True)

# delta (cross, 160f)
Xd, yd, ids_d, _ = cf.build_feature_matrix(
    [f"data/5GCore/correlations_o5g/{t}/correlations.csv" for t in TRIALS],
    cats=["infrastructure", "network", "pfcp", "pod_crash"], relabel={"cascade": "pod_crash"})
gd = np.array([re.search(r"\[([^\]]+)\]", s).group(1) for s in ids_d])
report("delta", Xd, yd, gd)
# raw
d = np.load("/tmp/op_raw_ablation.npz", allow_pickle=True)
Xr, yr, gr = d["X"], d["y"], d["g"]
report("raw", Xr, yr, gr)
# combined (align delta to raw order)
df = pd.concat([pd.read_pickle(f"/tmp/featcache/{t}.pkl") for t in TRIALS], ignore_index=True)
_, _, ids_r, _ = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
pos = {fid: i for i, fid in enumerate(ids_d)}
report("combined", np.hstack([Xr, Xd[[pos[fid] for fid in ids_r]]]), yr, gr)
print("[done]", flush=True)
