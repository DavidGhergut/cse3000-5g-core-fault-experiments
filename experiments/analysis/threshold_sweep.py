"""Sparsity-threshold robustness sweep (10-30%) for both models on aligned features.
Shows classification accuracy is stable across the filter threshold (we report 20%)."""
import sys, re
sys.path.insert(0, "experiments/david")
import numpy as np, pandas as pd
import classify_faults_rf as cf
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]
THRS = [0.10, 0.15, 0.20, 0.25, 0.30]
def rf(s): return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                         class_weight="balanced", random_state=s, n_jobs=-1)
def ms5(X, y, g):
    la, lt = [], []
    for s in range(5):
        la.append((cross_val_predict(rf(s), X, y, cv=LeaveOneOut()) == y).mean())
        lt.append((cross_val_predict(rf(s), X, y, cv=LeaveOneGroupOut(), groups=g) == y).mean())
    return np.mean(la) * 100, np.mean(lt) * 100

print("=== DELTA|rho| sparsity-threshold sweep (5 seeds) ===", flush=True)
paths = [f"data/5GCore/correlations_o5g/{t}/correlations.csv" for t in TRIALS]
cfg = cf.MODE_CONFIGS["operational"]
for mf in THRS:
    X, y, ids, _ = cf.build_feature_matrix(paths, cfg["cats"], cfg["relabel"], None, min_frac=mf)
    g = np.array([re.search(r"\[([^\]]+)\]", s).group(1) for s in ids])
    lo, lt = ms5(X, y, g)
    print(f"  thr={mf:.2f}  {X.shape[1]:4d}f  LOOCV {lo:5.1f}  LOTO {lt:5.1f}", flush=True)

print("=== RAW sparsity-threshold sweep (5 seeds) ===", flush=True)
df = pd.concat([pd.read_pickle(f"/tmp/featcache/{t}.pkl") for t in TRIALS], ignore_index=True)
for mf in THRS:
    X, y, ids, feat = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL, min_frac=mf)
    X = np.asarray(X); y = np.asarray(y)
    g = np.array([re.search(r"\[([^\]]+)\]", s).group(1) if re.search(r"\[([^\]]+)\]", s) else s for s in ids])
    lo, lt = ms5(X, y, g)
    print(f"  thr={mf:.2f}  {X.shape[1]:4d}f  LOOCV {lo:5.1f}  LOTO {lt:5.1f}", flush=True)
print("[done]", flush=True)
