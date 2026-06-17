"""Multi-classifier robustness for SQ4: RF / LogisticRegression / GradientBoosting on
raw shape-stat features vs Delta|rho| coupling features, single-modality (metrics) vs multi-modal,
under LOOCV and LOTO. Reproduces the paper's tab:cv multi-classifier claim with saved code.
Purpose: show (1) raw >> Delta|rho| across classifiers, (2) multi-modal ~= metrics across classifiers."""
import sys, os, re
sys.path.insert(0, "experiments/david")
import numpy as np
from pathlib import Path
import classify_extended_features as ext
import classify_faults_rf as rf
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

SEED = 42
TRIALS = ["boyan","boyan-2","trial4","trial5","trial6","trial7","trial8"]
def trial_of(s):
    m = re.search(r"\[([^\]]+)\]", s); return m.group(1) if m else s.split()[-1]

def classifiers():
    return {
        "RandomForest":     RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                                    class_weight="balanced", random_state=SEED, n_jobs=-1),
        "LogisticReg":      make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=3000, class_weight="balanced")),
        "GradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    }

def evaluate(X, y, groups, name):
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    print(f"\n  [{name}]  ({X.shape[0]} inst, {X.shape[1]} feat)")
    print(f"    {'classifier':<18}{'LOOCV':>9}{'LOTO':>9}")
    for cname, clf in classifiers().items():
        lo = (cross_val_predict(clf, X, y, cv=LeaveOneOut()) == y).mean()
        lt = (cross_val_predict(clf, X, y, cv=LeaveOneGroupOut(), groups=groups) == y).mean()
        print(f"    {cname:<18}{lo*100:>7.1f}%{lt*100:>8.1f}%")

# ---------- RAW features (paper construction: classify_extended_features, 8 NFs) ----------
ROOTS = []
for t in TRIALS:
    for c in [f"data/5GCore/final_dataset/{t}/C-fault-detection", f"data/5GCore/final_dataset/{t}"]:
        if os.path.isdir(c): ROOTS.append(Path(c)); break
print("loading raw shape-stat features ...", flush=True)
import pandas as pd
df = pd.concat([ext.load_dataset([r], ext.OPERATIONAL_CATEGORIES, ["mean","std","maxabs","slope"]) for r in ROOTS], ignore_index=True)
Xr, yr, idr, featr = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
gr = np.array([trial_of(s) for s in idr])
def mof(c):
    m = ext.modality_of(c); return "metrics" if m == "rtt" else m
metric_cols = [i for i, c in enumerate(featr) if mof(c) == "metrics"]

print("="*48)
print("RAW SHAPE-STAT FEATURES")
print("="*48)
evaluate(Xr, yr, gr, "raw — ALL modalities (multi-modal)")
evaluate(np.asarray(Xr)[:, metric_cols], yr, gr, "raw — metrics only (single-modality)")

# ---------- Delta|rho| features ----------
paths = [f"data/5GCore/correlations/{t}/correlations.csv" for t in TRIALS]
cfg = rf.MODE_CONFIGS["operational"]
out = rf.build_feature_matrix(paths, cfg["cats"], cfg["relabel"], rf.MODAL_SETS["cross"])
Xd, yd, idd = out[0], out[1], list(out[2])
gd = np.array([trial_of(s) for s in idd])
print("\n" + "="*48)
print("Δ|ρ| CROSS-MODAL COUPLING FEATURES")
print("="*48)
evaluate(Xd, yd, gd, "Δ|ρ| — all cross-modal")

print("\nseed=42, single run; HistGradientBoosting used for 'gradient boosting'.")
print("[done]")
