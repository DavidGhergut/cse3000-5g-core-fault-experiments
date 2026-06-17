"""Operational RAW modality ablation for the main-body SQ4 table:
singles + pairs + all, RF, LOOCV + LOTO. Correct per-trial labels (7 trials).
"""
import sys, os, re
sys.path.insert(0, "experiments/david")
import numpy as np, pandas as pd
from pathlib import Path
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

TRIALS = ["boyan","boyan-2","trial4","trial5","trial6","trial7","trial8"]
def trial_of(s): m = re.search(r"\[([^\]]+)\]", s); return m.group(1) if m else s
CACHE = "/tmp/op_raw_ablation.npz"

if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True); X,y,g,feat = d["X"],d["y"],d["g"],list(d["feat"]); print("cache",flush=True)
else:
    frames = []
    for t in TRIALS:
        root = None
        for c in [f"data/5GCore/final_dataset/{t}/C-fault-detection", f"data/5GCore/final_dataset/{t}"]:
            if os.path.isdir(c): root = Path(c); break
        print(f"  loading {t}", flush=True)
        df = ext.load_dataset([root], ext.OPERATIONAL_CATEGORIES, ["mean","std","maxabs","slope"])
        df["dataset"] = t
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    X, y, ids, feat = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
    X = np.asarray(X); y = np.asarray(y); g = np.array([trial_of(s) for s in ids])
    np.savez(CACHE, X=X, y=y, g=g, feat=np.array(feat, dtype=object))
print("trials:", pd.Series(g).value_counts().to_dict(), "n=", len(y), "feat=", len(feat), flush=True)

def mof(c):
    m = ext.modality_of(c); return "metrics" if m == "rtt" else m
mod = np.array([mof(c) for c in feat])
def cols(subset):
    want = set(subset.split("+")); return [i for i,m in enumerate(mod) if m in want]
def mkrf(): return RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)

maj = max(pd.Series(y).value_counts())/len(y)
print(f"majority-class baseline {maj*100:.1f}%\n")
print(f"{'subset':16}{'#feat':>6}{'LOOCV':>8}{'LOTO':>8}")
for s in ["metrics","logs","traces","metrics+logs","metrics+traces","logs+traces","metrics+logs+traces"]:
    ci = cols(s); X2 = X[:, ci]
    lo = (cross_val_predict(mkrf(), X2, y, cv=LeaveOneOut())==y).mean()
    lt = (cross_val_predict(mkrf(), X2, y, cv=LeaveOneGroupOut(), groups=g)==y).mean()
    label = "ALL" if s == "metrics+logs+traces" else s
    print(f"{label:16}{len(ci):>6}{lo*100:>7.1f}{lt*100:>7.1f}")
print("[done]")
