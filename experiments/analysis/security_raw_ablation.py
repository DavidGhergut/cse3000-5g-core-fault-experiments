"""Raw shape-stat classifier on the SECURITY faults (fills the gap Sehan flagged).
Per signal: z-score during vs pre baseline -> shape stats (mean/std/maxabs/slope).
Classes: nas / flood / auth. 6 fault types x 3 trials = 18 instances. LOOCV + LOTO (by trial).
"""
import sys, os, json
sys.path.insert(0, "experiments/david")
import numpy as np, pandas as pd
from pathlib import Path
import cross_correlation as cc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

SEC = "data/5GCore/final_dataset/security_faults"
CLASS = {"nas-registration-storm":"nas", "authentication-exhaustion":"auth",
         "sbi-http2-flood-amf":"flood", "sbi-http2-flood-nrf":"flood",
         "sbi-http2-flood-scp":"flood", "sbi-http2-flood-smf":"flood"}

def modality(k):
    if k.startswith(("cpu","mem","net_rx","net_tx")): return "metrics"
    if k.startswith("jaeger"): return "traces"
    if k.startswith("loki"): return "logs"
    return None

def shape_stats(z):
    z = np.nan_to_num(np.asarray(z, float))
    if len(z) < 2: return [0.0,0.0,0.0,0.0]
    t = np.arange(len(z))
    slope = float(np.polyfit(t, z, 1)[0])
    return [float(np.mean(z)), float(np.std(z)), float(np.max(np.abs(z))), slope]

rows = []
for fault, cls in CLASS.items():
    for trial in ["1","2","3"]:
        fdir = Path(SEC)/fault/trial
        tl = fdir/"timeline.json"
        if not tl.exists(): print("no timeline:", fault, trial); continue
        tj = json.loads(tl.read_text())
        try:
            pre,_,_ = cc.build_signals(fdir, "pre", tj["pre"]["start"], None, None)
            dur,_,_ = cc.build_signals(fdir, "during", tj["fault"]["start"], None, None)
        except Exception as e:
            print("skip", fault, trial, e); continue
        feats = {}
        for k in dur:
            if k not in pre or modality(k) is None: continue
            p = np.asarray(pre[k], float); d = np.asarray(dur[k], float)
            mu, sd = np.nanmean(p), np.nanstd(p)
            if np.isnan(mu): continue
            z = (d - mu) / (sd if sd > 1e-9 else 1.0)
            for st, v in zip(["mean","std","maxabs","slope"], shape_stats(z)):
                feats[f"{k}@@{st}"] = v
        feats["_y"] = cls; feats["_g"] = trial; feats["_f"] = fault
        rows.append(feats)

df = pd.DataFrame(rows).fillna(0.0)
meta = ["_y","_g","_f"]
featcols = [c for c in df.columns if c not in meta]
y = df["_y"].values; g = df["_g"].values
print(f"instances={len(df)}  classes={pd.Series(y).value_counts().to_dict()}  trials={pd.Series(g).value_counts().to_dict()}")
print(f"total raw features={len(featcols)}")

def cols_for(mod):
    return featcols if mod=="all" else [c for c in featcols if modality(c.split("@@")[0])==mod]
def mkrf(): return RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
def mklr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))

maj = max(pd.Series(y).value_counts())/len(y)
print(f"\nmajority-class baseline = {maj*100:.1f}%\n")
print(f"{'subset':9}{'#feat':>7}{'RF LOOCV':>10}{'RF LOTO':>9}{'LR LOOCV':>10}{'LR LOTO':>9}")
for mod in ["metrics","logs","traces","all"]:
    cols = cols_for(mod)
    if not cols: print(f"{mod:9}{0:>7}  (no features)"); continue
    X = df[cols].values
    rl = (cross_val_predict(mkrf(), X, y, cv=LeaveOneOut())==y).mean()
    rt = (cross_val_predict(mkrf(), X, y, cv=LeaveOneGroupOut(), groups=g)==y).mean()
    ll = (cross_val_predict(mklr(), X, y, cv=LeaveOneOut())==y).mean()
    lt = (cross_val_predict(mklr(), X, y, cv=LeaveOneGroupOut(), groups=g)==y).mean()
    print(f"{mod:9}{len(cols):>7}{rl*100:>9.1f}{rt*100:>9.1f}{ll*100:>10.1f}{lt*100:>9.1f}")
print("\n[done]")
