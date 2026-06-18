"""Raw shape-stat classifier on the SECURITY faults.

For each signal, z-scores the during window against the pre-fault baseline and
extracts shape stats (mean/std/maxabs/slope). Classes: nas / flood / auth
(6 fault types x 3 trials = 18 instances). Reports per-modality RF and logistic
regression accuracy under LOOCV and leave-one-trial-out.
"""

import json
from pathlib import Path

import _paths
import numpy as np
import pandas as pd
import cross_correlation as cc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

FAULT_CLASS = {
    "nas-registration-storm": "nas", "authentication-exhaustion": "auth",
    "sbi-http2-flood-amf": "flood", "sbi-http2-flood-nrf": "flood",
    "sbi-http2-flood-scp": "flood", "sbi-http2-flood-smf": "flood",
}
META_COLS = ["_y", "_g", "_f"]


def signal_modality(key):
    if key.startswith(("cpu", "mem", "net_rx", "net_tx")):
        return "metrics"
    if key.startswith("jaeger"):
        return "traces"
    if key.startswith("loki"):
        return "logs"
    return None


def shape_stats(z):
    """mean, std, max-abs, and linear slope of a z-scored array."""
    z = np.nan_to_num(np.asarray(z, float))
    if len(z) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    idx = np.arange(len(z))
    slope = float(np.polyfit(idx, z, 1)[0])
    return [float(np.mean(z)), float(np.std(z)), float(np.max(np.abs(z))), slope]


# Build one feature row per (security fault, trial).
rows = []
for fault, label in FAULT_CLASS.items():
    for trial in ["1", "2", "3"]:
        fault_dir = Path(_paths.SECURITY) / fault / trial
        timeline_path = fault_dir / "timeline.json"
        if not timeline_path.exists():
            print("no timeline:", fault, trial)
            continue
        timeline = json.loads(timeline_path.read_text())
        try:
            pre, _, _ = cc.build_signals(fault_dir, "pre", timeline["pre"]["start"], None, None)
            during, _, _ = cc.build_signals(fault_dir, "during", timeline["fault"]["start"], None, None)
        except Exception as err:
            print("skip", fault, trial, err)
            continue
        features = {}
        for key in during:
            if key not in pre or signal_modality(key) is None:
                continue
            pre_arr = np.asarray(pre[key], float)
            dur_arr = np.asarray(during[key], float)
            mean, std = np.nanmean(pre_arr), np.nanstd(pre_arr)
            if np.isnan(mean):
                continue
            z = (dur_arr - mean) / (std if std > 1e-9 else 1.0)
            for stat, val in zip(["mean", "std", "maxabs", "slope"], shape_stats(z)):
                features[f"{key}@@{stat}"] = val
        features["_y"], features["_g"], features["_f"] = label, trial, fault
        rows.append(features)

df = pd.DataFrame(rows).fillna(0.0)
feature_cols = [c for c in df.columns if c not in META_COLS]
y = df["_y"].values
groups = df["_g"].values
print(f"instances={len(df)}  classes={pd.Series(y).value_counts().to_dict()}  "
      f"trials={pd.Series(groups).value_counts().to_dict()}")
print(f"total raw features={len(feature_cols)}")


def cols_for(modality_name):
    if modality_name == "all":
        return feature_cols
    return [c for c in feature_cols if signal_modality(c.split("@@")[0]) == modality_name]


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


majority = max(pd.Series(y).value_counts()) / len(y)
print(f"\nmajority-class baseline = {majority * 100:.1f}%\n")
print(f"{'subset':9}{'#feat':>7}{'RF LOOCV':>10}{'RF LOTO':>9}{'LR LOOCV':>10}{'LR LOTO':>9}")
for modality_name in ["metrics", "logs", "traces", "all"]:
    cols = cols_for(modality_name)
    if not cols:
        print(f"{modality_name:9}{0:>7}  (no features)")
        continue
    X = df[cols].values
    rf_loocv = (cross_val_predict(make_rf(), X, y, cv=LeaveOneOut()) == y).mean()
    rf_loto = (cross_val_predict(make_rf(), X, y, cv=LeaveOneGroupOut(), groups=groups) == y).mean()
    lr_loocv = (cross_val_predict(make_lr(), X, y, cv=LeaveOneOut()) == y).mean()
    lr_loto = (cross_val_predict(make_lr(), X, y, cv=LeaveOneGroupOut(), groups=groups) == y).mean()
    print(f"{modality_name:9}{len(cols):>7}{rf_loocv * 100:>9.1f}{rf_loto * 100:>9.1f}"
          f"{lr_loocv * 100:>10.1f}{lr_loto * 100:>9.1f}")
print("\n[done]")
