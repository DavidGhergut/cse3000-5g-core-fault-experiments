"""Cache each trial's extended feature table to /tmp/featcache/<trial>.pkl.

Feature extraction is slow, so this loads every trial once (resource +
Open5GS + log + trace features, via classify_extended_features) and pickles
the result. The cross-validation and robustness scripts then read these
pickles instead of re-extracting. Re-run until all trials are cached.
"""

import os

import _paths  # noqa: F401  (anchors sys.path and dataset paths)
import classify_extended_features as ext

CACHE = "/tmp/featcache"
os.makedirs(CACHE, exist_ok=True)

for trial in _paths.TRIALS:
    out = f"{CACHE}/{trial}.pkl"
    if os.path.exists(out):
        print(f"{trial}: already cached, skip", flush=True)
        continue
    root = _paths.trial_dir(trial)
    print(f"{trial}: loading...", flush=True)
    df = ext.load_dataset([root], ext.OPERATIONAL_CATEGORIES, ["mean", "std", "maxabs", "slope"])
    df["dataset"] = trial
    df.to_pickle(out)
    print(f"{trial}: saved {df.shape}", flush=True)

done = [t for t in _paths.TRIALS if os.path.exists(f"{CACHE}/{t}.pkl")]
print(f"\nCACHED {len(done)}/{len(_paths.TRIALS)}: {done}", flush=True)
