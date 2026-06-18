"""Generate the security-fault Delta|rho| correlations (with Open5GS app metrics).

Runs cross_correlation.analyze_fault over the security layout
security_faults/<fault>/<trial>/ and writes correlations_o5g/security/correlations.csv
(the operational schema plus a trial column). Consumed by compute30_delta_abl.py.
"""

import _paths  # noqa: F401  (anchors sys.path and dataset paths)
import pandas as pd
import cross_correlation as cc

OUT = _paths.CORRELATIONS / "security"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORY = {
    "authentication-exhaustion": "security_auth",
    "nas-registration-storm": "security_nas",
    "sbi-http2-flood-amf": "security_flood",
    "sbi-http2-flood-nrf": "security_flood",
    "sbi-http2-flood-scp": "security_flood",
    "sbi-http2-flood-smf": "security_flood",
}

rows = []
for fault in sorted(_paths.SECURITY.iterdir()):
    if not fault.is_dir() or fault.name not in CATEGORY:
        continue
    for trial in sorted(fault.iterdir()):
        if not trial.is_dir() or not (trial / "timeline.json").exists():
            continue
        print(f"[sec] {fault.name}/{trial.name} ...", end=" ", flush=True)
        result = cc.analyze_fault(trial)
        for x in result:
            x["fault_name"] = fault.name
            x["trial"] = trial.name
            x["fault_category"] = CATEGORY[fault.name]
        rows.extend(result)
        print(f"{len(result)} pairs", flush=True)

df = pd.DataFrame(rows)
out_csv = OUT / "correlations.csv"
df.to_csv(out_csv, index=False)
print(f"\n[done] {len(df)} pairs -> {out_csv}", flush=True)
