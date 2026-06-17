"""Error-log volume per security attack -> reproduces tab:security_logblind.

Counts the non-header lines in loki/during/errors.csv for each security attack, averaged over
its three trials. The NRF SBI flood is a valid-request attack and produces ZERO error logs
(the security-side log blindspot, mirror of the operational trace blindspot).

Usage:  python3 security_log_volume.py [path/to/security_faults]
        (defaults to data/5GCore/final_dataset/security_faults under the dataset release)
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/5GCore/final_dataset/security_faults")
ATTACKS = [
    "nas-registration-storm", "sbi-http2-flood-amf", "sbi-http2-flood-nrf",
    "sbi-http2-flood-scp", "sbi-http2-flood-smf", "authentication-exhaustion",
]

print(f"{'attack':<28}{'mean error-log lines (during)':>30}")
print("-" * 58)
for a in ATTACKS:
    counts = []
    for t in ("1", "2", "3"):
        f = ROOT / a / t / "loki" / "during" / "errors.csv"
        if f.exists():
            counts.append(max(sum(1 for _ in f.open()) - 1, 0))  # minus header
    if counts:
        print(f"{a:<28}{round(sum(counts) / len(counts)):>30}")
