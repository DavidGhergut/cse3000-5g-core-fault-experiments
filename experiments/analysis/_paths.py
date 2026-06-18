"""Shared path configuration for the cross-modal analysis scripts.

Every analysis script imports from here so it runs correctly regardless of the
current working directory. By default the dataset is expected alongside these
scripts: extract the dataset release so that the trial directories live under

    experiments/analysis/final_dataset/<trial>/

Set the DATA_ROOT environment variable to point somewhere else.
"""

import os
import sys
from pathlib import Path

# Directory that contains the analysis scripts.
HERE = Path(__file__).resolve().parent

# Make sibling modules (cross_correlation, ccf_profile, ...) importable no
# matter where a script is launched from.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Root under which the dataset and generated artifacts live.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", HERE)).resolve()

FINAL_DATASET = DATA_ROOT / "final_dataset"
CORRELATIONS = DATA_ROOT / "correlations_o5g"
SECURITY = FINAL_DATASET / "security_faults"

# Where generated figures and tables are written.
OUTPUT = HERE / "output"

# The seven operational trials, in canonical order.
TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]


def trial_dir(trial):
    """Return a trial's fault directory, handling the C-fault-detection subdir."""
    base = FINAL_DATASET / trial
    nested = base / "C-fault-detection"
    return nested if nested.is_dir() else base


def corr_csv(trial):
    """Return the path to a trial's cross_correlation.py output."""
    return CORRELATIONS / trial / "correlations.csv"
