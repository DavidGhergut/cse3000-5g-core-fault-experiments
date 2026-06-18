# Multi-Modal Correlation of Observability Signals in a Cloud-Native 5G Core

This repository accompanies a TU Delft bachelor research project (CSE3000) on **cross-modal
correlation of observability signals** (metrics, logs, and distributed traces) for
characterizing faults in a cloud-native Open5GS 5G Core.

It has two parts:

1. **Testbed and experiment collection.** An Open5GS 5G Core on Kubernetes (kind), faulted with
   Chaos Mesh and observed through Prometheus (metrics), Grafana Loki (logs), and Grafana
   Beyla → Jaeger (traces), plus Kubernetes events, the NRF API, and UE round-trip time. Every
   fault is collected per phase (`pre`, `during`, `post`).
2. **Cross-modal correlation analysis** (`experiments/analysis/`). The pipeline that bins the
   three modalities into synchronized windows, computes the fault-induced change in Spearman
   correlation between cross-modal signal pairs (Δ|ρ|), and uses it to characterize faults and
   locate modality blindspots. This is the methodology behind the accompanying research paper.

## Layout

```
cluster-start.sh                     Recreate the kind cluster and deploy the full stack
kind/
  kind-config.yaml                   kind node config
  open5gs-values.yaml                Open5GS + UERANSIM Helm values
  monitoring/beyla-daemonset.yaml    eBPF (Beyla) span/metric collection
  chaos/*.yaml                       22 Chaos Mesh operational fault manifests
experiments/
  fault-detection/run_all.sh         Entry point; runs all 22 operational faults in sequence
  security/                          Security fault injection (run from inside the cluster)
    inject_*.sh                      HTTP/2 SBI floods (AMF/NRF/SCP/SMF), NAS registration storm,
                                     AUSF authentication exhaustion
    run_all_security.sh, run_security_experiment.sh
  analysis/                          Cross-modal correlation analysis (see below)
  lib/
    common.sh, run_fault.sh, traffic.sh, health_check.sh, provision_ues.sh
    collect_*.py, collect_ue_rtt.sh  Per-signal collectors
    hooks/<fault>.sh                 Optional per-fault setup/teardown hooks
```

## Prerequisites

- Linux with Docker, [`kind`](https://kind.sigs.k8s.io/), `kubectl`, and `helm`
- `python3` (the collectors use only the standard library)
- `sudo` for the one-time Docker iptables-chain fix
- `curl`, `lsof`, `awk`

### Docker Hub auth (required)

Recreating the cluster per fault makes many image pulls and will hit Docker Hub's anonymous limit
of 100 pulls per 6 hours. Create a gitignored auth file with a read-only personal access token:

```
kind/.dockerhub-auth      line 1: Docker Hub username
                          line 2: read-only PAT
```

`cluster-start.sh` injects it into the kind containerd config at runtime
(`kind/.kind-config.runtime.yaml`, also gitignored); the token is never committed.

## Run the experiments

```bash
cd experiments/fault-detection

# All 22 operational faults (long; run inside tmux/screen)
bash run_all.sh

# Resume from fault N (e.g. after a gate failure)
bash run_all.sh --from 7

# Run only specific faults
bash run_all.sh --only 19,20
```

Phase durations are env-overridable (`PRE_DURATION=600 FAULT_DURATION=300 POST_DURATION=300`).
Each fault recreates the cluster via `cluster-start.sh`, gates on NF readiness, injects, and
collects per phase. Security faults are injected from inside the running cluster via the scripts
in `experiments/security/`.

Each fault produces `prometheus/`, `jaeger/`, `loki/`, `events/`, `nrf/`, and `rtt/` subtrees
split into `pre/`, `during/`, `post/`, plus `health_pre.json`, `health_post.json`, and `meta.json`.

## Cross-modal correlation analysis (`experiments/analysis/`)

The analysis takes the collected per-phase telemetry and reproduces the paper's figures and
tables. Extract the dataset (see below) into `experiments/analysis/final_dataset/`, install the
dependencies with `pip install -r experiments/analysis/requirements.txt`, and run the whole
pipeline with `bash experiments/analysis/reproduce.sh`. See `experiments/analysis/README.md` for
the per-script breakdown. The main components:

- **Correlation pipeline:** `cross_correlation.py` (operational) and `regen_security_o5g.py`
  (security) compute the per-fault cross-modal Δ|ρ|; `classify_faults_rf.py` and
  `classify_extended_features.py` build the Δ|ρ| and raw shape-stat feature matrices.
- **Classification & ablations:** `compute30_cv.py`, `op_raw_ablation.py`, `raw_ablation_f1.py`,
  `compute30_delta_abl.py`, `multi_classifier_sq4.py` (RF / logistic regression / gradient
  boosting), `security_raw_ablation.py`, `lofo.py` (leave-one-fault-out), `fold_topk_selection.py`,
  `bin_sweep.py`, `threshold_sweep.py` (robustness).
- **Figures:** `heatmaps_o5g.py` (per-category coupling heatmap), `fig_partition_decoupling.py`
  (trace↔log decoupling under partitions), `fig_shap_intershap.py` + `raw_intershap.py` (attribution),
  `learning_curve.py`.
- **Statistics:** `significance_tests.py` (sign test / Mann–Whitney for the partition finding),
  `security_log_volume.py` (per-attack error-log volume → the security log-blindspot).

## Dataset

The full multi-modal fault dataset (metrics, logs, traces, events, RTT for every fault, per phase)
is published as a release of this repository:
**[dataset-v1](https://github.com/DavidGhergut/cse3000-5g-core-fault-experiments/releases/tag/dataset-v1)**.
Extract it into `experiments/analysis/final_dataset/` to run the analysis.

## Contribution

This repository was made for the course **CSE3000: Research Project** at TU Delft, under the topic
*Observability for Intelligent Fault Management in Cloud-native Beyond 5G Networks*. The 5G Core
testbed and experiment-collection pipeline were developed jointly by the project team (Boyan Bonev,
David Ghergut, Yana Mihaylova, Stoyan Kutsarov, and Victor Ilchev). The individual contribution in
this repository is the **cross-modal correlation analysis** (`experiments/analysis/`) by
**David Ghergut**: characterizing 5G Core faults by how the coupling between metrics, logs, and
traces changes during a fault, and identifying the architecture-grounded modality blindspots that
follow from it.
