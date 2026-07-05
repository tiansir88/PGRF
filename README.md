# PGRF: Prior-Guided Recurrent Residual Fusion

This repository contains the anonymized code for reproducing the PGRF experiments for serial-ECG mortality prediction.

The repository includes source code for cohort construction, frozen-backbone ECG sequence extraction, PGRF training, baselines, ablations, bootstrap analysis, high-risk stratification, per-endpoint evaluation, and paper figure generation. It does not include ECG waveforms, clinical tables, cached embeddings, checkpoints, logs, or generated outputs.

## Repository structure

```text
configs/                       Example path configuration
docs/                          Reproduction notes
scripts/
  01_data_preparation/          Cohort construction and leakage audits
  02_backbone_embeddings/       Frozen ECG backbone sequence-cache builders
  03_training/                  PGRF training, baselines, and controls
  04_ablation/                  PGRF module ablation
  05_evaluation/                Main tables, bootstrap, subgroup, endpoint analyses
  06_figures/                   Vector PDF figure generation
external_model_code/            Minimal MELP/MERL wrapper code used by the embedding script
```

## Environment

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Backbone-specific embedding extraction additionally requires the corresponding public code/checkpoints for ST-MEM, ECG-FM, CLEAR-HUG, MERL, and MELP. These weights are not redistributed.

## Path configuration

Before running the scripts, set:

```bash
export PGRF_PROJECT_DIR=/path/to/pgrf_project
export PGRF_BASE_DIR=/path/to/pgrf_project/training_manifest
export PGRF_MIMIC_ECG_ROOT=/path/to/mimic-iv-ecg
```

See `configs/default_paths.example.sh`.

## Reproduction order

```bash
# 1. Build temporal cohort and audit prediction time.
python scripts/01_data_preparation/build_temporal_cache.py
python scripts/01_data_preparation/temporal_audit.py
python scripts/01_data_preparation/audit_prediction_time.py

# 2. Build frozen-backbone ECG sequence caches.
python scripts/02_backbone_embeddings/build_foundation_sequence_cache.py --model stmem
python scripts/02_backbone_embeddings/build_foundation_sequence_cache.py --model ecgfm
python scripts/02_backbone_embeddings/build_clearhug_sequence_cache.py
python scripts/02_backbone_embeddings/build_merl_melp_sequence_cache.py --model merl
python scripts/02_backbone_embeddings/build_merl_melp_sequence_cache.py --model melp

# 3. Build the longitudinal-eligible cohort.
python scripts/02_backbone_embeddings/build_longitudinal_caches_fast.py

# 4. Train PGRF.
python scripts/03_training/train_pgrf.py

# 5. Run baselines, controls, and ablation.
python scripts/03_training/run_pooling_baselines.py
python scripts/03_training/run_utilization_control.py
python scripts/03_training/build_shuffled_history_cache.py
python scripts/03_training/summarize_shuffled_history_control.py
python scripts/04_ablation/run_ablation_stmem_temporal.py

# Optional: repeat the same key ablation on another backbone, e.g. CLEAR-HUG.
BACKBONE=CLEAR-HUG CACHE=/path/to/clearhug_temporal_count_ge2.npz \
RUN_NAME=pgrf_ablation_clearhug_temporal_v1 \
python scripts/04_ablation/run_ablation_stmem_temporal.py

# 6. Generate tables and figures.
python scripts/05_evaluation/create_ensemble_table.py
python scripts/05_evaluation/create_ensemble_bootstrap.py
python scripts/05_evaluation/create_patient_clustered_bootstrap.py
python scripts/05_evaluation/create_toprisk_history.py
python scripts/05_evaluation/create_per_endpoint_results.py
python scripts/05_evaluation/compute_history_subgroups.py
python scripts/06_figures/draw_patient_bootstrap_endpoint_figure.py
```

## Double-blind note

This repository is anonymized for review. It contains no author names, institutional paths, server addresses, passwords, checkpoints, logs, or private data.
