# Reproduction notes

## Data

The code assumes access to MIMIC-IV-ECG and linked MIMIC-IV clinical metadata. These datasets are not redistributed. Users should place the waveform and metadata files under their local paths and expose those paths through the environment variables documented in `configs/default_paths.example.sh`.

## Backbones

PGRF uses frozen record-level ECG embeddings. The scripts include builders for ST-MEM, ECG-FM, CLEAR-HUG, MERL, and MELP. Public model code and checkpoints must be obtained from their original sources. The downstream PGRF scripts operate on the generated sequence caches.

## Main cohort

The main experiments use the longitudinal-eligible cohort, defined as admissions with at least two ECG records available at or before the index ECG time. The index ECG is the earliest ECG within the admission, and prior same-patient ECGs before the index time provide the longitudinal history.

## Outputs

Generated outputs are intentionally excluded from the repository. By default, scripts write caches, tables, and figures under `PGRF_BASE_DIR` or the script-specific output directory.

## Random seeds

The main PGRF training script uses three seeds for the paper results. The final reported main table is based on calibrated probability ensembles across seeds.
