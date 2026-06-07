#!/usr/bin/env bash
# End-to-end reproduction driver for CLIP-QDA.
#
# Usage:
#   bash run_reproduce.sh                # full run (ViT-L/14@336px, as in paper)
#   DATA_ROOT=/path/to/MonuMAI bash run_reproduce.sh   # manual MonuMAI path

set -e

cd "$(dirname "$0")"

PY=${PYTHON:-python}
CLIP_FLAG=""
CDC_MAX_FLAG=""
if [ "${FAST:-0}" = "1" ]; then
  echo ">>> FAST mode: ViT-B/32 backbone, subsampled Cats/Dogs/Cars"
  CLIP_FLAG="--clip-model ViT-B/32"
  CDC_MAX_FLAG="--max-samples 600"
fi

DATA_ROOT_FLAG=""
if [ -n "${DATA_ROOT:-}" ]; then
  DATA_ROOT_FLAG="--data-root ${DATA_ROOT}"
fi

echo "Step 0: prepare data"
$PY scripts/00_prepare_data.py ${DATA_ROOT_FLAG} ${CDC_MAX_FLAG}

echo "Step 1: extract CLIP concept scores (MonuMAI)"
$PY scripts/01_extract_concept_scores.py --config configs/monumai.yaml \
    ${CLIP_FLAG} ${DATA_ROOT_FLAG}

echo "Step 1b: extract CLIP concept scores (Cats/Dogs/Cars)"
$PY scripts/01_extract_concept_scores.py --config configs/cats_dogs_cars.yaml \
    ${CLIP_FLAG} ${CDC_MAX_FLAG}

echo "Step 2: train + evaluate CLIP-QDA (Table 2 / MonuMAI)"
$PY scripts/02_train_eval_qda.py --config configs/monumai.yaml
$PY scripts/02_train_eval_qda.py --config configs/cats_dogs_cars.yaml

echo "Step 3: XAI experiments (Cats/Dogs/Cars)"
$PY scripts/03_run_xai_cats_dogs_cars.py --config configs/cats_dogs_cars.yaml

echo "Step 4: assemble result tables"
$PY scripts/04_make_results_tables.py

echo ">>> Done. See the results/ folder."
