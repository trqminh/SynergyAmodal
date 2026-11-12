#!/usr/bin/env bash
# Test-set predictions for the union model (box + scribble ControlNet-Union and reference IP-Adapter), all three
# hints at once, on the first 2000 records (layer 0) of the uCO3D test set, the set the reference runs used.
# Usage: bash scripts/eval_heldout_union.sh CKPT [device]   (default cuda:0; set CUDA_VISIBLE_DEVICES to pick the GPU)
# Writes preds/ and run.json under all_data/outputs/synergy_amodal_union/eval_heldout/; re-running skips finished records.
# Metrics: python scripts/collect_heldout_eval.py in the parent workspace.
set -e

cd "$(dirname "$0")/.."

PYTHON=/data/minht/TheGrid/hint-control-amodal-data/outputs/synergy_amodal/envs/synergyamodal/bin/python
CKPT=$1
DEVICE=${2:-cuda:0}
OUT=../all_data/outputs
ROOT=../all_data/datasets/hint_dataset_uco3d_ref_scribble_box_v1_test
STEP=$(basename "$CKPT" .ckpt | sed -E 's/.*step=([0-9]+)/step\1/')

"$PYTHON" eval_heldout.py --method union --config configs/union_hints.yaml --ckpt "$CKPT" --root "$ROOT" \
    --manifest manifests/test.provisional.jsonl --layer 0 --device "$DEVICE" \
    --out "$OUT/synergy_amodal_union/eval_heldout/reference/$STEP"
