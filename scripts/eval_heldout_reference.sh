#!/usr/bin/env bash
# Test-set predictions for SynergyAmodal and SynergyAmodal + reference IP-Adapter.
# Usage: bash scripts/eval_heldout_reference.sh CKPT [device] [REFERENCE_ONLY_CKPT]   (default cuda:0; set CUDA_VISIBLE_DEVICES to pick the GPU)
# Each run writes preds/ and run.json under all_data/outputs/; re-running skips finished records.
# Metrics: python scripts/collect_heldout_eval.py in the parent workspace.
set -e

cd "$(dirname "$0")/.."

PYTHON=/data/minht/TheGrid/hint-control-amodal-data/outputs/synergy_amodal/envs/synergyamodal/bin/python
CKPT=$1
DEVICE=${2:-cuda:0}
OUT=../all_data/outputs
CONFIG=configs/ip_adapter_reference.yaml
ROOT=../all_data/datasets/hint_dataset_uco3d_ref_scribble_box_v1_test
MANIFEST=manifests/test.provisional.jsonl
STEP=$(basename "$CKPT" .ckpt | sed -E 's/.*step=([0-9]+)/step\1/')

# SynergyAmodal alone: same records and noise, no reference.
"$PYTHON" eval_heldout.py --method synergy --config "$CONFIG" --root "$ROOT" --manifest "$MANIFEST" --device "$DEVICE" \
    --out "$OUT/synergy_amodal/eval_heldout/reference"

"$PYTHON" eval_heldout.py --method ip_adapter --config "$CONFIG" --ckpt "$CKPT" --root "$ROOT" --manifest "$MANIFEST" --device "$DEVICE" \
    --out "$OUT/synergy_amodal_ip_adapter/eval_heldout/reference/$STEP"

# Training-free reference_only attention (no checkpoint).
"$PYTHON" eval_heldout.py --method reference_only --config "$CONFIG" --root "$ROOT" --manifest "$MANIFEST" --device "$DEVICE" \
    --out "$OUT/synergy_amodal_reference_only/eval_heldout/reference"

# Trained reference_only attention (trainer_reference_only.py), only when its checkpoint is given as the third argument.
REF_CKPT=${3:-}
if [ -n "$REF_CKPT" ]; then
    REF_STEP=$(basename "$REF_CKPT" .ckpt | sed -E 's/.*step=([0-9]+)/step\1/')
    "$PYTHON" eval_heldout.py --method reference_only_trained --config configs/reference_only.yaml --ckpt "$REF_CKPT" --root "$ROOT" --manifest "$MANIFEST" --device "$DEVICE" \
        --out "$OUT/synergy_amodal_reference_only_trained/eval_heldout/reference/$REF_STEP"
fi
