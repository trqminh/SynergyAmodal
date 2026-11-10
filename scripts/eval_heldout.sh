#!/usr/bin/env bash
# Held-out predictions for SynergyAmodal and the hint ControlNets (box and scribble).
# Usage: bash scripts/eval_heldout.sh [device]   (default cuda:0; set CUDA_VISIBLE_DEVICES to pick the GPU)
# Each run writes preds/ and run.json under all_data/outputs/; re-running skips finished records.
# Metrics: python scripts/collect_heldout_eval.py in the parent workspace.
set -e

cd "$(dirname "$0")/.."

PYTHON=/data/minht/TheGrid/hint-control-amodal-data/outputs/synergy_amodal/envs/synergyamodal/bin/python
DEVICE=${1:-cuda:0}
OUT=../all_data/outputs
CN_LOGS=$OUT/synergy_amodal_controlnet/logs
BOX_CKPT=$CN_LOGS/sd2.1_deocclusion_controlnet_box/lightning_logs/version_1/checkpoints/epoch=59-step=50000.ckpt
SCRIBBLE_CKPT=$CN_LOGS/sd2.1_deocclusion_controlnet_scribble/lightning_logs/version_0/checkpoints/epoch=59-step=50000.ckpt

for HINT in box scribble; do
    CONFIG=configs/controlnet_$HINT.yaml
    if [ "$HINT" = box ]; then CKPT=$BOX_CKPT; else CKPT=$SCRIBBLE_CKPT; fi

    # SynergyAmodal alone: same records and noise, no hint.
    "$PYTHON" eval_heldout.py --method synergy --config "$CONFIG" --device "$DEVICE" \
        --out "$OUT/synergy_amodal/eval_heldout/$HINT"

    # SynergyAmodal + ControlNet, last checkpoint (step 50000).
    "$PYTHON" eval_heldout.py --method controlnet --config "$CONFIG" --ckpt "$CKPT" --device "$DEVICE" \
        --out "$OUT/synergy_amodal_controlnet/eval_heldout/$HINT/step50000"
done
