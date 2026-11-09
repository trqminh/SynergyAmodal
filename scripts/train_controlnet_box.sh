#!/usr/bin/env bash
# Usage: bash scripts/train_controlnet_box.sh [config overrides]
# Example: bash scripts/train_controlnet_box.sh training.resume_from_checkpoint=/path/to/last.ckpt
set -e

cd "$(dirname "$0")/.."

PYTHON=/data/minht/TheGrid/hint-control-amodal-data/outputs/synergy_amodal/envs/synergyamodal/bin/python

"$PYTHON" trainer_controlnet.py --config configs/controlnet_box.yaml 'training.devices=[0,3,4]' "$@"
