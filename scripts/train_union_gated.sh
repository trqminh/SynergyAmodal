#!/usr/bin/env bash
# Usage: bash scripts/train_union_gated.sh [config overrides]
# Example: bash scripts/train_union_gated.sh training.resume_from_checkpoint=/path/to/last.ckpt
set -e

cd "$(dirname "$0")/.."

PYTHON=/data/minht/TheGrid/hint-control-amodal-data/outputs/synergy_amodal/envs/synergyamodal/bin/python

"$PYTHON" trainer_union_gated.py --config configs/union_gated.yaml 'training.devices=[0,3,4]' "$@"
