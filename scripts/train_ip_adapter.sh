#!/usr/bin/env bash
# Usage: bash scripts/train_ip_adapter.sh [config overrides]
# Example: bash scripts/train_ip_adapter.sh training.resume_from_checkpoint=/path/to/last.ckpt
set -e

cd "$(dirname "$0")/.."

PYTHON=/data/minht/TheGrid/hint-control-amodal-data/outputs/synergy_amodal/envs/synergyamodal/bin/python

"$PYTHON" trainer_ip_adapter.py --config configs/ip_adapter_reference.yaml 'training.devices=[0,3,4]' "$@"
