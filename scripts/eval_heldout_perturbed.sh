#!/usr/bin/env bash
# A union model on clean and perturbed hints (hint_perturbation.py), all three hints given, on the first 2000 records
# (layer 0) of the uCO3D test set. One run per perturbation set; within a run only the named hints are wrong, so each
# run is also a conflict between the wrong hint, the right ones and the visible region.
#
# Usage: [METHOD=...] bash scripts/eval_heldout_perturbed.sh CKPT [device] [set ...]   (default cuda:0, all sets)
#   METHOD=union (default)        CKPT = union model checkpoint (box, scribble, IP-Adapter reference)
#   METHOD=union_reference_attn   CKPT = the reference_only checkpoint to combine with the union ControlNet
#                                 (trainer_union_gated.py with every gate set to 1, no training)
#   METHOD=union_gated            CKPT = gated union checkpoint
#   e.g. METHOD=union_reference_attn bash scripts/eval_heldout_perturbed.sh REF_CKPT cuda:0 clean scribble:shift:16
# Writes preds/, hints/ and run.json under all_data/outputs/<model>/eval_heldout/perturbed/<name>/<set>/;
# re-running skips finished records. Same noise in every run (seed 0), cfg_text 3.
# Scores: PYTHONPATH=. python scripts/collect_perturbed_eval.py --runs <that folder>, in the workspace.
set -e

cd "$(dirname "$0")/.."

PYTHON=/data/minht/TheGrid/hint-control-amodal-data/outputs/synergy_amodal/envs/synergyamodal/bin/python
METHOD=${METHOD:-union}
CKPT=$1
DEVICE=${2:-cuda:0}
shift $(( $# < 2 ? $# : 2 ))
OUT=../all_data/outputs
ROOT=../all_data/datasets/hint_dataset_uco3d_ref_scribble_box_v1_test
STEP=$(basename "$CKPT" .ckpt | sed -E 's/.*step=([0-9]+)/step\1/')

case $METHOD in
    union)
        MODEL_ARGS=(--config configs/union_hints.yaml --ckpt "$CKPT")
        RUNS=$OUT/synergy_amodal_union/eval_heldout/perturbed/$STEP ;;
    union_reference_attn)
        MODEL_ARGS=(--config configs/union_gated.yaml --set "init_reference_ckpt=$CKPT")
        RUNS=$OUT/synergy_amodal_union_gated/eval_heldout/perturbed/combined_reference_$STEP ;;
    union_gated)
        MODEL_ARGS=(--config configs/union_gated.yaml --ckpt "$CKPT")
        RUNS=$OUT/synergy_amodal_union_gated/eval_heldout/perturbed/$STEP ;;
    *) echo "unknown METHOD $METHOD"; exit 1 ;;
esac

# A set is one or more specs joined by '+'.
SETS=(
    clean
    box:loose:0.5 box:shift:0.5 box:cut:0.25
    scribble:shift:8 scribble:shift:16 scribble:shift:32 scribble:warp:16 scribble:drop:0.5 scribble:add:1
    reference:swap reference:swap_other
    scribble:shift:16+reference:swap
)
if [ $# -gt 0 ]; then SETS=("$@"); fi

for SET in "${SETS[@]}"; do
    ARGS=()
    if [ "$SET" != clean ]; then
        for SPEC in ${SET//+/ }; do ARGS+=(--perturb "$SPEC"); done
    fi
    echo "== $SET"
    "$PYTHON" eval_heldout.py --method "$METHOD" "${MODEL_ARGS[@]}" --root "$ROOT" \
        --manifest manifests/test.provisional.jsonl --layer 0 --cfg_text 3 --device "$DEVICE" "${ARGS[@]}" \
        --out "$RUNS/${SET//:/_}"
done
