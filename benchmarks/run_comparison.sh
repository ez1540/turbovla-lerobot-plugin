#!/usr/bin/env bash
# Train TurboVLA / SmolVLA / ACT under a matched protocol on one multi-task dataset.
#
# Everything that could confound the comparison is pinned identically: dataset, chunk size, batch
# size, step count, seed, checkpoint cadence, and (via resize224.yaml) the input resolution every
# policy sees. What is left varying is the architecture -- which is the thing under test.
#
# Roles:
#   turbovla  the candidate
#   smolvla   the incumbent language-conditioned baseline, architecture only (no smolvla_base
#             robotics pretraining) so both policies start from vision+language pretraining alone
#   act       the language-blind control: it cannot read the instruction, so its score is the floor
#             that any language conditioning has to beat on a multi-task dataset
#
# `--eval_steps=0` is deliberate: LeRobot applies image_transforms to the TRAIN split only, so the
# built-in eval would score a 224-trained policy on native-resolution frames. Score with
# benchmarks/eval_curve.py instead, which resizes on its own.
#
# Usage:  bash benchmarks/run_comparison.sh [STEPS]     (default 6000)

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-6000}"
DATASET=max-chr/libero_plus_object_language_all
BATCH=64
SEED=1000
SAVE_FREQ=500

# Both policies ship schedules written for their paper's full-length run: TurboVLA warms up over
# 10k steps and decays over 80k (the LIBERO recipe), SmolVLA warms up over 1k and decays over 30k.
# Left alone at a few thousand steps, TurboVLA would spend the ENTIRE run inside linear warmup --
# never reaching its peak LR -- while SmolVLA reaches peak at step 1k and trains near it throughout.
# That is a schedule artifact, and it would show up in the results as an architecture difference.
#
# So scale each policy's own schedule to the budget instead of imposing a single shared one: one
# full cosine cycle inside the budget for both, with each author's warmup *fraction* preserved
# (TurboVLA 1/8 of the run, SmolVLA 1/30). Peak learning rates are left at the authors' values,
# since those are tuned to the architecture and are not a function of run length. ACT has no
# scheduler at all -- a constant LR is its own recipe, and there is nothing to rescale.
TURBOVLA_WARMUP=$((STEPS / 8))
SMOLVLA_WARMUP=$((STEPS / 30))

common=(
  --dataset.repo_id="$DATASET"
  --batch_size="$BATCH"
  --steps="$STEPS"
  --seed="$SEED"
  --save_freq="$SAVE_FREQ"
  --eval_steps=0
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.chunk_size=12
  --policy.n_action_steps=12
  --wandb.enable=false
)

run() {  # run <gpu> <name> <extra flags...>
  local gpu="$1" name="$2"; shift 2
  echo "=== $name on GPU$gpu, $STEPS steps ==="
  CUDA_VISIBLE_DEVICES="$gpu" python benchmarks/train_with_yaml.py benchmarks/resize224.yaml \
    "${common[@]}" \
    --output_dir="outputs/cmp_${name}" \
    --job_name="cmp_${name}" \
    "$@" 2>&1 | tee "outputs/cmp_${name}.log"
}

mkdir -p outputs

case "${LANE:-both}" in
  gpu0)
    # DINOv3 is gated; dinov2-base is the ungated stand-in, and it is already cached locally.
    run 0 turbovla \
      --policy.type=turbovla \
      --policy.vision_backbone=facebook/dinov2-base \
      --policy.scheduler_warmup_steps="$TURBOVLA_WARMUP" \
      --policy.scheduler_decay_steps="$STEPS"
    ;;
  gpu1)
    run 1 smolvla \
      --policy.type=smolvla \
      --policy.load_vlm_weights=true \
      --policy.scheduler_warmup_steps="$SMOLVLA_WARMUP" \
      --policy.scheduler_decay_steps="$STEPS"
    run 1 act --policy.type=act
    ;;
  both)
    LANE=gpu0 bash "$0" "$STEPS" &
    LANE=gpu1 bash "$0" "$STEPS" &
    wait
    ;;
esac
