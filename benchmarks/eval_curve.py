#!/usr/bin/env python

# Copyright 2026 The TurboVLA-LeRobot contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Held-out action error across a training run's checkpoint series.

Why a curve and not a final number
----------------------------------
A single end-of-training number cannot distinguish the four things that matter, and they call for
opposite conclusions:

* one policy is ahead at every budget -- a claim robust to whatever budget was picked;
* the curves cross -- neither policy wins outright, and the useful result is *where* the crossover
  sits, since that tells a reader which one to choose for their budget;
* a policy is still descending steeply at the end -- the run was too short, and reporting its final
  number as a plateau would misrepresent it;
* a policy has turned back upward -- it is overfitting, and its best checkpoint is not its last.

Choosing the reporting budget *after* seeing the numbers is how a comparison gets discredited. Fix
the budget in advance, log the whole series, and report the shape that comes out.

Every policy is scored on the same held-out frames, at the same horizon, with the same normalizer,
so the only thing varying across a row is the checkpoint.

Usage
-----
    python benchmarks/eval_curve.py \
        --dataset max-chr/libero_plus_object_language_all \
        --run turbovla=outputs/cmp_turbovla \
        --run smolvla=outputs/cmp_smolvla \
        --max-frames-per-task 2
"""

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_per_task import (  # noqa: E402
    evaluate,
    load_checkpoint,
    select_frames,
    split_episodes,
)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import (  # noqa: E402
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

import lerobot_policy_turbovla  # noqa: F401,E402  (registers `turbovla`)


def find_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    """(step, path) for every numbered checkpoint in a run, oldest first.

    `last` is skipped: it is a symlink to one of the numbered directories, and following it would
    plot the same checkpoint twice under two different x values.
    """
    out = []
    ckpt_root = Path(run_dir) / "checkpoints"
    for child in sorted(ckpt_root.glob("*")):
        if child.is_symlink() or not re.fullmatch(r"\d+", child.name):
            continue
        model = child / "pretrained_model"
        if model.is_dir():
            out.append((int(child.name), str(model)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--eval-split", type=float, default=0.2, help="must match training")
    parser.add_argument("--horizon", type=int, default=0, help="0 = smallest chunk across runs")
    parser.add_argument("--max-frames-per-task", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    runs = []
    for entry in args.run:
        if "=" not in entry:
            raise SystemExit(f"--run expects NAME=DIR, got {entry!r}")
        name, path = entry.split("=", 1)
        ckpts = find_checkpoints(path)
        if not ckpts:
            raise SystemExit(f"no numbered checkpoints under {path}/checkpoints")
        runs.append((name, ckpts))

    meta = LeRobotDatasetMetadata(args.dataset)
    _, eval_episodes = split_episodes(meta, args.eval_split)

    # The horizon has to be common across every checkpoint of every run, because mean absolute
    # error grows with prediction horizon -- a policy scored over 12 steps is not comparable to one
    # scored over 50 regardless of which is better.
    native = {}
    for name, ckpts in runs:
        cfg = PreTrainedConfig.from_pretrained(ckpts[0][1])
        native[name] = len(cfg.action_delta_indices)
    horizon = args.horizon or min(native.values())
    if any(h < horizon for h in native.values()):
        raise SystemExit(f"--horizon {horizon} exceeds native chunks {native}")

    print(f"dataset  : {args.dataset}")
    print(f"held-out : {len(eval_episodes)} episodes (eval_split={args.eval_split})")
    print(f"horizon  : {horizon} steps  (native: {native})")

    target_cfg = PreTrainedConfig.from_pretrained(runs[0][1][0][1])
    target_cfg.chunk_size = horizon
    target_cfg.n_action_steps = min(target_cfg.n_action_steps, horizon)

    dataset = LeRobotDataset(
        args.dataset,
        episodes=eval_episodes,
        delta_timestamps=resolve_delta_timestamps(target_cfg, meta),
        return_uint8=True,
    )
    frames_by_task = select_frames(dataset, args.max_frames_per_task)
    n_frames = sum(len(v) for v in frames_by_task.values())
    print(f"frames   : {n_frames} over {len(frames_by_task)} tasks\n")

    curves: dict[str, list[tuple[int, float]]] = {}
    for name, ckpts in runs:
        curves[name] = []
        for step, path in ckpts:
            policy, pre, post, _ = load_checkpoint(path, args.device)
            per_task = evaluate(policy, pre, post, dataset, frames_by_task, args, horizon)
            mae = sum(per_task.values()) / len(per_task)
            curves[name].append((step, mae))
            print(f"  {name:<12} step {step:>6}  MAE {mae:.4f}", flush=True)
            del policy
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    names = [n for n, _ in runs]
    steps = sorted({s for c in curves.values() for s, _ in c})
    print(f"\n{'step':>8}" + "".join(f"{n:>12}" for n in names))
    print("-" * (8 + 12 * len(names)))
    for s in steps:
        row = f"{s:>8}"
        for n in names:
            hit = dict(curves[n]).get(s)
            row += f"{hit:>12.4f}" if hit is not None else f"{'-':>12}"
        print(row)

    print("\nbest checkpoint per run (lowest held-out MAE):")
    for n in names:
        step, mae = min(curves[n], key=lambda t: t[1])
        final_step, final_mae = curves[n][-1]
        note = "" if step == final_step else f"  <-- NOT the last checkpoint ({final_mae:.4f} at {final_step})"
        print(f"  {n:<12} {mae:.4f} at step {step}{note}")

    print(
        "\nMAE is an offline proxy: it rewards imitating the demonstration, which is not the same\n"
        "as task success. Read it together with the latency table and a real rollout, not alone."
    )


if __name__ == "__main__":
    main()
