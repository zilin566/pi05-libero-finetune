# VLA Internship Project Pitch

## 30-second version

I fine-tuned OpenPI's π0.5 on LIBERO with LoRA and built the full engineering loop from LeRobot data validation and quantile normalization to raw-action offline evaluation, closed-loop rollout, failure diagnosis and deployment profiling. The final SFT reached 9/10 on a fixed Task 5 comparison, 10/10 on Task 1 states 0–9 and 25/30 in a ten-task breadth screen, with 83 ms steady policy-query latency. I also explored reward-guided updates, but retained Frozen SFT when the stricter multi-seed gate did not confirm an RL improvement.

## Resume bullets

- Built an end-to-end OpenPI π0.5 + LIBERO pipeline covering LeRobot data/schema validation, quantile normalization, LoRA SFT, raw-action offline metrics, closed-loop rollout and predicate-based evaluation.
- Improved fixed-set action quality to `0.437` Action L2 and `89%` gripper-sign accuracy at step 13,999; achieved `9/10` on a fixed Task 5 protocol versus `0/10` for the old 5k checkpoint and `10/10` for the official policy.
- Evaluated cross-task behavior with Task 1 `10/10` and a LIBERO-10 three-state breadth screen of `25/30`; categorized five observed failures and quantified initial-state sensitivity with Wilson confidence intervals.
- Profiled bf16 policy serving at `83.25 ms` steady mean and `99.90 ms` P95 query latency with `24.11 GiB` observed peak VRAM, separating JAX cold-start cost from steady inference.

## Do not claim

- Do not call the 25/30 breadth screen an official LIBERO benchmark score.
- Do not translate normalized action error directly into centimeters.
- Do not claim real-robot deployment, RGB-D, INT8 quantization or unseen-object generalization.
- Do not say that RL outperformed SFT; the final Stage 3 gate rejected all candidates.
