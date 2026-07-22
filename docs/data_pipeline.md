# Data Quality and Manifest Pipeline

This project uses the public `physical-intelligence/libero` LeRobot dataset. It does **not** claim teleoperation collection, synthetic trajectory generation, RGB-D sensing or full-dataset relabeling.

## Why this still counts as data engineering

Training a VLA policy requires more than downloading parquet files. The model input must preserve alignment between two camera views, robot state, language instruction, episode/frame identity and the action target. A silent mismatch can produce a decreasing loss while breaking closed-loop control.

The implemented pipeline therefore separates four responsibilities:

1. **Inventory and schema discovery** — verify metadata, tasks and episode files.
2. **Multimodal alignment** — check camera/state/action/task fields at the same frame.
3. **Quality validation** — detect missing images, inconsistent dimensions, NaN/Inf, out-of-range actions and broken episode indices.
4. **Manifest cleaning** — keep source parquet files read-only and emit deterministic accepted/rejected episode lists with explicit reasons.

## Recorded evidence

| Evidence | Recorded result | Scope |
|---|---:|---|
| Episode files discovered | 1,693 | dataset inventory |
| Task texts loaded | 40 | metadata inventory |
| Quality-audit sample | 6 episodes / 1,908 frames | episode IDs 0, 1, 2, 10, 50, 100 |
| Camera format | agent + wrist, RGB, 256 × 256 | sampled frames |
| State / action dimension | 8 / 7 | all 1,908 sampled frames |
| Missing images | 0 | sampled frames |
| NaN/Inf state or action | 0 | sampled frames |
| Configured range violations | 0 | sampled frames |
| Flagged sampled episodes | 0/6 | sampled audit only |

Sources:

- `logs/day10_action_state_stats/action_state_stats.json`
- `logs/day11_clean_check/clean_check_summary.json`
- `logs/day12_data_format_check/day12_data_format_check.json`

These numbers are intentionally described as a **sampled audit**, not proof that every frame in the full dataset is clean.

## Reproducible cleaning command

```bash
python scripts/data_checks/audit_libero_dataset.py \
  --dataset-root /path/to/physical-intelligence/libero \
  --output-dir experiments/data_quality/latest
```

The command writes:

```text
experiments/data_quality/latest/
├── data_quality_report.json
├── accepted_episodes.jsonl
└── rejected_episodes.jsonl
```

No source parquet file is deleted or rewritten. Use `--strict` in CI or before training when any rejected episode should stop the pipeline.

## Validation rules

An episode is rejected when any configured hard rule fails, including:

- missing agent-view or wrist-view image;
- missing state/action/task/episode/frame fields;
- state dimension other than 8 or action dimension other than 7;
- NaN/Inf values;
- action magnitude outside the configured LIBERO range;
- inconsistent task or episode IDs within one parquet file;
- non-contiguous frame indices or non-monotonic timestamps;
- almost-all-zero action sequences;
- unmapped task instruction.

Thresholds are CLI-visible and saved in the JSON report so a later run cannot silently change the cleaning policy.

## Data-quality stress test

To verify that the training/evaluation loop reacts to bad supervision, a seeded corruption manifest injected finite ±3.0 values into roughly 10% of action frames across five episodes (`146` frames total). Relative to the clean normalized run, the dirty-data experiment increased mean training loss by `21.4%` and raw-space Action L2 by `7.96%`.

This is an ablation, not a synthetic data-generation claim. It demonstrates that the pipeline can create a traceable fault condition and measure its downstream effect.

## What is not included

- no teleoperation collection;
- no automatic language relabeling;
- no synthetic demonstration generation;
- no unseen-object or lighting-randomization benchmark;
- no real-robot sensor-noise validation.

Those would be separate future projects rather than claims added to this repository.
