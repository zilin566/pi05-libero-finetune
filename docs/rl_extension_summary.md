# Reward-guided Policy Improvement: Exploratory Extension

This document intentionally keeps the RL work separate from the main π0.5 SFT result.

## What was implemented

- Official LIBERO Task 8 predicates and one-shot reward shaping.
- Full `[10, 7]` action-chunk logging and exact executed-action linkage.
- Return-to-go reconstruction and deterministic query/trajectory weighting.
- Hard development/held-out/guard isolation in the dataset builder.
- Per-example reward-weighted flow-matching loss.
- LoRA-only and action-expert-only trainable-parameter allowlists.
- Multi-task SFT replay, Frozen-policy consistency and checkpoint SHA gates.

The method is accurately described as **reward-weighted flow matching / reward-weighted behavior cloning**. It is not PPO, GRPO or actor-critic RL.

## Result boundary

The small Stage 1 evaluation showed one exploratory held-out success (`0/5 → 1/5`) under a single environment seed, but development return decreased. It was therefore not treated as stable generalization.

Stage 2 increased Task 8 development success under its ten-episode protocol, but all candidates regressed on a Task 5 guard state and were rejected.

Stage 3 added action-expert-only LoRA, multi-task replay and Frozen-policy consistency. The final common protocol was:

```text
Task 8 states 0–4
policy seeds 7101–7104
20 episodes per policy
```

| Policy | Official success | Mean return |
|---|---:|---:|
| Frozen SFT | **4/20** | **1.3383** |
| Best Stage 3 candidate | 2/20 | 0.7371 |

No candidate passed the development gate. The registered final states 5–9 evaluation was not run or consumed, and Frozen SFT remained the release baseline.

## Accurate public wording

> I extended the π0.5 training pipeline with auditable reward-weighted flow matching and guard-task evaluation. Under the strict final multi-seed protocol, no RL candidate outperformed the Frozen SFT baseline, so the held-out set remained unconsumed and no candidate was promoted.

The value of this extension is the implemented online-data and safety-gating pipeline, together with an honestly retained negative result—not a claim that RL outperformed SFT.
