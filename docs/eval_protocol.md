# Evaluation Protocol

> **Project:** openpi / π0.5 + LIBERO LoRA fine-tuning and evaluation  
> **Frozen SFT checkpoint:** `pi05_libero_day22_bs4_e2 / step 13999`  
> **Document purpose:** define the fixed protocol used for offline action evaluation, closed-loop rollout, controlled robustness analysis, and deployment measurement.

---

## 1. Scope

This document defines the evaluation rules for:

1. fixed raw-action-space offline evaluation;
2. LIBERO closed-loop rollout;
3. Task 5 formal policy comparison;
4. Task 1 ten-state evaluation;
5. LIBERO-10 three-state-per-task screening;
6. Task 5 controlled initial-state robustness;
7. bf16 policy-query latency and VRAM measurement.

These evaluation families use different denominators and must be reported separately.

---

## 2. Frozen Model and Data Configuration

```text
config name             = pi05_libero_day22_bs4_e2
final saved step        = 13999
configured train steps  = 14000
initialization          = pi05_base
dataset                 = physical-intelligence/libero_day18_n100
training batch size     = 4
epoch coverage          ≈ 2.02
inference precision     = bfloat16
action horizon          = 10
LIBERO action dimension = 7
```

Checkpoint path:

```text
checkpoints/pi05_libero_day22_bs4_e2/day22_bs4_e2/13999
```

The following components are treated as frozen when policies are compared:

```text
evaluation observations and frame indices
task prompts
normalization statistics
action output transform
rollout script
success predicate
replan_steps
max_policy_steps
environment seed
predefined initial states
```

The checkpoint is the primary variable changed in formal policy comparisons.

---

## 3. Raw-Action-Space Offline Evaluation

### 3.1 Fixed Validation Set

```text
validation episodes      = 50–59
samples per episode      = 3
observations             = 30 per checkpoint
noise seed               = 2026
action horizon           = 10
LIBERO action dimension  = 7
```

All checkpoints use the same:

- observations;
- frame indices;
- ground-truth action chunks;
- prompts;
- normalization statistics;
- inference noise seed.

### 3.2 Evaluation Space

Training losses from normalized and non-normalized experiments are not directly comparable because the target scales differ.

All reported action metrics are computed after converting model predictions back to the original LIBERO action space.

Reported metrics:

```text
Action L2
RMSE
MAE
per-dimension RMSE
gripper RMSE
gripper sign accuracy
```

The offline metrics measure action prediction accuracy. They do not replace task-level rollout success.

---

## 4. Closed-Loop Rollout Protocol

### 4.1 Fixed Runtime Conditions

```text
task suite          = libero_10
environment seed    = 7
replan_steps        = 5
max_policy_steps    = 520
policy batch size   = 1
precision           = bfloat16
returned chunk      = [10, 7]
```

The policy returns an action chunk with shape `[10, 7]`. The rollout executes the first five actions and then requests a new chunk.

### 4.2 Initial-State Definition

```text
INITIAL_STATE_ID
= index of a predefined LIBERO initial state
```

`INITIAL_STATE_ID` is not a random seed.

### 4.3 Seed Definitions

```text
ENV_SEED
= NumPy / LIBERO environment-side seed
= not the policy seed

Policy RNG
= internal RNG state maintained by the Policy Server
```

The evaluated serving entry point does not expose a separate policy-seed CLI argument.

Therefore:

> Fixing `INITIAL_STATE_ID` and `ENV_SEED` does not guarantee trajectory-level determinism.

Repeated rollouts may differ slightly in actions, contact timing, or termination step while still producing the same task-level outcome.

---

## 5. Success Determination

The environment-reported task result is the source of truth.

A message such as:

```text
ROLLOUT SMOKE TEST: PASS
```

only confirms that the rollout pipeline completed without a runtime failure. It does not prove task success.

Task success must be read from the recorded environment result:

```text
success = True / False
```

or from the equivalent environment reward / termination condition used by the rollout script.

Visual inspection is used for diagnosis, not for overriding the formal success predicate.

---

## 6. Task 5 Formal Policy Comparison

Task:

```text
pick up the book and place it in the back compartment of the caddy
```

Protocol:

```text
task_id            = 5
initial states     = 0–9
environment seed   = 7
replan_steps       = 5
max_policy_steps   = 520
```

Compared policies:

```text
Official π0.5 LIBERO checkpoint
Old 5k SFT checkpoint
Second-round SFT step 13999
```

Reported statistics:

- successes / 10;
- observed success rate;
- Wilson 95% confidence interval;
- per-state result.

| Policy | Successes / 10 | Success Rate | Wilson 95% CI |
|---|---:|---:|---:|
| Official π0.5 LIBERO | 10/10 | 100% | [0.722, 1.000] |
| Old 5k | 0/10 | 0% | [0.000, 0.278] |
| Second-round 13,999 | 9/10 | 90% | [0.596, 0.982] |

---

## 7. Task 1 Ten-State Evaluation

Task:

```text
put both the cream cheese box and the butter in the basket
```

Protocol:

```text
task_id            = 1
initial states     = 0–9
environment seed   = 7
replan_steps       = 5
max_policy_steps   = 520
```

Result:

```text
successes          = 10/10
Wilson 95% CI      = [0.722, 1.000]
```

This result is reported separately from Task 5 and must not be merged into a synthetic combined success rate.

---

## 8. LIBERO-10 Three-State-Per-Task Screening

Protocol:

```text
task ids           = 0–9
initial states     = 0–2 for each task
rollouts           = 30
environment seed   = 7
replan_steps       = 5
max_policy_steps   = 520
```

Required name:

```text
LIBERO-10 three-state-per-task screening
```

Result:

```text
successes          = 25/30
observed rate      = 83.3%
Wilson 95% CI      = [0.664, 0.927]
```

This experiment is a breadth-oriented coverage screening. It is not the official LIBERO-10 benchmark because only three predefined initial states were evaluated for each task.

---

## 9. Controlled Initial-State Robustness

Target task:

```text
Task 5
```

Only intentionally changed variable:

```text
INITIAL_STATE_ID
```

Compared conditions:

```text
state 0            = 10 rollouts
state 2            = 10 rollouts
```

Fixed conditions:

```text
checkpoint          = step 13999
environment seed    = 7
replan_steps        = 5
max_policy_steps    = 520
Policy Server       = one continuously running process
run order           = alternating AB / BA
```

Results:

| Initial State | Successes / 10 | Rate | Wilson 95% CI |
|---:|---:|---:|---:|
| 0 | 9/10 | 90% | [0.596, 0.982] |
| 2 | 7/10 | 70% | [0.397, 0.892] |

Interpretation:

> State 2 had a lower observed success rate, but the sample size is small and the confidence intervals overlap. The result supports preliminary evidence of initial-state sensitivity, not a statistically significant difference.

---

## 10. Confidence Intervals

Success-rate uncertainty is reported using the Wilson score interval with:

```text
z = 1.96
```

For `x` successes in `n` trials:

```text
p̂ = x / n
```

Wilson intervals are preferred to the simple normal approximation because several reported sample sizes are small or include boundary outcomes such as `0/10` and `10/10`.

---

## 11. Latency Measurement Protocol

Fresh-server deployment measurement:

```text
total policy queries     = 35
cold-start queries       = 1
steady-state queries     = 34
```

The first query is excluded from steady-state statistics because it includes server-side tracing, JIT compilation, and first-inference initialization.

Reported latency metrics:

```text
cold-start first query
steady-state mean
steady-state P50
steady-state P95
```

The measurement represents:

```text
policy-query latency
```

It does not represent full real-robot end-to-end latency, which would also include sensor capture, image transport, network communication, safety checks, low-level control, and actuator response.

---

## 12. VRAM Measurement Protocol

VRAM is sampled using:

```text
nvidia-smi
sampling interval ≈ 200 ms
```

Reported metrics:

- GPU baseline before server loading;
- loaded-idle resident VRAM;
- observed sampled peak VRAM;
- peak above resident.

The peak must be described as:

```text
observed sampled peak VRAM
```

It is not an exact instantaneous CUDA allocator peak, and short-lived spikes may be missed.

---

## 13. Reporting Rules

Allowed statements:

```text
Task 5 formal result: 9/10
Task 1 ten-state result: 10/10
LIBERO-10 three-state-per-task screening: 25/30
Task 5 state 0 vs state 2: 9/10 vs 7/10
```

Disallowed or misleading statements:

```text
Official LIBERO-10 benchmark success rate = 83.3%
Task 1 and Task 5 combined success rate = 19/20
Fixed ENV_SEED guarantees identical trajectories
Observed peak VRAM is the exact CUDA allocator peak
A visual near-success overrides the environment predicate
```

---

## 14. Evidence Files

```text
notes/day23_second_sft_eval.md
notes/day24_notes.md

experiments/day24_robustness/determinism_check.json
experiments/day24_task1_eval/task1_states3_9_summary.json
experiments/day24_ten_task_screen/ten_task_screen_summary.json
experiments/day24_ten_task_screen/ten_task_screen_trials.csv
experiments/day24_controlled_robustness/task5_state0_vs_state2/robustness_summary.json
experiments/day24_controlled_robustness/task5_state0_vs_state2/robustness_trials.csv
experiments/day24_deployment/deployment_note.md
experiments/day24_deployment/latency_vram.csv
```
