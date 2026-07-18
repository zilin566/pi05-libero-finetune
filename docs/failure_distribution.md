# Failure Distribution

> **Scope:** LIBERO-10 three-state-per-task screening  
> **Checkpoint:** `pi05_libero_day22_bs4_e2 / step 13999`  
> **Total rollouts:** 30  
> **Observed failures:** 5

---

## 1. Summary

The screening result was:

```text
successes = 25/30
failures  = 5/30
```

The five observed failures were concentrated in:

- final placement outside the valid target region;
- object contact without a stable grasp;
- repeated re-grasp or oscillation during multi-object execution.

Task 8 was the clearest task-level weakness in this screening.

This document describes the observed failure sample. It does not estimate the model's universal causal failure distribution.

---

## 2. Labeling Method

Each failed rollout receives:

```text
exactly one primary observable label
zero or more secondary diagnostic labels
```

The primary label is mutually exclusive and records the most visible failure event.

Secondary labels are not mutually exclusive. They provide a finer diagnosis supported by one or more of:

- rollout logs;
- final object geometry;
- BDDL / predicate inspection;
- representative video review.

Termination by timeout is not automatically treated as the root cause. A timeout may be the final episode termination while the underlying failure is a grasp, placement, or recovery error.

---

## 3. Primary Label Dictionary

### `place_outside_target`

The object is transported near the intended target but is released or left outside the valid target region.

Typical evidence:

- the object reaches the target vicinity;
- the final success predicate remains false;
- one or more containment margins are negative.

### `contact_no_grasp`

The gripper reaches or contacts the object, but a stable grasp is not established.

Typical evidence:

- the gripper closes near the object;
- the object does not follow the end effector;
- repeated approach or grasp attempts occur.

### `timeout_or_oscillation`

The policy repeatedly replans or reattempts a failed stage without recovering before the step limit.

Typical evidence:

- repeated local motion;
- repeated gripper toggling or re-grasp;
- no meaningful stage progress;
- termination at `max_policy_steps`.

---

## 4. Failure Manifest

| Task / State | Furthest Stage Reached | Primary Label | Secondary Diagnosis | Termination |
|---|---|---|---|---|
| Task 5 / State 2 | transport and placement | `place_outside_target` | `book_wedged_at_compartment_edge` | timeout |
| Task 8 / State 0 | first moka pot completed | `contact_no_grasp` | `second_object_grasp_failure` | timeout after partial completion |
| Task 8 / State 1 | first moka pot completed | `timeout_or_oscillation` | `repeated_second_object_regrasp` | timeout after partial completion |
| Task 8 / State 2 | first-object grasp stage | `contact_no_grasp` | `first_object_grasp_failure` | timeout |
| Task 9 / State 1 | microwave placement stage | `place_outside_target` | `premature_release_at_microwave_threshold` | timeout |

---

## 5. Primary Failure Distribution

| Primary Label | Count | Share of Observed Failures |
|---|---:|---:|
| `place_outside_target` | 2 | 40% |
| `contact_no_grasp` | 2 | 40% |
| `timeout_or_oscillation` | 1 | 20% |
| **Total** | **5** | **100%** |

Interpretation:

> These percentages summarize only the five failures observed in the current 30-rollout screening. They must not be interpreted as stable causal frequencies for the policy as a whole.

---

## 6. Task-Level Analysis

### 6.1 Task 5

Task:

```text
pick up the book and place it in the back compartment of the caddy
```

Observed screening failure:

```text
Task 5 / State 2
```

The policy completed:

```text
approach
→ grasp
→ lift
→ transport
→ placement attempt
```

The final book position remained outside the valid containment region.

```text
primary label       = place_outside_target
secondary diagnosis = book_wedged_at_compartment_edge
```

The controlled robustness experiment provided additional evidence: all four Task 5 failures in the state 0 / state 2 comparison included an x-axis containment miss.

### 6.2 Task 8

Task:

```text
put both moka pots on the stove
```

Screening result:

```text
0/3
```

Observed behavior:

```text
state 0:
first moka pot completed
second-object grasp failed

state 1:
first moka pot completed
second-object re-grasp repeated without recovery

state 2:
first-object grasp failed
```

Task-level diagnostic label:

```text
multi_object_sequential_manipulation_failure
```

The evidence does not support complete task misunderstanding. In two of the three states, the first object was completed and the failure occurred during the transition to the second object.

Observed weak stages:

- second-object re-localization;
- second-object grasp;
- subtask transition;
- recovery after partial completion;
- long-horizon execution stability.

Task 8 is the preferred first RL target because:

1. it was the only task with `0/3` in the screening;
2. partial completion provides interpretable stage structure;
3. the success condition can be decomposed into object-level predicates;
4. Task 1 and Task 5 can serve as guard tasks for catastrophic forgetting.

### 6.3 Task 9

Task:

```text
put the yellow-white mug in the microwave and close it
```

Observed failure:

```text
Task 9 / State 1
```

The mug was moved toward the microwave but released near the opening before valid placement.

```text
primary label       = place_outside_target
secondary diagnosis = premature_release_at_microwave_threshold
```

The observed stage-ordering problem was:

```text
placement predicate still false
→ policy advanced toward microwave closing
```

The video supports this descriptive diagnosis, but it does not by itself prove a single internal causal mechanism.

---

## 7. Task 5 Predicate-Level Diagnosis

Formal BDDL goal:

```text
In(
    black_book_1,
    desk_caddy_1_back_contain_region
)
```

Source and rollout inspection indicate that success is determined by the book body center entering the target site's world-space bounds.

The predicate does not directly require:

- the full book bounding box to be inside;
- a specific book orientation;
- explicit gripper release;
- the arm to return to its initial pose.

Controlled failures:

| Initial State | x Margin | y Margin | z Margin | Failed Axes |
|---:|---:|---:|---:|---|
| 2 | -112.101 mm | +59.474 mm | +50.350 mm | x |
| 2 | -93.254 mm | +26.122 mm | +44.077 mm | x |
| 0 | -91.880 mm | +54.294 mm | +63.743 mm | x |
| 2 | -3.148 mm | -50.503 mm | +5.897 mm | x, y |

Summary:

```text
4/4 failures = x-axis containment miss
1/4 failures = additional y-axis miss
0/4 failures = z-axis miss
```

Supported diagnosis:

```text
primary:
place_outside_target

secondary:
x_axis_containment_miss
x_y_axis_containment_miss
```

The earlier hypothesis that insufficient z-direction insertion was the general failure mechanism is not supported by these four failures.

---

## 8. Representative Videos

Paths below are relative to this file in `docs/`.

### Successful Rollouts

- [Task 1: two-object sequential success](media/day24/success_task1_two_objects_to_basket.mp4)
- [Task 4: two mugs and two target plates](media/day24/success_task4_two_mugs_two_plates.mp4)
- [Task 5: successful book placement](media/day24/success_task5_book_back_compartment.mp4)

### Failed Rollouts

- [Task 5: placement outside target](media/day24/failure_task5_place_outside_target.mp4)
- [Task 8: second-object grasp failure](media/day24/failure_task8_second_object_grasp.mp4)
- [Task 9: microwave placement failure](media/day24/failure_task9_microwave_placement.mp4)

---

## 9. Limitations

1. Only five failures were observed.
2. Each LIBERO-10 task was evaluated on only three predefined initial states.
3. Reported percentages summarize the observed failure sample, not stable causal frequencies.
4. Video-derived labels are descriptive unless supported by logs, predicates, or final-state geometry.
5. Fixed environment seeds do not guarantee identical policy trajectories.
6. The controlled Task 5 geometry analysis contains only four failures.
7. No real-robot evaluation was performed.

---

## 10. Evidence Files

```text
experiments/day24_ten_task_screen/failure_manifest.md
experiments/day24_ten_task_screen/ten_task_screen_summary.json
experiments/day24_ten_task_screen/ten_task_screen_trials.csv
experiments/day24_state0_analysis/state0_failure_attribution.md
experiments/day24_controlled_robustness/task5_state0_vs_state2/robustness_failures.json
experiments/day24_controlled_robustness/task5_state0_vs_state2/robustness_summary.json
notes/day24_notes.md
docs/media/day24/
```
