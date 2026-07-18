# Task 5 State 0 Failure Attribution

## Evaluation target

- Suite: `libero_10`
- Task ID: `5`
- Task: `pick up the book and place it in the back compartment of the caddy`
- Checkpoint: `pi05_libero_day22_bs4_e2`, step `13999`
- Initial state ID: `0`
- Environment seed: `7`
- Replan steps: `5`
- Maximum policy steps: `520`

## Success predicate

The BDDL goal is:

```text
In(black_book_1, desk_caddy_1_back_contain_region)
For a containment site, check_contact() always returns True.
Therefore, task success is determined by whether the body center of
black_book_1 lies inside the axis-aligned world-space bounds of
back_contain_region.

The predicate does not directly require:

full-book geometric containment;
a particular book orientation;
gripper release;
physical contact with the caddy.
Repeat evaluation

Day 24 follow-up state-0 rollouts:

initial diagnostic rollout: 1/1 success;
repeated rollouts: 10/10 success;
total Day 24 follow-up: 11/11 success.

The original Day 23 state-0 rollout failed, so state 0 is not a
deterministic failure case.

Predicate margins over the 10 repeated rollouts
Axis	Minimum	Mean	Maximum
x	0.002 mm	12.084 mm	22.365 mm
y	26.423 mm	47.120 mm	61.406 mm
z	0.412 mm	4.303 mm	14.764 mm

The closest observed successful placements were:

x margin: 0.002 mm in run 8;
z margin: 0.412 mm in run 10.

The y margin remained comfortably positive in every repeated rollout.

Policy-query counts ranged from 34 to 79, and the final policy step
ranged from 179 to 404, showing substantial trajectory and
completion-time variability despite identical task-level success.

Attribution

Primary label:

stochastic_predicate_boundary_miss

Secondary description:

near_boundary_placement_variability

Confidence:

medium

The evidence supports a boundary-sensitive placement failure in the x
or z direction. The exact failed axis cannot be recovered because the
original Day 23 failure log did not record the book position and target
region bounds.

The failure should not be labeled definitively as
insufficient_insertion_depth, because the available evidence does not
prove that the original failure crossed the z boundary specifically.

Final interpretation

Task 5 state 0 is generally solvable by checkpoint 13999. The original
failure was an occasional rollout-level failure caused by residual
policy or observation nondeterminism combined with a placement that can
finish very close to the success-predicate boundary.
