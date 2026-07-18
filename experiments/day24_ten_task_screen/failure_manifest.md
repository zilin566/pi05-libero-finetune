# Day 24 Ten-Task Screening Failure Manifest

This is a breadth screening with three predefined initial states per task, not a formal benchmark.

- Overall: `25/30`
- Failed trials: `5`

## Per-task results

| Task | Result | Description |
|---:|---:|---|
| 0 | 3/3 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 3/3 | put both the cream cheese box and the butter in the basket |
| 2 | 3/3 | turn on the stove and put the moka pot on it |
| 3 | 3/3 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 3/3 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 2/3 | pick up the book and place it in the back compartment of the caddy |
| 6 | 3/3 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 3/3 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 0/3 | put both moka pots on the stove |
| 9 | 2/3 | put the yellow and white mug in the microwave and close it |

## Failed trials

### Task 5 / State 2

- Task: `pick up the book and place it in the back compartment of the caddy`
- Policy queries: `104`
- Executed policy steps: `520`
- Last environment step: `529`
- EEF movement L2: `0.2932 m`
- Gripper sign flips: `22`
- JSONL: `experiments/day24_ten_task_screen/task5/state2/ckptday22_s13999_screen_task5_state2.jsonl`
- Video: `experiments/day24_ten_task_screen/task5/state2/ckptday22_s13999_screen_task5_state2.mp4`

Failure label: `pending_video_review`

### Task 8 / State 0

- Task: `put both moka pots on the stove`
- Policy queries: `104`
- Executed policy steps: `520`
- Last environment step: `529`
- EEF movement L2: `0.1905 m`
- Gripper sign flips: `6`
- JSONL: `experiments/day24_ten_task_screen/task8/state0/ckptday22_s13999_screen_task8_state0.jsonl`
- Video: `experiments/day24_ten_task_screen/task8/state0/ckptday22_s13999_screen_task8_state0.mp4`

Failure label: `pending_video_review`

### Task 8 / State 1

- Task: `put both moka pots on the stove`
- Policy queries: `104`
- Executed policy steps: `520`
- Last environment step: `529`
- EEF movement L2: `0.3951 m`
- Gripper sign flips: `26`
- JSONL: `experiments/day24_ten_task_screen/task8/state1/ckptday22_s13999_screen_task8_state1.jsonl`
- Video: `experiments/day24_ten_task_screen/task8/state1/ckptday22_s13999_screen_task8_state1.mp4`

Failure label: `pending_video_review`

### Task 8 / State 2

- Task: `put both moka pots on the stove`
- Policy queries: `104`
- Executed policy steps: `520`
- Last environment step: `529`
- EEF movement L2: `0.3664 m`
- Gripper sign flips: `13`
- JSONL: `experiments/day24_ten_task_screen/task8/state2/ckptday22_s13999_screen_task8_state2.jsonl`
- Video: `experiments/day24_ten_task_screen/task8/state2/ckptday22_s13999_screen_task8_state2.mp4`

Failure label: `pending_video_review`

### Task 9 / State 1

- Task: `put the yellow and white mug in the microwave and close it`
- Policy queries: `104`
- Executed policy steps: `520`
- Last environment step: `529`
- EEF movement L2: `0.4041 m`
- Gripper sign flips: `2`
- JSONL: `experiments/day24_ten_task_screen/task9/state1/ckptday22_s13999_screen_task9_state1.jsonl`
- Video: `experiments/day24_ten_task_screen/task9/state1/ckptday22_s13999_screen_task9_state1.mp4`

Failure label: `pending_video_review`
