import collections
import importlib.util
import json
import os
import time
from pathlib import Path

import imageio
import numpy as np


TASK_SUITE_NAME = os.environ.get("TASK_SUITE_NAME", "libero_10")
TASK_ID = int(os.environ.get("TASK_ID", "5"))
INITIAL_STATE_ID = int(os.environ.get("INITIAL_STATE_ID", "0"))
ENV_SEED = int(os.environ.get("ENV_SEED", "7"))

HOST = os.environ.get("POLICY_HOST", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "8001"))

NUM_STEPS_WAIT = 10
MAX_POLICY_STEPS = int(
    os.environ.get("MAX_POLICY_STEPS", "30")
)
REPLAN_STEPS = int(
    os.environ.get("REPLAN_STEPS", "1")
)
CHECKPOINT_LABEL = os.environ.get(
    "CHECKPOINT_LABEL",
    "999",
)

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        "experiments/day20_rollout_smoke",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_DIR / (
    f"ckpt{CHECKPOINT_LABEL}_task{TASK_ID}_"
    f"state{INITIAL_STATE_ID}.jsonl"
)
VIDEO_PATH = OUTPUT_DIR / (
    f"ckpt{CHECKPOINT_LABEL}_task{TASK_ID}_"
    f"state{INITIAL_STATE_ID}.mp4"
)


def write_log(file_obj, record):
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_obj.flush()


def main():
    # 动态加载 openpi 官方 LIBERO 客户端，复用其接口与预处理。
    main_path = Path("examples/libero/main.py")
    spec = importlib.util.spec_from_file_location(
        "openpi_libero_main",
        main_path,
    )
    libero_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(libero_main)

    benchmark_dict = libero_main.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()

    task = task_suite.get_task(TASK_ID)
    initial_states = task_suite.get_task_init_states(TASK_ID)

    env, task_description = libero_main._get_libero_env(
        task,
        libero_main.LIBERO_ENV_RESOLUTION,
        ENV_SEED,
    )

    client = libero_main._websocket_client_policy.WebsocketClientPolicy(
        HOST,
        PORT,
    )

    replay_images = []
    action_plan = collections.deque()
    successful = False
    exception_message = None

    try:
        env.reset()
        obs = env.set_init_state(initial_states[INITIAL_STATE_ID])

        initial_eef_pos = np.asarray(obs["robot0_eef_pos"]).copy()
        initial_gripper = np.asarray(obs["robot0_gripper_qpos"]).copy()

        print("===== Rollout Configuration =====")
        print("task suite:", TASK_SUITE_NAME)
        print("task id:", TASK_ID)
        print("initial state id:", INITIAL_STATE_ID)
        print("task:", task_description)
        print("server:", f"{HOST}:{PORT}")
        print("replan steps:", REPLAN_STEPS)
        print("max policy steps:", MAX_POLICY_STEPS)

        with LOG_PATH.open("w", encoding="utf-8") as log_file:
            write_log(
                log_file,
                {
                    "event": "episode_start",
                    "task_suite": TASK_SUITE_NAME,
                    "task_id": TASK_ID,
                    "initial_state_id": INITIAL_STATE_ID,
                    "environment_seed": ENV_SEED,
                    "task_description": task_description,
                    "checkpoint": CHECKPOINT_LABEL,
                    "replan_steps": REPLAN_STEPS,
                    "initial_eef_pos": initial_eef_pos.tolist(),
                    "initial_gripper_qpos": initial_gripper.tolist(),
                },
            )

            total_steps = NUM_STEPS_WAIT + MAX_POLICY_STEPS

            for t in range(total_steps):
                # 先等待物体在仿真中稳定。
                if t < NUM_STEPS_WAIT:
                    action = np.asarray(
                        libero_main.LIBERO_DUMMY_ACTION,
                        dtype=np.float32,
                    )
                    obs, reward, done, info = env.step(action.tolist())

                    write_log(
                        log_file,
                        {
                            "event": "environment_step",
                            "phase": "wait",
                            "step": t,
                            "action": action.tolist(),
                            "reward": float(reward),
                            "done": bool(done),
                        },
                    )
                    continue

                # 与官方 main.py 完全一致的图像预处理。
                img = np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                )
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                img = libero_main.image_tools.convert_to_uint8(
                    libero_main.image_tools.resize_with_pad(
                        img,
                        libero_main.LIBERO_ENV_RESOLUTION,
                        libero_main.LIBERO_ENV_RESOLUTION,
                    )
                )
                wrist_img = libero_main.image_tools.convert_to_uint8(
                    libero_main.image_tools.resize_with_pad(
                        wrist_img,
                        libero_main.LIBERO_ENV_RESOLUTION,
                        libero_main.LIBERO_ENV_RESOLUTION,
                    )
                )

                replay_images.append(img)

                if not action_plan:
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            libero_main._quat2axisangle(
                                obs["robot0_eef_quat"]
                            ),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }

                    start_time = time.perf_counter()
                    response = client.infer(element)
                    inference_ms = (
                        time.perf_counter() - start_time
                    ) * 1000.0

                    action_chunk = np.asarray(
                        response["actions"],
                        dtype=np.float32,
                    )

                    if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                        raise ValueError(
                            "Unexpected action chunk shape: "
                            f"{action_chunk.shape}"
                        )

                    if not np.isfinite(action_chunk).all():
                        raise ValueError(
                            "Action chunk contains NaN or Inf."
                        )

                    action_plan.extend(
                        action_chunk[:REPLAN_STEPS]
                    )

                    print(
                        f"policy step={t - NUM_STEPS_WAIT:02d}, "
                        f"chunk={action_chunk.shape}, "
                        f"inference={inference_ms:.2f} ms, "
                        f"min={action_chunk.min():.4f}, "
                        f"max={action_chunk.max():.4f}, "
                        f"gripper={action_chunk[0, -1]:.4f}"
                    )

                    write_log(
                        log_file,
                        {
                            "event": "policy_query",
                            "step": t,
                            "inference_ms": inference_ms,
                            "state": state.tolist(),
                            "action_chunk_shape": list(
                                action_chunk.shape
                            ),
                            "action_chunk_min": float(
                                action_chunk.min()
                            ),
                            "action_chunk_max": float(
                                action_chunk.max()
                            ),
                            "first_action": action_chunk[0].tolist(),
                        },
                    )

                action = np.asarray(
                    action_plan.popleft(),
                    dtype=np.float32,
                )

                obs, reward, done, info = env.step(action.tolist())

                write_log(
                    log_file,
                    {
                        "event": "environment_step",
                        "phase": "policy",
                        "step": t,
                        "action": action.tolist(),
                        "reward": float(reward),
                        "done": bool(done),
                        "eef_pos": np.asarray(
                            obs["robot0_eef_pos"]
                        ).tolist(),
                        "gripper_qpos": np.asarray(
                            obs["robot0_gripper_qpos"]
                        ).tolist(),
                    },
                )

                if done:
                    successful = True
                    print("Environment reported task success.")
                    break

            final_eef_pos = np.asarray(obs["robot0_eef_pos"]).copy()
            final_gripper = np.asarray(
                obs["robot0_gripper_qpos"]
            ).copy()

            movement = float(
                np.linalg.norm(final_eef_pos - initial_eef_pos)
            )

            write_log(
                log_file,
                {
                    "event": "episode_end",
                    "success": successful,
                    "final_eef_pos": final_eef_pos.tolist(),
                    "final_gripper_qpos": final_gripper.tolist(),
                    "eef_movement_l2": movement,
                },
            )

            print("\n===== Smoke Test Result =====")
            print("success:", successful)
            print("initial eef pos:", initial_eef_pos)
            print("final eef pos:", final_eef_pos)
            print("EEF movement L2:", movement)
            print("initial gripper:", initial_gripper)
            print("final gripper:", final_gripper)

    except Exception as exc:
        exception_message = f"{type(exc).__name__}: {exc}"
        print("\nROLLOUT SMOKE TEST: FAILED")
        print(exception_message)
        raise

    finally:
        if replay_images:
            imageio.mimwrite(
                VIDEO_PATH,
                [np.asarray(frame) for frame in replay_images],
                fps=10,
            )

        env.close()

        print("\n===== Outputs =====")
        print("log:", LOG_PATH)
        print("video:", VIDEO_PATH)
        print("exception:", exception_message)

    print("\nROLLOUT SMOKE TEST: PASS")
    if not successful:
        print(
            "The interface completed successfully, but the task "
            "did not finish within this short smoke test."
        )


if __name__ == "__main__":
    main()
