import collections
import hashlib
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


def array_sha256(array):
    """计算数组内容哈希，用于检查 observation/action 是否完全一致。"""
    contiguous = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def collect_back_region_diagnostics(env):
    """读取 Task 5 中书本中心点与 back containment region 的几何关系。"""
    core_env = getattr(env, "env", env)

    object_states = getattr(core_env, "object_states_dict", {})
    object_sites = getattr(core_env, "object_sites_dict", {})

    book_name = None
    if "black_book_1" in object_states:
        book_name = "black_book_1"
    else:
        book_name = next(
            (
                name
                for name in object_states
                if "black_book" in name
                and not name.endswith("_region")
            ),
            None,
        )

    site_name = next(
        (
            name
            for name in object_states
            if name.endswith("back_contain_region")
        ),
        None,
    )

    if book_name is None or site_name is None:
        return {
            "available": False,
            "reason": "Task does not expose black book and back containment site.",
            "book_name": book_name,
            "site_name": site_name,
        }

    book_state = object_states[book_name]
    site_state = object_states[site_name]

    # SiteObjectState.object_name 是 MuJoCo 中实际使用的 site 名称。
    site_sim_name = site_state.object_name

    site_model = object_sites.get(site_sim_name)
    if site_model is None:
        site_model = object_sites.get(site_name)

    if site_model is None:
        return {
            "available": False,
            "reason": "Back containment SiteObject was not found.",
            "book_name": book_name,
            "site_name": site_name,
            "site_sim_name": site_sim_name,
            "available_site_keys": sorted(object_sites.keys()),
        }

    sim = core_env.sim

    book_body_id = core_env.obj_body_id[book_name]
    book_position = np.asarray(
        sim.data.body_xpos[book_body_id],
        dtype=np.float64,
    ).copy()
    book_quaternion = np.asarray(
        sim.data.body_xquat[book_body_id],
        dtype=np.float64,
    ).copy()

    site_position = np.asarray(
        sim.data.get_site_xpos(site_sim_name),
        dtype=np.float64,
    ).copy()
    site_matrix = np.asarray(
        sim.data.get_site_xmat(site_sim_name),
        dtype=np.float64,
    ).reshape(3, 3).copy()

    local_half_size = np.asarray(
        site_model.size,
        dtype=np.float64,
    ).copy()

    # 与 SiteObject.in_box() 完全保持一致。
    world_aabb_half_size = np.abs(
        site_matrix @ local_half_size
    )

    lower_bound = site_position - world_aabb_half_size
    upper_bound = site_position + world_aabb_half_size

    # LIBERO 源码对 z 下边界额外放宽 1 cm。
    lower_bound[2] -= 0.01

    lower_margin = book_position - lower_bound
    upper_margin = upper_bound - book_position
    nearest_margin = np.minimum(lower_margin, upper_margin)

    axis_pass = (
        (book_position > lower_bound)
        & (book_position < upper_bound)
    )

    contact_result = bool(site_state.check_contact(book_state))
    contain_result = bool(site_state.check_contain(book_state))
    in_result = bool(contact_result and contain_result)

    return {
        "available": True,
        "book_name": book_name,
        "site_name": site_name,
        "site_sim_name": site_sim_name,
        "book_world_pos": book_position.tolist(),
        "book_world_quat": book_quaternion.tolist(),
        "site_world_pos": site_position.tolist(),
        "site_world_matrix": site_matrix.tolist(),
        "site_local_pos": np.asarray(
            site_model.site_pos,
            dtype=np.float64,
        ).tolist(),
        "site_local_quat": np.asarray(
            site_model.site_quat,
            dtype=np.float64,
        ).tolist(),
        "site_local_half_size": local_half_size.tolist(),
        "site_world_aabb_half_size": world_aabb_half_size.tolist(),
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist(),
        "lower_margin": lower_margin.tolist(),
        "upper_margin": upper_margin.tolist(),
        "nearest_axis_margin": nearest_margin.tolist(),
        "axis_pass": {
            "x": bool(axis_pass[0]),
            "y": bool(axis_pass[1]),
            "z": bool(axis_pass[2]),
        },
        "contact_result": contact_result,
        "contain_result": contain_result,
        "in_result": in_result,
        "environment_success": bool(env.check_success()),
    }


def print_back_region_diagnostics(label, diagnostics):
    print(f"\n===== {label} Predicate Diagnostics =====")

    if not diagnostics.get("available", False):
        print("available: False")
        print("reason:", diagnostics.get("reason"))
        return

    margins_mm = (
        np.asarray(
            diagnostics["nearest_axis_margin"],
            dtype=np.float64,
        )
        * 1000.0
    )

    print("book world pos:", diagnostics["book_world_pos"])
    print("site world pos:", diagnostics["site_world_pos"])
    print("site local half-size:", diagnostics["site_local_half_size"])
    print("lower bound:", diagnostics["lower_bound"])
    print("upper bound:", diagnostics["upper_bound"])
    print("nearest margins mm:", margins_mm.tolist())
    print("axis pass:", diagnostics["axis_pass"])
    print("contact:", diagnostics["contact_result"])
    print("contain:", diagnostics["contain_result"])
    print("In predicate:", diagnostics["in_result"])


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

        initial_predicate_diagnostics = (
            collect_back_region_diagnostics(env)
        )

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
                    "initial_predicate_diagnostics": (
                        initial_predicate_diagnostics
                    ),
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
                            "image_sha256": array_sha256(img),
                            "wrist_image_sha256": array_sha256(
                                wrist_img
                            ),
                            "action_chunk_sha256": array_sha256(
                                action_chunk
                            ),
                            "action_chunk": action_chunk.tolist(),
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

            final_predicate_diagnostics = (
                collect_back_region_diagnostics(env)
            )

            write_log(
                log_file,
                {
                    "event": "episode_end",
                    "success": successful,
                    "final_eef_pos": final_eef_pos.tolist(),
                    "final_gripper_qpos": final_gripper.tolist(),
                    "eef_movement_l2": movement,
                    "final_predicate_diagnostics": (
                        final_predicate_diagnostics
                    ),
                },
            )

            print("\n===== Smoke Test Result =====")
            print("success:", successful)
            print("initial eef pos:", initial_eef_pos)
            print("final eef pos:", final_eef_pos)
            print("EEF movement L2:", movement)
            print("initial gripper:", initial_gripper)
            print("final gripper:", final_gripper)

            print_back_region_diagnostics(
                "Initial",
                initial_predicate_diagnostics,
            )
            print_back_region_diagnostics(
                "Final",
                final_predicate_diagnostics,
            )

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
