import numpy as np
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wan_va" / "utils" / "Simple_Remote_Infer"))
from deploy.websocket_client_policy import WebsocketClientPolicy

from libero.libero import benchmark
import time
import math
import random
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
try:
    from lerobot.datasets.utils import write_json
except ImportError:
    def write_json(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
import os
import imageio
import cv2


EVAL_PROTOCOL_OFFICIAL_LONG = "official_long"
EVAL_PROTOCOL_FASTWAM_LEROBOT = "fastwam_lerobot"
EVAL_PROTOCOLS = (
    EVAL_PROTOCOL_OFFICIAL_LONG,
    EVAL_PROTOCOL_FASTWAM_LEROBOT,
)


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs, eval_protocol):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Official LingBot-VA LIBERO-Long videos use a vertical flip. The LeRobot
    dataset built from FastWAM videos uses FastWAM's 180-degree rotation.
    """
    if eval_protocol == EVAL_PROTOCOL_OFFICIAL_LONG:
        image_slice = (slice(None, None, -1), slice(None))
    elif eval_protocol == EVAL_PROTOCOL_FASTWAM_LEROBOT:
        image_slice = (slice(None, None, -1), slice(None, None, -1))
    else:
        raise ValueError(f"Unsupported eval protocol: {eval_protocol}")

    agentview = np.ascontiguousarray(obs["agentview_image"][image_slice])
    eye_in_hand = np.ascontiguousarray(
        obs["robot0_eye_in_hand_image"][image_slice]
    )
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def _settling_action(eval_protocol):
    if eval_protocol == EVAL_PROTOCOL_OFFICIAL_LONG:
        return np.zeros(7, dtype=np.float32)
    if eval_protocol == EVAL_PROTOCOL_FASTWAM_LEROBOT:
        # FastWAM uses -1 as the environment-domain "open gripper" command.
        return np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)
    raise ValueError(f"Unsupported eval protocol: {eval_protocol}")


def _policy_action_to_env_action(policy_action, eval_protocol):
    """Convert a model-domain action into an independent environment action."""
    env_action = np.asarray(policy_action, dtype=np.float32).copy()
    if env_action.shape != (7,):
        raise ValueError(
            f"Expected one 7-D LIBERO action, got shape {env_action.shape}"
        )
    if not np.isfinite(env_action).all():
        raise ValueError(f"Action contains non-finite values: {env_action}")

    if eval_protocol == EVAL_PROTOCOL_OFFICIAL_LONG:
        return env_action
    if eval_protocol == EVAL_PROTOCOL_FASTWAM_LEROBOT:
        # LeRobot training actions use 0=closed, 1=open. Panda/robosuite uses
        # +1=close, -1=open. Keep the original model action untouched because
        # it is sent back to the server as the action-conditioning KV cache.
        env_action[-1] = np.sign(1.0 - 2.0 * env_action[-1])
        return env_action
    raise ValueError(f"Unsupported eval protocol: {eval_protocol}")


def init_single_env(env_in, init_state, eval_protocol, num_steps_wait):
    obs = env_in.reset()
    if init_state is not None:
        obs = env_in.set_init_state(init_state)
    settling_action = _settling_action(eval_protocol)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env_in.step(settling_action)
    return _extract_obs(obs, eval_protocol)


def env_one_step(env_in, policy_action, eval_protocol):
    env_action = _policy_action_to_env_action(policy_action, eval_protocol)
    obs, _, done, _ = env_in.step(env_action)
    return _extract_obs(obs, eval_protocol), done, env_action


def run_one(
    model,
    libero_benchmark,
    task_idx,
    out_dir,
    episode_idx,
    seed,
    eval_protocol,
    max_env_steps,
    num_steps_wait,
    save_video_enabled=False,
    save_action_trace=False,
):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    reset_initial_state = str(libero_benchmark).endswith("_ood")
    init_states = None if reset_initial_state else benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    cur_env.seed(seed + episode_idx)
    first_obs = init_single_env(
        cur_env,
        None if reset_initial_state else init_states[episode_idx % init_states.shape[0]],
        eval_protocol,
        num_steps_wait,
    )

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = [] if save_video_enabled else None
    policy_action_trace = [] if save_action_trace else None
    env_action_trace = [] if save_action_trace else None
    done = False
    first = True
    action_trace_written = False
    rollout_truncated = False
    while cur_env.env.timestep < max_env_steps:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret['action']

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                if cur_env.env.timestep >= max_env_steps:
                    rollout_truncated = True
                    break
                ee_action = action[:, i, j]
                observes, done, env_action = env_one_step(
                    cur_env, ee_action, eval_protocol
                )
                if not done and cur_env.env.timestep >= max_env_steps:
                    rollout_truncated = True
                if policy_action_trace is not None:
                    policy_action_trace.append(np.asarray(ee_action).copy())
                    env_action_trace.append(env_action.copy())
                if not action_trace_written:
                    print(
                        "[action-trace] "
                        f"protocol={eval_protocol} "
                        f"model_gripper={float(ee_action[-1]):.6f} "
                        f"env_gripper={float(env_action[-1]):.1f}"
                    )
                    action_trace_written = True
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    if full_obs_list is not None:
                        full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done or rollout_truncated:
                break

        first = False

        if done or rollout_truncated:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    episode_dir = (
        Path(out_dir)
        / libero_benchmark
        / f"{task_idx}_{prompt.replace(' ', '_')}"
    )
    if save_video_enabled:
        out_file = episode_dir / f"{episode_idx}_{done}.mp4"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        save_video(
            real_obs_list=full_obs_list,
            save_path=out_file,
            fps=60,
            video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
        )

    if save_action_trace:
        episode_dir.mkdir(exist_ok=True, parents=True)
        policy_actions = np.asarray(policy_action_trace, dtype=np.float32)
        env_actions = np.asarray(env_action_trace, dtype=np.float32)
        trace_path = episode_dir / f"{episode_idx}_{done}_actions.npz"
        np.savez_compressed(
            trace_path,
            policy_actions=policy_actions,
            env_actions=env_actions,
            eval_protocol=np.asarray(eval_protocol),
            final_env_timestep=np.asarray(cur_env.env.timestep),
            max_env_steps=np.asarray(max_env_steps),
            rollout_truncated=np.asarray(rollout_truncated),
        )
        if len(policy_actions):
            arm_norms = np.linalg.norm(policy_actions[:, :6], axis=1)
            env_gripper_values, env_gripper_counts = np.unique(
                env_actions[:, -1], return_counts=True
            )
            gripper_histogram = dict(
                zip(env_gripper_values.tolist(), env_gripper_counts.tolist())
            )
            expected_env_gripper = np.sign(
                1.0 - 2.0 * policy_actions[:, -1]
            )
            mapping_ok = (
                eval_protocol != EVAL_PROTOCOL_FASTWAM_LEROBOT
                or np.array_equal(env_actions[:, -1], expected_env_gripper)
            )
            print(
                "[action-summary] "
                f"count={len(policy_actions)} "
                f"finite={bool(np.isfinite(policy_actions).all())} "
                f"arm_norm_p50={float(np.quantile(arm_norms, 0.50)):.6f} "
                f"arm_norm_p95={float(np.quantile(arm_norms, 0.95)):.6f} "
                f"arm_norm_max={float(arm_norms.max()):.6f} "
                f"policy_gripper_range="
                f"[{float(policy_actions[:, -1].min()):.6f},"
                f"{float(policy_actions[:, -1].max()):.6f}] "
                f"env_gripper_histogram={gripper_histogram} "
                f"mapping_ok={mapping_ok} "
                f"trace={trace_path}"
            )

    cur_env.close()
    return done


def _select_task_ids(libero_benchmark, num_tasks, task_range, task_sample_ratio, task_sample_seed):
    if task_range is None:
        task_ids = list(range(num_tasks))
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        task_ids = list(range(task_range[0], task_range[1]))

    if task_sample_ratio is not None:
        if not (0.0 < task_sample_ratio <= 1.0):
            raise ValueError(f"task_sample_ratio must be in (0, 1], got {task_sample_ratio}")
        if task_sample_ratio < 1.0:
            n_sample = max(1, int(math.ceil(len(task_ids) * task_sample_ratio)))
            rng = random.Random(f"{task_sample_seed}:{libero_benchmark}")
            task_ids = sorted(rng.sample(task_ids, n_sample))
    return task_ids


def _shard_task_ids(task_ids, task_shard_index, task_shard_count):
    if task_shard_count < 1:
        raise ValueError(f"task_shard_count must be at least 1, got {task_shard_count}")
    if not 0 <= task_shard_index < task_shard_count:
        raise ValueError(
            "task_shard_index must be in "
            f"[0, {task_shard_count}), got {task_shard_index}"
        )
    return task_ids[task_shard_index::task_shard_count]


def run(
    libero_benchmark,
    host,
    port,
    out_dir,
    test_num,
    task_range=None,
    task_sample_ratio=None,
    task_sample_seed=42,
    task_shard_index=0,
    task_shard_count=1,
    create_task_list_only=False,
    resume=False,
    seed=42,
    eval_protocol=EVAL_PROTOCOL_OFFICIAL_LONG,
    max_env_steps=800,
    num_steps_wait=5,
    save_video_enabled=False,
    save_action_trace=False,
):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if isinstance(libero_benchmark, str):
        libero_benchmarks = [libero_benchmark]
    else:
        libero_benchmarks = libero_benchmark

    model = None
    if num_steps_wait < 0:
        raise ValueError(f"num_steps_wait must be non-negative, got {num_steps_wait}")
    if max_env_steps <= num_steps_wait:
        raise ValueError(
            f"max_env_steps must be greater than num_steps_wait, got "
            f"{max_env_steps} <= {num_steps_wait}"
        )
    random.seed(seed)
    np.random.seed(seed)
    print(f"[eval] protocol={eval_protocol}")

    for suite_name in libero_benchmarks:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[suite_name]()
        num_tasks = benchmark_instance.get_num_tasks()
        sampled_task_ids = _select_task_ids(
            suite_name, num_tasks, task_range, task_sample_ratio, task_sample_seed
        )
        task_ids = _shard_task_ids(
            sampled_task_ids, task_shard_index, task_shard_count
        )
        progress_bar = tqdm(task_ids, total=len(task_ids))

        print(
            f"#################### Use benchmark: {suite_name}, num_tasks: {num_tasks}, "
            f"sampled_tasks: {len(sampled_task_ids)}, eval_tasks: {len(task_ids)}, "
            f"shard: {task_shard_index}/{task_shard_count} #############"
        )
        if (task_sample_ratio is not None and task_sample_ratio < 1.0) or create_task_list_only:
            ratio_label = "all" if task_sample_ratio is None else f"{task_sample_ratio:g}"
            shard_label = (
                ""
                if task_shard_count == 1
                else f"_shard_{task_shard_index:02d}-of-{task_shard_count:02d}"
            )
            task_file = Path(out_dir) / (
                f"{suite_name}_sample_ratio_{ratio_label}_seed_{task_sample_seed}"
                f"{shard_label}.txt"
            )
            task_file.parent.mkdir(exist_ok=True, parents=True)
            task_file.write_text("".join(f"{task_id}\n" for task_id in task_ids))
        if create_task_list_only:
            continue

        if model is None:
            model = WebsocketClientPolicy(host=host, port=port)

        video_save_root_dict = None

        for task_idx in progress_bar:
            out_file = Path(out_dir) / f"{suite_name}_{task_idx}.json"
            episode_start = 0
            succ_num = 0.0
            if resume and out_file.exists():
                result = json.loads(out_file.read_text())
                result_protocol = result.get("eval_protocol")
                if result_protocol is None:
                    if eval_protocol != EVAL_PROTOCOL_OFFICIAL_LONG:
                        raise ValueError(
                            f"Cannot resume {out_file}: it has no eval_protocol "
                            f"metadata, but this run uses {eval_protocol}. Use a "
                            "new output directory."
                        )
                elif result_protocol != eval_protocol:
                    raise ValueError(
                        f"Cannot resume {out_file}: stored eval_protocol="
                        f"{result_protocol}, current={eval_protocol}"
                    )
                result_max_env_steps = int(result.get("max_env_steps", 800))
                if result_max_env_steps != max_env_steps:
                    raise ValueError(
                        f"Cannot resume {out_file}: stored max_env_steps="
                        f"{result_max_env_steps}, current={max_env_steps}"
                    )
                result_num_steps_wait = int(result.get("num_steps_wait", 5))
                if result_num_steps_wait != num_steps_wait:
                    raise ValueError(
                        f"Cannot resume {out_file}: stored num_steps_wait="
                        f"{result_num_steps_wait}, current={num_steps_wait}"
                    )
                episode_start = int(result["total_num"])
                succ_num = float(result["succ_num"])
                if episode_start >= test_num:
                    print(f"Skip completed task {suite_name}/{task_idx}")
                    continue

            if video_save_root_dict is not None and task_idx in video_save_root_dict:
                video_save_list = os.listdir(os.path.join(out_dir, suite_name, video_save_root_dict[task_idx]))
                video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
                succ_num = float(len(video_states))
                episode_start = len(video_save_list)

            episode_list = range(episode_start, test_num)
            for episode_idx in tqdm(episode_list, total=len(episode_list)):
                res_i = run_one(
                    model=model,
                    libero_benchmark=suite_name,
                    task_idx=task_idx,
                    out_dir=out_dir,
                    episode_idx=episode_idx,
                    seed=seed,
                    eval_protocol=eval_protocol,
                    max_env_steps=max_env_steps,
                    num_steps_wait=num_steps_wait,
                    save_video_enabled=save_video_enabled,
                    save_action_trace=save_action_trace,
                )
                succ_num += res_i
                succ_rate = succ_num / (episode_idx + 1)
                print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
                out_file.parent.mkdir(exist_ok=True, parents=True)
                write_json({
                    "succ_num": succ_num,
                    "total_num": episode_idx + 1.,
                    "succ_rate": succ_rate,
                    "eval_protocol": eval_protocol,
                    "max_env_steps": max_env_steps,
                    "num_steps_wait": num_steps_wait,
                    "initial_state_mode": "reset" if suite_name.endswith("_ood") else "fixed",
                    }, out_file
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Inference server host",
    )
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        nargs="+",
        default=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        choices=[
            "libero_10",
            "libero_goal",
            "libero_spatial",
            "libero_object",
            "libero_90",
            "libero_100",
            "libero_mix",
            "libero_spatial_ood",
            "libero_object_ood",
            "libero_goal_ood",
        ],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=None,
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--task-sample-ratio",
        type=float,
        default=None,
        help="Task sample ratio. Uses the same suite-seeded sampling as ImageWAM.",
    )
    parser.add_argument(
        "--task-sample-seed",
        type=int,
        default=42,
        help="Task sample seed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="LIBERO environment and client RNG seed",
    )
    parser.add_argument(
        "--eval-protocol",
        type=str,
        choices=EVAL_PROTOCOLS,
        default=EVAL_PROTOCOL_OFFICIAL_LONG,
        help=(
            "Atomic observation/action convention. Use fastwam_lerobot for "
            "models trained on the FastWAM-derived LeRobot dataset."
        ),
    )
    parser.add_argument(
        "--task-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index applied after global task sampling",
    )
    parser.add_argument(
        "--task-shard-count",
        type=int,
        default=1,
        help="Number of round-robin shards applied after global task sampling",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=800,
        help="Maximum environment timestep per episode (including settling)",
    )
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=5,
        help="Settling steps executed immediately after reset",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--create-task-list-only",
        action="store_true",
        help="Create sampled task lists and exit without connecting to the server",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose result already contains all requested episodes",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video_enabled",
        action="store_true",
        help="Save a two-camera rollout video for each episode",
    )
    parser.add_argument(
        "--save-action-trace",
        action="store_true",
        help="Save model-domain and environment-domain actions as compressed NPZ",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
