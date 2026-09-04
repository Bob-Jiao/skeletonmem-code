"""Render LingBot-VA decoded predictions beside OOD simulator rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = Path("/root/zouyude/FastWAM/experiments/libero")
UTIL_DIR = PROJECT_ROOT / "wan_va" / "utils" / "Simple_Remote_Infer"
for path in (PROJECT_ROOT, COMMON_DIR, UTIL_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from evaluation.libero.client import (  # noqa: E402
    EVAL_PROTOCOL_FASTWAM_LEROBOT,
    _extract_obs,
    _policy_action_to_env_action,
    _settling_action,
    construct_single_env,
)
from libero.libero import benchmark  # noqa: E402
from qualitative_video_utils import (  # noqa: E402
    CAPTURE_STRIDE,
    MAX_CONTROL_STEPS,
    SUITES,
    TASK_IDS,
    is_complete,
    observation_pair,
    open_video_writer,
    pair_frame,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--max-control-steps", type=int)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--task-ids", nargs="+", type=int, default=TASK_IDS)
    return parser.parse_args()


def interpolate_video_frame(video: np.ndarray, local_step: int, total_actions: int) -> Image.Image:
    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] == 0:
        raise ValueError(f"Expected decoded video [T,H,W,3], got {video.shape}.")
    position = min(1.0, float(local_step) / float(max(1, total_actions))) * float(video.shape[0] - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    first = Image.fromarray(video[lower])
    if lower == upper:
        return first
    return Image.blend(first, Image.fromarray(video[upper]), alpha=position - lower)


def run_episode(*, args, model: WebsocketClientPolicy, suite: str, task_id: int) -> dict[str, Any]:
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    task_description = task.language
    env = construct_single_env(
        {
            "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
            "camera_heights": 128,
            "camera_widths": 128,
        }
    )
    if env is None:
        raise RuntimeError(f"Failed to create LIBERO env for {suite}/task{task_id}.")
    env.seed(int(args.seed))
    max_steps = int(MAX_CONTROL_STEPS[suite])
    if args.max_control_steps is not None:
        max_steps = min(max_steps, int(args.max_control_steps))
    stem = f"{suite}_task{task_id:02d}"
    running_video = args.output_dir / f"{stem}.running.mp4"
    final_video = args.output_dir / f"{stem}.mp4"
    writer = open_video_writer(running_video, args.fps)
    control_step = 0
    env_step = 0
    chunk_index = 0
    frame_count = 0
    last_capture_step = -1
    success = False
    first_chunk = True

    try:
        obs = env.reset()
        for _ in range(int(args.num_wait_steps)):
            obs, _, done, _ = env.step(_settling_action(EVAL_PROTOCOL_FASTWAM_LEROBOT))
            env_step += 1
            if done:
                raise RuntimeError("Episode terminated during settling.")
        current_obs = _extract_obs(obs, EVAL_PROTOCOL_FASTWAM_LEROBOT)
        model.infer(dict(reset=True, prompt=task_description))

        while control_step < max_steps and not success:
            response = model.infer(
                dict(obs=current_obs, prompt=task_description, return_video=True)
            )
            action = np.asarray(response["action"])
            decoded_video = np.asarray(response["video"], dtype=np.uint8)
            if action.ndim != 3 or action.shape[0] != 7:
                raise ValueError(f"Unexpected action shape: {action.shape}")
            if action.shape[2] % 4 != 0:
                raise ValueError(f"Action horizon is not divisible by 4: {action.shape}")
            action_per_frame = action.shape[2] // 4
            start_idx = 1 if first_chunk else 0
            total_actions = int((action.shape[1] - start_idx) * action.shape[2])
            local_step = 0
            key_frames = []

            def capture(terminal: bool = False) -> None:
                nonlocal frame_count, last_capture_step
                if control_step == last_capture_step:
                    return
                generated = interpolate_video_frame(decoded_video, local_step, total_actions)
                simulator = observation_pair(
                    current_obs["observation.images.agentview_rgb"],
                    current_obs["observation.images.eye_in_hand_rgb"],
                )
                writer.append_data(
                    pair_frame(
                        generated,
                        simulator,
                        generated_label="generated: native decoded rollout",
                        detail=f"step {control_step}/{max_steps}" + (" terminal" if terminal else ""),
                    )
                )
                frame_count += 1
                last_capture_step = control_step

            if control_step % CAPTURE_STRIDE == 0:
                capture()
            for frame_idx in range(start_idx, action.shape[1]):
                for action_idx in range(action.shape[2]):
                    if control_step >= max_steps:
                        break
                    policy_action = action[:, frame_idx, action_idx]
                    env_action = _policy_action_to_env_action(
                        policy_action, EVAL_PROTOCOL_FASTWAM_LEROBOT
                    )
                    obs, _, done, _ = env.step(env_action)
                    current_obs = _extract_obs(obs, EVAL_PROTOCOL_FASTWAM_LEROBOT)
                    local_step += 1
                    control_step += 1
                    env_step += 1
                    success = bool(done)
                    terminal = success or control_step >= max_steps
                    if control_step % CAPTURE_STRIDE == 0 or terminal:
                        capture(terminal=terminal)
                    if terminal:
                        break
                    if action_idx % action_per_frame == action_per_frame - 1:
                        key_frames.append(current_obs)
                if success or control_step >= max_steps:
                    break

            chunk_index += 1
            first_chunk = False
            print(
                f"[lingbot-va] {suite}/task{task_id} chunk={chunk_index} "
                f"control={control_step}/{max_steps} frames={frame_count} success={success}",
                flush=True,
            )
            if not success and control_step < max_steps:
                model.infer(
                    dict(
                        obs=key_frames,
                        compute_kv_cache=True,
                        imagine=False,
                        state=action,
                    )
                )
    finally:
        writer.close()
        env.close()

    running_video.replace(final_video)
    result = {
        "complete": True,
        "model": "lingbot-va",
        "suite": suite,
        "task_id": task_id,
        "task_description": task_description,
        "seed": int(args.seed),
        "initial_state_mode": "reset",
        "success": success,
        "control_steps": control_step,
        "env_steps_including_settling": env_step,
        "replan_chunks": chunk_index,
        "video_frames": frame_count,
        "fps": float(args.fps),
        "layout": ["native decoded generated rollout", "simulator rendering"],
        "generated_semantics": "native Wan VAE decode of the video latent returned with each action chunk",
        "checkpoint": str(args.checkpoint.resolve()),
    }
    write_json_atomic(args.output_dir / f"{stem}.json", result)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = WebsocketClientPolicy(host=args.host, port=int(args.port))
    if args.smoke_output_dir is not None:
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.output_dir = args.smoke_output_dir
        smoke_args.output_dir.mkdir(parents=True, exist_ok=True)
        smoke_args.max_control_steps = 8
        if not is_complete(smoke_args.output_dir, SUITES[0], 0):
            run_episode(
                args=smoke_args,
                model=model,
                suite=SUITES[0],
                task_id=0,
            )
        write_json_atomic(
            smoke_args.output_dir / "smoke_complete.json",
            {"complete": True, "model": "lingbot-va", "control_steps": 8},
        )
    new_results = []
    for suite in SUITES:
        for task_id in args.task_ids:
            if is_complete(args.output_dir, suite, int(task_id)):
                print(f"[lingbot-va] skip complete {suite}/task{task_id}", flush=True)
                continue
            new_results.append(
                run_episode(
                    args=args,
                    model=model,
                    suite=suite,
                    task_id=int(task_id),
                )
            )
    write_json_atomic(
        args.output_dir / "model_complete.json",
        {"complete": True, "model": "lingbot-va", "new_results": new_results},
    )


if __name__ == "__main__":
    main()
