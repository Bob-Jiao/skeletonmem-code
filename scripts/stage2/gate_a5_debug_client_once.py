#!/usr/bin/env python3
"""Minimal A5 LIBERO client debug probe.

This script is intentionally not a benchmark runner. It prints flush-marked
milestones to locate where the official client blocks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def extract_obs(obs):
    import numpy as np

    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(obs["agentview_image"][::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    report_lines = []

    def mark(name: str, extra: str = "") -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {name} {extra}".rstrip()
        report_lines.append(line)
        print(line, flush=True)
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_report).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    mark("start")
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[args.benchmark]()
    mark("benchmark_ok", f"num_tasks={benchmark_instance.get_num_tasks()}")
    prompt = benchmark_instance.get_task(args.task_idx).language
    mark("task_ok", prompt)

    mark("connect_start", f"port={args.port}")
    model = WebsocketClientPolicy(port=args.port)
    mark("connect_ok")

    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(args.task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    mark("env_construct_start")
    env = OffScreenRenderEnv(**env_args)
    mark("env_construct_ok")

    init_states = benchmark_instance.get_task_init_states(args.task_idx)
    mark("init_states_ok", f"shape={tuple(init_states.shape)}")

    mark("env_reset_start")
    env.reset()
    env.set_init_state(init_states[args.episode_idx % init_states.shape[0]])
    raw_obs = None
    for _ in range(5):
        raw_obs, _, _, _ = env.step([0.0] * 7)
    obs = extract_obs(raw_obs)
    mark("env_reset_ok", f"keys={list(obs.keys())}")

    mark("model_reset_infer_start")
    reset_ret = model.infer(dict(reset=True, prompt=prompt))
    mark("model_reset_infer_ok", f"type={type(reset_ret)}")

    mark("model_obs_infer_start")
    ret = model.infer(dict(obs=obs, prompt=prompt))
    action = ret["action"]
    mark("model_obs_infer_ok", f"action_shape={getattr(action, 'shape', None)}")

    env.close()
    mark("probe_pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
