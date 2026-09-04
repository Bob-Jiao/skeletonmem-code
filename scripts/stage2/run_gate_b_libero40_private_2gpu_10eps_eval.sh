#!/usr/bin/env bash
set -Eeuo pipefail

run_id=${1:?usage: run_gate_b_libero40_private_2gpu_10eps_eval.sh <run_id> <transformer_dir> [test_num=10]}
transformer=${2:?usage: run_gate_b_libero40_private_2gpu_10eps_eval.sh <run_id> <transformer_dir> [test_num=10]}
test_num=${3:-10}

root=/data/jiaoguanbo/skeletonmem
repo=${root}/repos/lingbot-va-private-mirror-main
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python
libero_root=/data/jiaoguanbo/LIBERO
icd=/data/jiaoguanbo/lingbot-va/outputs/0729_robotwin_mem_pick_step1500_first_call_bundle_20260729/0729_nvidia_icd_egl.json
out_root=${root}/results/stage2/gate_b/${run_id}
result_dir=${out_root}/results
log_dir=${root}/logs/stage2/gate_b/${run_id}
libero_config_dir=${out_root}/libero_config
ws_port_base=${WS_PORT_BASE:-39266}
master_port_base=${MASTER_PORT_BASE:-39366}

if [[ ! "$test_num" =~ ^[1-9][0-9]*$ ]]; then
  echo "test_num must be a positive integer: ${test_num}" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "missing python: ${python_bin}" >&2
  exit 3
fi
if [[ ! -d "${repo}" ]]; then
  echo "missing repo: ${repo}" >&2
  exit 4
fi
if [[ ! -s "${transformer}/config.json" || ! -s "${transformer}/diffusion_pytorch_model.safetensors" ]]; then
  echo "invalid transformer dir: ${transformer}" >&2
  exit 5
fi
if [[ ! -d "${libero_root}" ]]; then
  echo "missing LIBERO root: ${libero_root}" >&2
  exit 6
fi
if [[ ! -s "${icd}" ]]; then
  echo "missing EGL ICD: ${icd}" >&2
  exit 7
fi
if [[ -e "${out_root}/COMPLETE" ]]; then
  echo "eval already complete: ${out_root}/COMPLETE"
  exit 0
fi

mkdir -p "${result_dir}" "${log_dir}" "${libero_config_dir}"
install -m 0644 "${root}/runtime/libero_config/config.yaml" "${libero_config_dir}/config.yaml"

server_pids=()
client_pids=()

cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  for pid in "${client_pids[@]}"; do kill "${pid}" 2>/dev/null; done
  for pid in "${server_pids[@]}"; do kill "${pid}" 2>/dev/null; done
  for pid in "${client_pids[@]}"; do wait "${pid}" 2>/dev/null; done
  for pid in "${server_pids[@]}"; do wait "${pid}" 2>/dev/null; done
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cat >"${out_root}/launch_manifest.json" <<EOF
{
  "run_id": "${run_id}",
  "repo": "${repo}",
  "checkpoint": "$(realpath -e "${transformer}")",
  "out_root": "${out_root}",
  "test_num": ${test_num},
  "rollouts": 400,
  "eval_protocol": "fastwam_lerobot",
  "initial_state_mode": "fixed",
  "max_env_steps": 800,
  "num_steps_wait": 5,
  "seed": 42,
  "save_video": true,
  "save_action_trace": true,
  "shards": [
    {"gpu": 0, "port": ${ws_port_base}, "suites": ["libero_10", "libero_goal"]},
    {"gpu": 1, "port": $((ws_port_base + 1)), "suites": ["libero_spatial", "libero_object"]}
  ]
}
EOF

echo "[config] repo=${repo}"
echo "[config] checkpoint=${transformer}"
echo "[config] out=${out_root}"
echo "[config] test_num=${test_num} eval_protocol=fastwam_lerobot save_video=1 save_action_trace=1"
echo "[config] shard0 suites=libero_10,libero_goal gpu=0 port=${ws_port_base}"
echo "[config] shard1 suites=libero_spatial,libero_object gpu=1 port=$((ws_port_base + 1))"

for shard in 0 1; do
  port=$((ws_port_base + shard))
  master_port=$((master_port_base + shard))
  server_log="${log_dir}/server_gpu${shard}.log"
  env \
    CUDA_VISIBLE_DEVICES="${shard}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${master_port}" \
    RANK=0 \
    LOCAL_RANK=0 \
    WORLD_SIZE=1 \
    HOME="${root}/runtime/home" \
    XDG_CACHE_HOME="${root}/runtime/cache" \
    HF_HOME="${root}/runtime/cache/huggingface" \
    LD_LIBRARY_PATH="/data/jiaoguanbo/runtime/nvidia-gl-595/usr/lib/x86_64-linux-gnu:/data/zouyude/conda/envs/fastwam/lib:${LD_LIBRARY_PATH:-}" \
    VK_ICD_FILENAMES="${icd}" \
    __EGL_VENDOR_LIBRARY_FILENAMES="${icd}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${repo}/wan_va:${repo}" \
    PYTHONUNBUFFERED=1 \
    "${python_bin}" "${repo}/wan_va/wan_va_server.py" \
      --config-name libero_all \
      --port "${port}" \
      --transformer-path "${transformer}" \
      >"${server_log}" 2>&1 &
  server_pids+=("$!")
done

echo "[server] pids ${server_pids[*]}"
for shard in 0 1; do
  port=$((ws_port_base + shard))
  server_log="${log_dir}/server_gpu${shard}.log"
  deadline=$((SECONDS + 1800))
  while ! curl --max-time 2 -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; do
    if ! kill -0 "${server_pids[$shard]}" 2>/dev/null; then
      echo "[ERROR] server shard ${shard} exited before ready" >&2
      tail -100 "${server_log}" >&2
      exit 10
    fi
    if ((SECONDS >= deadline)); then
      echo "[ERROR] timed out waiting for server shard ${shard}" >&2
      tail -100 "${server_log}" >&2
      exit 11
    fi
    sleep 5
  done
  echo "[server] ready port=${port}"
done

for shard in 0 1; do
  port=$((ws_port_base + shard))
  client_log="${log_dir}/client_gpu${shard}.log"
  if [[ "${shard}" -eq 0 ]]; then
    suites=(libero_10 libero_goal)
    client_cvd=0
  else
    suites=(libero_spatial libero_object)
    client_cvd=1,0
  fi
  env \
    CUDA_VISIBLE_DEVICES="${client_cvd}" \
    MUJOCO_EGL_DEVICE_ID=0 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    HOME="${root}/runtime/home" \
    XDG_CACHE_HOME="${root}/runtime/cache" \
    HF_HOME="${root}/runtime/cache/huggingface" \
    LD_LIBRARY_PATH="/data/jiaoguanbo/runtime/nvidia-gl-595/usr/lib/x86_64-linux-gnu:/data/zouyude/conda/envs/fastwam/lib:${LD_LIBRARY_PATH:-}" \
    VK_ICD_FILENAMES="${icd}" \
    __EGL_VENDOR_LIBRARY_FILENAMES="${icd}" \
    LIBERO_CONFIG_PATH="${libero_config_dir}" \
    PYTHONPATH="${libero_root}:${repo}" \
    PYTHONUNBUFFERED=1 \
    "${python_bin}" "${repo}/evaluation/libero/client.py" \
      --libero-benchmark "${suites[@]}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --test-num "${test_num}" \
      --seed 42 \
      --eval-protocol fastwam_lerobot \
      --resume \
      --save-video \
      --save-action-trace \
      --out-dir "${result_dir}" \
      >"${client_log}" 2>&1 &
  client_pids+=("$!")
done

echo "[client] pids ${client_pids[*]}"
wait "${client_pids[0]}"
echo "[client] shard0 complete"
wait "${client_pids[1]}"
echo "[client] shard1 complete"

env \
  LIBERO_CONFIG_PATH="${libero_config_dir}" \
  LD_LIBRARY_PATH="/data/jiaoguanbo/runtime/nvidia-gl-595/usr/lib/x86_64-linux-gnu:/data/zouyude/conda/envs/fastwam/lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="${libero_root}:${repo}" \
  "${python_bin}" - "${result_dir}" "${test_num}" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
test_num = int(sys.argv[2])
suites = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
missing = []
incomplete = []
protocol_bad = []
total_success = 0
total_rollouts = 0

for suite in suites:
    for task_idx in range(10):
        path = result_dir / f"{suite}_{task_idx}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        record = json.loads(path.read_text())
        total = int(record.get("total_num", 0))
        if total != test_num:
            incomplete.append(f"{path}: {total}/{test_num}")
        if record.get("eval_protocol") != "fastwam_lerobot":
            protocol_bad.append(f"{path}: {record.get('eval_protocol')!r}")
        total_success += int(record.get("succ_num", 0))
        total_rollouts += total

if missing or incomplete or protocol_bad:
    if missing:
        print("[ERROR] missing:\n" + "\n".join(missing), file=sys.stderr)
    if incomplete:
        print("[ERROR] incomplete:\n" + "\n".join(incomplete), file=sys.stderr)
    if protocol_bad:
        print("[ERROR] protocol mismatch:\n" + "\n".join(protocol_bad), file=sys.stderr)
    raise SystemExit(1)

print(f"[validate] total_success={total_success}/{total_rollouts}")
PY

touch "${out_root}/COMPLETE"
echo "[complete] ${out_root}/COMPLETE"
