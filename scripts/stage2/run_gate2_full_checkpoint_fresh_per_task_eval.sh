#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?usage: run_gate2_full_checkpoint_fresh_per_task_eval.sh <run_id> <eval_model> <expected_transformer_sha|auto> [test_num=10] [task_start=0] [task_end=3]}
model=${2:?usage: run_gate2_full_checkpoint_fresh_per_task_eval.sh <run_id> <eval_model> <expected_transformer_sha|auto> [test_num=10] [task_start=0] [task_end=3]}
expected_transformer_sha=${3:?usage: run_gate2_full_checkpoint_fresh_per_task_eval.sh <run_id> <eval_model> <expected_transformer_sha|auto> [test_num=10] [task_start=0] [task_end=3]}
test_num=${4:-10}
task_start=${5:-0}
task_end=${6:-3}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/repos/lingbot-va-gatea-official
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python
icd=/data/jiaoguanbo/lingbot-va/outputs/0729_robotwin_mem_pick_step1500_first_call_bundle_20260729/0729_nvidia_icd_egl.json

out_root=${root}/results/stage2/gate_a/a5_closed_loop/${run_id}
log_root=${root}/logs/stage2/gate_a
mkdir -p "${out_root}" "${log_root}"

if [ ! -x "${python_bin}" ]; then
  echo "missing python: ${python_bin}" >&2
  exit 2
fi
if [ ! -d "${source_root}" ]; then
  echo "missing source_root: ${source_root}" >&2
  exit 3
fi
if [ ! -d "${model}" ]; then
  echo "missing eval model: ${model}" >&2
  exit 4
fi
if [ ! -f "${model}/transformer/diffusion_pytorch_model.safetensors" ]; then
  echo "missing transformer safetensors under eval model: ${model}" >&2
  exit 5
fi
if [ ! -f "${icd}" ]; then
  echo "missing EGL ICD: ${icd}" >&2
  exit 6
fi
if [ "${task_start}" -ge "${task_end}" ]; then
  echo "invalid task range: ${task_start} ${task_end}" >&2
  exit 7
fi

actual_transformer_sha=$(sha256sum "${model}/transformer/diffusion_pytorch_model.safetensors" | awk '{print $1}')
if [ "${expected_transformer_sha}" != "auto" ] && [ "${actual_transformer_sha}" != "${expected_transformer_sha}" ]; then
  echo "transformer sha mismatch: ${actual_transformer_sha}" >&2
  exit 8
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${source_root}:/data/jiaoguanbo/LIBERO:${source_root}/wan_va:${PYTHONPATH:-}
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export HF_HOME=${root}/runtime/cache/huggingface
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export LINGBOT_VA_LIBERO_MODEL_PATH=${model}
export LINGBOT_VA_WEBSOCKET_HOST=127.0.0.1
export LIBERO_CONFIG_PATH=${root}/runtime/libero_config
export LD_LIBRARY_PATH=/data/jiaoguanbo/runtime/nvidia-gl-595/usr/lib/x86_64-linux-gnu:/data/zouyude/conda/envs/fastwam/lib:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=${icd}
export __EGL_VENDOR_LIBRARY_FILENAMES=${icd}
export MUJOCO_GL=egl

server_pid=""

cleanup_server() {
  if [ -n "${server_pid}" ] && ps -p "${server_pid}" >/dev/null 2>&1; then
    kill -INT "${server_pid}" >/dev/null 2>&1 || true
    sleep 8
  fi
  if [ -n "${server_pid}" ] && ps -p "${server_pid}" >/dev/null 2>&1; then
    kill -TERM "${server_pid}" >/dev/null 2>&1 || true
    sleep 5
  fi
  if [ -n "${server_pid}" ] && ps -p "${server_pid}" >/dev/null 2>&1; then
    kill -KILL "${server_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${server_pid}" ]; then
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
  server_pid=""
}

trap cleanup_server EXIT

for task_idx in $(seq "${task_start}" $((task_end - 1))); do
  port=$((29220 + task_idx))
  master_port=$((29230 + task_idx))
  next_task=$((task_idx + 1))
  task_name="task${task_idx}"
  server_save_root=${out_root}/server_outputs_${task_name}
  client_out=${out_root}/official_client_${task_name}_${test_num}eps
  server_log=${log_root}/${run_id}_server_${task_name}.log
  client_log=${log_root}/${run_id}_client_${task_name}.log

  mkdir -p "${server_save_root}" "${client_out}"

  {
    echo "RUN_ID=${run_id}"
    echo "PHASE=server_${task_name}_fresh"
    echo "MODEL=${model}"
    echo "TRANSFORMER_SHA256=${actual_transformer_sha}"
    echo "PORT=${port}"
    echo "MASTER_PORT=${master_port}"
    echo "SAVE_ROOT=${server_save_root}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "EVAL_PROTOCOL=LIBERO_OFFICIAL_INIT_POOL_NO_HELDOUT_SUBSET"
  } > "${server_log}"

  (
    cd "${source_root}"
    exec "${python_bin}" -m torch.distributed.run \
      --nproc_per_node 1 \
      --master_port "${master_port}" \
      wan_va/wan_va_server.py \
      --config-name libero \
      --port "${port}" \
      --save_root "${server_save_root}"
  ) >> "${server_log}" 2>&1 &
  server_pid=$!

  server_ready=0
  for _ in $(seq 1 180); do
    if grep -q "server listening" "${server_log}"; then
      server_ready=1
      break
    fi
    if ! ps -p "${server_pid}" >/dev/null 2>&1; then
      echo "server exited before ready for ${task_name}; see ${server_log}" >&2
      exit 10
    fi
    sleep 5
  done
  if [ "${server_ready}" != "1" ]; then
    echo "server did not become ready for ${task_name}; see ${server_log}" >&2
    exit 11
  fi

  {
    echo "RUN_ID=${run_id}"
    echo "PHASE=client_${task_name}_${test_num}eps"
    echo "MODEL=${model}"
    echo "TRANSFORMER_SHA256=${actual_transformer_sha}"
    echo "OUT=${client_out}"
    echo "PORT=${port}"
    echo "TEST_NUM=${test_num}"
    echo "TASK_RANGE=${task_idx} ${next_task}"
    echo "INIT_STATE_RULE=episode_idx % init_states.shape[0]"
    echo "EVAL_PROTOCOL=LIBERO_OFFICIAL_INIT_POOL_NO_HELDOUT_SUBSET"
  } > "${client_log}"

  (
    cd "${source_root}"
    exec "${python_bin}" evaluation/libero/client.py \
      --libero-benchmark libero_10 \
      --port "${port}" \
      --test-num "${test_num}" \
      --task-range "${task_idx}" "${next_task}" \
      --out-dir "${client_out}"
  ) >> "${client_log}" 2>&1

  cleanup_server

  {
    echo "CLEANUP_AFTER_${task_name}"
    pgrep -af "wan_va_server.py|evaluation/libero/client.py|torch.distributed.run|torchrun" || true
  } >> "${client_log}"
done

echo "Gate2 checkpoint eval completed: ${run_id}"
