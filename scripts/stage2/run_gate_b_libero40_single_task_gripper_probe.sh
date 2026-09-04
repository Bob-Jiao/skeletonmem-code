#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?usage: run_gate_b_libero40_single_task_gripper_probe.sh <run_id> <checkpoint_or_eval_model> <suite> <task_idx> [test_num=1] [remap=none|libero40_open01_to_cmd] [gpu=0]}
model=${2:?usage: run_gate_b_libero40_single_task_gripper_probe.sh <run_id> <checkpoint_or_eval_model> <suite> <task_idx> [test_num=1] [remap=none|libero40_open01_to_cmd] [gpu=0]}
suite=${3:?usage: run_gate_b_libero40_single_task_gripper_probe.sh <run_id> <checkpoint_or_eval_model> <suite> <task_idx> [test_num=1] [remap=none|libero40_open01_to_cmd] [gpu=0]}
task_idx=${4:?usage: run_gate_b_libero40_single_task_gripper_probe.sh <run_id> <checkpoint_or_eval_model> <suite> <task_idx> [test_num=1] [remap=none|libero40_open01_to_cmd] [gpu=0]}
test_num=${5:-1}
remap=${6:-none}
gpu=${7:-0}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/repos/lingbot-va-gatea-official
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python
icd=/data/jiaoguanbo/lingbot-va/outputs/0729_robotwin_mem_pick_step1500_first_call_bundle_20260729/0729_nvidia_icd_egl.json

out_root=${root}/results/stage2/gate_b/libero40_full_train/eval/${run_id}
log_root=${root}/logs/stage2/gate_b/libero40_eval
mkdir -p "${out_root}" "${log_root}"

case "${suite}" in
  libero_object|libero_goal|libero_spatial|libero_10) ;;
  *)
    echo "unsupported suite: ${suite}" >&2
    exit 2
    ;;
esac
case "${remap}" in
  none|libero40_open01_to_cmd) ;;
  *)
    echo "unsupported remap: ${remap}" >&2
    exit 3
    ;;
esac
if [ "${task_idx}" -lt 0 ] || [ "${task_idx}" -gt 9 ]; then
  echo "task_idx must be in [0, 9]: ${task_idx}" >&2
  exit 4
fi
if [ ! -f "${model}/transformer/diffusion_pytorch_model.safetensors" ]; then
  echo "missing transformer safetensors under model: ${model}" >&2
  exit 5
fi
if [ ! -f "${icd}" ]; then
  echo "missing EGL ICD: ${icd}" >&2
  exit 6
fi

export CUDA_VISIBLE_DEVICES=${gpu}
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
export LINGBOT_VA_LIBERO_CLIENT_ACTION_DEBUG_STEPS=32
if [ "${remap}" = "none" ]; then
  unset LINGBOT_VA_LIBERO_CLIENT_GRIPPER_REMAP || true
else
  export LINGBOT_VA_LIBERO_CLIENT_GRIPPER_REMAP=${remap}
fi

port=$((29620 + task_idx))
master_port=$((29640 + task_idx))
task_name="${suite}_task${task_idx}_${remap}"
server_save_root=${out_root}/server_outputs_${task_name}
client_out=${out_root}/${suite}/official_client_task${task_idx}_${test_num}eps_${remap}
server_log=${log_root}/${run_id}_${task_name}_server.log
client_log=${log_root}/${run_id}_${task_name}_client.log

mkdir -p "${server_save_root}" "${client_out}"

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

{
  echo "RUN_ID=${run_id}"
  echo "PHASE=server_${task_name}_fresh"
  echo "MODEL=${model}"
  echo "SUITE=${suite}"
  echo "TASK=${task_idx}"
  echo "PORT=${port}"
  echo "MASTER_PORT=${master_port}"
  echo "SAVE_ROOT=${server_save_root}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "GRIPPER_REMAP=${remap}"
  echo "EVAL_PROTOCOL=LIBERO40_SINGLE_TASK_GRIPPER_CONTRACT_PROBE"
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
    echo "server exited before ready; see ${server_log}" >&2
    exit 10
  fi
  sleep 5
done
if [ "${server_ready}" != "1" ]; then
  echo "server did not become ready; see ${server_log}" >&2
  exit 11
fi

{
  echo "RUN_ID=${run_id}"
  echo "PHASE=client_${task_name}_${test_num}eps"
  echo "MODEL=${model}"
  echo "SUITE=${suite}"
  echo "TASK=${task_idx}"
  echo "OUT=${client_out}"
  echo "PORT=${port}"
  echo "TEST_NUM=${test_num}"
  echo "TASK_RANGE=${task_idx} $((task_idx + 1))"
  echo "GRIPPER_REMAP=${remap}"
  echo "EVAL_PROTOCOL=LIBERO40_SINGLE_TASK_GRIPPER_CONTRACT_PROBE"
} > "${client_log}"

(
  cd "${source_root}"
  exec "${python_bin}" evaluation/libero/client.py \
    --libero-benchmark "${suite}" \
    --port "${port}" \
    --test-num "${test_num}" \
    --task-range "${task_idx}" "$((task_idx + 1))" \
    --out-dir "${client_out}"
) >> "${client_log}" 2>&1

cleanup_server

echo "LIBERO40 gripper probe completed: ${run_id} ${suite} task${task_idx} remap=${remap}"
echo "client_log=${client_log}"
echo "server_log=${server_log}"
