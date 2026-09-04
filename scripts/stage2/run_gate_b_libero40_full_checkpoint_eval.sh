#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?usage: run_gate_b_libero40_full_checkpoint_eval.sh <run_id> <checkpoint_or_eval_model> [test_num=50] [suite_list=all]}
model=${2:?usage: run_gate_b_libero40_full_checkpoint_eval.sh <run_id> <checkpoint_or_eval_model> [test_num=50] [suite_list=all]}
test_num=${3:-50}
suite_list=${4:-all}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/repos/lingbot-va-gatea-official
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python
icd=/data/jiaoguanbo/lingbot-va/outputs/0729_robotwin_mem_pick_step1500_first_call_bundle_20260729/0729_nvidia_icd_egl.json

out_root=${root}/results/stage2/gate_b/libero40_full_train/eval/${run_id}
log_root=${root}/logs/stage2/gate_b/libero40_eval
mkdir -p "${out_root}" "${log_root}"

if [ "${test_num}" -lt 50 ]; then
  echo "test_num must be at least 50 per task for LIBERO40 eval: ${test_num}" >&2
  exit 2
fi
if [ ! -x "${python_bin}" ]; then
  echo "missing python: ${python_bin}" >&2
  exit 3
fi
if [ ! -d "${source_root}" ]; then
  echo "missing source_root: ${source_root}" >&2
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

if [ "${suite_list}" = "all" ]; then
  suites=(libero_object libero_goal libero_spatial libero_10)
else
  IFS=',' read -r -a suites <<< "${suite_list}"
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

echo "RUN_ID=${run_id}"
echo "MODEL=${model}"
echo "TEST_NUM=${test_num}"
echo "SUITES=${suites[*]}"
echo "EVAL_PROTOCOL=LIBERO40_FOUR_SUITE_PER_TASK_MIN50_FRESH_SERVER"

if [ "${LIBERO40_EVAL_DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1"
  exit 0
fi

for suite_idx in "${!suites[@]}"; do
  suite=${suites[${suite_idx}]}
  for task_idx in $(seq 0 9); do
    port=$((29320 + suite_idx * 20 + task_idx))
    master_port=$((29330 + suite_idx * 20 + task_idx))
    task_name="${suite}_task${task_idx}"
    server_save_root=${out_root}/server_outputs_${task_name}
    client_out=${out_root}/${suite}/official_client_task${task_idx}_${test_num}eps
    server_log=${log_root}/${run_id}_${task_name}_server.log
    client_log=${log_root}/${run_id}_${task_name}_client.log

    mkdir -p "${server_save_root}" "${client_out}"

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
      echo "EVAL_PROTOCOL=LIBERO40_FOUR_SUITE_PER_TASK_MIN50_FRESH_SERVER"
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
      echo "SUITE=${suite}"
      echo "TASK=${task_idx}"
      echo "OUT=${client_out}"
      echo "PORT=${port}"
      echo "TEST_NUM=${test_num}"
      echo "TASK_RANGE=${task_idx} $((task_idx + 1))"
      echo "INIT_STATE_RULE=episode_idx % init_states.shape[0]"
      echo "EVAL_PROTOCOL=LIBERO40_FOUR_SUITE_PER_TASK_MIN50_FRESH_SERVER"
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
  done
done

echo "LIBERO40 checkpoint eval completed: ${run_id}"
