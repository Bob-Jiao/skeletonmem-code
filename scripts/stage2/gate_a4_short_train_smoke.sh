#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="/data/jiaoguanbo/skeletonmem"
SRC="${ROOT}/repos/lingbot-va-gatea-official"
RESULT_DIR="${ROOT}/results/stage2/gate_a"
LOG_DIR="${ROOT}/logs/stage2/gate_a"
SAVE_ROOT="${RESULT_DIR}/a4_short_train_${RUN_ID}"
TRAIN_LOG="${LOG_DIR}/gate_a4_short_train_${RUN_ID}.log"
GPU_LOG="${LOG_DIR}/gate_a4_short_gpu_${RUN_ID}.csv"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${SAVE_ROOT}"

export HF_HOME="${ROOT}/runtime/cache/huggingface"
export XDG_CACHE_HOME="${ROOT}/runtime/cache"
unset LEROBOT_HOME || true
export HF_LEROBOT_HOME="${ROOT}/runtime/cache/lerobot"

export LINGBOT_VA_DATASET_PATH="/data/jiaoguanbo/LIBERO/libero-long-lerobot"
export LINGBOT_VA_LIBERO_MODEL_PATH="/data/jiaoguanbo/models/lingbot-va-base"
export LINGBOT_VA_LOAD_WORKERS="${LINGBOT_VA_LOAD_WORKERS:-0}"
export LINGBOT_VA_INIT_WORKERS="${LINGBOT_VA_INIT_WORKERS:-1}"
export LINGBOT_VA_TRAIN_STEPS="${LINGBOT_VA_TRAIN_STEPS:-10}"
export LINGBOT_VA_GRAD_ACCUM="${LINGBOT_VA_GRAD_ACCUM:-10}"
export LINGBOT_VA_SAVE_INTERVAL="${LINGBOT_VA_SAVE_INTERVAL:-10}"
unset LINGBOT_VA_EPISODE_FILTER_PATH || true
unset LINGBOT_VA_EPISODE_FILTER_KEY || true
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "utc,gpu_index,memory_used_mib,memory_free_mib,utilization_gpu,utilization_memory"
  while true; do
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits \
      | awk -v ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{gsub(/, /,","); print ts "," $0}'
    sleep 5
  done
} > "${GPU_LOG}" &
MONITOR_PID="$!"

cleanup() {
  kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "${SRC}"

echo "RUN_ID=${RUN_ID}" | tee "${TRAIN_LOG}"
echo "SAVE_ROOT=${SAVE_ROOT}" | tee -a "${TRAIN_LOG}"
echo "TRAIN_STEPS=${LINGBOT_VA_TRAIN_STEPS}" | tee -a "${TRAIN_LOG}"
echo "GRAD_ACCUM=${LINGBOT_VA_GRAD_ACCUM}" | tee -a "${TRAIN_LOG}"
echo "SAVE_INTERVAL=${LINGBOT_VA_SAVE_INTERVAL}" | tee -a "${TRAIN_LOG}"
echo "DATA_PROTOCOL=LIBERO_OFFICIAL_FULLDATA_NO_EPISODE_FILTER" | tee -a "${TRAIN_LOG}"

/data/jiaoguanbo/conda/envs/lingbot-va/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port "${MASTER_PORT:-29542}" \
  --tee 3 \
  -m wan_va.train \
  --config-name libero_train \
  --save-root "${SAVE_ROOT}" 2>&1 | tee -a "${TRAIN_LOG}"

echo "RUN_ID=${RUN_ID}"
echo "TRAIN_LOG=${TRAIN_LOG}"
echo "GPU_LOG=${GPU_LOG}"
echo "SAVE_ROOT=${SAVE_ROOT}"
