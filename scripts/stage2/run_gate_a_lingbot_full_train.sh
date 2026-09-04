#!/usr/bin/env bash
set -euo pipefail

benchmark=${1:?usage: run_gate_a_lingbot_full_train.sh <libero|robotwin> <dataset_path> <run_name> [steps=4000] [master_port] [save_interval=1000] [grad_accum=10]}
dataset_path=${2:?usage: run_gate_a_lingbot_full_train.sh <libero|robotwin> <dataset_path> <run_name> [steps=4000] [master_port] [save_interval=1000] [grad_accum=10]}
run_name=${3:?usage: run_gate_a_lingbot_full_train.sh <libero|robotwin> <dataset_path> <run_name> [steps=4000] [master_port] [save_interval=1000] [grad_accum=10]}
steps=${4:-4000}
master_port=${5:-29661}
save_interval=${6:-1000}
grad_accum=${7:-10}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/repos/lingbot-va-gatea-official
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python

case "${benchmark}" in
  libero)
    config=libero_train
    export LINGBOT_VA_LIBERO_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-base
    ;;
  robotwin)
    config=robotwin_train
    export LINGBOT_VA_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-base
    ;;
  *)
    echo "unknown benchmark: ${benchmark}" >&2
    exit 2
    ;;
esac

if [ ! -d "${dataset_path}" ]; then
  echo "dataset_path does not exist: ${dataset_path}" >&2
  exit 3
fi
if [ ! -f "${dataset_path}/empty_emb.pt" ]; then
  echo "missing empty_emb.pt at dataset root: ${dataset_path}/empty_emb.pt" >&2
  exit 4
fi

output=${root}/results/stage2/gate_a/full_train/${run_name}
mkdir -p "${output}"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=${source_root}:${source_root}/wan_va
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export HF_HOME=${root}/runtime/cache/huggingface
export STAGE1_COMPILE_CACHE_ROOT=${root}/runtime/cache/gate_a_train_compile_${run_name}
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export LINGBOT_VA_DATASET_PATH=${dataset_path}
export LINGBOT_VA_TRAIN_STEPS=${steps}
export LINGBOT_VA_LOAD_WORKERS=0
export LINGBOT_VA_INIT_WORKERS=1
export LINGBOT_VA_GRAD_ACCUM=${grad_accum}
export LINGBOT_VA_SAVE_INTERVAL=${save_interval}
unset LINGBOT_VA_EPISODE_FILTER_PATH || true
unset LINGBOT_VA_EPISODE_FILTER_KEY || true

cd "${root}"
echo "DATA_PROTOCOL=LIBERO_OFFICIAL_FULLDATA_NO_EPISODE_FILTER"
echo "TRAIN_STEPS=${steps}"
echo "SAVE_INTERVAL=${save_interval}"
echo "GRAD_ACCUM=${grad_accum}"
exec "${python_bin}" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port "${master_port}" \
  "${root}/scripts/stage1/ranked_train_entry.py" \
  --config-name "${config}" \
  --save-root "${output}"
