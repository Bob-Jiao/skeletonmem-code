#!/usr/bin/env bash
set -euo pipefail

dataset_path=${1:?usage: run_lingbot_selector_train_smoke.sh <dataset_path> <run_name> [steps] [master_port] [save_interval=1000]}
run_name=${2:?usage: run_lingbot_selector_train_smoke.sh <dataset_path> <run_name> [steps] [master_port] [save_interval=1000]}
steps=${3:-1}
master_port=${4:-29641}
save_interval=${5:-1000}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/repos/lingbot-va-gatea-official
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python
output=${root}/results/stage2/lingbot_selector_train/${run_name}

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=${source_root}:${source_root}/wan_va
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export HF_HOME=${root}/runtime/cache/huggingface
export STAGE1_COMPILE_CACHE_ROOT=${root}/runtime/cache/stage2_train_compile_${run_name}
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export LINGBOT_VA_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-base
export LINGBOT_VA_DATASET_PATH=${dataset_path}
export LINGBOT_VA_TRAIN_STEPS=${steps}
export LINGBOT_VA_LOAD_WORKERS=0
export LINGBOT_VA_INIT_WORKERS=1
export LINGBOT_VA_GRAD_ACCUM=1
export LINGBOT_VA_SAVE_INTERVAL=${save_interval}
unset LINGBOT_VA_EPISODE_FILTER_PATH || true
unset LINGBOT_VA_EPISODE_FILTER_KEY || true

mkdir -p "${output}"
cd "${root}"
echo "DATA_PROTOCOL=FULLDATA_NO_EPISODE_FILTER"
exec "${python_bin}" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port "${master_port}" \
  "${root}/scripts/stage1/ranked_train_entry.py" \
  --config-name robotwin_train \
  --save-root "${output}"
