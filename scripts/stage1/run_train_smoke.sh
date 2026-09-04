#!/usr/bin/env bash
set -euo pipefail

benchmark=${1:?usage: run_train_smoke.sh <robotwin|libero> [master_port]}
master_port=${2:-29571}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/worktrees/lingbot-va-stage1
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python

case "${benchmark}" in
  robotwin)
    config=robotwin_train
    dataset=${root}/data/stage1/robotwin_official_2ep
    export LINGBOT_VA_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-base
    ;;
  libero)
    config=libero_train
    dataset=${root}/data/stage1/libero_contract_proxy_2ep
    export LINGBOT_VA_LIBERO_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-base
    ;;
  *)
    echo "unknown benchmark: ${benchmark}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=${source_root}:${source_root}/wan_va
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export HF_HOME=${root}/runtime/cache/huggingface
export STAGE1_COMPILE_CACHE_ROOT=${root}/runtime/cache/train_compile
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export LINGBOT_VA_DATASET_PATH=${dataset}
export LINGBOT_VA_TRAIN_STEPS=1
export LINGBOT_VA_LOAD_WORKERS=0
export LINGBOT_VA_INIT_WORKERS=1
export LINGBOT_VA_GRAD_ACCUM=1
export LINGBOT_VA_SAVE_INTERVAL=1000

output=${root}/results/stage1/train_${benchmark}_2gpu
mkdir -p "${output}"
cd "${root}"
exec "${python_bin}" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port "${master_port}" \
  "${root}/scripts/stage1/ranked_train_entry.py" \
  --config-name "${config}" \
  --save-root "${output}"
