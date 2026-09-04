#!/usr/bin/env bash
set -euo pipefail

benchmark=${1:?usage: run_server.sh <robotwin|libero> [gpu] [port]}
gpu=${2:-0}
port=${3:-29056}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/worktrees/lingbot-va-stage1
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python

case "${benchmark}" in
  robotwin)
    config=robotwin
    export LINGBOT_VA_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-posttrain-robotwin
    ;;
  libero)
    config=libero
    export LINGBOT_VA_LIBERO_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-posttrain-libero-long
    ;;
  *)
    echo "unknown benchmark: ${benchmark}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONPATH=${source_root}:${source_root}/wan_va
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export TORCHINDUCTOR_CACHE_DIR=${root}/runtime/cache/torchinductor
export TRITON_CACHE_DIR=${root}/runtime/cache/triton
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1

cd "${root}"
exec "${python_bin}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=$((port + 1000)) \
  "${source_root}/wan_va/wan_va_server.py" \
  --config-name "${config}" \
  --port "${port}" \
  --save_root "${root}/results/stage1/${benchmark}_server"
