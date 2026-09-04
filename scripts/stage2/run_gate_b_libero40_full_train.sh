#!/usr/bin/env bash
set -euo pipefail

run_name=${1:?usage: run_gate_b_libero40_full_train.sh <libero40_run_name> [steps=4000] [master_port=29671] [save_interval=1000] [grad_accum=10] [dataset_path=/data/jiaoguanbo/LIBERO/libero_all] [gripper_remap=none|libero40_open01_to_cmd] [episode_filter_path=] [episode_filter_key=]}
steps=${2:-4000}
master_port=${3:-29671}
save_interval=${4:-1000}
grad_accum=${5:-10}
dataset_path=${6:-/data/jiaoguanbo/LIBERO/libero_all}
gripper_remap=${7:-none}
episode_filter_path=${8:-}
episode_filter_key=${9:-}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/repos/lingbot-va-gatea-official
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python

case "${run_name}" in
  libero40_*) ;;
  *)
    echo "run_name must start with 'libero40_' to avoid confusion with LIBERO-LONG runs: ${run_name}" >&2
    exit 2
    ;;
esac

if [ "${dataset_path}" = "/data/jiaoguanbo/LIBERO/libero-long-lerobot" ]; then
  echo "refusing LIBERO-LONG dataset path in Gate B LIBERO40 script: ${dataset_path}" >&2
  exit 3
fi
if [ ! -d "${dataset_path}" ]; then
  echo "dataset_path does not exist: ${dataset_path}" >&2
  exit 4
fi
if [ ! -f "${dataset_path}/empty_emb.pt" ]; then
  echo "missing empty_emb.pt at dataset root: ${dataset_path}/empty_emb.pt" >&2
  exit 5
fi
case "${gripper_remap}" in
  none|libero40_open01_to_cmd) ;;
  *)
    echo "unsupported gripper_remap: ${gripper_remap}" >&2
    exit 9
    ;;
esac
if [ -n "${episode_filter_path}" ] && [ ! -f "${episode_filter_path}" ]; then
  echo "episode_filter_path does not exist: ${episode_filter_path}" >&2
  exit 10
fi

for suite in \
  libero_10_no_noops_lingbot \
  libero_goal_no_noops_lingbot \
  libero_object_no_noops_lingbot \
  libero_spatial_no_noops_lingbot; do
  suite_dir="${dataset_path}/${suite}"
  if [ ! -f "${suite_dir}/meta/info.json" ]; then
    echo "missing suite meta/info.json: ${suite_dir}/meta/info.json" >&2
    exit 6
  fi
  if [ ! -d "${suite_dir}/data" ]; then
    echo "missing or broken suite data directory: ${suite_dir}/data" >&2
    exit 7
  fi
  if [ ! -d "${suite_dir}/latents" ]; then
    echo "missing suite latents directory: ${suite_dir}/latents" >&2
    exit 8
  fi
done

output=${root}/results/stage2/gate_b/libero40_full_train/${run_name}
mkdir -p "${output}"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=${source_root}:${source_root}/wan_va
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export HF_HOME=${root}/runtime/cache/huggingface
export STAGE1_COMPILE_CACHE_ROOT=${root}/runtime/cache/gate_b_libero40_train_compile_${run_name}
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export LINGBOT_VA_LIBERO_MODEL_PATH=/data/jiaoguanbo/models/lingbot-va-base
export LINGBOT_VA_DATASET_PATH=${dataset_path}
export LINGBOT_VA_TRAIN_STEPS=${steps}
export LINGBOT_VA_LOAD_WORKERS=0
export LINGBOT_VA_INIT_WORKERS=1
export LINGBOT_VA_GRAD_ACCUM=${grad_accum}
export LINGBOT_VA_SAVE_INTERVAL=${save_interval}
if [ "${gripper_remap}" = "none" ]; then
  unset LINGBOT_VA_TRAIN_GRIPPER_ACTION_REMAP || true
else
  export LINGBOT_VA_TRAIN_GRIPPER_ACTION_REMAP=${gripper_remap}
fi
if [ -n "${episode_filter_path}" ]; then
  export LINGBOT_VA_EPISODE_FILTER_PATH=${episode_filter_path}
  export LINGBOT_VA_EPISODE_FILTER_KEY=${episode_filter_key}
else
  unset LINGBOT_VA_EPISODE_FILTER_PATH || true
  unset LINGBOT_VA_EPISODE_FILTER_KEY || true
fi

cd "${root}"
echo "EXPERIMENT_FAMILY=LIBERO40_MIXED"
echo "DATA_PROTOCOL=LIBERO40_MIXED_FULLDATA_NO_EPISODE_FILTER"
echo "DATASET_PATH=${dataset_path}"
echo "RUN_NAME=${run_name}"
echo "TRAIN_STEPS=${steps}"
echo "SAVE_INTERVAL=${save_interval}"
echo "GRAD_ACCUM=${grad_accum}"
echo "GRIPPER_REMAP=${gripper_remap}"
echo "EPISODE_FILTER_PATH=${episode_filter_path}"
echo "EPISODE_FILTER_KEY=${episode_filter_key}"
echo "OUTPUT=${output}"
if [ "${LIBERO40_DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1"
  exit 0
fi
exec "${python_bin}" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port "${master_port}" \
  "${root}/scripts/stage1/ranked_train_entry.py" \
  --config-name libero_train \
  --save-root "${output}"
