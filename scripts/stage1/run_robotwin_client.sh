#!/usr/bin/env bash
set -euo pipefail

port=${1:-29056}
task_name=${2:-adjust_bottle}
test_num=${3:-1}
gpu=${4:-1}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/worktrees/lingbot-va-stage1
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python
runtime_root=${root}/runtime/robotwin

export ROBOTWIN_ROOT=/data/jiaoguanbo/RoboTwin
export ROBOTWIN_RUN_ROOT=${runtime_root}
export CUDA_VISIBLE_DEVICES=${gpu}
export SAPIEN_RENDER_DEVICE=cuda:0
export SERVER_HOST=127.0.0.1
export PYTHONPATH=${source_root}:${ROBOTWIN_ROOT}
export PATH=${root}/runtime/bin:${PATH}
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export MPLCONFIGDIR=${root}/runtime/cache/matplotlib
export TORCH_EXTENSIONS_DIR=${root}/runtime/cache/torch_extensions
export LD_LIBRARY_PATH=${root}/runtime/system-libs/usr/lib/x86_64-linux-gnu:/data/jiaoguanbo/conda/envs/lingbot-va/lib/python3.10/site-packages/sapien/oidn_library:/data/jiaoguanbo/runtime/nvidia-gl-595/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/data/jiaoguanbo/runtime/nvidia-gl-595/usr/share/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/jiaoguanbo/runtime/nvidia-gl-595/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export TORCH_CUDA_ARCH_LIST=8.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=${NO_PROXY}
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${runtime_root}"
cd "${runtime_root}"
exec "${python_bin}" -m evaluation.robotwin.eval_polict_client_openpi \
  --config "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml" \
  --overrides \
  --task_name "${task_name}" \
  --task_config demo_clean \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting lingbot_va_official \
  --seed 0 \
  --policy_name ACT \
  --save_root "${root}/results/stage1/robotwin_eval" \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num "${test_num}" \
  --port "${port}"
