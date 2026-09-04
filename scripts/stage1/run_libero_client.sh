#!/usr/bin/env bash
set -euo pipefail

port=${1:-29056}
task_start=${2:-0}
task_end=${3:-1}
test_num=${4:-1}

root=/data/jiaoguanbo/skeletonmem
source_root=${root}/worktrees/lingbot-va-stage1
python_bin=/data/jiaoguanbo/conda/envs/lingbot-va/bin/python

export PYTHONPATH=${source_root}:/data/jiaoguanbo/LIBERO
export SERVER_HOST=127.0.0.1
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=${NO_PROXY}
export HOME=${root}/runtime/home
export XDG_CACHE_HOME=${root}/runtime/cache
export MPLCONFIGDIR=${root}/runtime/cache/matplotlib
export LD_LIBRARY_PATH=${root}/runtime/system-libs/usr/lib/x86_64-linux-gnu:/data/jiaoguanbo/conda/envs/lingbot-va/lib/python3.10/site-packages/sapien/oidn_library:/data/jiaoguanbo/runtime/nvidia-gl-595/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/data/jiaoguanbo/runtime/nvidia-gl-595/usr/share/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/jiaoguanbo/runtime/nvidia-gl-595/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONDONTWRITEBYTECODE=1

cd "${root}"
exec "${python_bin}" "${source_root}/evaluation/libero/client.py" \
  --libero-benchmark libero_10 \
  --port "${port}" \
  --test-num "${test_num}" \
  --task-range "${task_start}" "${task_end}" \
  --out-dir "${root}/results/stage1/libero_eval"
