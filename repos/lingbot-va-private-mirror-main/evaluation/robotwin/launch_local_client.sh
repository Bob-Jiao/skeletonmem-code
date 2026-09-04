#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 6 )); then
    echo "Usage: $0 TASK_NAME SAVE_ROOT [GPU_ID] [PORT] [TEST_NUM] [SEED]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LINGBOT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
ROBOTWIN_ENV=${ROBOTWIN_ENV:-/data/shared/zouyude/conda/envs/robotwin-lingbot-va}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/root/zouyude/RoboTwin-lingbot-va}
NVIDIA_GL_RUNTIME=${NVIDIA_GL_RUNTIME:-/root/shared/zouyude/train/lawam/robotwin_eval_runtime/nvidia-gl-595}

TASK_NAME=$1
SAVE_ROOT=$2
GPU_ID=${3:-${GPU_ID:-0}}
PORT=${4:-${PORT:-29056}}
TEST_NUM=${5:-${TEST_NUM:-100}}
SEED=${6:-${SEED:-0}}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
SAVE_VIDEOS=${SAVE_VIDEOS:-1}

case "${TASK_CONFIG}" in
    demo_clean|demo_randomized) ;;
    *) echo "TASK_CONFIG must be demo_clean or demo_randomized, found: ${TASK_CONFIG}" >&2; exit 2 ;;
esac
case "${SAVE_VIDEOS}" in
    0|1) ;;
    *) echo "SAVE_VIDEOS must be 0 or 1, found: ${SAVE_VIDEOS}" >&2; exit 2 ;;
esac

for path in \
    "${ROBOTWIN_ENV}/bin/python" \
    "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml" \
    "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml" \
    "${NVIDIA_GL_RUNTIME}/usr/share/vulkan/icd.d/nvidia_icd.json" \
    "${NVIDIA_GL_RUNTIME}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done

mkdir -p -- "${SAVE_ROOT}"
SAVE_ROOT=$(readlink -f -- "${SAVE_ROOT}")
cd -- "${LINGBOT_ROOT}"

RUNTIME_LIB="${NVIDIA_GL_RUNTIME}/usr/lib/x86_64-linux-gnu"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export LD_LIBRARY_PATH="${RUNTIME_LIB}:/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_GL_RUNTIME}/usr/share/vulkan/icd.d/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_GL_RUNTIME}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export PYTHONPATH="${LINGBOT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ROBOTWIN_ROOT
export TASK_CONFIG
export SAVE_VIDEOS
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ws_proxy wss_proxy WS_PROXY WSS_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${no_proxy}"

echo "RoboTwin root: ${ROBOTWIN_ROOT}"
echo "Task: ${TASK_NAME}; episodes: ${TEST_NUM}; seed index: ${SEED}"
echo "Task config: ${TASK_CONFIG}"
echo "Physical GPU: ${GPU_ID}; policy port: ${PORT}"
echo "Save evaluation videos: ${SAVE_VIDEOS}"
echo "Results: ${SAVE_ROOT}"

exec env PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    "${ROBOTWIN_ENV}/bin/python" -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --train_config_name 0 \
    --model_name 0 \
    --ckpt_setting 0 \
    --seed "${SEED}" \
    --policy_name ACT \
    --save_root "${SAVE_ROOT}" \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num "${TEST_NUM}" \
    --port "${PORT}"
