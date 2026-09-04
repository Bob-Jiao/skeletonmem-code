#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 5 )); then
    echo "Usage: $0 TRANSFORMER_DIR [GPU_ID] [PORT] [MASTER_PORT] [SAVE_ROOT]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LINGBOT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
LINGBOT_ENV=${LINGBOT_ENV:-/data/shared/zouyude/conda/envs/lingbot-va}
BASE_MODEL=${LINGBOT_VA_BASE_MODEL:-/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base}
TRANSFORMER_DIR=$(readlink -f -- "$1")
GPU_ID=${2:-${GPU_ID:-0}}
PORT=${3:-${PORT:-29056}}
MASTER_PORT=${4:-${MASTER_PORT:-29061}}
SAVE_ROOT=${5:-${SAVE_ROOT:-${LINGBOT_ROOT}/visualization/robotwin}}

for executable in "${LINGBOT_ENV}/bin/python"; do
    [[ -x "${executable}" ]] || { echo "Missing executable: ${executable}" >&2; exit 1; }
done
for path in \
    "${BASE_MODEL}/vae" \
    "${BASE_MODEL}/tokenizer" \
    "${BASE_MODEL}/text_encoder" \
    "${TRANSFORMER_DIR}/config.json"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done

"${LINGBOT_ENV}/bin/python" - "${TRANSFORMER_DIR}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    mode = json.load(handle).get("attn_mode")
if mode not in {"torch", "flashattn"}:
    raise SystemExit(
        f"Evaluation requires attn_mode=torch/flashattn, found {mode!r} in {sys.argv[1]}"
    )
print(f"Validated evaluation attention mode: {mode}")
PY

mkdir -p -- "${SAVE_ROOT}"
SAVE_ROOT=$(readlink -f -- "${SAVE_ROOT}")
cd -- "${LINGBOT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LINGBOT_VA_BASE_MODEL="${BASE_MODEL}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

echo "LingBot root: ${LINGBOT_ROOT}"
echo "Base model: ${BASE_MODEL}"
echo "Transformer: ${TRANSFORMER_DIR}"
echo "Physical GPU: ${GPU_ID}; policy port: ${PORT}; master port: ${MASTER_PORT}"

exec "${LINGBOT_ENV}/bin/python" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --transformer-path "${TRANSFORMER_DIR}" \
    --port "${PORT}" \
    --save_root "${SAVE_ROOT}"
