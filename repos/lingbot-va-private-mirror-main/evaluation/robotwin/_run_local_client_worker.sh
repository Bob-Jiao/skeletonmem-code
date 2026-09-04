#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 )); then
    echo "Usage: $0 GPU_ID PORT SAVE_ROOT TEST_NUM SEED TASK..." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLIENT_LAUNCHER=${ROBOTWIN_CLIENT_LAUNCHER:-${SCRIPT_DIR}/launch_local_client.sh}
GPU_ID=$1
PORT=$2
SAVE_ROOT=$3
TEST_NUM=$4
SEED=$5
shift 5

status=0
for task_name in "$@"; do
    echo "===== Starting ${task_name} on GPU ${GPU_ID}, port ${PORT} ====="
    if "${CLIENT_LAUNCHER}" \
        "${task_name}" "${SAVE_ROOT}" "${GPU_ID}" "${PORT}" "${TEST_NUM}" "${SEED}"; then
        echo "===== Completed ${task_name} ====="
    else
        task_status=$?
        status=1
        echo "===== ${task_name} exited with status ${task_status}; continuing =====" >&2
    fi
done

exit "${status}"
