#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    echo "Usage: bash evaluation/libero/launch_eval_4gpu.sh {full|lora} {standard|plus}" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

MODE="$1"
EVAL_SET="${2:-standard}"
STEP="${STEP:-20000}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-1800}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-fastwam_lerobot}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

REPO_ROOT="${REPO_ROOT:-/data/zouyude/lingbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:-/root/shared/zouyude/train/lingbot-va}"
BASE_MODEL_ROOT="${BASE_MODEL_ROOT:-/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base}"
EVAL_ROOT="${EVAL_ROOT:-/root/shared/zouyude/eval/lingbot-va}"

SERVER_PY="${SERVER_PY:-/data/shared/zouyude/conda/envs/lingbot-va/bin/python}"
STANDARD_CONFIG_SOURCE="${STANDARD_CONFIG_SOURCE:-${REPO_ROOT}/evaluation/libero/config_standard.yaml}"
PLUS_CONFIG_PATH="${PLUS_CONFIG_PATH:-/root/.libero}"
TASK_SAMPLE_SEED="${TASK_SAMPLE_SEED:-42}"

case "$EVAL_SET" in
    standard)
        TEST_NUM="${TEST_NUM:-50}"
        CLIENT_PY="${CLIENT_PY:-/data/shared/zouyude/conda/envs/libero/bin/python}"
        LIBERO_ROOT="${LIBERO_ROOT:-/data/zouyude/LIBERO}"
        OUTPUT_TAG="libero"
        FULL_WS_PORT_DEFAULT=35056
        FULL_MASTER_PORT_DEFAULT=35156
        LORA_WS_PORT_DEFAULT=36056
        LORA_MASTER_PORT_DEFAULT=36156
        CLIENT_SAMPLE_ARGS=()
        ;;
    plus)
        TEST_NUM="${TEST_NUM:-1}"
        CLIENT_PY="${CLIENT_PY:-/data/shared/zouyude/conda/envs/libero_plus/bin/python}"
        LIBERO_ROOT="${LIBERO_ROOT:-/root/zouyude/LIBERO-plus}"
        OUTPUT_TAG="libero_plus"
        FULL_WS_PORT_DEFAULT=37056
        FULL_MASTER_PORT_DEFAULT=37156
        LORA_WS_PORT_DEFAULT=38056
        LORA_MASTER_PORT_DEFAULT=38156
        TASK_SAMPLE_RATIO="${TASK_SAMPLE_RATIO:-0.15}"
        CLIENT_SAMPLE_ARGS=(
            --task-sample-ratio "$TASK_SAMPLE_RATIO"
            --task-sample-seed "$TASK_SAMPLE_SEED"
        )
        ;;
    *)
        usage
        exit 2
        ;;
esac

case "$MODE" in
    full)
        TRANSFORMER="${TRAIN_ROOT}/libero-all-full/checkpoints/checkpoint_step_${STEP}/transformer"
        OUT_ROOT="${OUT_ROOT:-${EVAL_ROOT}/full_step${STEP}_${OUTPUT_TAG}_fastwam_lerobot}"
        WS_PORT_BASE="${WS_PORT_BASE:-$FULL_WS_PORT_DEFAULT}"
        MASTER_PORT_BASE="${MASTER_PORT_BASE:-$FULL_MASTER_PORT_DEFAULT}"
        SERVER_MODEL_ARGS=(--transformer-path "$TRANSFORMER")
        ;;
    lora)
        TRANSFORMER="${BASE_MODEL_ROOT}/transformer"
        ADAPTER="${TRAIN_ROOT}/libero-all-lora-r64-4gpu/checkpoints/checkpoint_step_${STEP}/adapter"
        OUT_ROOT="${OUT_ROOT:-${EVAL_ROOT}/lora_step${STEP}_${OUTPUT_TAG}_fastwam_lerobot}"
        WS_PORT_BASE="${WS_PORT_BASE:-$LORA_WS_PORT_DEFAULT}"
        MASTER_PORT_BASE="${MASTER_PORT_BASE:-$LORA_MASTER_PORT_DEFAULT}"
        SERVER_MODEL_ARGS=(
            --transformer-path "$TRANSFORMER"
            --lora-adapter-path "$ADAPTER"
        )
        ;;
    *)
        usage
        exit 2
        ;;
esac

RESULT_DIR="${OUT_ROOT}/results"
LOG_DIR="${OUT_ROOT}/logs"
LIBERO_CONFIG_DIR="${OUT_ROOT}/libero_config"
if [[ "$EVAL_SET" == "standard" ]]; then
    CLIENT_LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
else
    CLIENT_LIBERO_CONFIG_PATH="$PLUS_CONFIG_PATH"
fi

GPU_IDS=(0 1 2 3)
# MuJoCo's EGL id is local to the CUDA-visible list and must remain 0.
# Putting the target GPU first distributes rendering; appending literal 0
# also satisfies the robosuite visibility check used by this environment.
CLIENT_CVDS=(0 1,0 2,0 3,0)

SERVER_PIDS=()
CLIENT_PIDS=()
declare -A CLIENT_SHARD_BY_PID=()

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    for pid in "${CLIENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    for pid in "${CLIENT_PIDS[@]}"; do
        wait "$pid" 2>/dev/null
    done
    for pid in "${SERVER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null
    done
    exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$REPO_ROOT"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$LIBERO_CONFIG_DIR"
rm -f "${OUT_ROOT}/COMPLETE"

[[ -x "$SERVER_PY" ]] || {
    echo "[ERROR] Missing server Python: $SERVER_PY" >&2
    exit 1
}
[[ -x "$CLIENT_PY" ]] || {
    echo "[ERROR] Missing client Python: $CLIENT_PY" >&2
    exit 1
}
[[ -s "${TRANSFORMER}/config.json" ]] || {
    echo "[ERROR] Invalid transformer directory: $TRANSFORMER" >&2
    exit 1
}
if [[ "$EVAL_SET" == "standard" ]]; then
    [[ -s "$STANDARD_CONFIG_SOURCE" ]] || {
        echo "[ERROR] Missing standard LIBERO config: $STANDARD_CONFIG_SOURCE" >&2
        exit 1
    }
else
    [[ -s "${PLUS_CONFIG_PATH}/config.yaml" ]] || {
        echo "[ERROR] Missing LIBERO-Plus config: ${PLUS_CONFIG_PATH}/config.yaml" >&2
        exit 1
    }
fi
[[ "$EVAL_PROTOCOL" == "fastwam_lerobot" ]] || {
    echo "[ERROR] This launcher evaluates FastWAM-derived checkpoints and requires EVAL_PROTOCOL=fastwam_lerobot" >&2
    exit 1
}
[[ "$PREFLIGHT_ONLY" == "0" || "$PREFLIGHT_ONLY" == "1" ]] || {
    echo "[ERROR] PREFLIGHT_ONLY must be 0 or 1, got $PREFLIGHT_ONLY" >&2
    exit 1
}
if [[ "$MODE" == "full" ]]; then
    [[ -s "${TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || {
        echo "[ERROR] Missing full checkpoint weights: $TRANSFORMER" >&2
        exit 1
    }
else
    [[ -s "${TRANSFORMER}/diffusion_pytorch_model.safetensors.index.json" ]] || {
        echo "[ERROR] Missing base transformer weights: $TRANSFORMER" >&2
        exit 1
    }
    [[ -s "${ADAPTER}/pytorch_lora_weights.safetensors" ]] || {
        echo "[ERROR] Missing LoRA adapter weights: $ADAPTER" >&2
        exit 1
    }
fi

if [[ "$EVAL_SET" == "standard" ]]; then
    install -m 0644 "$STANDARD_CONFIG_SOURCE" "${LIBERO_CONFIG_DIR}/config.yaml"
fi

echo "[config] mode=$MODE eval_set=$EVAL_SET step=$STEP test_num=$TEST_NUM"
echo "[config] eval_protocol=$EVAL_PROTOCOL"
if [[ "$EVAL_SET" == "plus" ]]; then
    echo "[config] task_sample_ratio=$TASK_SAMPLE_RATIO task_sample_seed=$TASK_SAMPLE_SEED"
fi
echo "[config] transformer=<$TRANSFORMER>"
if [[ "$MODE" == "lora" ]]; then
    echo "[config] adapter=<$ADAPTER>"
fi
echo "[config] output=<$OUT_ROOT>"
echo "[config] inherited CUDA_VISIBLE_DEVICES=<${CUDA_VISIBLE_DEVICES-UNSET}>"
echo "[config] inherited NVIDIA_VISIBLE_DEVICES=<${NVIDIA_VISIBLE_DEVICES-UNSET}>"

echo "[preflight] checking the four logical CUDA devices"
for shard in 0 1 2 3; do
    target="${GPU_IDS[$shard]}"
    env CUDA_VISIBLE_DEVICES="$target" "$SERVER_PY" -c \
        'import torch; assert torch.cuda.device_count() == 1, torch.cuda.device_count(); print(torch.cuda.get_device_name(0))'
done

echo "[preflight] creating one $EVAL_SET LIBERO environment on every client mapping"
for shard in 0 1 2 3; do
    client_cvd="${CLIENT_CVDS[$shard]}"
    echo "[preflight] shard=$shard CUDA_VISIBLE_DEVICES=<$client_cvd> MUJOCO_EGL_DEVICE_ID=<0>"
    env \
        CUDA_VISIBLE_DEVICES="$client_cvd" \
        MUJOCO_EGL_DEVICE_ID=0 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        LIBERO_CONFIG_PATH="$CLIENT_LIBERO_CONFIG_PATH" \
        PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}" \
        PYTHONUNBUFFERED=1 \
        "$CLIENT_PY" -c \
        'from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; b = benchmark.get_benchmark_dict()["libero_10"](); s = b.get_task_init_states(0); e = OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(0), camera_heights=128, camera_widths=128); e.reset(); e.set_init_state(s[0]); e.close(); print("LIBERO_EGL_PREFLIGHT_OK")'
done

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[preflight] mode=$MODE eval_set=$EVAL_SET passed; no servers launched"
    exit 0
fi

echo "[server] launching four one-GPU replicas"
for shard in 0 1 2 3; do
    target="${GPU_IDS[$shard]}"
    port=$((WS_PORT_BASE + shard))
    master_port=$((MASTER_PORT_BASE + shard))
    server_log="${LOG_DIR}/server_gpu${shard}.log"

    echo "[server] shard=$shard gpu=$target port=$port master_port=$master_port log=$server_log"
    env \
        CUDA_VISIBLE_DEVICES="$target" \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT="$master_port" \
        RANK=0 \
        LOCAL_RANK=0 \
        WORLD_SIZE=1 \
        TOKENIZERS_PARALLELISM=false \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH="${REPO_ROOT}/wan_va:${REPO_ROOT}" \
        PYTHONUNBUFFERED=1 \
        "$SERVER_PY" wan_va/wan_va_server.py \
        --config-name libero_all \
        --port "$port" \
        "${SERVER_MODEL_ARGS[@]}" \
        >"$server_log" 2>&1 &
    SERVER_PIDS+=("$!")
done

echo "[server] waiting for all health endpoints"
READY=(0 0 0 0)
deadline=$((SECONDS + SERVER_TIMEOUT_SECONDS))
while true; do
    all_ready=1
    for shard in 0 1 2 3; do
        if [[ "${READY[$shard]}" -eq 1 ]]; then
            continue
        fi

        port=$((WS_PORT_BASE + shard))
        pid="${SERVER_PIDS[$shard]}"
        if curl --max-time 2 -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
            READY[$shard]=1
            echo "[server] shard=$shard ready on port=$port"
            continue
        fi

        all_ready=0
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] server shard=$shard exited before becoming ready" >&2
            tail -100 "${LOG_DIR}/server_gpu${shard}.log" >&2
            exit 1
        fi
    done

    if [[ "$all_ready" -eq 1 ]]; then
        break
    fi
    if ((SECONDS >= deadline)); then
        echo "[ERROR] Timed out waiting for servers after ${SERVER_TIMEOUT_SECONDS}s" >&2
        for shard in 0 1 2 3; do
            echo "===== server_gpu${shard}.log =====" >&2
            tail -60 "${LOG_DIR}/server_gpu${shard}.log" >&2
        done
        exit 1
    fi
    sleep 5
done

echo "[client] launching four task shards"
for shard in 0 1 2 3; do
    client_cvd="${CLIENT_CVDS[$shard]}"
    port=$((WS_PORT_BASE + shard))
    client_log="${LOG_DIR}/client_gpu${shard}.log"

    echo "[client] shard=$shard CUDA_VISIBLE_DEVICES=<$client_cvd> EGL=<0> port=$port log=$client_log"
    env \
        CUDA_VISIBLE_DEVICES="$client_cvd" \
        MUJOCO_EGL_DEVICE_ID=0 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        LIBERO_CONFIG_PATH="$CLIENT_LIBERO_CONFIG_PATH" \
        PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}" \
        PYTHONUNBUFFERED=1 \
        "$CLIENT_PY" evaluation/libero/client.py \
        --libero-benchmark libero_10 libero_goal libero_spatial libero_object \
        --host 127.0.0.1 \
        --port "$port" \
        --test-num "$TEST_NUM" \
        --seed 42 \
        --eval-protocol "$EVAL_PROTOCOL" \
        "${CLIENT_SAMPLE_ARGS[@]}" \
        --task-shard-index "$shard" \
        --task-shard-count 4 \
        --resume \
        --out-dir "$RESULT_DIR" \
        >"$client_log" 2>&1 &

    pid=$!
    CLIENT_PIDS+=("$pid")
    CLIENT_SHARD_BY_PID["$pid"]="$shard"
done

echo "[client] monitoring all shards; any failure aborts immediately"
while ((${#CLIENT_PIDS[@]} > 0)); do
    done_pid=""
    if wait -n -p done_pid "${CLIENT_PIDS[@]}"; then
        rc=0
    else
        rc=$?
    fi

    shard="${CLIENT_SHARD_BY_PID[$done_pid]:-unknown}"
    if [[ "$rc" -ne 0 ]]; then
        echo "[ERROR] client shard=$shard pid=${done_pid:-unknown} exited with rc=$rc" >&2
        if [[ "$shard" != "unknown" ]]; then
            tail -120 "${LOG_DIR}/client_gpu${shard}.log" >&2
        fi
        exit "$rc"
    fi

    echo "[client] shard=$shard completed successfully"
    next_pids=()
    for pid in "${CLIENT_PIDS[@]}"; do
        if [[ "$pid" != "$done_pid" ]]; then
            next_pids+=("$pid")
        fi
    done
    CLIENT_PIDS=("${next_pids[@]}")
    unset 'CLIENT_SHARD_BY_PID[$done_pid]'
done

env \
    LIBERO_CONFIG_PATH="$CLIENT_LIBERO_CONFIG_PATH" \
    PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}" \
    "$CLIENT_PY" - \
    "$RESULT_DIR" "$TEST_NUM" "$EVAL_SET" \
    "${TASK_SAMPLE_RATIO-}" "$TASK_SAMPLE_SEED" "$EVAL_PROTOCOL" 800 <<'PY'
import json
import math
import random
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
test_num = int(sys.argv[2])
eval_set = sys.argv[3]
ratio_text = sys.argv[4]
sample_seed = int(sys.argv[5])
eval_protocol = sys.argv[6]
max_env_steps = int(sys.argv[7])
suites = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
missing = []
incomplete = []
protocol_mismatches = []
step_limit_mismatches = []
evaluated_tasks = 0

if eval_set == "plus":
    from libero.libero import benchmark

for suite in suites:
    if eval_set == "standard":
        task_ids = range(10)
    else:
        num_tasks = benchmark.get_benchmark_dict()[suite]().get_num_tasks()
        ratio = float(ratio_text)
        sample_count = max(1, int(math.ceil(num_tasks * ratio)))
        rng = random.Random(f"{sample_seed}:{suite}")
        task_ids = sorted(rng.sample(range(num_tasks), sample_count))

    for task_id in task_ids:
        evaluated_tasks += 1
        path = result_dir / f"{suite}_{task_id}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        result = json.loads(path.read_text())
        total = int(result["total_num"])
        if total < test_num:
            incomplete.append(f"{path}: {total}/{test_num}")
        if result.get("eval_protocol") != eval_protocol:
            protocol_mismatches.append(
                f"{path}: {result.get('eval_protocol')!r}/{eval_protocol!r}"
            )
        if int(result.get("max_env_steps", -1)) != max_env_steps:
            step_limit_mismatches.append(
                f"{path}: {result.get('max_env_steps')!r}/{max_env_steps!r}"
            )

if missing or incomplete or protocol_mismatches or step_limit_mismatches:
    if missing:
        print("[ERROR] Missing result files:\n" + "\n".join(missing), file=sys.stderr)
    if incomplete:
        print("[ERROR] Incomplete result files:\n" + "\n".join(incomplete), file=sys.stderr)
    if protocol_mismatches:
        print(
            "[ERROR] Protocol mismatches:\n" + "\n".join(protocol_mismatches),
            file=sys.stderr,
        )
    if step_limit_mismatches:
        print(
            "[ERROR] max_env_steps mismatches:\n"
            + "\n".join(step_limit_mismatches),
            file=sys.stderr,
        )
    raise SystemExit(1)

print(
    f"[validate] all {evaluated_tasks} {eval_set} tasks have "
    f"{test_num} episodes with protocol={eval_protocol}"
)
PY

touch "${OUT_ROOT}/COMPLETE"
echo "[complete] $MODE $EVAL_SET LIBERO evaluation finished: ${OUT_ROOT}/COMPLETE"
