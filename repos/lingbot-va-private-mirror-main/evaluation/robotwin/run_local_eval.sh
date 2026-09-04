#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 TRANSFORMER_DIR SAVE_ROOT" >&2
    echo 'Optional env: GPU_IDS="0 1 2 3" TEST_NUM=100 SEED=0 TASK_CONFIG=demo_clean SAVE_VIDEOS=1 MAX_RETRY_ROUNDS=3 TASKS="task_a task_b"' >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRANSFORMER_DIR=$(readlink -f -- "$1")
SAVE_ROOT=$2
START_PORT=${START_PORT:-29556}
START_MASTER_PORT=${START_MASTER_PORT:-29661}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-900}
TEST_NUM=${TEST_NUM:-100}
SEED=${SEED:-0}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
SAVE_VIDEOS=${SAVE_VIDEOS:-1}
MAX_RETRY_ROUNDS=${MAX_RETRY_ROUNDS:-3}
CHECK_PYTHON=${CHECK_PYTHON:-/data/shared/zouyude/conda/envs/lingbot-va/bin/python}
CHECK_SCRIPT=${SCRIPT_DIR}/check_eval_completeness.py
SERVER_LAUNCHER=${ROBOTWIN_SERVER_LAUNCHER:-${SCRIPT_DIR}/launch_local_server.sh}
WORKER_LAUNCHER=${ROBOTWIN_WORKER_LAUNCHER:-${SCRIPT_DIR}/_run_local_client_worker.sh}

case "${TASK_CONFIG}" in
    demo_clean|demo_randomized) ;;
    *) echo "TASK_CONFIG must be demo_clean or demo_randomized, found: ${TASK_CONFIG}" >&2; exit 2 ;;
esac
case "${SAVE_VIDEOS}" in
    0|1) ;;
    *) echo "SAVE_VIDEOS must be 0 or 1, found: ${SAVE_VIDEOS}" >&2; exit 2 ;;
esac
if ! [[ "${TEST_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEST_NUM must be a positive integer, found: ${TEST_NUM}" >&2
    exit 2
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer, found: ${SEED}" >&2
    exit 2
fi
if ! [[ "${MAX_RETRY_ROUNDS}" =~ ^[0-9]+$ ]]; then
    echo "MAX_RETRY_ROUNDS must be a non-negative integer, found: ${MAX_RETRY_ROUNDS}" >&2
    exit 2
fi
export TASK_CONFIG SAVE_VIDEOS

read -r -a gpu_ids <<< "${GPU_IDS}"
if (( ${#gpu_ids[@]} == 0 )); then
    echo "GPU_IDS must contain at least one GPU" >&2
    exit 2
fi

default_tasks=(
    stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe
    adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch
    shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket
    pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot
    stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox
    click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes
    place_empty_cup blocks_ranking_rgb
)
if [[ -n "${TASKS:-}" ]]; then
    read -r -a tasks <<< "${TASKS}"
else
    tasks=("${default_tasks[@]}")
fi
if (( ${#tasks[@]} == 0 )); then
    echo "No tasks selected" >&2
    exit 2
fi

declare -A seen_tasks=()
for task_name in "${tasks[@]}"; do
    if [[ -n "${seen_tasks[${task_name}]:-}" ]]; then
        echo "Duplicate task selected: ${task_name}" >&2
        exit 2
    fi
    seen_tasks[${task_name}]=1
done

for executable in "${CHECK_PYTHON}" "${SERVER_LAUNCHER}" "${WORKER_LAUNCHER}"; do
    [[ -x "${executable}" ]] || { echo "Missing executable: ${executable}" >&2; exit 1; }
done
[[ -f "${CHECK_SCRIPT}" ]] || { echo "Missing checker: ${CHECK_SCRIPT}" >&2; exit 1; }

mkdir -p -- "${SAVE_ROOT}"
SAVE_ROOT=$(readlink -f -- "${SAVE_ROOT}")
mkdir -p -- "${SAVE_ROOT}/logs" "${SAVE_ROOT}/server_visualization"
echo "Task config: ${TASK_CONFIG}"
echo "Save evaluation videos: ${SAVE_VIDEOS}"
echo "Maximum automatic retry rounds: ${MAX_RETRY_ROUNDS}"
server_pids=()
worker_pids=()
server_generations=()
incomplete_tasks=()
cleaned=0

cleanup() {
    if (( cleaned )); then
        return
    fi
    cleaned=1
    set +e
    for pid in "${worker_pids[@]}" "${server_pids[@]}"; do
        [[ -n "${pid}" ]] && kill -TERM -- "-${pid}" 2>/dev/null
    done
    for _ in {1..20}; do
        any_alive=0
        for pid in "${worker_pids[@]}" "${server_pids[@]}"; do
            if [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
                any_alive=1
            fi
        done
        (( any_alive == 0 )) && break
        sleep 0.25
    done
    for pid in "${worker_pids[@]}" "${server_pids[@]}"; do
        [[ -n "${pid}" ]] && kill -KILL -- "-${pid}" 2>/dev/null
    done
    wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_is_open() {
    "${CHECK_PYTHON}" - "$1" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.5)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
PY
}

process_group_alive() {
    local pid=$1
    kill -0 -- "-${pid}" 2>/dev/null
}

check_completeness() {
    local output checker_status
    if output=$("${CHECK_PYTHON}" "${CHECK_SCRIPT}" "${SAVE_ROOT}" \
        --test-num "${TEST_NUM}" \
        --seed "${SEED}" \
        --report "${SAVE_ROOT}/completeness.json" \
        --missing-only \
        "${tasks[@]}"); then
        checker_status=0
    else
        checker_status=$?
    fi
    if (( checker_status > 1 )); then
        echo "Completeness checker failed with status ${checker_status}" >&2
        return "${checker_status}"
    fi

    incomplete_tasks=()
    if [[ -n "${output}" ]]; then
        mapfile -t incomplete_tasks <<< "${output}"
    fi
}

show_incomplete_details() {
    "${CHECK_PYTHON}" "${CHECK_SCRIPT}" "${SAVE_ROOT}" \
        --test-num "${TEST_NUM}" \
        --seed "${SEED}" \
        --report "${SAVE_ROOT}/completeness.json" \
        --verbose \
        "${tasks[@]}" || true
}

start_server() {
    local slot=$1
    local gpu_id=${gpu_ids[$slot]}
    local port=$(( START_PORT + slot ))
    local master_port=$(( START_MASTER_PORT + slot ))
    local generation=$(( ${server_generations[$slot]:-0} + 1 ))
    local log_file="${SAVE_ROOT}/logs/server_gpu${gpu_id}_port${port}.log"
    server_generations[$slot]=${generation}

    {
        echo "===== Starting server generation ${generation}: GPU ${gpu_id}, policy ${port}, master ${master_port} ====="
    } >>"${log_file}"
    echo "Starting server ${slot}: GPU ${gpu_id}, port ${port}, generation ${generation}"
    setsid "${SERVER_LAUNCHER}" \
        "${TRANSFORMER_DIR}" "${gpu_id}" "${port}" "${master_port}" \
        "${SAVE_ROOT}/server_visualization" >>"${log_file}" 2>&1 &
    server_pids[$slot]=$!
}

wait_for_server() {
    local slot=$1
    local gpu_id=${gpu_ids[$slot]}
    local port=$(( START_PORT + slot ))
    local pid=${server_pids[$slot]}
    local deadline=$(( SECONDS + SERVER_TIMEOUT ))
    local log_file="${SAVE_ROOT}/logs/server_gpu${gpu_id}_port${port}.log"

    while ! port_is_open "${port}"; do
        if ! process_group_alive "${pid}"; then
            echo "Server ${slot} exited before opening port ${port}" >&2
            tail -n 80 "${log_file}" >&2
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for policy port ${port}" >&2
            tail -n 80 "${log_file}" >&2
            return 1
        fi
        sleep 2
    done
    echo "Server ${slot} is ready on port ${port}"
}

stop_server() {
    local slot=$1
    local pid=${server_pids[$slot]:-}
    [[ -n "${pid}" ]] || return 0
    if process_group_alive "${pid}"; then
        kill -TERM -- "-${pid}" 2>/dev/null || true
        for _ in {1..20}; do
            process_group_alive "${pid}" || break
            sleep 0.25
        done
        if process_group_alive "${pid}"; then
            kill -KILL -- "-${pid}" 2>/dev/null || true
        fi
    fi
    wait "${pid}" 2>/dev/null || true
    server_pids[$slot]=""
}

ensure_server() {
    local slot=$1
    local pid=${server_pids[$slot]:-}
    local port=$(( START_PORT + slot ))
    local master_port=$(( START_MASTER_PORT + slot ))
    if [[ -n "${pid}" ]] && process_group_alive "${pid}" && port_is_open "${port}"; then
        return 0
    fi

    echo "Server ${slot} is not healthy; restarting it before the next round" >&2
    stop_server "${slot}"
    for _ in {1..40}; do
        if ! port_is_open "${port}" && ! port_is_open "${master_port}"; then
            break
        fi
        sleep 0.25
    done
    if port_is_open "${port}" || port_is_open "${master_port}"; then
        echo "Cannot restart server ${slot}: port ${port} or ${master_port} is still in use" >&2
        return 1
    fi
    start_server "${slot}"
    wait_for_server "${slot}"
}

run_task_round() {
    local round=$1
    shift
    local -a round_tasks=("$@")
    local round_worker_count=${#round_tasks[@]}
    if (( round_worker_count > server_count )); then
        round_worker_count=${server_count}
    fi
    local worker slot index gpu_id port log_file pid worker_exit
    local round_failed=0
    local -a worker_tasks=()
    worker_pids=()

    for (( worker=0; worker<round_worker_count; worker++ )); do
        slot=$(( (worker + round) % server_count ))
        worker_tasks=()
        for (( index=worker; index<${#round_tasks[@]}; index+=round_worker_count )); do
            worker_tasks+=("${round_tasks[$index]}")
        done
        gpu_id=${gpu_ids[$slot]}
        port=$(( START_PORT + slot ))
        log_file=$(printf '%s/logs/client_round%02d_worker%d_gpu%s.log' \
            "${SAVE_ROOT}" "${round}" "${worker}" "${gpu_id}")
        echo "Round ${round}, worker ${worker}: GPU ${gpu_id}, port ${port}, tasks: ${worker_tasks[*]}"
        setsid "${WORKER_LAUNCHER}" \
            "${gpu_id}" "${port}" "${SAVE_ROOT}" "${TEST_NUM}" "${SEED}" \
            "${worker_tasks[@]}" >"${log_file}" 2>&1 &
        worker_pids+=("$!")
    done

    for pid in "${worker_pids[@]}"; do
        if wait "${pid}"; then
            :
        else
            worker_exit=$?
            round_failed=1
            echo "A client worker exited with status ${worker_exit}; completeness will decide retries" >&2
        fi
    done
    worker_pids=()
    return "${round_failed}"
}

if check_completeness; then
    :
else
    exit $?
fi
if (( ${#incomplete_tasks[@]} == 0 )); then
    echo "All ${#tasks[@]} selected tasks were already complete. Results: ${SAVE_ROOT}"
    exit 0
fi
echo "Resuming with ${#incomplete_tasks[@]}/${#tasks[@]} incomplete tasks"

server_count=${#gpu_ids[@]}
if (( ${#incomplete_tasks[@]} < server_count )); then
    server_count=${#incomplete_tasks[@]}
fi

for (( slot=0; slot<server_count; slot++ )); do
    port=$(( START_PORT + slot ))
    master_port=$(( START_MASTER_PORT + slot ))
    if port_is_open "${port}" || port_is_open "${master_port}"; then
        echo "Port ${port} or ${master_port} is already in use" >&2
        exit 1
    fi
done
for (( slot=0; slot<server_count; slot++ )); do
    start_server "${slot}"
done
for (( slot=0; slot<server_count; slot++ )); do
    wait_for_server "${slot}" || exit 1
done

round=0
round_tasks=("${incomplete_tasks[@]}")
while true; do
    for (( slot=0; slot<server_count; slot++ )); do
        ensure_server "${slot}" || exit 1
    done

    echo "===== Evaluation round ${round}: ${#round_tasks[@]} task(s) ====="
    if run_task_round "${round}" "${round_tasks[@]}"; then
        echo "All client workers in round ${round} exited successfully"
    else
        echo "One or more client workers failed in round ${round}" >&2
    fi

    if check_completeness; then
        :
    else
        exit $?
    fi
    if (( ${#incomplete_tasks[@]} == 0 )); then
        echo "All ${#tasks[@]} task evaluations are complete. Results: ${SAVE_ROOT}"
        exit 0
    fi
    if (( round >= MAX_RETRY_ROUNDS )); then
        echo "Exhausted ${MAX_RETRY_ROUNDS} automatic retry round(s); ${#incomplete_tasks[@]} task(s) remain incomplete" >&2
        show_incomplete_details
        exit 1
    fi

    round=$(( round + 1 ))
    round_tasks=("${incomplete_tasks[@]}")
    echo "Scheduling retry ${round}/${MAX_RETRY_ROUNDS} for ${#round_tasks[@]} incomplete task(s)"
done
