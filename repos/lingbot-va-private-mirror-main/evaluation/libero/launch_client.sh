TEST_NUM=${TEST_NUM:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-29056}
LIBERO_BENCHMARK=${LIBERO_BENCHMARK:-"libero_10 libero_goal libero_spatial libero_object"}
TASK_SAMPLE_RATIO=${TASK_SAMPLE_RATIO:-0.15}
TASK_SAMPLE_SEED=${TASK_SAMPLE_SEED:-42}
SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-outputs/libero_plus}
LIBERO_ENV=${LIBERO_ENV:-libero_plus}
TASK_SHARD_INDEX=${TASK_SHARD_INDEX:-0}
TASK_SHARD_COUNT=${TASK_SHARD_COUNT:-1}
RESUME=${RESUME:-0}
SAVE_VIDEO=${SAVE_VIDEO:-0}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-official_long}

range_args=()
if [ -n "${START:-}" ] && [ -n "${END:-}" ]; then
    range_args=(--task-range "$START" "$END")
fi

resume_args=()
if [ "$RESUME" = "1" ]; then
    resume_args=(--resume)
fi

video_args=()
if [ "$SAVE_VIDEO" = "1" ]; then
    video_args=(--save-video)
fi

conda run --no-capture-output -n "$LIBERO_ENV" \
    env LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/root/zouyude/LIBERO-plus \
    python evaluation/libero/client.py \
    --libero-benchmark $LIBERO_BENCHMARK \
    --host "$HOST" \
    --port "$PORT" \
    --test-num "$TEST_NUM" \
    "${range_args[@]}" \
    --task-sample-ratio "$TASK_SAMPLE_RATIO" \
    --task-sample-seed "$TASK_SAMPLE_SEED" \
    --seed "$SEED" \
    --eval-protocol "$EVAL_PROTOCOL" \
    --task-shard-index "$TASK_SHARD_INDEX" \
    --task-shard-count "$TASK_SHARD_COUNT" \
    "${resume_args[@]}" \
    "${video_args[@]}" \
    --out-dir "$OUT_DIR"
