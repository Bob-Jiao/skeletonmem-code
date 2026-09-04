
save_root=${SAVE_ROOT:-'visualization/'}
SAVE_DEBUG_DATA=${SAVE_DEBUG_DATA:-0}
PORT=${PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
LINGBOT_ENV=${LINGBOT_ENV:-lingbot-va}

save_args=()
if [ "$SAVE_DEBUG_DATA" = "1" ]; then
    mkdir -p "$save_root"
    save_args=(--save-debug-data --save_root "$save_root")
fi

conda run --no-capture-output -n "$LINGBOT_ENV" python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port "$PORT" \
    "${save_args[@]}"
