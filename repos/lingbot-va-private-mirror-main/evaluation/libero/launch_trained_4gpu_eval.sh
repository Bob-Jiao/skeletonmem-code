#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash evaluation/libero/launch_trained_4gpu_eval.sh \
    {full-4gpu|lora-4gpu|lora-action-4gpu} {standard|plus} STEP

Examples:
  bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu standard 20000
  bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu plus 20000
  bash evaluation/libero/launch_trained_4gpu_eval.sh lora-4gpu standard 20000
  bash evaluation/libero/launch_trained_4gpu_eval.sh lora-4gpu plus 20000
  bash evaluation/libero/launch_trained_4gpu_eval.sh lora-action-4gpu standard 16000
  bash evaluation/libero/launch_trained_4gpu_eval.sh lora-action-4gpu plus 16000
EOF
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

VARIANT="$1"
EVAL_SET="$2"
STEP="$3"

if [[ ! "$STEP" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] STEP must be a positive integer, got: $STEP" >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-/data/zouyude/lingbot-va}"
SOURCE_TRAIN_ROOT="${TRAIN_ROOT:-/root/shared/zouyude/train/lingbot-va}"
BASE_MODEL_ROOT="${BASE_MODEL_ROOT:-/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base}"
EVAL_ROOT="${EVAL_ROOT:-/root/shared/zouyude/eval/lingbot-va}"
SERVER_PY="${SERVER_PY:-/data/shared/zouyude/conda/envs/lingbot-va/bin/python}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-fastwam_lerobot}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
TASK_SAMPLE_SEED="${TASK_SAMPLE_SEED:-42}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${REPO_ROOT}/evaluation/libero/launch_eval_4gpu.sh}"

if [[ "$EVAL_PROTOCOL" != "fastwam_lerobot" ]]; then
    echo "[ERROR] 4-GPU-trained checkpoints require EVAL_PROTOCOL=fastwam_lerobot" >&2
    exit 2
fi
if [[ "$PREFLIGHT_ONLY" != "0" && "$PREFLIGHT_ONLY" != "1" ]]; then
    echo "[ERROR] PREFLIGHT_ONLY must be 0 or 1, got: $PREFLIGHT_ONLY" >&2
    exit 2
fi

case "$EVAL_SET" in
    standard)
        OUTPUT_TAG="libero"
        TEST_NUM="${TEST_NUM:-50}"
        TASK_SAMPLE_RATIO=""
        ;;
    plus)
        OUTPUT_TAG="libero_plus"
        TEST_NUM="${TEST_NUM:-1}"
        TASK_SAMPLE_RATIO="${TASK_SAMPLE_RATIO:-0.15}"
        "$SERVER_PY" - "$TASK_SAMPLE_RATIO" <<'PY'
import math
import sys

ratio = float(sys.argv[1])
if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
    raise SystemExit(f"TASK_SAMPLE_RATIO must be finite and in (0, 1], got {ratio}")
PY
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ ! "$TEST_NUM" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] TEST_NUM must be a positive integer, got: $TEST_NUM" >&2
    exit 2
fi

case "$VARIANT" in
    full-4gpu)
        DELEGATE_MODE="full"
        TRAINING_RUN="libero-all-full-4gpu"
        DELEGATE_RUN_NAME="libero-all-full"
        CHECKPOINT_KIND="full"
        ;;
    lora-4gpu)
        DELEGATE_MODE="lora"
        TRAINING_RUN="libero-all-lora-r64-4gpu"
        DELEGATE_RUN_NAME="libero-all-lora-r64-4gpu"
        CHECKPOINT_KIND="lora"
        ;;
    lora-action-4gpu)
        DELEGATE_MODE="lora"
        TRAINING_RUN="libero-all-lora-r64-action-4gpu"
        DELEGATE_RUN_NAME="libero-all-lora-r64-4gpu"
        CHECKPOINT_KIND="lora"
        ;;
    *)
        usage
        exit 2
        ;;
esac

TRAINING_RUN_DIR="${SOURCE_TRAIN_ROOT}/${TRAINING_RUN}"
CHECKPOINT_DIR="${TRAINING_RUN_DIR}/checkpoints/checkpoint_step_${STEP}"
BASE_TRANSFORMER="${BASE_MODEL_ROOT}/transformer"

if [[ "$CHECKPOINT_KIND" == "full" ]]; then
    SELECTED_MODEL="${CHECKPOINT_DIR}/transformer"
    REQUIRED_FILES=(
        "${SELECTED_MODEL}/config.json"
        "${SELECTED_MODEL}/diffusion_pytorch_model.safetensors"
    )
else
    SELECTED_MODEL="${CHECKPOINT_DIR}/adapter"
    REQUIRED_FILES=(
        "${CHECKPOINT_DIR}/_SUCCESS"
        "${CHECKPOINT_DIR}/manifest.json"
        "${SELECTED_MODEL}/adapter_config.json"
        "${SELECTED_MODEL}/pytorch_lora_weights.safetensors"
        "${BASE_TRANSFORMER}/config.json"
        "${BASE_TRANSFORMER}/diffusion_pytorch_model.safetensors.index.json"
    )
fi

[[ -x "$SERVER_PY" ]] || {
    echo "[ERROR] Missing server Python: $SERVER_PY" >&2
    exit 1
}
[[ -s "$BASE_LAUNCHER" ]] || {
    echo "[ERROR] Missing base evaluation launcher: $BASE_LAUNCHER" >&2
    exit 1
}
for required_file in "${REQUIRED_FILES[@]}"; do
    [[ -e "$required_file" ]] || {
        echo "[ERROR] Missing checkpoint file: $required_file" >&2
        exit 1
    }
    if [[ ! -d "$required_file" && "$required_file" != *"/_SUCCESS" ]]; then
        [[ -s "$required_file" ]] || {
            echo "[ERROR] Empty checkpoint file: $required_file" >&2
            exit 1
        }
    fi
done

TRAINING_RUN_REAL="$(realpath -e "$TRAINING_RUN_DIR")"
CHECKPOINT_REAL="$(realpath -e "$CHECKPOINT_DIR")"
SELECTED_MODEL_REAL="$(realpath -e "$SELECTED_MODEL")"
BASE_TRANSFORMER_REAL="$(realpath -e "$BASE_TRANSFORMER")"
EXPECTED_CHECKPOINT_REAL="${TRAINING_RUN_REAL}/checkpoints/checkpoint_step_${STEP}"
if [[ "$CHECKPOINT_REAL" != "$EXPECTED_CHECKPOINT_REAL" ]]; then
    echo "[ERROR] Checkpoint escaped the selected training run:" >&2
    echo "        expected=$EXPECTED_CHECKPOINT_REAL" >&2
    echo "        resolved=$CHECKPOINT_REAL" >&2
    exit 1
fi

if [[ "$CHECKPOINT_KIND" == "lora" ]]; then
    "$SERVER_PY" - "$CHECKPOINT_REAL/manifest.json" "$STEP" "$BASE_TRANSFORMER_REAL" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
expected_base = os.path.realpath(sys.argv[3])
manifest = json.loads(manifest_path.read_text())

if manifest.get("training_mode") != "lora":
    raise SystemExit(f"Not a LoRA checkpoint manifest: {manifest_path}")
if int(manifest.get("step", -1)) != expected_step:
    raise SystemExit(
        f"LoRA manifest step mismatch: {manifest.get('step')} != {expected_step}"
    )
topology = manifest.get("topology", {})
if int(topology.get("world_size", -1)) != 4:
    raise SystemExit(
        f"Expected a 4-GPU LoRA checkpoint, got world_size={topology.get('world_size')}"
    )
if topology.get("packing_enabled") is not True:
    raise SystemExit("Expected a packed LoRA checkpoint, but packing_enabled is not true")
if int(topology.get("global_episodes_per_update", -1)) != 128:
    raise SystemExit(
        "Expected global_episodes_per_update=128, got "
        f"{topology.get('global_episodes_per_update')}"
    )
manifest_base = os.path.realpath(manifest.get("base_transformer_path", ""))
if manifest_base != expected_base:
    raise SystemExit(
        f"LoRA base transformer mismatch: {manifest_base} != {expected_base}"
    )

index_path = Path(expected_base) / "diffusion_pytorch_model.safetensors.index.json"
index = json.loads(index_path.read_text())
for shard_name in set(index.get("weight_map", {}).values()):
    shard_path = index_path.parent / shard_name
    if not shard_path.is_file() or shard_path.stat().st_size == 0:
        raise SystemExit(f"Missing base transformer shard: {shard_path}")
PY
fi

OUT_ROOT="${OUT_ROOT:-${EVAL_ROOT}/${TRAINING_RUN}_step${STEP}_${OUTPUT_TAG}_${EVAL_PROTOCOL}}"
mkdir -p "$OUT_ROOT"
OUT_ROOT_REAL="$(realpath -e "$OUT_ROOT")"
MANIFEST_PATH="${OUT_ROOT}/launch_manifest.json"

if find "${OUT_ROOT}/results" -maxdepth 1 -type f -name '*.json' -print -quit \
    2>/dev/null | read -r _ && [[ ! -s "$MANIFEST_PATH" ]]; then
    echo "[ERROR] Refusing to adopt existing results without launch_manifest.json:" >&2
    echo "        $OUT_ROOT" >&2
    echo "        Choose a new OUT_ROOT." >&2
    exit 1
fi

command -v flock >/dev/null 2>&1 || {
    echo "[ERROR] flock is required to protect the result directory" >&2
    exit 1
}
exec 9>"${OUT_ROOT}/.launch.lock"
if ! flock -n 9; then
    echo "[ERROR] Another evaluation already owns OUT_ROOT: $OUT_ROOT" >&2
    exit 1
fi

# The long-running legacy launcher may already be executing in other jobs, so it
# is deliberately left untouched. A private train-root view maps its legacy
# fixed child name to the explicitly selected 4-GPU training run.
DELEGATE_TRAIN_ROOT="${OUT_ROOT}/.resolved_train_root"
mkdir -p "$DELEGATE_TRAIN_ROOT"
DELEGATE_RUN_LINK="${DELEGATE_TRAIN_ROOT}/${DELEGATE_RUN_NAME}"
if [[ -L "$DELEGATE_RUN_LINK" ]]; then
    EXISTING_TARGET="$(realpath -e "$DELEGATE_RUN_LINK")"
    if [[ "$EXISTING_TARGET" != "$TRAINING_RUN_REAL" ]]; then
        echo "[ERROR] Existing model view points to a different training run:" >&2
        echo "        link=$DELEGATE_RUN_LINK" >&2
        echo "        existing=$EXISTING_TARGET" >&2
        echo "        requested=$TRAINING_RUN_REAL" >&2
        exit 1
    fi
elif [[ -e "$DELEGATE_RUN_LINK" ]]; then
    echo "[ERROR] Model-view path exists and is not a symlink: $DELEGATE_RUN_LINK" >&2
    exit 1
else
    ln -s "$TRAINING_RUN_REAL" "$DELEGATE_RUN_LINK"
fi

REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null || true)" ]]; then
    REPO_DIRTY="true"
else
    REPO_DIRTY="false"
fi

env \
    MANIFEST_PATH="$MANIFEST_PATH" \
    VARIANT="$VARIANT" \
    CHECKPOINT_KIND="$CHECKPOINT_KIND" \
    TRAINING_RUN="$TRAINING_RUN" \
    STEP="$STEP" \
    EVAL_SET="$EVAL_SET" \
    EVAL_PROTOCOL="$EVAL_PROTOCOL" \
    TEST_NUM="$TEST_NUM" \
    TASK_SAMPLE_RATIO="$TASK_SAMPLE_RATIO" \
    TASK_SAMPLE_SEED="$TASK_SAMPLE_SEED" \
    TRAINING_RUN_REAL="$TRAINING_RUN_REAL" \
    CHECKPOINT_REAL="$CHECKPOINT_REAL" \
    SELECTED_MODEL_REAL="$SELECTED_MODEL_REAL" \
    BASE_TRANSFORMER_REAL="$BASE_TRANSFORMER_REAL" \
    OUT_ROOT_REAL="$OUT_ROOT_REAL" \
    PREFLIGHT_ONLY="$PREFLIGHT_ONLY" \
    REPO_ROOT="$REPO_ROOT" \
    REPO_COMMIT="$REPO_COMMIT" \
    REPO_DIRTY="$REPO_DIRTY" \
    WRAPPER_PATH="${REPO_ROOT}/evaluation/libero/launch_trained_4gpu_eval.sh" \
    BASE_LAUNCHER="$BASE_LAUNCHER" \
    "$SERVER_PY" - <<'PY'
import datetime as dt
import hashlib
import json
import os
import struct
from pathlib import Path


def file_record(path):
    path = Path(path)
    stat = path.stat()
    record = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 16 * 1024 * 1024:
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def safetensors_schema(path):
    path = Path(path)
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise SystemExit(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        if header_size <= 0 or header_size > 256 * 1024 * 1024:
            raise SystemExit(f"Invalid safetensors header size {header_size}: {path}")
        header = json.loads(handle.read(header_size))
    header.pop("__metadata__", None)
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return {
        "key_count": len(header),
        "schema_sha256": hashlib.sha256(canonical).hexdigest(),
    }


kind = os.environ["CHECKPOINT_KIND"]
checkpoint = Path(os.environ["CHECKPOINT_REAL"])
selected_model = Path(os.environ["SELECTED_MODEL_REAL"])
if kind == "full":
    config_path = selected_model / "config.json"
    weights_path = selected_model / "diffusion_pytorch_model.safetensors"
else:
    config_path = selected_model / "adapter_config.json"
    weights_path = selected_model / "pytorch_lora_weights.safetensors"

checkpoint_files = {
    "config": file_record(config_path),
    "weights": {
        **file_record(weights_path),
        **safetensors_schema(weights_path),
    },
}
if kind == "lora":
    checkpoint_files["training_manifest"] = file_record(
        checkpoint / "manifest.json"
    )

identity = {
    "variant": os.environ["VARIANT"],
    "checkpoint_kind": kind,
    "training_run": os.environ["TRAINING_RUN"],
    "step": int(os.environ["STEP"]),
    "checkpoint_dir": str(checkpoint),
    "selected_model": str(selected_model),
    "base_transformer": (
        os.environ["BASE_TRANSFORMER_REAL"] if kind == "lora" else None
    ),
    "eval_set": os.environ["EVAL_SET"],
    "eval_protocol": os.environ["EVAL_PROTOCOL"],
    "test_num": int(os.environ["TEST_NUM"]),
    "task_sample_ratio": (
        float(os.environ["TASK_SAMPLE_RATIO"])
        if os.environ["TASK_SAMPLE_RATIO"]
        else None
    ),
    "task_sample_seed": int(os.environ["TASK_SAMPLE_SEED"]),
    "env_seed": 42,
    "task_shard_count": 4,
    "max_env_steps": 800,
}
identity_bytes = json.dumps(
    identity, sort_keys=True, separators=(",", ":")
).encode()
identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()

manifest_path = Path(os.environ["MANIFEST_PATH"])
now = dt.datetime.now(dt.timezone.utc).isoformat()
if manifest_path.exists():
    old = json.loads(manifest_path.read_text())
    if old.get("identity_sha256") != identity_sha256:
        raise SystemExit(
            "Existing launch manifest does not match this evaluation identity: "
            f"{manifest_path}"
        )
    if old.get("checkpoint_files") != checkpoint_files:
        raise SystemExit(
            "Checkpoint files changed since the launch manifest was created: "
            f"{manifest_path}"
        )
    created_at = old.get("created_at_utc", now)
else:
    created_at = now

manifest = {
    "schema_version": 1,
    "identity_sha256": identity_sha256,
    "identity": identity,
    "created_at_utc": created_at,
    "last_invoked_at_utc": now,
    "preflight_only": os.environ["PREFLIGHT_ONLY"] == "1",
    "resolved": {
        "training_run_dir": os.environ["TRAINING_RUN_REAL"],
        "output_root": os.environ["OUT_ROOT_REAL"],
    },
    "checkpoint_files": checkpoint_files,
    "code": {
        "repo_root": os.environ["REPO_ROOT"],
        "git_commit": os.environ["REPO_COMMIT"],
        "git_dirty": os.environ["REPO_DIRTY"] == "true",
        "wrapper": file_record(os.environ["WRAPPER_PATH"]),
        "base_launcher": file_record(os.environ["BASE_LAUNCHER"]),
    },
}

temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
os.replace(temporary, manifest_path)
print(f"[manifest] identity_sha256={identity_sha256}")
print(f"[manifest] path=<{manifest_path}>")
PY

echo "[selector] variant=$VARIANT training_run=$TRAINING_RUN step=$STEP"
echo "[selector] selected_model=<$SELECTED_MODEL_REAL>"
if [[ "$CHECKPOINT_KIND" == "lora" ]]; then
    echo "[selector] base_transformer=<$BASE_TRANSFORMER_REAL>"
fi
echo "[selector] output=<$OUT_ROOT_REAL>"

DELEGATE_ENV=(
    "REPO_ROOT=$REPO_ROOT"
    "TRAIN_ROOT=$DELEGATE_TRAIN_ROOT"
    "BASE_MODEL_ROOT=$BASE_MODEL_ROOT"
    "EVAL_ROOT=$EVAL_ROOT"
    "OUT_ROOT=$OUT_ROOT"
    "STEP=$STEP"
    "SERVER_PY=$SERVER_PY"
    "EVAL_PROTOCOL=$EVAL_PROTOCOL"
    "PREFLIGHT_ONLY=$PREFLIGHT_ONLY"
    "TEST_NUM=$TEST_NUM"
    "TASK_SAMPLE_SEED=$TASK_SAMPLE_SEED"
)
if [[ "$EVAL_SET" == "plus" ]]; then
    DELEGATE_ENV+=("TASK_SAMPLE_RATIO=$TASK_SAMPLE_RATIO")
fi
env "${DELEGATE_ENV[@]}" bash "$BASE_LAUNCHER" "$DELEGATE_MODE" "$EVAL_SET"
