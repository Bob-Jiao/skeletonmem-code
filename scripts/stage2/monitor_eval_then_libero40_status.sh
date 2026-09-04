#!/usr/bin/env bash
set -euo pipefail

tag=${1:?usage: monitor_eval_then_libero40_status.sh <tag> <supervisor_pid> [interval_seconds=600]}
supervisor_pid=${2:?usage: monitor_eval_then_libero40_status.sh <tag> <supervisor_pid> [interval_seconds=600]}
interval=${3:-600}

root=/data/jiaoguanbo/skeletonmem
out=${root}/results/stage2/gate_b/libero40_full_train/${tag}_monitor_snapshots.jsonl
closed_loop=${root}/results/stage2/gate_a/a5_closed_loop
gate_b=${root}/results/stage2/gate_b/libero40_full_train

mkdir -p "$(dirname "${out}")"

while ps -p "${supervisor_pid}" >/dev/null 2>&1; do
  TAG="${tag}" CLOSED_LOOP="${closed_loop}" GATE_B="${gate_b}" python - <<'PY' >> "${out}"
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

tag = os.environ["TAG"]
closed_loop = Path(os.environ["CLOSED_LOOP"])
gate_b = Path(os.environ["GATE_B"])

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()

runs = {}
for run_dir in sorted(closed_loop.glob(f"{tag}*")):
    jsons = sorted(run_dir.glob("official_client_task*_10eps/libero_10_*.json"))
    if not jsons:
        continue
    tasks = {}
    succ = 0.0
    total = 0.0
    for p in jsons:
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            tasks[p.name] = {"error": str(exc)}
            continue
        s = float(data.get("succ_num", 0.0))
        n = float(data.get("total_num", 0.0))
        tasks[p.name] = {"succ_num": s, "total_num": n, "succ_rate": float(data.get("succ_rate", s / n if n else 0.0))}
        succ += s
        total += n
    runs[run_dir.name] = {"succ_num": succ, "total_num": total, "succ_rate": succ / total if total else 0.0, "tasks": tasks}

train_dirs = sorted(p for p in gate_b.glob(f"libero40_full4k_gacc10_after_{tag}*") if p.is_dir())
train = []
for d in train_dirs:
    checkpoints = sorted((d / "checkpoints").glob("checkpoint_step_*")) if (d / "checkpoints").exists() else []
    train.append({"path": str(d), "checkpoints": [p.name for p in checkpoints]})

snap = {
    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "tag": tag,
    "gpu": sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"),
    "processes": sh("pgrep -af 'wan_va_server.py|evaluation/libero/client.py|torch.distributed.run|run_liberolong_eval_then_libero40_train|run_liberolong_full4k_two_round_eval|run_gate_b_libero40_full_train|ranked_train_entry.py' || true"),
    "eval_runs": runs,
    "libero40_train": train,
}
print(json.dumps(snap, sort_keys=True))
PY
  sleep "${interval}"
done

TAG="${tag}" python - <<'PY' >> "${out}"
import json
import os
from datetime import datetime, timezone
print(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "tag": os.environ["TAG"],
    "event": "supervisor_pid_exited",
}))
PY
