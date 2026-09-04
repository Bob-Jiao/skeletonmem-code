#!/usr/bin/env bash
set -euo pipefail

tag=${1:-full4k_two_round_eval_$(date -u +%Y%m%dT%H%M%SZ)}
test_num=${2:-10}

root=/data/jiaoguanbo/skeletonmem
scripts=${root}/scripts/stage2
eval_root=${root}/results/stage2/gate_a
closed_loop_root=${eval_root}/a5_closed_loop
log_root=${root}/logs/stage2/gate_a
summary_json=${eval_root}/${tag}_summary.json
summary_md=${eval_root}/${tag}_summary.md
prepare_log=${log_root}/${tag}_prepare_eval_models.log

steps=(1000 2000 3000 4000)
screen_task_start=0
screen_task_end=3
qual_task_start=0
qual_task_end=10

mkdir -p "${closed_loop_root}" "${log_root}"

echo "TAG=${tag}"
echo "TEST_NUM=${test_num}"
echo "ROUND1=steps_1000_2000_3000_4000_task0to2_${test_num}eps"
echo "ROUND2=selected_checkpoint_task0to9_${test_num}eps_and_step4000_task0to9_${test_num}eps"

"${scripts}/prepare_liberolong_full4k_eval_models.sh" > "${prepare_log}" 2>&1

for step in "${steps[@]}"; do
  run_id=${tag}_round1_step${step}_task0to2_${test_num}eps
  model=${eval_root}/full4k_eval_model_step${step}
  echo "ROUND1_START step=${step} run_id=${run_id}"
  "${scripts}/run_gate2_full_checkpoint_fresh_per_task_eval.sh" \
    "${run_id}" \
    "${model}" \
    auto \
    "${test_num}" \
    "${screen_task_start}" \
    "${screen_task_end}"
  echo "ROUND1_DONE step=${step} run_id=${run_id}"
done

best_step=$(
  TAG="${tag}" CLOSED_LOOP_ROOT="${closed_loop_root}" TEST_NUM="${test_num}" python - <<'PY'
import json
import os
from pathlib import Path

tag = os.environ["TAG"]
root = Path(os.environ["CLOSED_LOOP_ROOT"])
test_num = int(os.environ["TEST_NUM"])
steps = [1000, 2000, 3000, 4000]
rows = []
for step in steps:
    run_id = f"{tag}_round1_step{step}_task0to2_{test_num}eps"
    succ = 0.0
    total = 0.0
    tasks = {}
    for task in range(3):
        p = root / run_id / f"official_client_task{task}_{test_num}eps" / f"libero_10_{task}.json"
        data = json.loads(p.read_text())
        s = float(data.get("succ_num", 0.0))
        n = float(data.get("total_num", 0.0))
        succ += s
        total += n
        tasks[str(task)] = {"succ_num": s, "total_num": n, "succ_rate": float(data.get("succ_rate", s / n if n else 0.0))}
    rows.append({"step": step, "succ_num": succ, "total_num": total, "succ_rate": succ / total if total else 0.0, "tasks": tasks})

# Screening selection rule: maximize aggregate success on task0-2; break ties by later checkpoint.
best = sorted(rows, key=lambda r: (r["succ_num"], r["step"]))[-1]
print(best["step"])
PY
)

echo "ROUND1_SELECTION best_step=${best_step}"

round2_steps=("${best_step}")
if [ "${best_step}" != "4000" ]; then
  round2_steps+=("4000")
fi

for step in "${round2_steps[@]}"; do
  run_id=${tag}_round2_step${step}_task0to9_${test_num}eps
  model=${eval_root}/full4k_eval_model_step${step}
  echo "ROUND2_START step=${step} run_id=${run_id}"
  "${scripts}/run_gate2_full_checkpoint_fresh_per_task_eval.sh" \
    "${run_id}" \
    "${model}" \
    auto \
    "${test_num}" \
    "${qual_task_start}" \
    "${qual_task_end}"
  echo "ROUND2_DONE step=${step} run_id=${run_id}"
done

TAG="${tag}" CLOSED_LOOP_ROOT="${closed_loop_root}" TEST_NUM="${test_num}" BEST_STEP="${best_step}" SUMMARY_JSON="${summary_json}" SUMMARY_MD="${summary_md}" python - <<'PY'
import datetime as dt
import json
import os
from pathlib import Path

tag = os.environ["TAG"]
root = Path(os.environ["CLOSED_LOOP_ROOT"])
test_num = int(os.environ["TEST_NUM"])
best_step = int(os.environ["BEST_STEP"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])

def load_run(run_id, tasks):
    out = {"run_id": run_id, "tasks": {}, "succ_num": 0.0, "total_num": 0.0}
    for task in tasks:
        p = root / run_id / f"official_client_task{task}_{test_num}eps" / f"libero_10_{task}.json"
        data = json.loads(p.read_text())
        s = float(data.get("succ_num", 0.0))
        n = float(data.get("total_num", 0.0))
        out["tasks"][str(task)] = {
            "succ_num": s,
            "total_num": n,
            "succ_rate": float(data.get("succ_rate", s / n if n else 0.0)),
            "json": str(p),
        }
        out["succ_num"] += s
        out["total_num"] += n
    out["succ_rate"] = out["succ_num"] / out["total_num"] if out["total_num"] else 0.0
    return out

round1 = {}
for step in [1000, 2000, 3000, 4000]:
    run_id = f"{tag}_round1_step{step}_task0to2_{test_num}eps"
    round1[str(step)] = load_run(run_id, range(3))

round2_steps = [best_step] if best_step == 4000 else [best_step, 4000]
round2 = {}
for step in round2_steps:
    run_id = f"{tag}_round2_step{step}_task0to9_{test_num}eps"
    round2[str(step)] = load_run(run_id, range(10))

doc = {
    "tag": tag,
    "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "protocol": {
        "dataset": "LIBERO-LONG / libero_10 official init pool",
        "train_run": "full4k_gacc10_20260818T134819Z",
        "round1": "checkpoint_step_1000/2000/3000/4000, task0-2, 10 episodes per task",
        "round2": "selected checkpoint on task0-9, plus checkpoint_step_4000 on task0-9 if not selected",
        "selection_rule": "maximize aggregate task0-2 success in round1; tie-break by later checkpoint",
        "important_constraint": "checkpoint_step_4000 task0-9 is always evaluated in round2",
    },
    "selected_step_from_round1": best_step,
    "round1": round1,
    "round2": round2,
}
summary_json.write_text(json.dumps(doc, indent=2) + "\n")

lines = [
    f"# LIBERO-LONG full4k two-round closed-loop eval summary",
    "",
    f"- tag: `{tag}`",
    f"- created UTC: `{doc['created_utc']}`",
    f"- train run: `full4k_gacc10_20260818T134819Z`",
    f"- selection rule: maximize aggregate task0-2 success; tie-break by later checkpoint",
    f"- hard constraint: `checkpoint_step_4000` is always evaluated on task0-9 in Round 2",
    "",
    "## Round 1: task0-2 screening",
    "",
    "| step | success | total | rate |",
    "| --- | ---: | ---: | ---: |",
]
for step in [1000, 2000, 3000, 4000]:
    r = round1[str(step)]
    lines.append(f"| {step} | {int(r['succ_num'])} | {int(r['total_num'])} | {r['succ_rate']:.4f} |")
lines.extend([
    "",
    f"Selected step for Round 2: `{best_step}`.",
    "",
    "## Round 2: task0-9 qualification",
    "",
    "| step | success | total | rate |",
    "| --- | ---: | ---: | ---: |",
])
for step in round2_steps:
    r = round2[str(step)]
    lines.append(f"| {step} | {int(r['succ_num'])} | {int(r['total_num'])} | {r['succ_rate']:.4f} |")
lines.extend([
    "",
    "Interpretation rule:",
    "",
    "- Round 1 is checkpoint screening only, not a paper-level task coverage result.",
    "- Round 2 is the first all-task local anchor check; it still estimates rollout variance for a single train seed.",
    "- Selector/accounting baselines remain blocked until the full-data all-task anchor is interpretable.",
])
summary_md.write_text("\n".join(lines) + "\n")
print(summary_json)
print(summary_md)
PY

echo "TWO_ROUND_EVAL_DONE tag=${tag}"
echo "SUMMARY_JSON=${summary_json}"
echo "SUMMARY_MD=${summary_md}"
