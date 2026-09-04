#!/usr/bin/env bash
set -euo pipefail

tag=${1:-full4k_eval_then_libero40_$(date -u +%Y%m%dT%H%M%SZ)}
test_num=${2:-10}
libero40_run_name=${3:-libero40_full4k_gacc10_after_${tag}}
libero40_steps=${4:-4000}
libero40_master_port=${5:-29681}
libero40_save_interval=${6:-1000}
libero40_grad_accum=${7:-10}
libero40_dataset=${8:-/data/jiaoguanbo/LIBERO/libero_all}

root=/data/jiaoguanbo/skeletonmem
report=${root}/results/stage2/gate_b/libero40_full_train/${libero40_run_name}_supervisor_report.md
mkdir -p "$(dirname "${report}")"

write_report_header() {
  cat > "${report}" <<EOF
# LIBERO-LONG eval then LIBERO40 train supervisor

Status: running

- tag: \`${tag}\`
- started UTC: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`
- Round 1 eval: LIBERO-LONG checkpoints 1000/2000/3000/4000, task0-2, ${test_num} episodes/task
- Round 2 eval: selected checkpoint task0-9 plus forced checkpoint_step_4000 task0-9, ${test_num} episodes/task
- LIBERO40 run name: \`${libero40_run_name}\`
- LIBERO40 dataset: \`${libero40_dataset}\`
- LIBERO40 steps: \`${libero40_steps}\`
- LIBERO40 save interval: \`${libero40_save_interval}\`
- LIBERO40 grad accumulation: \`${libero40_grad_accum}\`
- LIBERO40 world size: \`2\`

## Timeline

- \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`: supervisor started.
EOF
}

append_report() {
  echo "- \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`: $*" >> "${report}"
}

write_report_header

append_report "starting LIBERO-LONG two-round closed-loop eval."
"/data/jiaoguanbo/skeletonmem/scripts/stage2/run_liberolong_full4k_two_round_eval.sh" "${tag}" "${test_num}"
append_report "LIBERO-LONG two-round closed-loop eval completed successfully."

append_report "starting LIBERO40 mixed full-data 4k training."
"/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_b_libero40_full_train.sh" \
  "${libero40_run_name}" \
  "${libero40_steps}" \
  "${libero40_master_port}" \
  "${libero40_save_interval}" \
  "${libero40_grad_accum}" \
  "${libero40_dataset}"
append_report "LIBERO40 mixed full-data training completed successfully."

{
  echo
  echo "Status: completed"
  echo
  echo "- completed UTC: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
  echo "- eval summary json: \`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/${tag}_summary.json\`"
  echo "- eval summary md: \`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/${tag}_summary.md\`"
  echo "- LIBERO40 output: \`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/${libero40_run_name}\`"
} >> "${report}"
