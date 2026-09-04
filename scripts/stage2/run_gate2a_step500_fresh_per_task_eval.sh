#!/usr/bin/env bash
set -euo pipefail

run_id=${1:-gate2a_step500_task0to2_10eps_fresh_per_task_20260818T000000Z}
test_num=${2:-10}

root=/data/jiaoguanbo/skeletonmem
model=${root}/results/stage2/gate_a/a4_full500_eval_model_step500
expected_transformer_sha=d47f8aeaa505549362f35a33596d08c3c78597d925cb4dafbcfe365037c8a44b

exec "${root}/scripts/stage2/run_gate2_full_checkpoint_fresh_per_task_eval.sh" \
  "${run_id}" \
  "${model}" \
  "${expected_transformer_sha}" \
  "${test_num}" \
  0 \
  3
