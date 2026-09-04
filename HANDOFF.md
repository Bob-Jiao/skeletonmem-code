# SkeletonMem / Temporal Supervision Accounting Handoff

Last updated: 2026-09-03

## Active state snapshot: 2026-09-03

If this section conflicts with older notes below, this section is authoritative.
The 2026-08-21 notes remain useful for lineage and for the LIBERO40 gripper
contract diagnosis, but they are no longer the latest project state.

Current strict paper positioning:

```text
Temporal curation resource-equivalent counterfactual audit
```

Do not frame the work as a memory paper, a new selector paper, a generic data
compression paper, or a retained-ratio leaderboard. The paper should test how a
temporal curation method consumes resources, and should attribute closed-loop
effects to resource-specific counterfactuals: selected support, dense-data
access schedule, target exposure/replay, and training/curation compute.

Active collaboration split:

- Codex owns code implementation, training, evaluation, raw result production,
  logs, tables, Resource Cards, and abnormal-run records.
- The researcher owns the research system, theory/formalism, experimental
  design, controls, pre-registration rules, result interpretation, paper
  narrative, and reviewer defense.
- Codex should not turn "better result" into "better selector" without the
  researcher's identification review.
- Codex should hand over evidence artifacts; the researcher decides whether they
  support claims and what the next highest-information experiment is.

Active positioning document:

```text
/data/jiaoguanbo/skeletonmem/TEMPORAL_CURATION_RESOURCE_AUDIT.md
```

Current theory interface:

```text
Temporal Curation Method = S + C

S = selected temporal support:
    selector, retained manifest, support size, nominal retention ratio

C = consumption protocol:
    remapping, sampling probability, warmup, dense anchors, replay,
    optimizer steps, schedule

B = L(S, C) = (U_ret, U_eligible, U_seen, E_A, {n_i}, R_bar, K_train, K_curate)

J = F(S, C, B, theta_0, E)
```

The retention ratio `q` is only a coarse property of `S`. It does not identify
`C`, `B`, or final closed-loop performance.

Canonical thesis:

```text
Nominal temporal retention does not identify the resources consumed by a
sequential robot imitation learner. Resource-specific counterfactual evaluation
is necessary to attribute closed-loop gains to temporal data selection rather
than to training schedule.
```

Use this resource vector throughout:

```text
B = (U_ret, U_eligible, U_seen, E_A, {n_i}, R_bar, K_train, K_curate)
```

where:

- `U_ret`: action-target support in the retained manifest;
- `U_eligible`: whole-run action-target support theoretically accessible;
- `U_seen`: support that actually enters effective loss;
- `E_A`: total effective action-target exposure;
- `{n_i}`: per-target exposure counts;
- `R_bar = E_A / U_seen`: mean replay multiplier;
- `K_train`: training cost;
- `K_curate`: curation cost.

Current empirical baseline anchor:

```text
LIBERO40 / libero_all 20k full-data LingBot-VA baseline
run: libero_all_private_repro_g32_20k_stateful_2gpu_20260828
checkpoint: checkpoint_step_20000/transformer
eval: libero40_eval_g32_20k_step20000_4suite_10eps_20260902
protocol: 40 tasks x 10 episodes/task = 400 rollouts
result: 385/400 = 96.25%
overall Wilson 95% CI: about 93.91% to 97.71%
report: /data/jiaoguanbo/skeletonmem/results/stage2/gate_b/reports/libero40_eval_g32_20k_step20000_20260902_frozen_summary.md
```

Scope guard:

- This is the current full-data baseline anchor for matched-resource experiments.
- It is not yet the larger `50 episodes/task` paper-final LIBERO protocol.
- `10 episodes/task` results are screening/frozen anchors unless upgraded.
- Do not use the invalid earlier LIBERO40 4k gripper-mismatch checkpoint as a
  control anchor.

Main claims now required:

1. Retention and run-level usage are not equivalent.
2. Support breadth and exposure depth separately matter for closed-loop control.
3. FrameSkip-style gains must be decomposed into selection quality,
   dense-access schedule bundle, and additional exposure on retained support.

Main experiment matrix, assuming pre-registered `q=20%`:

| ID | Support | Dense access | Target exposure | Implementation |
| --- | --- | --- | --- | --- |
| `F-L` | 100% | N/A | about 20% | `Full-20k` checkpoint at 4k |
| `F-H` | 100% | N/A | 100% | `Full-20k` final checkpoint |
| `R-L` | 20% random hard-pruned | no | about 20% | `Random-Hard-20k` checkpoint at 4k |
| `R-H` | 20% random hard-pruned | no | 100% | `Random-Hard-20k` final checkpoint |
| `RR-O` | nominal 20% random | yes | 100% | original remap schedule |
| `FS-O` | nominal 20% importance | yes | 100% | original FrameSkip-style schedule |
| `FS-S` | strict 20% importance | no | 100% | strict support trained to 20k |
| `FS-R` | strict 20% importance | no | about 20% | `FS-S` checkpoint at 4k |

Pre-registered budget switch rule:

- If `Full-4k` success is `60%-85%`, keep `q=20%` and `4k/20k`.
- If `Full-4k` is below `60%` or extremely unstable, switch to `q=50%` and
  `10k/20k`.
- This choice must be made from the Full learning curve only, before inspecting
  FrameSkip-style results.

Current lineage audit bundle:

```text
/data/jiaoguanbo/skeletonmem/results/lineage_audit/first_lineage_audit_bundle_20260903T135825Z
```

Status:

- Full 100-step tracer-OFF and tracer-ON dry runs completed with the same seed
  and packing seed.
- Actual lineage parquet exists for rank0/rank1 with required fields present.
- Resource Card exists as JSON/Markdown with whole-run, phase, and source-view
  summaries.
- Synthetic lineage tests passed `12/12`.
- Invariance report shows exact replay-vs-ON batch signatures for both ranks,
  zero logged loss/grad/lr/pack differences at rank0 tqdm precision, bitwise
  identical final transformer checkpoint, and no extra dataloader iteration by
  lineage sample-draw accounting.
- This validates the tracer for the Full path only. FrameSkip-style remapping is
  still covered by synthetic tests, not by actual training.

Immediate priority order:

1. Implement/validate the canonical raw-time lineage tracer and Resource Cards.
2. Audit existing Full 10k/20k artifacts with exact `U_*`, `E_A`, replay, and
   compute fields; call `320k/640k` global sample draws, not exposure.
3. Freeze the FrameSkip-style port: reference commit, Original-vs-Port table,
   remap object, warmup, 5:1 mixed schedule, retained manifest SHA256.
4. Run 100-200 step dry runs for Full, Random-Hard, `RR-O`, `FS-O`, and `FS-S`
   to validate batch mix, chunk alignment, and no cross-episode leakage.
5. Only then start the matched-resource training matrix.

Next Codex-to-researcher deliverable:

```text
two-run same-format Resource Cards for a proposed matched comparison, after the
researcher approves which labels may be called exposure-matched or
mean-replay-matched
```

It must include:

- exact field definitions;
- synthetic test results for overlap/remap/padding;
- whether DDP aggregation is exact or still approximate;
- batch-order/training-result invariance check;
- raw Resource Card values, not claim-level interpretation.

Hard wording rules:

- Say `resource-specific matched protocols`, not "all resources matched".
- Say `dense-access schedule bundle`, not "pure dense access effect".
- Say `additional exposure on retained support`, not "pure replay causal effect".
- Do not demand multiple retained-set seeds for deterministic FrameSkip. Freeze
  one manifest and vary training seeds; Random baselines need subset seeds.
- Do not report retained ratio without `U_ret`, `U_eligible`, `U_seen`, `E_A`,
  replay concentration, training compute, curation compute, and closed-loop
  uncertainty.

## Active state snapshot: 2026-08-21

If this section conflicts with older notes below, treat this section as the active state.

## Critical update: LIBERO40 gripper action contract and remap eval

Last checked: `2026-08-21T10:01:29Z`.

Current high-priority finding:

```text
The first LIBERO40 step4000 quick eval failure is dominated by a gripper action
semantic mismatch, not yet by a proved lack of training steps.
```

Confirmed contract facts:

- LIBERO eval environment command convention:
  - `action[6] = -1.0` opens the gripper.
  - `action[6] = +1.0` closes the gripper.
- LIBERO-LONG training data uses the same executable convention:
  - raw `action6=-1` corresponds to open;
  - raw `action6=+1` corresponds to close.
- LIBERO40 `libero_all` training data uses a different convention:
  - raw `action6` is `0/1`;
  - statistics and state correlation show `1=open`, `0=closed/closing`;
  - this is already present in the upstream source under
    `/data/yaoyifei/dataset/fastwam/libero_mujoco3.3.2/*_lerobot`, not introduced
    by the local symlink copy.
- The active training dataloader, server, and eval client did not originally
  remap this convention. The model can therefore fit low action loss while
  producing `+1` at task start, which the environment executes as close.

Important evidence:

- Failed LIBERO40 server output at task start predicted gripper values near
  `+1`.
- Long successful anchor predicts gripper values near `-1` at task start under
  the same server/client structure.
- Inference-side probe with `exec_action6 = 1 - 2 * model_action6` confirmed
  that the executed gripper command changes from close-like to open-like.

Diagnostic reports:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_gripper_action_contract_diagnosis_20260821.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_gripper_remap_probe_and_contract_audit_20260821.md
```

Code/script changes currently present:

```text
/data/jiaoguanbo/skeletonmem/repos/lingbot-va-gatea-official/evaluation/libero/client.py
/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_b_libero40_single_task_gripper_probe.sh
/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_b_libero40_full_checkpoint_eval_2gpu_5eps.sh
```

The client change is guarded by environment variable and is off by default:

```text
LINGBOT_VA_LIBERO_CLIENT_GRIPPER_REMAP=libero40_open01_to_cmd
```

When enabled:

```text
exec_action6 = 1 - 2 * model_action6
```

This eval mode treats the checkpoint output using the LIBERO40 training-data
semantics:

```text
model/data 1 -> env -1  # open
model/data 0 -> env +1  # close
```

Single-task remap probe already completed:

```text
run_id: libero40_gripper_probe_object_t0_1eps_remap_20260821
suite/task: libero_object task0
rollouts: 1
remap: libero40_open01_to_cmd
result: 0/1 success
```

Interpretation of the single-task probe:

- Remap took effect; logs show `model_gripper≈+1` and `exec_gripper≈-1`.
- The episode still failed, so a pure inference-side remap does not prove the
  checkpoint is good.
- The probe does prove the gripper action API mismatch is real and executable.

Active eval currently running:

```text
run_id: libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b
model: /data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_eval_model_step4000
checkpoint: LIBERO40 step4000
rollouts: 5 per task
suite/task coverage: all 40 LIBERO tasks
total rollouts planned: 200
gripper remap: libero40_open01_to_cmd
GPU split:
  GPU0: libero_object, libero_goal
  GPU1: libero_spatial, libero_10
```

Active eval process state at `2026-08-21T10:01:29Z`:

```text
main pid:   165481
shard0 pid: 165501
shard1 pid: 165502
GPU0 server: libero_object task0, port 29420
GPU1 server: libero_spatial task0, port 29520
GPU0 client: libero_object task0, test_num=5
GPU1 client: libero_spatial task0, test_num=5
GPU memory/utilization observed:
  GPU0: 25741 / 81920 MB, util 36%
  GPU1: 25807 / 81920 MB, util 40%
```

At that check time:

```text
No per-task json result had been written yet for the active remap eval.
Both first-task clients were connected and running rollouts.
```

Active eval logs:

```text
/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/libero40_eval/libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b_main.log
/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/libero40_eval/libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b_shard0.log
/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/libero40_eval/libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b_shard1.log
```

Active eval output root:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/eval/libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b
```

How to monitor:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b|wan_va_server.py|evaluation/libero/client.py'
find /data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/eval/libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b -name '*.json' -print -exec cat {} \;
nvidia-smi
```

Current interpretation gate:

- If remap eval recovers non-trivial success, the previous all-failure eval was
  primarily an eval action API mismatch.
- If remap eval remains near-zero, then continue debugging training sufficiency,
  mixed-suite learning, task/prompt/order, and action-scale issues.
- Do not treat either the original no-remap quick eval or the current remap eval
  as paper-level LIBERO40 benchmark evidence; both are diagnostic.
- Do not resume selector/accounting baselines based on LIBERO40 until this
  contract issue is resolved and a valid full-data reference is established.

Additional contract risk found:

```text
libero_all meta/tasks.jsonl and preprocess_manifest task order generally do not
match the official LIBERO benchmark task_idx order.
```

This is not necessarily a model-conditioning bug because the policy is
language-conditioned, but it is a reporting/sharding risk. Always report by
official benchmark suite + official benchmark task language, not by assuming
dataset `task_index == benchmark task_idx`.

Current paper framing:

```text
Temporal Supervision Accounting for Closed-Loop Robot Imitation
```

Do not frame the work as a memory paper, a new skeleton selector paper, or a generic data compression paper. The current ICRA-facing claim is that retained-data ratio is not a unit-consistent cost for temporal robot imitation learning. The paper should introduce a canonical temporal lineage, a supervision ledger, matched-resource protocols, and closed-loop control-sufficiency curves.

Current Gate status:

```text
Gate A: LIBERO-LONG full-data 4k local anchor exists, but paper-level full reference distribution is not yet solidified.
Gate B0: LIBERO40 / libero_all data contract passed.
Gate B: LIBERO40 full-data 4k training complete; step4000 quick eval was paused early after repeated all-failure rollouts.
```

Key completed Gate A facts:

- LIBERO-LONG full-data run `full4k_gacc10_20260818T134819Z` completed `4000` optimizer steps with `gradient_accumulation_steps=10`.
- Checkpoints exist at steps `1000`, `2000`, `3000`, and `4000`.
- Two-round eval run `full4k_eval_then_libero40_20260819T122727Z` completed.
- Round 1 task0-2, 10 episodes/task:
  - step1000: `22/30`
  - step2000: `26/30`
  - step3000: `26/30`
  - step4000: `27/30`
- Round 2 step4000 task0-9, 10 episodes/task: `91/100`.
- Interpretation: `91/100` is a strong local full-data control anchor, not a final paper-level full reference distribution. It is one training seed, 10 rollout/task, local setup, and current checkpoint rule.

Key completed Gate B0 / Gate B facts:

- Active mixed-suite data path: `/data/jiaoguanbo/LIBERO/libero_all`.
- Data contract: `1712` episodes/parquet, `3424` videos, `3424` latents.
- Suites:
  - Object: `457` episodes
  - Long/libero_10: `388` episodes
  - Goal: `433` episodes
  - Spatial: `434` episodes
- Dynamic schema dataloader patch is required and applied in the active clone because mixed-suite parquet includes extra `observation.states.*` columns.
- LIBERO40 full-data run `libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z` completed successfully.
- Training window: `2026-08-20T01:02:52Z` to `2026-08-20T16:12:09Z`.
- Final logged train state: `4000/4000`, latent loss `0.0880`, action loss `0.0757`, total loss `0.1636`, `eff_labels=22064`, max mem `56.50GB`.
- Checkpoints `1000/2000/3000/4000` exist; each safetensors file is `10177831668` bytes.
- `checkpoint_step_4000/transformer/diffusion_pytorch_model.safetensors` sha256:
  `df1df614a50daf953c3d81b3555c6bfd00219f92abc43701bd11db497a419221`.

LIBERO40 mixed-data ledger, computed before eval:

```text
C_unique-raw   = 277713
C_unique-label = 1961820
C_seen-label   = 91676228
R_replay       = 46.73019339185043
C_compute      = 4000 steps, gradient_accumulation_steps=10, world_size=2, about 30.28 GPU-hours
```

Ledger file:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z_ledger_20260821.json
```

Paused LIBERO40 eval:

```text
run id: libero40_full4k_step4000_all_suites_5eps_2gpu_20260821T065000Z
model: /data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_eval_model_step4000
protocol: 40 tasks x 5 rollouts/task = 200 rollouts
shard0 / GPU0: libero_object + libero_goal
shard1 / GPU1: libero_spatial + libero_10
runner: /data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_b_libero40_full_checkpoint_eval_2gpu_5eps.sh
output: /data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/eval/libero40_full4k_step4000_all_suites_5eps_2gpu_20260821T065000Z
logs: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/libero40_eval/
```

Pause decision:

- The eval was intentionally stopped on 2026-08-21 after early tasks repeatedly failed.
- Completed/partial observed JSON at pause:
  - `libero_object` task0/task1/task2: `0/5`, `0/5`, `0/5`;
  - `libero_object` task3: `0/1`;
  - `libero_spatial` task0/task1/task2: `0/5`, `0/5`, `0/5`;
  - `libero_spatial` task3: `0/1`;
  - aggregate observed quick-eval prefix: `0/32`.
- Since train/test suite are intended to be consistent for LIBERO40, this early all-failure pattern is evidence of a training/configuration/step-budget problem, not useful statistical benchmark signal.
- Current interpretation: LIBERO40 4k mixed-suite training is not a valid control anchor. Treat it as a failed compatibility/training-sufficiency probe until diagnosed.

Current research route:

1. Do not resume LIBERO40 eval until the training/configuration issue is diagnosed.
2. Do not let LIBERO40 become the main paper bottleneck.
3. Prioritize Long full reference solidification:
   - add at least a second full-data training seed;
   - add extra rollouts for weak Long tasks, especially task2/task4/task8 if still weak under the current checkpoint;
   - separate checkpoint selection from final test evaluation;
   - report task-level confidence intervals and failure taxonomy;
   - output a full-data cost ledger and manifest.
4. Implement an automated temporal ledger before any compressed-policy training:
   - canonical raw-time lineage;
   - action target lineage;
   - window/chunk overlap accounting;
   - valid action mask accounting;
   - gradient accumulation and optimizer-step accounting;
   - per-label exposure histogram where logs allow it;
   - replay multiplier and replay concentration/entropy;
   - GPU/wall-clock/peak-memory accounting.
5. Run the first scientific counterfactual only after the ledger can recompute the completed full-data run:
   - Full reference;
   - Random-50 fixed-step, as the common retained-ratio reporting baseline;
   - Random-50 label-repetition-matched;
   - Random-50 total-exposure-matched;
   - Random-50 compute-matched.

Actions still not allowed:

- Do not start selector/skeleton/memory baselines before the full reference ledger and Random-50 protocol contract exist.
- Do not claim `91/100` as the final full-data oracle.
- Do not claim LIBERO40 5-rollout eval as a statistical benchmark.
- Do not report retained percentage without `C_unique-label`, `C_seen-label`, replay multiplier, and compute.
- Do not use segment retention as the supervision unit without mapping back to the canonical raw-time lineage.

## Historical status before 2026-08-21

## Latest status: 2026-08-19 LIBERO-LONG 4k eval-to-LIBERO40-train supervisor running

Current Gate A status:

```text
FULLDATA_4K_GACC10_TRAINING_COMPLETE_TWO_ROUND_EVAL_THEN_LIBERO40_4K_SUPERVISOR_RUNNING
```

Current execution decision:

```text
CONDITIONAL_GO_TO_LIBEROLONG_4K_TWO_ROUND_EVAL_THEN_LIBERO40_4K_MIXED_TRAIN_HOLD_SELECTOR_BASELINES
```

Active run:

```text
full4k_gacc10_20260818T134819Z
```

Active run record:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_gacc10_20260818T134819Z_run_record.md
```

Active train log:

```text
/data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/full4k_gacc10_20260818T134819Z_train.log
```

Active protocol files:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/libero_full_data_protocol_update_20260818.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full_data_4k_training_protocol_20260818.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_two_round_eval_protocol_20260819.md
```

Important interpretation update:

- LIBERO future experiments follow the official full-data convention.
- We do not use a separate `45 train / 5 held-out` split as the main LIBERO closed-loop evaluation contract.
- The old custom `45/5` split and first accounting audit products were deleted as non-active artifacts.
- Future command files explicitly unset `LINGBOT_VA_EPISODE_FILTER_PATH` and `LINGBOT_VA_EPISODE_FILTER_KEY`.
- The full-data baseline now follows the original paper scale: `4000` train steps, saving every `1000` steps, with `gradient_accumulation_steps=10`.
- Aborted run `full4k_20260818T133130Z` used `gradient_accumulation_steps=1`; it was stopped and its training output/log were deleted.
- Run `full4k_gacc10_20260818T134819Z` used `gradient_accumulation_steps=10`; 20-step supervision passed and the run completed.
- `checkpoint_step_1000`, `checkpoint_step_2000`, `checkpoint_step_3000`, and `checkpoint_step_4000` have been observed under the run.
- `checkpoint_step_4000` was saved successfully at `2026-08-19 12:13:42 UTC`; final displayed total loss was `0.1255`.
- Eval model directories have been prepared for all four checkpoints:
  - `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_eval_model_step1000`
  - `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_eval_model_step2000`
  - `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_eval_model_step3000`
  - `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_eval_model_step4000`
- Two-round eval contract:
  - Round 1: `1k/2k/3k/4k x task0-2 x 10 episodes`;
  - Round 2: selected checkpoint on `task0-9 x 10 episodes`;
  - `checkpoint_step_4000` must also run `task0-9 x 10 episodes` regardless of Round 1 selection.
- `/data` currently has sufficient free space again; latest observed state was about `19T` available.
- LIBERO Object/Goal/Spatial data acquisition is now allowed only as CPU/I/O-only Gate B0 under `/data/jiaoguanbo/LIBERO`; it must not use GPU, start training, or interrupt the active run.
- Candidate Object/Goal/Spatial sources are HuggingFace LeRobot image/parquet datasets, not confirmed LingBot-ready Wan latent posttrain datasets.
- `/data/zouyude/data/lingbot-va/libero_all` was audited as a possible source tree, but the active multi-suite training source is now the extracted copy under `/data/jiaoguanbo/LIBERO/libero_all`.
- `/data/zouyude/data/lingbot-va/libero_all.zip` was extracted and audited; it contains LingBot-VA style Object/Goal/Spatial/Long parquet, videos, latents, and meta.
- The zip datasets require the dynamic schema dataloader patch because their parquet includes extra `observation.states.*` columns.
- `/data/jiaoguanbo/LIBERO/libero_all` is retained as the fixed B0 staging path. Its `data`/`videos` symlinks were relinked from unavailable `/data/shared/yaoyifei/...` targets to available `/data/yaoyifei/dataset/fastwam/libero_mujoco3.3.2/...` targets.
- After relink, parquet/videos/latents counts match the expected `1712/3424/3424`, and CPU-only multi-suite dataloader smoke passed.
- Gate B LIBERO40 training script is prepared but not launched. It writes future outputs under `results/stage2/gate_b/libero40_full_train/` and requires run names to start with `libero40_` to avoid confusion with the active LIBERO-LONG `full4k_gacc10_20260818T134819Z` run.

Key new result:

- local full-data step500 training completed from base;
- checkpoint saved at:
  `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a4_medium_train_a4full500_20260817T151000Z/checkpoints/checkpoint_step_500`;
- step500 held-out validation passed:
  - avg latent loss: `0.130291`;
  - avg action loss: `0.089883`;
  - avg total loss: `0.220174`;
- step500 A5 dedicated task0 run:
  - `1/1` success;
  - then `3/3` success confirmation;
- step500 A5 task0-2 pilot:
  - task0: `0/1`;
  - task1: `1/1`;
  - task2: `1/1`.

Interpretation:

- The A5 eval contract is credible.
- The local full-data checkpoint is no longer a zero-success baseline: step500 can close the loop.
- step500 is an engineering anchor only; it is too short for a final mixed-task LIBERO-LONG full-data baseline.
- The broader multi-task LIBERO-LONG baseline must be rebuilt as a 4k full-data checkpoint trajectory.

Main report:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_step500_full_training_validation_a5_report.md
```

Gate 1 result:

- Contract Pass: `PASS`.
- Main run:
  - task0: `2/3`;
  - task1: `2/3`;
  - task2: `2/3`;
  - aggregate: `6/9`.
- Post sentinel:
  - task0 episode index `0`: `0/1`.
- Sentinel mismatch:
  - main task0 episode0 was `True`;
  - post-sentinel task0 episode0 was `False`.

Interpretation:

- step500 fresh-start eval has non-zero multi-task control signal;
- the A5 contract and artifact lineage are usable;
- sentinel mismatch prevents treating step500 as a statistical full-data anchor;
- next stage is Gate 2A planning or video-level diagnosis, not selector/accounting baselines.

Gate 1 protocol:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_step500_fresh_start_gate1_protocol.md
```

Gate 1 result report:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_step500_fresh_start_gate1_result.md
```

Gate 2A protocol:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_step500_gate2a_protocol.md
```

Gate 2A result:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_step500_gate2a_result.md
```

Gate 2A completed test:

- fixed local full-data step500 checkpoint;
- task0-2 only;
- 10 episodes per task;
- fresh server per task, to reduce the within-server cross-task order risk exposed by Gate 1;
- official client init-state proxy remains `episode_idx % init_states.shape[0]`;
- no selector, skeleton, random/uniform, FrameSkip, accounting baseline, or budget curve.

Gate 2A attempt 0 note:

- run id `gate2a_step500_task0to2_10eps_fresh_per_task_20260818T000000Z`;
- failed before any rollout because the runner omitted `/data/jiaoguanbo/LIBERO` from `PYTHONPATH`;
- task0 server reached listening state, but client import failed with `ModuleNotFoundError: No module named 'libero'`;
- this is a runner environment failure, not a policy/eval-contract result;
- retry run id is `gate2a_step500_task0to2_10eps_fresh_per_task_retry1_20260818T000000Z`.

Gate 2A retry1 result, reinterpreted under full-data official-init protocol:

- Contract Pass: `PASS`, `30/30` rollouts completed with JSON/videos/logs;
- task0: `7/10`, Wilson 95% `[0.3968, 0.8922]`;
- task1: `7/10`, Wilson 95% `[0.3968, 0.8922]`;
- task2: `4/10`, Wilson 95% `[0.1682, 0.6873]`;
- aggregate reference only: `18/30`, Wilson 95% `[0.4232, 0.7541]`;
- interpretation: positive official-init closed-loop control signal on all three tasks, but task-level intervals remain too wide and task2 is weaker;
- decision: do not expand step500 as the final anchor; run full-data 4k training and evaluate checkpoints at `1000/2000/3000/4000` first.

Active next step:

```text
Run two-round official-init closed-loop eval for checkpoints 1000/2000/3000/4000 from
full4k_gacc10_20260818T134819Z.
```

Prepared eval scripts:

```text
/data/jiaoguanbo/skeletonmem/scripts/stage2/prepare_liberolong_full4k_eval_models.sh
/data/jiaoguanbo/skeletonmem/scripts/stage2/run_liberolong_full4k_two_round_eval.sh
/data/jiaoguanbo/skeletonmem/scripts/stage2/run_liberolong_eval_then_libero40_train.sh
```

User-approved long supervisor:

```text
Run LIBERO-LONG two-round eval first. If and only if eval exits successfully, start LIBERO40 mixed full-data 4k training on /data/jiaoguanbo/LIBERO/libero_all.
```

Active supervisor:

```text
tag: full4k_eval_then_libero40_20260819T122727Z
pid: 1808361
log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/full4k_eval_then_libero40_20260819T122727Z.log
pidfile: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/full4k_eval_then_libero40_20260819T122727Z.pid
libero40 run name: libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z
supervisor report: /data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z_supervisor_report.md
monitor pid: 1836129
monitor snapshots: /data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/full4k_eval_then_libero40_20260819T122727Z_monitor_snapshots.jsonl
```

Active training command:

```bash
/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_a_lingbot_full_train.sh \
  libero \
  /data/jiaoguanbo/LIBERO/libero-long-lerobot \
  full4k_gacc10_20260818T134819Z \
  4000 \
  29661 \
  1000 \
  10
```

Prepared LIBERO40 training script, not started:

```text
/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_b_libero40_full_train.sh
```

Prepared LIBERO40 protocol:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero40_full_data_training_protocol_20260819.md
```

Naming rule:

```text
LIBERO-LONG current run: full4k_gacc10_20260818T134819Z
LIBERO40 future runs: libero40_full4k_gacc10_<timestamp>
```

After the 4k trajectory exists, prepare eval models for steps `1000`, `2000`, `3000`, and `4000`, then evaluate task0-2 with `10` official init states per task before expanding any checkpoint to `20-30` episodes per task.

Current observed 4k checkpoints:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full_train/full4k_gacc10_20260818T134819Z/checkpoints/checkpoint_step_1000
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full_train/full4k_gacc10_20260818T134819Z/checkpoints/checkpoint_step_2000
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full_train/full4k_gacc10_20260818T134819Z/checkpoints/checkpoint_step_3000
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full_train/full4k_gacc10_20260818T134819Z/checkpoints/checkpoint_step_4000
```

## 0.1 Gate B0: LIBERO multi-suite data acquisition / audit

User requested preparing the standard LIBERO Object/Goal/Spatial/Long benchmark coverage. Current judgment:

- `CONDITIONAL GO` for data acquisition and audit only.
- This benchmark figure is useful for credibility and standard comparison, but it is not the core paper novelty.
- Core novelty remains temporal supervision accounting, matched exposure/replay/compute protocols, and control-sufficiency curves.
- Local data currently confirmed: `/data/jiaoguanbo/LIBERO/libero-long-lerobot` and `/data/jiaoguanbo/LIBERO/libero_all`.
- Object/Goal/Spatial/Long LingBot-VA style data has now been located inside `/data/zouyude/data/lingbot-va/libero_all.zip` and passed format/dataloader contract audit after a dynamic schema patch and symlink relink.
- The extracted copy is retained at `/data/jiaoguanbo/LIBERO/libero_all`.
- The current candidate repos are `lerobot/libero_object_image`, `lerobot/libero_goal_image`, and `lerobot/libero_spatial_image`.
- These are HuggingFace LeRobot image/parquet candidates for common LIBERO-style benchmark coverage.
- They are not the same data contract as the current Long dataset `robbyant/libero-long-lerobot`, which already includes LingBot-compatible Wan latents.
- The original LIBERO project also provides official raw Spatial/Object/Goal datasets; raw datasets would require a separate conversion/latent-prep path.

Gate B0 constraints:

- data may be downloaded only under `/data/jiaoguanbo/LIBERO`;
- logs, manifests, audit reports must be written under `/data/jiaoguanbo/skeletonmem`;
- no GPU use;
- no training/evaluation launch;
- do not modify or restart `full4k_gacc10_20260818T134819Z`;
- if downloaded data is raw LIBERO demos rather than LingBot-ready LeRobot parquet plus Wan latents, record it as `DATA_PRESENT_BUT_TRAINING_CONTRACT_NOT_READY`.
- if downloaded data is LeRobot image/parquet without Wan latents, also record it as `DATA_PRESENT_BUT_TRAINING_CONTRACT_NOT_READY` for LingBot until latent preparation is solved.

Gate B0 output fields:

- suite name;
- source repo / URL;
- local path;
- file counts and byte size;
- episodes and tasks if metadata exists;
- data format;
- latent availability;
- `empty_emb.pt` provenance;
- task mapping compatibility;
- whether current LingBot `libero_train` dataloader can consume it directly.

Current Gate B0 execution:

```text
ZIP_DATA_CONTRACT_PASS_DYNAMIC_SCHEMA_PATCH_APPLIED_SYMLINK_RELINKED_DATALOADER_SMOKE_PASS
```

Object partial download:

```text
source: lerobot/libero_object_image
target: /data/jiaoguanbo/LIBERO/libero-object-image-lerobot
log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_b0/libero_object_download_retry_no_xet_20260819.log
summary: /data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero_object_download_retry_no_xet_summary.json
```

Mode:

```text
CPU/I/O only, max_workers=1, HF_HUB_DISABLE_XET=1, HF_HUB_ENABLE_HF_TRANSFER=0
```

Final observed HF progress before pause:

- log reached `10 / 103` files;
- target directory reached about `793M`;
- file count: `23`;
- parquet data files: `8`;
- incomplete files: `1`;
- process `1623100` was terminated by user request;
- no residual download process remained;
- Goal/Spatial intentionally not started;
- partial local copy was later removed during disk-pressure cleanup.

Zouyude `libero_all` audit:

```text
source: /data/zouyude/data/lingbot-va/libero_all
decision: HOLD_NO_COPY_MISSING_PARQUET_DATA
```

Audit result:

- four suite directories exist:
  - `libero_10_no_noops_lingbot`
  - `libero_goal_no_noops_lingbot`
  - `libero_object_no_noops_lingbot`
  - `libero_spatial_no_noops_lingbot`
- each suite has `meta`, two-camera Wan latents, `empty_emb.pt`, `text_embeddings.pt`, `preprocess_manifest.json`, and `validation_summary.json`;
- latent coverage is complete at one agentview and one eye-in-hand latent per episode;
- no suite has `data/chunk-*/episode_*.parquet`;
- current `LatentLeRobotDataset.load_hf_dataset()` requires `root/data` parquet;
- the original source roots recorded in `preprocess_manifest.json` under `/data/shared/yaoyifei/dataset/fastwam/libero_mujoco3.3.2/` are not visible from this workspace.

Reports:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_source_audit.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_source_audit.json
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_copy_decision_20260819.md
```

Zouyude `libero_all.zip` extraction audit:

```text
source zip: /data/zouyude/data/lingbot-va/libero_all.zip
audit target: /data/jiaoguanbo/LIBERO/libero_all
decision: DATA_CONTRACT_PASS_WITH_DYNAMIC_SCHEMA_PATCH_SYMLINK_RELINKED
```

Zip audit result:

- zip has `8645` entries and `8596` files;
- extracted file count matched zip non-directory file count;
- `libero_object_no_noops_lingbot`: `457` episodes, `10` tasks, `67309` frames, `457` parquet, `914` videos, `914` latents;
- `libero_10_no_noops_lingbot`: `388` episodes, `10` tasks, `104280` frames, `388` parquet, `776` videos, `776` latents;
- `libero_goal_no_noops_lingbot`: `433` episodes, `10` tasks, `52895` frames, `433` parquet, `866` videos, `866` latents;
- `libero_spatial_no_noops_lingbot`: `434` episodes, `10` tasks, `53229` frames, `434` parquet, `868` videos, `868` latents;
- totals: `1712` episodes, `1712` parquet files, `3424` videos, `3424` latent files, `0` incomplete files.

Schema / dataloader result:

- zip parquet includes `observation.state`, `observation.states.ee_state`, `observation.states.joint_state`, `observation.states.gripper_state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`;
- old Long parquet only has `episode_index`, `index`, `frame_index`, `task_index`, `timestamp`, `action`, `observation.state`;
- the active repo dataloader now builds non-video parquet `Features` dynamically from `meta/info.json`;
- `data`/`videos` symlinks were relinked to `/data/yaoyifei/dataset/fastwam/libero_mujoco3.3.2/...`;
- zip multi-suite smoke passed with `len=1712`, sub lengths `[457, 388, 433, 434]`;
- existing Long regression smoke passed with `len=500`.

Gate B0 reports:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_zip_extracted_audit.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_zip_extracted_audit.json
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_zip_extracted_contract_report_20260819.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_dynamic_schema_dataloader.patch
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/disk_pressure_cleanup_20260819.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_symlink_relink_20260819.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero40_full_data_training_protocol_20260819.md
```

At latest cleanup:

- no residual `unzip`, HF download, `wan_va_server.py`, LIBERO eval client, old Gate 2A runner, or `full4k_gacc10_20260818T134819Z` training process remained;
- both A100 GPUs were observed idle after completion;
- `/data` latest observed free space is about `18T`; the previous `checkpoint_step_4000` disk risk is resolved for this run.

## 0. Hard constraints

- All new code, scripts, logs, manifests, reports, figures, and experiment outputs for this project must stay under:
  `/data/jiaoguanbo/skeletonmem`
- Do not modify:
  `/data/jiaoguanbo/lingbot-va`
- Benchmark data may live under:
  `/data/jiaoguanbo/LIBERO`
- Checkpoints may be read from:
  `/data/jiaoguanbo/models`
  and other existing read-only locations, but must be recorded by absolute path and source type.
- Do not start selector / skeleton / memory / budget comparison until a stable full-data closed-loop baseline is qualified.
- Do not start Object/Goal/Spatial training until the active LIBERO-LONG 4k checkpoint trajectory is evaluated and the mixed-suite training exposure/accounting plan is fixed.
- Future LIBERO40 training outputs must use `results/stage2/gate_b/libero40_full_train/libero40_*`; do not place them under Gate A Long output directories.

## 1. Research direction

The project is no longer framed as “a better event/frame selector.” FrameSkip-like work already covers much of that method story.

Current positioning:

> Temporal Supervision Accounting / Control-Sufficiency Curves for robot imitation learning.

Core question:

> When a paper says it trains with 20% or 50% of demonstration data, what did it actually reduce: unique experience, unique labels, seen label exposure, replay multiplier, or compute?

Main technical accounting variables:

- `C_unique-raw`: unique raw timesteps / observations retained.
- `C_unique-label`: unique supervised action labels after projection into policy training format.
- `C_seen-label`: effective labels actually seen by the optimizer.
- `C_compute`: steps, wall-clock, GPU-hours, FLOPs, memory.
- replay multiplier: `C_seen-label / C_unique-label`.

Current proposal:

- `/data/jiaoguanbo/skeletonmem/CONTROL_SKELETON_PROPOSAL.md`
- version at handoff: `v3.7`

Current plan:

- `/data/jiaoguanbo/skeletonmem/RESEARCH_PLAN.md`

## 2. Isolated LingBot-VA worktree

Active isolated clone:

```text
/data/jiaoguanbo/skeletonmem/repos/lingbot-va-gatea-official
```

Official commit:

```text
7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
```

This clone is modified only for Gate A runtime compatibility. The main `/data/jiaoguanbo/lingbot-va` tree is not modified.

Runtime patch:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a2_runtime_compat.patch
```

Patch scope:

- flash-attn fallback to `custom_sdpa`;
- LIBERO parquet features compatibility for `datasets==2.21.0`;
- environment overrides for dataset/model/train paths;
- train metrics logging, including effective action labels and memory;
- held-out episode filter support;
- websocket client host fix:
  - official default `0.0.0.0` caused client connect hang;
  - skeletonmem clone now defaults to `127.0.0.1`;
  - override env: `LINGBOT_VA_WEBSOCKET_HOST=127.0.0.1`.

## 3. LIBERO-LONG data and latent status

Dataset:

```text
/data/jiaoguanbo/LIBERO/libero-long-lerobot
```

Official source:

```text
robbyant/libero-long-lerobot
```

Audit status:

- download complete;
- episodes: `500`;
- raw frames: `138090`;
- parquet files: `500`;
- action_config nominal segments: `500`;
- agentview latents: `500 / 500`;
- eye-in-hand latents: `500 / 500`;
- missing official camera latents: `0`;
- videos present for both cameras.

Important caveat:

- Official HF dataset does not include `empty_emb.pt`.
- User authorized copying:

```text
source: /data/zouyude/data/lingbot-va/libero_all/empty_emb.pt
target: /data/jiaoguanbo/LIBERO/libero-long-lerobot/empty_emb.pt
sha256: 99e5568f3f700b675315dd9999a392a1eee8428d7ab8f3cefd945b093ef6a9c7
shape: [512, 4096]
dtype: torch.bfloat16
```

Data audit artifacts:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_liberolong_data_audit.generated.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_liberolong_data_audit.json
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_liberolong_manifest.yaml
```

## 4. Gate A training and validation status

Current Gate A status:

```text
PARTIAL_PASS_REFERENCE_EVAL_CONTRACT_VALIDATED_BUT_LOCAL_FULLDATA_BASELINE_NOT_YET_QUALIFIED
```

Meaning:

- training/data/validation/closed-loop pipeline can run;
- released `lingbot-va-posttrain-libero-long` succeeds under the same A5 closed-loop contract for `libero_10 task0`, including a `3/3` confirmation run;
- local 50-step finetune did not produce a stable full-data baseline;
- selector/skeleton/accounting comparison is still not allowed.

### A2 / A3 sanity

Passed:

- single episode lineage;
- dataloader sanity;
- model config/input contract sanity;
- one-step two-card FSDP tiny training sanity.

Artifacts:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a2_single_episode_lineage.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a2_dataloader_sanity.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a2_model_forward_sanity.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a3_tiny_training_sanity.md
```

### A4-short

Status: pass.

- 10 full-data training steps;
- checkpoint save/reload smoke passed;
- first total loss: `0.659`;
- last total loss: `0.3781`;
- reload total loss: `0.3889`.

### A4-medium train50

Status: pass.

Checkpoint:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a4_medium_train_a4medium50_20260817T083843Z/checkpoints/checkpoint_step_50
```

Metrics:

- steps: `50`;
- first total loss: `0.6093`;
- last total loss: `0.2919`;
- max total loss: `0.6639`;
- min total loss: `0.2764`;
- max allocated GPU memory: `56.61 GB`.

Artifacts:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a4_medium_training_report.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a4_medium_metrics.jsonl
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a4_medium_train_loss_curve.png
```

### Historical held-out diagnostic validation

Held-out split status:

- deterministic task-balanced diagnostic used during early A4 validation;
- per task: last `5` episodes as validation;
- train / val: `450 / 50` episodes;
- generated split/accounting artifacts are now deleted and must not be used by future commands.

Split:

```text
deleted: /data/jiaoguanbo/skeletonmem/results/stage2/accounting/liberolong_heldout_split.yaml
```

Full held-out validation for step-50 checkpoint:

- validation segments: `50`;
- global batches evaluated: `50`;
- global seen labels: `96012`;
- seed: `0`;
- avg latent loss: `0.149725`;
- avg action loss: `0.140073`;
- avg total loss: `0.289798`.

Artifacts:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_validation_full_report.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_validation_full_metrics.jsonl
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_train_val_loss_overview.png
```

## 5. Accounting prototype result

Accounting audit:

```text
deleted: /data/jiaoguanbo/skeletonmem/results/stage2/accounting/liberolong_accounting_audit.md
deleted: /data/jiaoguanbo/skeletonmem/results/stage2/accounting/liberolong_accounting_audit.json
deleted: /data/jiaoguanbo/skeletonmem/results/stage2/accounting/liberolong_accounting_overview.png
```

Key numbers:

- raw frames: `138090`;
- nominal LingBot segments: `500`;
- full official scalar labels / `C_unique-label`: `971656`;
- train split official scalar labels: `875644`;
- val split official scalar labels: `96012`;
- A4-medium train50 observed `C_seen-label`: `1943312`;
- replay multiplier vs full unique labels: `2.0x`.

Interpretation:

`50 optimizer steps` under current official dataloader/sampler/gradient-accumulation already equals about two full-dataset label exposures. Future “less data” claims must report unique labels, seen labels, replay multiplier, and compute separately.

## 6. A5 closed-loop status

Status:

```text
PARTIAL_PASS_REFERENCE_EVAL_CONTRACT_VALIDATED_BUT_LOCAL_FULLDATA_BASELINE_NOT_YET_QUALIFIED
```

### Rendering issue and HANDOFF fix

Initial LIBERO rendering probes failed:

- default `MUJOCO_GL=egl`: EGL device display initialization failed;
- `MUJOCO_GL=osmesa`: OpenGL `glGetError` initialization failed.

The root `/data/jiaoguanbo/HANDOFF_RMBench.md` recorded a prior rendering fix. Reusing that fixed LIBERO env construction.

Required env:

```bash
export LD_LIBRARY_PATH=/data/jiaoguanbo/runtime/nvidia-gl-595/usr/lib/x86_64-linux-gnu:/data/zouyude/conda/envs/fastwam/lib:${LD_LIBRARY_PATH}
export VK_ICD_FILENAMES=/data/jiaoguanbo/lingbot-va/outputs/0729_robotwin_mem_pick_step1500_first_call_bundle_20260729/0729_nvidia_icd_egl.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/jiaoguanbo/lingbot-va/outputs/0729_robotwin_mem_pick_step1500_first_call_bundle_20260729/0729_nvidia_icd_egl.json
export MUJOCO_GL=egl
```

Probe result:

- `OffScreenRenderEnv` construct: pass;
- reset: pass;
- one zero-action step: pass.

Report:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_closed_loop_dependency_probe.md
```

### Websocket issue

Official `WebsocketClientPolicy` default host was `0.0.0.0`, which caused client connection to hang. Patched isolated clone to default to `127.0.0.1`.

Debug client proved:

- websocket connect: pass;
- env construct/reset: pass;
- server reset infer: pass;
- first obs-to-action infer: pass;
- action shape: `(7, 4, 4)`.

Debug report:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_debug_client_127_once_20260817T100526Z/debug_client_once_report.txt
```

### Official 1-rollout smoke

Completed official LIBERO client rollout:

- benchmark: `libero_10`;
- task range: `0 1`;
- instruction: `put both the alphabet soup and the tomato sauce in the basket`;
- test episodes: `1`;
- wall time: about `419 s`;
- saved frames: `796`;
- success: `0 / 1`;
- success rate: `0.0`.

Artifacts:

```text
report: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_closed_loop_smoke_report.md
client log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/gate_a5_official_client_a5_official_client_1rollout_20260817T100616Z.log
server log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/gate_a5_debug_server_a5_debug_server_20260817T100221Z.log
result json: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_official_client_1rollout_20260817T100616Z/libero_10_0.json
video: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_official_client_1rollout_20260817T100616Z/libero_10/0_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/0_False.mp4
```

Interpretation:

- closed-loop pipeline runs end-to-end;
- `0/1` success is not a stable baseline;
- 50-step local finetune is too short to judge final learning;
- next step should not be selector comparison.

### Released checkpoint reference eval

Completed released checkpoint reference eval under the same A5 contract:

- model symlink: `/data/jiaoguanbo/models/lingbot-va-posttrain-libero-long`;
- model real path: `/data/zouyude/ckpts/lingbot-va/lingbot-va-posttrain-libero-long`;
- benchmark: `libero_10`;
- task range: `0 1`;
- instruction: `put both the alphabet soup and the tomato sauce in the basket`;
- test episodes: `1`;
- wall time: about `168.20 s`;
- saved frames: `271`;
- success: `1 / 1`;
- success rate: `1.0`.

Additional runtime requirement observed during this run:

- set `PYTHONPATH=/data/jiaoguanbo/skeletonmem/repos/lingbot-va-gatea-official:/data/jiaoguanbo/LIBERO`;
- set `LIBERO_CONFIG_PATH=/data/jiaoguanbo/skeletonmem/runtime/libero_config`.

Artifacts:

```text
report: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_released_checkpoint_reference_eval_report.md
debug report: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_released_ref_20260817T114816Z/debug_client_once_report.txt
client log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/gate_a5_released_ref_official_client_1rollout_20260817T114816Z.log
server log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/gate_a5_released_ref_server_20260817T114816Z.log
result json: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_released_ref_20260817T114816Z/official_client_1rollout/libero_10_0.json
video: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_released_ref_20260817T114816Z/official_client_1rollout/libero_10/0_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/0_True.mp4
```

Interpretation:

- A5 eval contract is credible for at least `libero_10 task0`;
- local step-50 failure is now most likely insufficient training rather than a broken eval path;
- local full-data closed-loop baseline is still not qualified.

### Released checkpoint 3-episode confirmation

Completed a small multi-episode confirmation after the 1-rollout pass:

- model symlink: `/data/jiaoguanbo/models/lingbot-va-posttrain-libero-long`;
- model real path: `/data/zouyude/ckpts/lingbot-va/lingbot-va-posttrain-libero-long`;
- benchmark: `libero_10`;
- task range: `0 1`;
- instruction: `put both the alphabet soup and the tomato sauce in the basket`;
- test episodes: `3`;
- wall time: about `482.95 s`;
- saved videos: `0_True.mp4`, `1_True.mp4`, `2_True.mp4`;
- success: `3 / 3`;
- success rate: `1.0`.

Artifacts:

```text
report: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_released_checkpoint_task0_3eps_confirmation_report.md
client log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/gate_a5_released_ref_task0_3eps_client_20260817T120333Z.log
server log: /data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/gate_a5_released_ref_task0_3eps_server_20260817T120333Z.log
result json: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_released_ref_task0_3eps_20260817T120333Z/official_client_3eps/libero_10_0.json
videos: /data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a5_closed_loop/a5_released_ref_task0_3eps_20260817T120333Z/official_client_3eps/libero_10/0_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/
```

Interpretation:

- released reference is stable enough on `libero_10 task0` to proceed with longer local full-data training;
- local step-50 failure should not be interpreted as an eval contract failure.

## 7. Current decision

Gate A current decision:

```text
PARTIAL_PASS_REFERENCE_EVAL_CONTRACT_VALIDATED_BUT_LOCAL_FULLDATA_BASELINE_NOT_YET_QUALIFIED
```

Why partial pass:

- official data and latent chain are usable;
- training and validation run;
- closed-loop pipeline runs and saves video/result;
- released checkpoint succeeds on A5 reference eval, including `3/3` on `libero_10 task0`;
- but local full-data baseline still has no stable non-zero closed-loop signal.

Still not allowed:

- selector sweep;
- skeleton comparison;
- random/uniform subset training;
- budget-performance curve;
- memory module.

## 8. Recommended next step

Immediate next step:

1. Resume local full-data training from the existing Gate A setup beyond step 50, starting with a moderate target such as `200` or `500` steps.
2. Evaluate the longer local full-data checkpoint under the exact same A5 contract.
3. If local full-data success becomes non-zero and repeatable, then qualify Gate A baseline and only then start accounting selector/skeleton comparisons.

Do not start selector, skeleton, FrameSkip, random/uniform, or budget-curve experiments until local full-data closed-loop success is non-zero and repeatable.

## 9. Useful paths

Main docs:

```text
/data/jiaoguanbo/skeletonmem/CONTROL_SKELETON_PROPOSAL.md
/data/jiaoguanbo/skeletonmem/RESEARCH_PLAN.md
/data/jiaoguanbo/skeletonmem/HANDOFF.md
```

Gate A reports:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_liberolong_manifest.yaml
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_current_plan.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_command_log_index.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_full_training_report.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_closed_loop_report.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_final_decision.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_released_checkpoint_reference_eval_report.md
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_released_checkpoint_task0_3eps_confirmation_report.md
```

Scripts:

```text
deleted: /data/jiaoguanbo/skeletonmem/scripts/stage2/accounting_split_and_audit.py
/data/jiaoguanbo/skeletonmem/scripts/stage2/gate_a4_validation_loss.py
/data/jiaoguanbo/skeletonmem/scripts/stage2/gate_a5_debug_client_once.py
```

Eval model dir:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a4_medium_eval_model_step50
```

## 10. Cleanup state

At handoff time:

- no matching `wan_va_server.py` process should remain;
- no matching LIBERO client process should remain;
- GPUs were checked idle after A5 released checkpoint reference eval.

Before launching any next run, recheck:

```bash
ps -eo pid,ppid,stat,etimes,cmd | rg 'wan_va_server|evaluation/libero/client.py|torch.distributed.run'
nvidia-smi
```

## 11. 2026-08-21 LIBERO40 eval stopped and current diagnosis

Current LIBERO40 status:

```text
STOPPED_DO_NOT_CONTINUE_CURRENT_STEP4000_EVAL
```

The remapped LIBERO40 eval was stopped after all observed rollouts failed:

- run id: `libero40_full4k_step4000_all_suites_5eps_2gpu_gripper_open01_evalsem_20260821b`;
- model: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_eval_model_step4000`;
- partial observed results:
  - `libero_object task0`: `0/5`;
  - `libero_object task1`: `0/4`;
  - `libero_spatial task0`: `0/5`;
  - `libero_spatial task1`: `0/4`;
- observed failed videos before stop: `18`;
- post-stop process/GPU state: no matching LIBERO eval/server/client process remains; GPU0/GPU1 memory used was checked as `0 MiB`.

Report:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_full4k_step4000_gripper_remap_eval_stopped_and_diagnosis_20260821.md
```

Confirmed action contract diagnosis:

- LIBERO40 raw `action[6]` is `0/1`, where `1=open` and `0=closed/closing`;
- LIBERO-LONG and the LIBERO env command contract use `-1=open`, `+1=close`;
- current LIBERO40 training used the Long-style gripper norm config `q01=-1`, `q99=1`;
- therefore LIBERO40 gripper targets were trained as `{0, +1}`, not canonical env commands `{-1, +1}`;
- eval-side remap `exec_action6 = 1 - 2 * model_action6` was only a diagnostic execution fix, not a training fix;
- visual contract probe did not show a vertical camera flip bug. Training video raw frames match eval saved frame orientation after the client-side `[::-1]`.

Additional fit diagnosis:

- successful LIBERO-LONG 4k final 500-step action loss mean: `0.0361`;
- LIBERO40 mixed 4k final 500-step action loss mean: `0.0751`;
- LIBERO40 4k mixed run is action-underfit relative to the Long anchor, before considering the gripper semantic mismatch.
- saved eval action tensors show object-task gripper predictions are mostly open-state values:
  - `libero_object task0`: frac gripper `>0.75` = `0.9345`;
  - `libero_object task1`: frac gripper `>0.75` = `0.8767`;
  - `libero_spatial task0`: frac gripper `<0.25` = `0.5280`, but still `0/5`;
  - `libero_spatial task1`: frac gripper `>0.75` = `0.8156`.

Current checkpoint label:

```text
INVALID_ACTION_CONTRACT_AND_UNDERFIT_FOR_LIBERO40_CLOSED_LOOP
```

Do not spend more rollout budget evaluating this checkpoint.

Recommended next diagnostic:

1. Fix LIBERO40 training-side canonical gripper mapping before action normalization:

```text
action6_env_cmd = 1 - 2 * action6_open01
```

2. Run a cheap corrected smoke train first, not full LIBERO40:

```text
single suite or 1-2 object tasks, 500-1000 steps, no eval-side remap
```

3. Check generated gripper distribution and closed-loop success on the same easy task. Only if this smoke produces a real close phase and non-zero success should corrected mixed LIBERO40 full-data training resume.

## 12. 2026-08-21 corrected LIBERO40 full15k active run

Current status:

```text
CORRECTED_LIBERO40_FULL15K_RUNNING
```

Training-side gripper canonicalization was implemented behind this explicit env flag:

```text
LINGBOT_VA_TRAIN_GRIPPER_ACTION_REMAP=libero40_open01_to_cmd
```

Mapping:

```text
action6_env_cmd = 1 - 2 * action6_open01
```

Smoke gate:

- run: `libero40_grippercanon_smoke_first40eps_200s_gacc10_20260821`;
- steps: `200`;
- save interval: `100`;
- episode filter: first 40 episode ids per suite;
- result: completed cleanly;
- checkpoints: `checkpoint_step_100`, `checkpoint_step_200`;
- final logged loss at step `199`: latent `0.1081`, action `0.1356`, total `0.2437`;
- last-50-step mean action loss: `0.1318`;
- decision: `SMOKE_PASS_TRAINING_CONTRACT_AND_CHECKPOINT_SAVE_VALIDATED`.

Full corrected run:

- run: `libero40_grippercanon_full15k_gacc10_20260821`;
- status: running;
- pid file: `/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/libero40_grippercanon_full15k_gacc10_20260821.pid`;
- log: `/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/libero40_grippercanon_full15k_gacc10_20260821.log`;
- output: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_grippercanon_full15k_gacc10_20260821`;
- dataset: `/data/jiaoguanbo/LIBERO/libero_all`;
- episode filter: none;
- train steps: `15000`;
- save interval: `3000`;
- grad accumulation: `10`;
- world size: `2`;
- launched with `setsid` so the torchrun process is not tied to the Codex shell.

Initial full-run health check:

- training loop entered;
- latest observed during launch check: step `7`;
- step 7 loss: latent `0.2020`, action `0.3371`, total `0.5391`;
- GPU memory during launch check: about `71.5 GB` per GPU.

Step-50 follow-up health check:

- process state: running;
- pid: `373824`;
- latest observed step: `50`;
- step 50 loss: latent `0.1496`, action `0.1826`, total `0.3321`;
- last-20-step mean action loss: `0.1779`;
- checkpoint status: none expected yet; first save is `checkpoint_step_3000`;
- interpretation: `RUN_HEALTHY_PAST_EARLY_WARMUP_WAIT_FOR_STEP3000_CHECKPOINT`.

Step-100 follow-up health check:

- process state: running;
- pid: `373824`;
- latest observed step: `101`;
- step 101 loss: latent `0.1325`, action `0.1513`, total `0.2837`;
- last-20-step mean action loss: `0.1531`;
- last-50-step mean action loss: `0.1603`;
- checkpoint status: none expected yet; first save is `checkpoint_step_3000`;
- interpretation: `RUN_HEALTHY_STEP100_LOSS_STABLE_WAIT_FOR_STEP3000_CHECKPOINT`.

Expected checkpoints:

```text
checkpoint_step_3000
checkpoint_step_6000
checkpoint_step_9000
checkpoint_step_12000
checkpoint_step_15000
```

Run record:

```text
/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/libero40_grippercanon_full15k_launch_record_20260821.md
```

Immediate next checks:

1. Confirm the run continues past step 50 and action loss decreases into the normal range.
2. Wait for `checkpoint_step_3000`; do not eval before that checkpoint exists.
3. Do not run selector/accounting baselines while this repair full-data run is active.
