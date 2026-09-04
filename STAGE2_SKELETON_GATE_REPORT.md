# Stage2 Skeleton Supervision Gate

## Current Scope

This stage tests one question first:

> Under the same effective supervision budget, does an event/stage skeleton get closer to the full-data baseline than random or uniform selection?

The first task family is `pick_objects_in_order`. All generated scripts, manifests, tables, and reports are kept under `/data/jiaoguanbo/skeletonmem`.

## What Was Done

- Selected the first audit dataset:
  `/data/jiaoguanbo/datasets/robotwin-mem-lingbot-official-hf-tailfixed/pick_objects_in_order-demo_clean`
- Used the original RoboTwin-MeM metadata for oracle event references:
  `/data/jiaoguanbo/Robotwin-MeM/lerobot_2.1/pick_objects_in_order`
- Added the Stage2 audit and selector script:
  `/data/jiaoguanbo/skeletonmem/scripts/stage2/skeleton_gate_audit.py`
- Generated selector manifests and a budget table:
  `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_gate_v1`

## Data Audit

The selected dataset has:

- `50` episodes
- `56156` raw frames / unique action labels
- `850` nominal LingBot-style training segments
- `3` camera latent streams
- Segment metadata with `segment_id`, `frame_ids`, `action_loss_first`, `action_loss_last`, padding counts, and success-tail flags
- External oracle keyframes for all `50` episodes

The mapping is clear enough for Stage2:

- raw episode time: `(episode_index, frame_index)`
- training segment: `(episode_index, segment_id, start_frame, end_frame)`
- action supervision range: `[action_loss_first, action_loss_last]`
- observation latent frames: `frame_ids`

## Selectors

Implemented five selectors:

- `full`: full-data baseline, exported only at `100%`
- `random`: episode-stratified random segment selection
- `uniform`: uniform segment coverage across each episode
- `motion_gripper`: ranks segments by action and gripper change
- `event_stage`: oracle selector using keyframes, stage anchors, and success-tail metadata

Important distinction:

- `event_stage` is currently an oracle selector because it uses metadata not guaranteed available online.
- `random`, `uniform`, and `motion_gripper` are deployable-style baselines.

## Budget Contract

The primary budget is `unique_action_labels`, not nominal segment count.

Secondary budget fields are:

- `training_samples`
- `nominal_segments`
- `action_label_exposures`
- `action_overlap_ratio`
- `unique_observation_timesteps`
- `latent_files`
- `latent_bytes`

The full dataset has `107356` action-label exposures but only `56156` unique action labels, giving an overlap ratio of `1.91x`. This confirms that segment count alone would overstate the real supervision budget.

## Current Result

The dry-run budget table is:

`/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_gate_v1/budget_table.csv`

The generated manifests are trainable-data candidates, not copied datasets yet. They should be used to build masked or filtered training subsets next.

## Immediate Next Step

Before training, the next step is to connect these manifests to a lightweight policy training path:

- Decide whether the first quick baseline uses ACT raw-state/action data or LingBot-style latent/action segments.
- Implement a dataset wrapper/exporter that consumes a manifest and exposes only selected samples or selected action-loss timesteps.
- Run one smoke train/validation pass for `full`, `random-25`, `uniform-25`, `motion-25`, and `event-stage-25`.

## Expected Evidence

A positive signal means:

- `event_stage` has lower validation action loss than `random` and `uniform` at matched `unique_action_labels`.
- The gap persists at both `25%` and `50%` budgets, or appears clearly in one budget with consistent failure cases.
- Full-data baseline trains and validates normally.

A negative signal means:

- The task may not expose enough stage/causal dependence in this training metric.
- The oracle event definition may not align with action-loss value.
- The budget may need to move from segment selection to timestep-level loss masking.

## Proxy Training Smoke

ACT's local implementation expects HDF5 image episodes, while the current audited data is LeRobot parquet plus LingBot latent segments. To avoid starting Stage2 with a large data conversion, the first smoke uses a lightweight future-action proxy:

- input: `observation.state[t]` plus normalized episode progress
- target: `action[t+16]`
- train split: episodes `0-39`
- validation split: episodes `40-49`
- model: two-layer MLP
- training: `5` epochs, batch size `2048`, seed `0`

This is not a replacement for closed-loop success. It is a cheap gate for whether selector choice has any visible signal before spending rollout or LingBot fine-tuning cost.

Results:

| selector | budget | train samples | best val loss |
| --- | ---: | ---: | ---: |
| full | 100% | 44265 | 0.1348 |
| random | 25% | 11449 | 0.2415 |
| uniform | 25% | 11520 | 0.8481 |
| motion/gripper | 25% | 11136 | 3.4617 |
| event/stage | 25% | 11392 | 0.9086 |
| random | 50% | 22221 | 0.1838 |
| uniform | 50% | 23040 | 0.3314 |
| motion/gripper | 50% | 22464 | 0.2387 |
| event/stage | 50% | 22464 | 0.2701 |

Artifacts:

- results: `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_proxy_v2/proxy_loss_results.csv`
- curve: `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_proxy_v2/proxy_loss_curve.png`
- failure examples: `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_proxy_v2/*_failure_examples.json`
- configs and per-run histories: `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_proxy_v2`

Interpretation:

- The first proxy result does not support the strong claim that the current `event_stage` selector is better than random/uniform.
- At `25%`, random is clearly better than event/stage and uniform under this future-action proxy.
- At `50%`, random remains best among compressed selectors, with motion/gripper and event/stage behind it.
- The highest-error validation examples for `event_stage-25%` cluster around timestep `598-601`, while `random-25%` clusters around `418-421`, suggesting that phase coverage is a real diagnostic target.
- This does not yet falsify the research direction, because the proxy is smooth state-action prediction, not memory-dependent closed-loop control. It does show that the current event/stage heuristic is too sparse or too tail/event biased for generic future-action loss.

Next diagnostic before closed-loop:

- Add selector coverage plots over episode time.
- Check whether `event_stage` over-selects keyframe/tail windows and under-covers ordinary transport phases.
- Add a timestep-level loss mask variant instead of segment-level filtering.
- Re-run the proxy with at least `3` random seeds for `random` and `event_stage`.
- Only promote to expensive closed-loop if the selector has either validation-loss advantage or a clearly justified task-specific success hypothesis.

## Closed-Loop Audit

A preliminary closed-loop path check was attempted with the Stage1 LingBot server/client setup. The standard RoboTwin client cannot evaluate `pick_objects_in_order` because that task is not present in `/data/jiaoguanbo/RoboTwin/envs`; it exists under `/data/jiaoguanbo/EventVLA/RoboTwin-Mem/envs`.

Existing RoboTwin-MeM LingBot closed-loop results were summarized read-only:

- summary: `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_closed_loop_existing_summary.csv`
- source root: `/data/jiaoguanbo/EventVLA/RoboTwin-Mem/eval_result/pick_objects_in_order/LingBotVA/demo_clean`
- rows: `30`
- success range: `0.0` to `0.0`
- reward range: `0.0` to `0.56`

This means `pick_objects_in_order` currently does not have a usable LingBot full-success baseline in the existing records. Stage2 should not spend compute on selector closed-loop comparison for this task until either:

- the EventVLA/RoboTwin-MeM LingBot evaluation path is reproduced cleanly under `/data/jiaoguanbo/skeletonmem`, or
- a task with a nonzero full-data closed-loop baseline is selected for the first selector comparison.

## Multi-Seed Proxy Result

The future-action proxy was repeated with three seeds. The aggregate file is:

- `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_proxy_multiseed/proxy_loss_aggregate.csv`
- `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_proxy_multiseed/proxy_loss_aggregate_curve.png`

Mean best validation loss:

| selector | budget | seeds | mean best val loss | std |
| --- | ---: | ---: | ---: | ---: |
| full | 100% | 3 | 0.1338 | 0.0013 |
| random | 25% | 3 | 0.2277 | 0.0120 |
| uniform | 25% | 3 | 0.8669 | 0.0190 |
| event/stage | 25% | 3 | 0.8713 | 0.0401 |
| motion/gripper | 25% | 3 | 3.5829 | 0.1293 |
| random | 50% | 3 | 0.1818 | 0.0019 |
| motion/gripper | 50% | 3 | 0.2373 | 0.0021 |
| event/stage | 50% | 3 | 0.2752 | 0.0125 |
| uniform | 50% | 3 | 0.3341 | 0.0390 |

Interpretation:

- The negative proxy trend is stable across seeds.
- The current `event_stage` selector is not a good generic supervised-data selector for this task.
- The result is still useful: it catches a bad selector cheaply before expensive LingBot fine-tuning or rollout.

## Coverage Diagnostic

The coverage diagnostic is:

- `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_coverage_v1/selector_coverage_by_time.csv`
- `/data/jiaoguanbo/skeletonmem/results/stage2/pick_objects_in_order_coverage_v1/selector_coverage_by_time.png`

Key finding:

- At `25%`, `event_stage` covers early/keyframe regions heavily but leaves many later normalized-time bins at `0%` coverage.
- At `50%`, this improves in the first half but still leaves the last `40%` of normalized episode time uncovered.
- This explains why it performs poorly for a generic future-action proxy: the selector is too stage-anchor biased and misses ordinary continuation/control phases.

## Baseline Selection Rule After Stage1

Stage1 did not prove stable full-data fine-tuning on every target benchmark. It proved:

- LIBERO released checkpoint evaluation smoke: `1/1` success on one LIBERO-10 task.
- RoboTwin released checkpoint evaluation smoke: `1/1` success on `adjust_bottle`.
- RoboTwin two-GPU LingBot fine-tune smoke: one optimizer step works on two official `adjust_bottle` episodes.
- LIBERO two-GPU fine-tune smoke: contract-level only; official LIBERO-LONG training data was not available locally.

Therefore the Stage2 rule is:

1. Prefer a benchmark/task whose full-data closed-loop baseline is nonzero and reproducible on this machine.
2. Use `pick_objects_in_order` as a memory-heavy audit case, not as the first main closed-loop selector comparison, until full-data success is recovered.
3. Treat `adjust_bottle` as the current stable RoboTwin engineering baseline, but not as the strongest scientific memory task because it is short-horizon and weakly memory-dependent.
4. For the next main experiment, first run a full-data sanity check on candidate tasks, then apply the five selectors only to tasks that pass.

Candidate priority:

| candidate | strength | risk | current status |
| --- | --- | --- | --- |
| LIBERO released checkpoint task | stable evaluation path | may be near ceiling and not memory-heavy | good for evaluator sanity |
| RoboTwin `adjust_bottle` | verified evaluation and train smoke | weak memory/skeleton signal | good engineering baseline |
| RoboTwin-MeM `pick_objects_in_order` | strong memory/stage structure | existing LingBot closed-loop success is `0` | audit only for now |
| RMBench / RoboTwin-Mem alternatives | better memory-task fit possible | full baseline not yet checked | next candidate search |

## Current Stage2 Decision

The first Skeleton Supervision Gate is partially complete:

- data audit: complete for `pick_objects_in_order`
- raw-time to training-sample mapping: complete
- budget table and overlap audit: complete
- five selectors and 25/50/100 manifests: complete
- exposure-matched proxy training: complete, three seeds
- budget/loss curves and failure examples: complete
- closed-loop comparison: blocked for this task because no stable nonzero LingBot full baseline is available

Practical conclusion:

> Do not spend more compute trying to prove `event_stage` on `pick_objects_in_order` right now. First choose a stable full-baseline task, then port the selector audit and proxy gate there.
