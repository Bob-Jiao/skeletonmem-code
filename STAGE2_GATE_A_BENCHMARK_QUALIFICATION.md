# Stage2-Gate A: Full-Data Benchmark Qualification

## Current Decision

下一轮只做 Gate A，不做 selector sweep、不做 budget 曲线、不推进 memory-task 主实验。

Gate A 要回答的问题是：

> 在 full supervision 下，LingBot-VA 在本机是否能通过官方训练流程稳定学会至少一个任务，并产生非零、可重复的 closed-loop signal？

只有通过 Gate A 的任务，才允许进入后续 Skeleton Supervision Gate。

## Hard Rule

所有新增文件只写入：

- `/data/jiaoguanbo/skeletonmem`

其他目录只允许读取，或作为下载目标；不能修改外部源码、数据或结果。

源码与 checkpoint 规则：

- Gate A 只使用新的 Stage2 专用 LingBot-VA worktree：
  `/data/jiaoguanbo/skeletonmem/worktrees/lingbot-va-stage2-gatea`
- 不混用主目录源码：
  `/data/jiaoguanbo/lingbot-va`
- checkpoint 可以放在主目录共享位置，例如：
  `/data/jiaoguanbo/models`
- 每个 run 必须显式记录初始化 checkpoint 路径，并区分 `base`、`released posttrain`、`local finetune`。
- Stage2 worktree 当前基于官方 commit：
  `robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`
- 最小运行适配 patch：
  `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/lingbot_stage2_gatea_minimal_runtime.patch`

## Evidence From Stage1

Stage1 的准确结论是工程链路验证，不是 full-data baseline 证明。

| item | evidence | conclusion |
| --- | --- | --- |
| LIBERO released checkpoint eval | `/data/jiaoguanbo/skeletonmem/results/stage1/libero_eval/libero_10_0.json` | clean path `1/1` success |
| old LIBERO eval record | `/data/jiaoguanbo/lingbot-va/outputs/libero/libero_10_0.json` | old local record `49/50`; historical context only, not Gate A qualification evidence |
| RoboTwin released checkpoint eval | `/data/jiaoguanbo/skeletonmem/results/stage1/robotwin_eval/stseed-10000/metrics/adjust_bottle/res.json` | `adjust_bottle` `1/1` success |
| RoboTwin two-GPU train smoke | `/data/jiaoguanbo/skeletonmem/logs/stage1/train_robotwin_2gpu_retry2.log` | one optimizer step works |
| LIBERO two-GPU train smoke | `/data/jiaoguanbo/skeletonmem/logs/stage1/train_libero_2gpu.log` | contract proxy only, not official full-data training |

Therefore:

- LIBERO eval is currently the most stable evaluation path.
- LIBERO official full-data fine-tuning is not verified because official LingBot LIBERO latent training data has not been found locally.
- RoboTwin `adjust_bottle` is the strongest engineering sanity baseline, but it is weak as a scientific skeleton/memory task.
- RoboTwin-Mem / RMBench must not be used for selector comparison until their full-data baseline passes Gate A.

## Dataset Audit

Audit script:

- `/data/jiaoguanbo/skeletonmem/scripts/stage2/gate_a_dataset_audit.py`

Output:

- `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a_dataset_audit.csv`

Key findings:

| candidate | local data status | current Gate A status |
| --- | --- | --- |
| LIBERO official LingBot latent data | not found | blocked by missing full training data |
| LIBERO contract proxy | 2 episodes, 4 latent files | only validates training contract |
| RoboTwin `adjust_bottle` | 50 raw episodes found, but only few latent files in audit path | engineering baseline candidate after latent coverage check/regeneration |
| RoboTwin-Mem `pick_objects_in_order` | 50 episodes, 850 segments, 2550 latent files | data-qualified, but closed-loop full success currently `0` |
| RMBench `cover_blocks` tailfixed | 50 episodes, 768 segments, 2304 latent files | data-qualified, but full closed-loop status unknown |

## Frozen Candidate Manifest

Candidate manifest:

- `/data/jiaoguanbo/skeletonmem/configs/stage2/gate_a/benchmark_qualification_candidates.yaml`

The manifest separates:

- evaluation-path evidence;
- full-data training-data evidence;
- policy/config contract;
- Gate A status;
- next required action.

## Training Entry

Full-data LingBot smoke script:

- `/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_a_lingbot_full_train.sh`

The script now uses:

- source root: `/data/jiaoguanbo/skeletonmem/worktrees/lingbot-va-stage2-gatea`
- output root: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full_train`
- compile/runtime cache: `/data/jiaoguanbo/skeletonmem/runtime/cache`

It requires:

```bash
run_gate_a_lingbot_full_train.sh <libero|robotwin> <dataset_path> <run_name> [steps] [master_port]
```

It fails early if:

- `dataset_path` does not exist;
- `dataset_path/empty_emb.pt` is missing.

This is intentional: for Gate A, we should not silently train on incomplete or incorrectly rooted data.

The script was syntax-checked after switching to the Stage2 worktree. A previous 1-step contract-proxy LIBERO smoke ran successfully before this source-root switch, so it is only historical evidence for the generic two-GPU training path. If needed, repeat a 1-step smoke with the Stage2 worktree before any longer Gate A run.

## Gate A Passing Criteria

A task passes Gate A only if all conditions hold:

1. official or frozen local full-data root is recorded;
2. observation camera order matches the LingBot config;
3. raw action dimension and model action channel mapping match;
4. `empty_emb.pt`, `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`, `data`, and `latents` are present;
5. full-data fine-tune has finite train loss and a meaningful validation/loss signal;
6. checkpoint selection rule is frozen before rollout;
7. fixed closed-loop eval protocol gives nonzero success or meaningful nonzero stage progress;
8. at least a second seed or repeated eval does not completely collapse;
9. failure modes are not dominated by observation/action interface mismatch;
10. all commands, logs, checkpoints, and results are written under `/data/jiaoguanbo/skeletonmem`.

## Immediate Next Actions

1. Locate or download official LingBot LIBERO latent training data.
   - If found: run LIBERO full-data short fine-tune first.
   - If not found quickly: do not spend time on format migration in this stage.
2. For RoboTwin `adjust_bottle`, verify or regenerate full latent coverage.
   - Use it as pipeline qualification and negative/sanity baseline only.
3. For RoboTwin-Mem / RMBench, run only full-data qualification probes after one stable engineering baseline exists.
   - No selector sweep until full-data signal is nonzero.

## Current Non-Completion Reason

The old Stage2 objective included selector comparison and preliminary closed-loop evaluation. That objective is not complete because no task has yet been qualified with full-data fine-tuning and stable nonzero closed-loop signal under the Gate A rules.
