# SkeletonMem / CSS 一个月研究计划

> 版本：v3.0  
> 日期：2026-09-03  
> 当前定位：Temporal Curation Resource-Equivalent Counterfactual Audit  
> 最终定位：[TEMPORAL_CURATION_RESOURCE_AUDIT.md](./TEMPORAL_CURATION_RESOURCE_AUDIT.md)  
> 详细提案：[CONTROL_SKELETON_PROPOSAL.md](./CONTROL_SKELETON_PROPOSAL.md)

## 0. Active Route Override: 2026-09-03

如果本节与下方 2026-08-21 或更早计划冲突，以本节为准。

当前论文必须严格定位为：

> 对 temporal curation 的资源等价反事实审计。

不要再把工作讲成 memory、selector、data compression，或 retained-ratio
leaderboard。核心问题是：

> 一个 temporal curation 方法声称只保留了 `q%` 时间数据时，闭环收益到底来自
> retained temporal support，还是来自 dense-data access schedule、额外 target
> exposure/replay、optimizer trajectory、训练计算或筛选计算？

当前协作分工：

- Codex 负责代码实现、训练、评估、原始结果产出、日志、表格、Resource Card
  和异常记录。
- 研究者负责研究体系、理论形式化、实验设计、控制变量、预注册规则、结果
  解释、论文叙事和 reviewer defense。
- Codex 每次实验批次只交付 evidence artifacts；是否足以支撑论文结论以及
  下一步最高信息增益实验由研究者判断。

关键主张：

```text
q_retain does not identify B
```

其中：

```text
B = (U_ret, U_eligible, U_seen, E_A, {n_i}, R_bar, K_train, K_curate)
```

理论接口：

```text
Temporal Curation Method = S + C
B = L(S, C)
J = F(S, C, B, theta_0, E)
```

其中 `S` 是 selector/retained manifest/support size，`C` 是 remapping、
sampling probability、warmup、dense anchors、replay 和训练步数。`q` 只描述
`S` 的粗粒度属性，不能识别 `C`、`B` 或最终性能。

必须使用的术语：

- `resource-specific matched protocols`;
- `dense-access schedule bundle`;
- `additional exposure on retained support`;
- `FrameSkip-style port`，除非逐项证明与参考实现完全一致。

当前可用 full-data anchor：

- LIBERO40 / `libero_all` private mirror 20k full-data run:
  `libero_all_private_repro_g32_20k_stateful_2gpu_20260828`;
- eval: `libero40_eval_g32_20k_step20000_4suite_10eps_20260902`;
- result: `385/400 = 96.25%`;
- report:
  `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/reports/libero40_eval_g32_20k_step20000_20260902_frozen_summary.md`;
- scope: current matched-resource baseline anchor, not yet paper-final
  `50 episodes/task` official-style evaluation.

主实验默认预算：

```text
q = 20%
Full reference = 20k optimizer steps
low exposure point = corresponding 4k checkpoint
```

主矩阵：

| ID | Support | Dense access | Target exposure | 实现 |
| --- | --- | --- | --- | --- |
| `F-L` | 100% | N/A | about 20% | `Full-20k` 的 4k checkpoint |
| `F-H` | 100% | N/A | 100% | `Full-20k` final checkpoint |
| `R-L` | 20% random hard-pruned | 无 | about 20% | `Random-Hard-20k` 的 4k checkpoint |
| `R-H` | 20% random hard-pruned | 无 | 100% | `Random-Hard-20k` final checkpoint |
| `RR-O` | nominal 20% random | 有 | 100% | original remap schedule |
| `FS-O` | nominal 20% importance | 有 | 100% | original FrameSkip-style schedule |
| `FS-S` | strict 20% importance | 无 | 100% | strict support, 训练到 20k |
| `FS-R` | strict 20% importance | 无 | about 20% | `FS-S` 的 4k checkpoint |

低曝光点必须来自同一 schedule 的 early checkpoint，不能后验选 checkpoint。

预算切换只能根据 Full learning curve 预注册：

- `Full-4k` 成功率在 `60%-85%`：主线使用 `q=20%`, `4k/20k`;
- `Full-4k` 低于 `60%` 或极不稳定：主线切换到 `q=50%`, `10k/20k`;
- 不能看过 FrameSkip-style 结果后再决定预算。

实验种子规则：

- Random baseline 需要多个 subset seeds 和 training seeds；
- 确定性 FrameSkip 不要求多个 retained-set seeds；
- FrameSkip retained manifest 固定一个 SHA256，增加 training seeds；
- 核心条件至少 `2-3` 个 training seeds。

P0-P6 checklist:

- [x] P0：固化最终论文定位为 resource-equivalent counterfactual audit。
- [x] P0.1：冻结当前 LIBERO40 20k full-data anchor：`385/400`。
- [x] P1：实现 canonical raw-time lineage tracer；Full 100-step first audit bundle 已完成。
- [x] P1.1：canonical action ID 使用 absolute target timestep。
- [x] P1.2：记录 requested/remapped anchors、absolute context timestamps、absolute action-target timestamps。
- [x] P1.3：记录 valid mask、loss weights，并完成 DDP-aware aggregation。
- [x] P1.4：synthetic overlap/remap/padding tests 全部通过；`12/12` pass。
- [x] P1.5：证明 tracer 不改变 batch 顺序或训练结果；100-step OFF/ON checkpoint bitwise identical。
- [x] P2：生成 first Full 100-step Resource Card。
- [x] P2.1：报告 `U_ret`, `U_eligible`, `U_seen`, `E_A`, `{n_i}`, `R_bar`。
- [x] P2.2：报告 exposure p10/p50/p90/p99、zero-exposure ratio、`N_eff/U_A`。
- [x] P2.3：报告 training GPU-hours、wall-clock、valid targets/s、peak VRAM；curation GPU-hours 在 Full dry-run 中为 N/A/未触发。
- [ ] P2.4：复算 existing Full 10k/20k；将 `320k/640k` 命名为 global sample draws，不命名为 exposure。
- [ ] P3：FrameSkip-style port。
- [ ] P3.1：冻结参考代码 commit 和 Original-vs-Port 对照表。
- [ ] P3.2：明确 remap 对象、warmup steps、5:1 mixed schedule。
- [ ] P3.3：冻结 retained manifest 和 SHA256，所有 FS counterfactual 共用同一 manifest。
- [ ] P3.4：100-200 step dry run 验证 full/pruned batch 比例、action chunk 对齐、无跨 episode 越界。
- [ ] P4：机制矩阵校准 `F-L/F-H/R-L/R-H`。
- [ ] P5：FrameSkip-style 分解 `RR-O/FS-O/FS-S/FS-R`。
- [ ] P6：核心条件升级到 `50 episodes/task`，报告 paired task-stratified bootstrap、training-seed variance、failure-stage breakdown 和 paired-difference 95% CI。

当前明确不做：

- 不把 `q_retain` 或 nominal retained ratio 当作资源充分统计量；
- 不用一个 counterfactual 同时解释 selection、dense access 和 replay；
- 不把 Random 2x2 机制矩阵当作 FrameSkip 的硬性 go/no-go；
- 不虚构 deterministic FrameSkip 的 retained-set seeds；
- 不报告没有 ledger 验证的 matched relation；
- 不把 10 episodes/task screening 写成 paper-final 官方 benchmark。

## 0. Active Route Override: 2026-08-21

如果本节与下方历史周计划冲突，以本节为准。

当前论文不再定位为 data compression、memory 或 selector 论文。主线是：

> Robot imitation learning currently lacks a unit-consistent notion of temporal data efficiency. We introduce temporal supervision accounting, matched-resource interventions, and closed-loop control-sufficiency curves to ask what “using 50% robot data” actually means.

中文定调：

> 机器人模仿学习中的时间数据效率不能由 retained-data ratio 单独判断。窗口重叠、action chunk 和 optimizer replay 会让“保留了多少数据”与策略实际消耗的独立动作监督、总 label exposure 和训练计算脱钩。本文建立时间监督账本，并在闭环控制下测量不同资源定义的充分性。

当前事实状态：

- Gate A / LIBERO-LONG full-data 4k run `full4k_gacc10_20260818T134819Z` 已完成。
- Long two-round eval `full4k_eval_then_libero40_20260819T122727Z` 已完成：
  - Round 1 task0-2, 10 episodes/task: step1000 `22/30`, step2000 `26/30`, step3000 `26/30`, step4000 `27/30`;
  - Round 2 step4000 task0-9, 10 episodes/task: `91/100`.
- `91/100` 是 strong local full-data control anchor，不是最终 paper-level full reference distribution。
- Gate B0 / `/data/jiaoguanbo/LIBERO/libero_all` 数据合同通过：`1712` episodes, `3424` videos, `3424` latents。
- Gate B / LIBERO40 full-data 4k training 已完成：
  - run `libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z`;
  - final loss: latent `0.0880`, action `0.0757`, total `0.1636`;
  - step4000 transformer sha256 `df1df614a50daf953c3d81b3555c6bfd00219f92abc43701bd11db497a419221`.
- LIBERO40 mixed-data ledger 已复算：
  - `C_unique-raw = 277713`;
  - `C_unique-label = 1961820`;
  - `C_seen-label = 91676228`;
  - `R_replay = 46.73`;
  - `C_compute = 4000 steps, gacc=10, world_size=2, about 30.28 GPU-hours`.
- mixed-suite quick eval 已暂停：
  - run `libero40_full4k_step4000_all_suites_5eps_2gpu_20260821T065000Z`;
  - 原计划 40 tasks x 5 rollouts/task = 200 rollouts;
  - 暂停时 observed prefix 为 `0/32`;
  - 完成/部分完成：Object task0-2 `0/5`、Object task3 `0/1`、Spatial task0-2 `0/5`、Spatial task3 `0/1`;
  - train/test suite 按当前理解是一致的，因此该早期全失败更像训练配置、训练步数或 mixed-suite 训练充分性问题，而不是统计评估波动；
  - 这不是最终 LIBERO40 benchmark，不能用作统计结论。

论文主贡献收敛为三项：

1. **Canonical temporal supervision lineage**  
   将任意训练样本映射为：
   `episode -> raw timestep -> window/segment -> action target -> batch exposure`。核心原则是：model segment 不是独立监督单位，所有预算先定义在 canonical raw timeline，再投影到 policy training format。

2. **Temporal supervision ledger**  
   每个 run 必须自动报告：
   `C_unique-raw`, `C_unique-label`, `C_seen-label`, `C_compute`, replay multiplier, replay concentration/entropy, optimizer steps, gradient accumulation, GPU-hours, peak memory。

3. **Matched-resource protocols + control-sufficiency curves**  
   对同一个 retained ratio，在 fixed-step、label-repetition-matched、total-exposure-matched、compute-matched 下分别测闭环表现，报告 `S(C_unique-label)`, `S(C_seen-label)`, `S(C_compute)` 和区间化的 `B90/B95`。

P0-P4 checklist:

- [x] P0：暂停当前 LIBERO40 5-rollout 双卡 eval；observed prefix `0/32`，判定为 failed training-sufficiency / compatibility probe。
- [ ] P1：固化 Long full reference distribution。
- [ ] P1.1：补 Long full-data second training seed。
- [ ] P1.2：对弱任务补额外 rollout，优先 task2/task4/task8。
- [ ] P1.3：固定 checkpoint selection 和 final evaluation 分离规则。
- [ ] P1.4：输出 Long full-data ledger、manifest、task-level confidence interval 和 failure taxonomy。
- [ ] P2：实现 automated temporal ledger，并先复算已完成 full-data runs。
- [ ] P2.1：实现 raw-time lineage、action target lineage、window/chunk overlap、valid mask。
- [ ] P2.2：实现 exposure histogram、replay multiplier、replay entropy/concentration。
- [ ] P2.3：实现 compute ledger：steps、gacc、world size、wall-clock、GPU-hours、peak memory。
- [ ] P3：只在 ledger 可复算 full run 后启动 Random-50 protocol intervention。
- [ ] P3.1：Full reference。
- [ ] P3.2：Random-50 fixed-step。
- [ ] P3.3：Random-50 label-repetition-matched。
- [ ] P3.4：Random-50 total-exposure-matched。
- [ ] P3.5：Random-50 compute-matched。
- [ ] P4：如果 P3 出现 protocol-dependent divergence，再加入 Uniform、FrameSkip-style、Motion/Event-stage、25% budget 和 cross-task/policy validation。

当前明确不做：

- 不启动 selector/skeleton/memory baseline。
- 不把 Event-stage 作为主方法。
- 不把 `91/100` 写成最终 headline result。
- 不用 `5 rollouts/task` 的 LIBERO40 快速 eval 写统计 benchmark claim。
- 不报告 retained ratio 而缺少 `C_unique-label`, `C_seen-label`, replay multiplier 和 compute。

## 1. 工作区规则

1. 所有新增或修改的代码、脚本、日志、patch、manifest、报告和实验结果只能写入 `/data/jiaoguanbo/skeletonmem`。
2. `/data/jiaoguanbo/skeletonmem` 外的现有源码目录只读，不修改、不移动、不删除。
3. benchmark 数据允许下载到 `/data/jiaoguanbo/LIBERO`。
4. checkpoint 可从 `/data/jiaoguanbo/models` 或已有只读路径读取，但来源和绝对路径必须写入 manifest。
5. 在 Long full reference distribution 和 automated ledger 可复算前，不启动 selector / skeleton / memory 对比。
6. LIBERO40 mixed-suite 工作只作为兼容性和 scaling evidence；当前 4k mixed training 不能作为 control anchor，主线优先级低于 Long full reference 和 Random-50 accounting intervention。

## 2. 当前论文故事

我们不再把工作讲成“提出一个更好的 event/frame selector”。FrameSkip 已经覆盖了这条方法型故事。

新的核心问题是：

> 机器人模仿学习中所谓“少用 20%/50% 数据”，到底少用了什么？是 unique experience 真的减少，还是相同少量 label 被重复曝光更多次？

论文主线：

- 建立 temporal supervision accounting；
- 区分 `C_unique-raw`、`C_unique-label`、`C_seen-label`、`C_compute`；
- 报告 replay multiplier；
- 用 control-sufficiency curves 测量不同任务和策略达到 full-data 闭环性能所需的时间监督密度；
- 把 FrameSkip-style 方法作为第一强 baseline 和审计对象。

## 3. 当前已完成结果

### 2.1 LingBot-VA / LIBERO-LONG Gate A

已完成：

- A1 数据与 latent 审计；
- A2 single episode lineage / dataloader / model contract sanity；
- A3 one-step tiny training sanity；
- A4-short 10-step full-data smoke；
- A4-medium train50；
- LIBERO official full-data protocol; the previous deterministic held-out split products have been deleted and are not active inputs；
- completed full-data baseline protocol: `4000` training steps, checkpoint save every `1000` steps；
- 第一版 accounting audit；
- A4 validation smoke；
- A4 historical validation diagnostics。
- released checkpoint A5 reference eval；
- local step500 Gate2A official-init closed-loop qualification；
- aborted `gradient_accumulation_steps=1` 的 4k run 已记录并删除训练产物；
- completed 4k full-data baseline run: `full4k_gacc10_20260818T134819Z`，`gradient_accumulation_steps=10`，20-step supervision pass，training complete。
- run saved `checkpoint_step_1000`, `checkpoint_step_2000`, `checkpoint_step_3000`, and `checkpoint_step_4000`。
- `checkpoint_step_4000` saved successfully at `2026-08-19 12:13:42 UTC`; final displayed total loss was `0.1255`。
- LIBERO Object/Goal/Spatial Gate B0 established: candidate HuggingFace LeRobot image/parquet datasets identified and CPU-only downloader dry-run passed; after user restored proxy, Object partially downloaded with no-Xet single-worker mode, then was stopped by user request and later removed due disk pressure. These datasets are standard LeRobot image candidates, not confirmed LingBot-ready Wan latent datasets。
- `/data/zouyude/data/lingbot-va/libero_all` audited: four suite-level LingBot latent/meta directories are present, but that source tree itself is not the active training source。
- `/data/zouyude/data/lingbot-va/libero_all.zip` extracted and audited: Object/Goal/Spatial/Long LingBot-VA style parquet/videos/latents/meta are complete. A dynamic schema dataloader patch was added because these parquet files include extra `observation.states.*` columns. The extracted copy is retained at `/data/jiaoguanbo/LIBERO/libero_all`。
- `libero_all` symlinks were relinked from unavailable `/data/shared/yaoyifei/...` targets to `/data/yaoyifei/dataset/fastwam/libero_mujoco3.3.2/...`; parquet/videos/latents counts and CPU-only dataloader smoke now pass。
- Long two-round checkpoint eval 已完成，step4000 在 task0-9 上得到 `91/100` local anchor。
- Gate B LIBERO40 full-data 4k training 已完成：`libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z`。
- LIBERO40 step4000 quick eval 已暂停：`libero40_full4k_step4000_all_suites_5eps_2gpu_20260821T065000Z`，observed prefix `0/32`，只作为 failed mixed-suite training-sufficiency evidence。

当前状态：

`LONG_FULLDATA_4K_TWO_ROUND_EVAL_COMPLETE_LOCAL_ANCHOR_LIBERO40_4K_TRAIN_COMPLETE_QUICK_EVAL_RUNNING`

### 2.2 关键数值

LIBERO-LONG：

- episodes: `500`
- tasks: `10`
- raw frames: `138090`
- LingBot nominal segments: `500`
- full `C_unique-label`: `971656`

Historical held-out diagnostic：

- future LIBERO closed-loop experiments use the official full-data / official-init-state pool, not a separate `45 train / 5 val` main test split
- old custom split/accounting artifacts have been deleted and are not active inputs
- historical train episodes: `450`
- historical val episodes: `50`
- historical train official scalar labels: `875644`
- historical val official scalar labels: `96012`

A4-medium train50：

- optimizer steps: `50`
- grad accumulation: `10`
- world size: `2`
- observed `C_seen-label`: `1943312`
- replay multiplier vs full unique labels: `2.0x`
- first total loss: `0.6093`
- last total loss: `0.2919`
- max allocated GPU memory: `56.61 GB`
- checkpoint step 50 saved.

A4 validation smoke：

- checkpoint: `checkpoint_step_50`
- validation segments: `50`
- smoke evaluated: `5` batches per rank, `10` global batches
- global seen labels: `19908`
- avg latent loss: `0.160899`
- avg action loss: `0.148452`
- avg total loss: `0.309351`

解释：

validation smoke 只证明 step-50 checkpoint 可以在 held-out split 上完成两卡 FSDP 加载和 loss 评估；它还不是完整 validation curve，也不能证明 closed-loop success。

A4 full held-out validation：

- evaluated validation segments: `50`
- evaluated batches per rank: `25`
- global batches evaluated: `50`
- global seen labels: `96012`
- seed: `0`
- avg latent loss: `0.149725`
- avg action loss: `0.140073`
- avg total loss: `0.289798`

解释：

full validation 已确认 checkpoint step 50 的 open-loop held-out loss 管线可运行；当前仍不能证明 closed-loop success。

A5 dependency probe：

- HANDOFF 文件中确实记录过类似渲染问题；
- 复用 `/data/jiaoguanbo/runtime/nvidia-gl-595` 和 `0729_nvidia_icd_egl.json` 后，LIBERO `OffScreenRenderEnv` construct/reset/one-step 已通过；
- websocket client 默认连接 `0.0.0.0` 会卡住，已在 skeletonmem 隔离 clone 中改为默认 `127.0.0.1`；
- official 1-rollout smoke 已完成，保存视频，success `0/1`；
- 当前剩余任务是确认 released checkpoint 在同一 eval contract 下是否有非零 signal，或延长 full-data training。

解释：

`50 optimizer steps` 在当前官方 dataloader / sampler / grad-accum 设置下，已经约等于完整数据集的两遍 label exposure。这是当前最重要的 accounting 发现。

Released checkpoint reference eval：

- checkpoint: `/data/jiaoguanbo/models/lingbot-va-posttrain-libero-long`
- same A5 contract；
- task0: `1/1` then `3/3`；
- interpretation: LIBERO env/rendering/websocket/action interface/instruction contract are credible under this local setup.

Local step500 Gate2A：

- checkpoint: local full-data step500 from `/data/jiaoguanbo/models/lingbot-va-base`
- task0: `7/10`
- task1: `7/10`
- task2: `4/10`
- aggregate reference only: `18/30`
- interpretation: local training can produce non-zero closed-loop control, but step500 is an engineering anchor only and is too short for a final mixed-task full-data baseline.

Active 4k full-data baseline：

- run id: `full4k_gacc10_20260818T134819Z`
- train steps: `4000`
- save interval: `1000`
- gradient accumulation: `10`
- data protocol: `LIBERO_OFFICIAL_FULLDATA_NO_EPISODE_FILTER`
- episode filters: unset
- 20-step supervision: `PASS`
- saved checkpoints observed: `checkpoint_step_1000`, `checkpoint_step_2000`, `checkpoint_step_3000`, `checkpoint_step_4000`
- final checkpoint: `checkpoint_step_4000`, size about `9.5G`
- storage state: `/data` latest observed free space is about `18T`
- run record: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_gacc10_20260818T134819Z_run_record.md`
- log: `/data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/full4k_gacc10_20260818T134819Z_train.log`

Active 4k trajectory closed-loop eval：

- protocol: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_two_round_eval_protocol_20260819.md`
- prepare script: `/data/jiaoguanbo/skeletonmem/scripts/stage2/prepare_liberolong_full4k_eval_models.sh`
- runner script: `/data/jiaoguanbo/skeletonmem/scripts/stage2/run_liberolong_full4k_two_round_eval.sh`
- prepared eval models: `full4k_eval_model_step1000`, `full4k_eval_model_step2000`, `full4k_eval_model_step3000`, `full4k_eval_model_step4000`
- Round 1: checkpoints `1000/2000/3000/4000`, task0-2, `10` official init episodes per task
- Round 2: selected checkpoint on task0-9, `10` episodes per task
- hard constraint: `checkpoint_step_4000` is evaluated on task0-9 in Round 2 regardless of selection
- selector/accounting baselines remain blocked until this all-task full-data anchor is interpretable.

Approved chained execution after user instruction:

- supervisor script: `/data/jiaoguanbo/skeletonmem/scripts/stage2/run_liberolong_eval_then_libero40_train.sh`
- execution order:
  1. LIBERO-LONG Round 1 checkpoint screening;
  2. LIBERO-LONG Round 2 all-task qualification, with forced `checkpoint_step_4000` task0-9 evaluation;
  3. if eval completes without runner/pipeline failure, start LIBERO40 mixed full-data 4k training.
- LIBERO40 default train command inside supervisor:
  - dataset: `/data/jiaoguanbo/LIBERO/libero_all`
  - steps: `4000`
  - save interval: `1000`
  - grad accumulation: `10`
  - world size: `2`
  - output family: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/`

Active supervisor instance:

- tag: `full4k_eval_then_libero40_20260819T122727Z`
- pid: `1808361`
- log: `/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b/full4k_eval_then_libero40_20260819T122727Z.log`
- LIBERO40 run name: `libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z`
- rule: LIBERO40 training starts only if both LIBERO-LONG eval rounds exit successfully.

Discarded 4k run：

- run id: `full4k_20260818T133130Z`
- reason: used `gradient_accumulation_steps=1`, not original-scale `10`
- status: stopped, recorded, training output/log deleted
- no checkpoint from this run should be used.

### 2.3 当前产物

- A4-medium train curve: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a4_medium_train_loss_curve.png`
- A4 validation smoke report: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_validation_smoke_report.md`
- A4 full validation report: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_validation_full_report.md`
- A4 train/validation overview: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_train_val_loss_overview.png`
- A5 dependency probe: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_closed_loop_dependency_probe.md`
- A5 closed-loop smoke report: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_closed_loop_smoke_report.md`
- accounting overview: deleted with old custom split/accounting artifacts; historical numeric observations remain in md only
- historical held-out diagnostic split artifact: deleted; not an active protocol input.
- accounting report: deleted with old custom split/accounting artifacts; not an active protocol input
- Gate A manifest: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_liberolong_manifest.yaml`
- Gate2A step500 result: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_step500_gate2a_result.md`
- full-data 4k active protocol: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full_data_4k_training_protocol_20260818.md`
- active 4k run record: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_gacc10_20260818T134819Z_run_record.md`
- prepared LIBERO40 training protocol: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero40_full_data_training_protocol_20260819.md`

### 2.4 LIBERO multi-suite Gate B0

User-proposed next benchmark coverage:

- `libero_object`
- `libero_goal`
- `libero_spatial`
- `libero_10` / `libero_long`

Current judgment:

- `CONDITIONAL GO` for data acquisition and audit only；
- multi-suite full-data bar chart is useful for standard LIBERO presentation and reader trust；
- it is not the core novelty; core novelty remains accounting and matched temporal supervision protocols；
- current local data audit confirms `libero-long-lerobot` and retained `libero_all` staging copy exists；
- Object/Goal/Spatial must be downloaded or located under `/data/jiaoguanbo/LIBERO`, then audited before training。`/data/zouyude/data/lingbot-va/libero_all.zip` 已解压并固定为 `/data/jiaoguanbo/LIBERO/libero_all`（LingBot-VA multi-suite audit staging）。

Current B0 status:

- candidate repos: `lerobot/libero_object_image`, `lerobot/libero_goal_image`, `lerobot/libero_spatial_image`；
- source type: HuggingFace LeRobot image/parquet datasets; not the same as `robbyant/libero-long-lerobot` LingBot-ready Wan latent posttrain data；
- prepared script: `/data/jiaoguanbo/skeletonmem/scripts/stage2/download_libero_multisuite_cpu_only.py`；
- status report: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero_multisuite_data_gate_b0_status_20260819.md`；
- dry-run summary: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero_multisuite_download_summary.json`；
- Object partial target: `/data/jiaoguanbo/LIBERO/libero-object-image-lerobot`；
- Object partial size: about `793M`, `23` files, `8` parquet data files, `1` incomplete file before cleanup；
- Object log: `/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b0/libero_object_download_retry_no_xet_20260819.log`；
- Object status: stopped by user request; no residual download process; partial copy removed during disk-pressure cleanup；
- Goal/Spatial: not started。
- Zouyude source audit: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_source_audit.md`；
- Zouyude copy decision: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_copy_decision_20260819.md`；
- Zouyude status: `DATA_CONTRACT_PASS_WITH_EXTRACTED_COPY_KEPT`。
- Zouyude zip source: `/data/zouyude/data/lingbot-va/libero_all.zip`；
- zip extracted audit: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_zip_extracted_audit.md`；
- zip contract report: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_zip_extracted_contract_report_20260819.md`；
- dynamic schema patch: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_dynamic_schema_dataloader.patch`；
- disk cleanup report: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/disk_pressure_cleanup_20260819.md`；
- LIBERO40 training protocol: `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero40_full_data_training_protocol_20260819.md`；
- zip status: `DATA_CONTRACT_PASS_WITH_DYNAMIC_SCHEMA_PATCH_EXTRACTED_COPY_KEPT_AT_FIXED_PATH`。

Zip audit result:

- suites: `libero_object_no_noops_lingbot`, `libero_10_no_noops_lingbot`, `libero_goal_no_noops_lingbot`, `libero_spatial_no_noops_lingbot`；
- total episodes: `1712`；
- total parquet files: `1712`；
- total video files: `3424`；
- total latent files: `3424`；
- incomplete files: `0`；
- dataloader smoke after patch: zip multi-suite `len=1712` passed, existing Long `len=500` regression passed。

Officialness caveat:

- The LIBERO project website provides official raw datasets for Spatial/Object/Goal/100.
- The current B0 download uses HuggingFace LeRobot image/parquet candidate repos for the common LeRobot-style benchmark figure.
- For LingBot training, this is not yet an official training contract because Wan latents, feature names, action mapping, and `empty_emb.pt` provenance remain unaudited.

Gate B0 outputs:

- source repo / URL；
- local path；
- file and episode counts；
- format: raw LIBERO demos vs LeRobot parquet；
- Wan latent availability；
- `empty_emb.pt` provenance；
- task mapping compatibility；
- whether current LingBot `libero_train` dataloader can consume it directly。

Gate B0 must not:

- use GPU；
- start training；
- interrupt `full4k_gacc10_20260818T134819Z`；
- be described as a completed multi-suite experiment before audit passes。

## 4. 一个月最低成稿标准

一个月内不追求覆盖所有 benchmark。最低可成稿标准是：

1. 一个稳定可复现的 full-data training + closed-loop pipeline；
2. 一个 official full-data / official-init-state evaluation contract；
3. full-data closed-loop 有非零、可重复 signal；
4. 至少一个主任务 / benchmark 上完成 accounting comparison；
5. 至少比较 Full、Random、Uniform、FrameSkip-style；
6. 每个设置同时报告四类成本和 replay multiplier；
7. 至少两套协议：
   - repetition-matched；
   - compute/step-matched；
8. 至少一张 control-sufficiency curve；
9. 明确展示 retained ratio 与 true supervision cost 不一致。

如果 closed-loop 难以稳定，但 accounting 结果非常清晰，则退为 measurement / protocol paper。

## 5. 四周计划

### Week 1：完成 Gate A qualification

目标：证明 LingBot-VA / LIBERO-LONG full-data 链路可用于后续实验，并建立足够强的 full-data local anchor。

已完成：

- 实现 validation loss loop；
- 用历史 `450/50` held-out split 生成 step-50 diagnostic validation loss；
- released checkpoint reference eval；
- local step500 Gate2A official-init closed-loop qualification；
- 启动 official-scale 4k full-data training；
- 4k run 20-step supervision pass。

仍需完成：

- `checkpoint_step_4000` 已保存并完成核验；
- 准备 1k/2k/3k/4k eval models；
- 用同一 official-init contract 跑 task0-2 checkpoint trajectory；
- 确认 full-data 4k trajectory 中哪个 checkpoint 能作为 local anchor；
- 记录失败类型：interface、low-level control、task progress、insufficient training。
- 并行执行 Gate B0：在不占用 GPU 的前提下下载/审计 LIBERO Object/Goal/Spatial。

通过标准：

- train / validation loss 都 finite；
- checkpoint 可保存、加载；
- closed-loop trajectory 至少有非零、可复现 signal 或明确可诊断失败；
- 所有命令和结果可复现。

### Week 2：Multi-suite benchmark coverage + Accounting gate 最小闭环

目标：先补标准 LIBERO suite 覆盖，再把成本审计跑完整。不急着证明 selector 更好。

Benchmark coverage:

- Long：使用当前 4k checkpoint trajectory；
- Object/Goal/Spatial：只有 Gate B0 证明数据格式和 latent contract 可用后，才进入 full-data baseline；
- 若 Object/Goal/Spatial 只有 raw demos 而没有 LingBot-ready latents，则本周只记录 data-present / training-contract-blocked，不强行开训；
- 标准图报告 per-suite success，不只报 macro average。

实验对象：

- Full；
- Random；
- Uniform；
- FrameSkip-style / equivalent；
- diagnostic event-window。

初始预算：

- `25%`
- `50%`
- `100%`

每个设置必须输出：

- `C_unique-raw`
- `C_unique-label`
- `C_seen-label`
- `C_compute`
- replay multiplier
- train loss
- validation loss
- 如果可行，closed-loop success / stage progress

优先协议：

1. compute/step-matched：工程上最容易先跑；
2. repetition-matched：判断是否真实压缩；
3. total-label-exposure-matched：有余力再补。

### Week 3：Control-sufficiency curve 与任务边界

目标：从“审计工具”变成“有科学结论的曲线”。

需要补：

- 在阈值附近增加预算点，如 `10% / 35% / 65% / 75%`；
- 估计 `B90 / B95`；
- 至少加入一个非纯 pick/place 的任务族或低成本任务；
- 分析删除失败集中在哪些时间区域；
- 如果 FrameSkip-style 只在 replay-heavy 协议下有效，要明确写成 replay allocation。

### Week 4：论文打包

目标：形成完整论文版本。

必须完成：

- 主图 1：cost accounting overview；
- 主图 2：control-sufficiency curve；
- 主图 3：三协议或至少两协议对比；
- 主图 4：FrameSkip-style audit；
- 主表：Full / Random / Uniform / FrameSkip-style 的 success/loss + 四成本；
- failure case / timeline；
- related work matrix；
- limitations；
- reproducibility checklist。

## 6. 图片结果设计

优先级从高到低：

1. **Cost accounting overview**  
   raw frames → unique labels → seen labels → compute。

2. **Replay multiplier figure**  
   展示同样 retention ratio 下不同协议的 replay multiplier。

3. **Control-sufficiency curve**  
   x 轴分别用 `C_unique-label` 和 `C_seen-label`，y 轴用 validation loss / closed-loop success。

4. **FrameSkip-style audit**  
   不只报告 retained frames，也报告 seen labels、anchors/warmup、compute。

5. **Task / failure timeline**  
   标出 retained / removed / failure-sensitive windows。

## 7. Historical next step as of 2026-08-19

This section is superseded by `Active Route Override: 2026-08-21`. It is retained only for process lineage.

当前下一步只做三件事：

1. **执行 LIBERO-LONG 4k checkpoint trajectory eval**
   - `full4k_gacc10_20260818T134819Z` 已完成；
   - 已确认 `checkpoint_step_1000`、`checkpoint_step_2000`、`checkpoint_step_3000` 和 `checkpoint_step_4000`；
   - 导出 1k/2k/3k/4k eval models；
   - 跑 official-init task0-2 trajectory eval，先每 task `10` episodes。

2. **Gate B0：LIBERO Object/Goal/Spatial 数据下载与审计**
   - 只做 CPU/I/O；
   - 下载到 `/data/jiaoguanbo/LIBERO`；
   - 写入 logs / manifest / audit report 到 `/data/jiaoguanbo/skeletonmem`；
   - 不启动训练和 eval。
   - Object partial download has been stopped by user request；Goal/Spatial not started。
   - `libero_all.zip` 来源已解压到 `/data/jiaoguanbo/LIBERO/libero_all` 并保留；`/data/zouyude/data/lingbot-va/libero_all` 仅作源文件备查。

3. **整理 multi-suite 训练合同**
   - mixed-suite full-data baseline 脚本已准备：`/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_b_libero40_full_train.sh`；
   - 命名规则：LIBERO-LONG 使用当前 `full4k_gacc10_...` 历史 run；LIBERO40 必须使用 `libero40_full4k_gacc10_...`；
   - historical note: as of 2026-08-21, LIBERO40 full-data 4k training has completed and the mixed-suite quick eval was paused after an all-failure prefix.

## 8. 当前不做

- selector sweep；
- skeleton / memory module；
- RoboTwin-Mem / RMBench 主实验；
- 大规模超参搜索；
- 多 benchmark 同时展开；
- 修改 `/data/jiaoguanbo/skeletonmem` 外的源码。

## 9. 风险与止损

### 主要风险

- full-data closed-loop 不稳定；
- LIBERO-LONG validation/rollout 成本过高；
- FrameSkip-style 复现成本高；
- accounting 结果清晰但 closed-loop 差异不明显；
- `empty_emb.pt` 不是官方 HF 数据的一部分，需要如实记录。

### 止损条件

- full-data checkpoint 无法产生任何可诊断 closed-loop signal；
- validation loss loop 与 training pipeline 无法可靠对齐；
- closed-loop failure 主要来自 interface bug；
- 一个月内无法得到至少一张 sufficiency curve。

### Pivot

如果 closed-loop 失败但 accounting 审计扎实，则转成：

> Temporal Supervision Accounting for Robot Imitation Learning: A Protocol Study

重点讲 retained ratio、seen labels、replay multiplier 和 compute 的错配。

## 10. 决策记录

- 2026-08-14：不再把首个题目限定为 memory。
- 2026-08-17：放弃“更好的 event/frame selector”主故事，转为 temporal supervision accounting。
- 2026-08-17：FrameSkip-style 成为第一强 baseline 和审计对象。
- 2026-08-17：正式采用四类成本与 replay multiplier。
- 2026-08-17：训练协议改为 repetition-matched、total-label-exposure-matched、compute/step-matched。
- 2026-08-17：LIBERO-LONG deterministic held-out split 固定为每 task 最后 5 episodes，后续降级为 historical diagnostic。
- 2026-08-17：A4-medium train50 发现当前设置下 `C_seen-label = 2.0x C_unique-label`。
- 2026-08-17：A4 validation smoke 与 full held-out validation 通过。
- 2026-08-17：根据 HANDOFF 的 NVIDIA EGL ICD 方案，A5 LIBERO env probe 通过。
- 2026-08-17：修复 websocket client host 后，official 1-rollout smoke 完成，结果 `0/1`；Gate A 当前为 pipeline partial pass。
- 2026-08-18：LIBERO 主协议改为 official full-data / official-init-state pool；旧 custom held-out split/accounting 产物删除。
- 2026-08-18：released `lingbot-va-posttrain-libero-long` checkpoint 在同一 A5 contract 下 task0 `1/1` 与 `3/3`，eval contract 可信。
- 2026-08-18：local step500 Gate2A 得到 task0/task1/task2 `7/10`, `7/10`, `4/10`；step500 定位为 engineering anchor。
- 2026-08-18：`full4k_20260818T133130Z` 因 `gradient_accumulation_steps=1` 与 original-scale protocol 不匹配而中止，训练产物与日志删除。
- 2026-08-18：启动 `full4k_gacc10_20260818T134819Z`，`4000` steps、save interval `1000`、`gradient_accumulation_steps=10`；20-step supervision pass。
- 2026-08-19：`full4k_gacc10_20260818T134819Z` 已保存 `checkpoint_step_1000` 与 `checkpoint_step_2000`，训练继续。
- 2026-08-19：建立 LIBERO Object/Goal/Spatial Gate B0；允许 CPU/I/O-only 下载和审计，但不占用 GPU、不启动训练、不改变当前 4k run。
- 2026-08-19：识别候选源 `lerobot/libero_object_image`, `lerobot/libero_goal_image`, `lerobot/libero_spatial_image`；CPU-only 下载脚本 dry-run 通过；实际下载因 shell 网络代理 403 暂未开始。
- 2026-08-19：用户恢复代理后，`lerobot/libero_object_image` no-Xet single-worker 下载开始；明确该源是 HuggingFace LeRobot image/parquet 候选，不是已确认的 LingBot-ready Wan latent official training data。
- 2026-08-19：按用户要求停止 Object 下载；曾保留约 `793M` partial data，后因 `/data` 满盘删除，Goal/Spatial 未启动。
- 2026-08-19：审计 `/data/zouyude/data/lingbot-va/libero_all`；四套 suite 的 meta/latents 完整，但缺 `data/` parquet，原始 source_root 不再用于训练来源。
- 2026-08-19：解压并审计 `/data/zouyude/data/lingbot-va/libero_all.zip`；确认四套 LIBERO LingBot-VA 数据包含 parquet/videos/latents/meta，总计 `1712` episodes，并已固定保留在 `/data/jiaoguanbo/LIBERO/libero_all`。
- 2026-08-19：为 zip 数据 richer parquet schema 添加 dynamic schema dataloader patch；zip multi-suite 和现有 Long dataloader smoke 均通过。
- 2026-08-19：HF Object partial 和历史 smoke checkpoint dirs 已清理；保留 `libero_all` staging copy、报告、patch、源 zip 和 active 4k checkpoints。
- 2026-08-19：`full4k_gacc10_20260818T134819Z` 已保存 `checkpoint_step_3000`；`/data` 后续恢复到约 `19T` 可用，但 `checkpoint_step_4000` 仍需完成后核验。
- 2026-08-19：准备 Gate B LIBERO40 full-data 训练脚本；强制 `libero40_` run-name 前缀，输出隔离到 `results/stage2/gate_b/libero40_full_train/`；完成 syntax、guardrail 和 dry-run 检查，未启动训练。
- 2026-08-19：`full4k_gacc10_20260818T134819Z` 完成 `4000/4000` steps；`checkpoint_step_4000` 保存成功，大小约 `9.5G`；最终显示 total loss `0.1255`；双 A100 已空闲。
- 2026-08-19：Long 4k two-round eval 完成；Round 1 step1000/2000/3000/4000 在 task0-2 上分别为 `22/30`, `26/30`, `26/30`, `27/30`；Round 2 step4000 task0-9 为 `91/100`。
- 2026-08-20：LIBERO40 / `libero_all` full-data 4k training 完成；run `libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z`，final loss latent `0.0880`, action `0.0757`, total `0.1636`。
- 2026-08-21：LIBERO40 mixed-data ledger 复算完成：`C_unique-raw=277713`, `C_unique-label=1961820`, `C_seen-label=91676228`, `R_replay=46.73`, compute about `30.28 GPU-hours`。
- 2026-08-21：启动后暂停 LIBERO40 step4000 quick eval：原计划 `40 tasks x 5 rollouts/task`，双卡分片；暂停时 observed prefix `0/32`。当前判定是 LIBERO40 4k mixed training 不能作为 control anchor，需先诊断训练配置或训练步数是否不足。
- 2026-08-21：论文路线收敛为 Temporal Supervision Accounting for Closed-Loop Robot Imitation；下一科学实验优先级为 Long full reference solidification、automated temporal ledger、Random-50 protocol intervention。
