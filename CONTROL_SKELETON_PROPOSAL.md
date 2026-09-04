# Temporal Supervision Accounting for Closed-Loop Robot Imitation

> Supersession note, 2026-09-03: this proposal is historical background.
> The active paper contract is now
> [TEMPORAL_CURATION_RESOURCE_AUDIT.md](./TEMPORAL_CURATION_RESOURCE_AUDIT.md).
> The strict positioning is a resource-equivalent counterfactual audit of
> temporal curation, not a memory paper, selector paper, retained-ratio
> leaderboard, or generic compression paper. When this file conflicts with the
> active contract, the active contract wins.

> 版本：v4.0  
> 日期：2026-08-21  
> 当前阶段：LIBERO-LONG full-data 4k local anchor 已完成并完成两轮 closed-loop eval；LIBERO40 / `libero_all` data contract 已通过，LIBERO40 full-data 4k training 已完成；LIBERO40 `5 rollouts/task` 双卡 quick eval 已因 observed prefix `0/32` 暂停，定位为 failed training-sufficiency / compatibility probe。  
> 硬约束：所有代码、脚本、日志、patch、manifest、报告只写入 `/data/jiaoguanbo/skeletonmem`；benchmark 数据允许下载到 `/data/jiaoguanbo/LIBERO`；Long full reference distribution 与 automated temporal ledger 固化前，不启动 selector / skeleton / memory baseline；LIBERO40 quick eval 不得被写成统计 benchmark。

## 0. Active Thesis: 2026-08-21

本文不主张“我们提出了一个更好的 frame/event selector”，也不主张“我们解决了机器人 memory”。当前最稳的 ICRA 定调是：

> Temporal data efficiency in robot imitation learning is not identifiable from retained-data ratio alone: overlapping windows, action chunks, and optimizer replay decouple retained experience from the supervision and compute actually consumed by a policy. We introduce a temporal accounting protocol and evaluate control sufficiency under matched resource definitions.

中文：

> 机器人模仿学习中的时间数据效率，不能由“保留了多少数据”判断。由于窗口重叠、action chunk 和优化过程中的重复 replay，保留比例与策略实际消耗的独立经验、动作监督和训练计算并不等价。我们提出时间监督审计协议，并在严格资源匹配下测量闭环控制充分性。

核心问题：

```text
When researchers report “using 50% robot demonstration data,” 50% of what?
```

必须区分四类资源：

```text
C_raw       = C_unique-raw
C_label     = C_unique-label
C_exposure  = C_seen-label
C_compute
```

并报告平均 label replay multiplier：

```text
R_replay = C_exposure / C_label
```

可选诊断是 replay concentration / entropy：

```text
p_i = n_i / sum_j n_j
H_replay = - sum_i p_i log p_i
```

其中 `n_i` 是第 `i` 个 unique action label 被训练看到的次数。`R_replay` 是平均重复倍率，不描述重复是否集中在少量 label 上，因此 entropy/concentration 对解释 replay allocation 很重要。

## 0.1 Current Empirical Anchors

LIBERO-LONG local full-data anchor:

- run: `full4k_gacc10_20260818T134819Z`;
- train: `4000` optimizer steps, `gradient_accumulation_steps=10`, save every `1000`;
- two-round eval tag: `full4k_eval_then_libero40_20260819T122727Z`;
- Round 1 task0-2, 10 episodes/task:
  - step1000: `22/30`;
  - step2000: `26/30`;
  - step3000: `26/30`;
  - step4000: `27/30`;
- Round 2 step4000 task0-9, 10 episodes/task: `91/100`.

Interpretation:

- `91/100` is a qualified local full-data control anchor.
- It proves the local LingBot-VA data, latent, training, checkpoint, server/client, and evaluation loop are connected.
- It is not the final paper-level full reference distribution because it is one training seed, 10 rollouts/task, local setup, and current checkpoint rule.

LIBERO40 mixed-suite anchor:

- data path: `/data/jiaoguanbo/LIBERO/libero_all`;
- data contract: `1712` episodes/parquet, `3424` videos, `3424` latents;
- train run: `libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z`;
- train complete: `2026-08-20T16:12:09Z`;
- final train loss: latent `0.0880`, action `0.0757`, total `0.1636`;
- step4000 sha256: `df1df614a50daf953c3d81b3555c6bfd00219f92abc43701bd11db497a419221`;
- ledger:
  - `C_unique-raw = 277713`;
  - `C_unique-label = 1961820`;
  - `C_seen-label = 91676228`;
  - `R_replay = 46.73`;
  - `C_compute = 4000 steps, gacc=10, world_size=2, about 30.28 GPU-hours`.

Paused LIBERO40 quick eval:

- run: `libero40_full4k_step4000_all_suites_5eps_2gpu_20260821T065000Z`;
- planned protocol: `40` tasks x `5` rollouts/task = `200` rollouts;
- GPU0: Object + Goal;
- GPU1: Spatial + Long/libero_10;
- paused observed prefix: `0/32`;
- completed/partial tasks at pause: Object task0-2 `0/5`, Object task3 `0/1`, Spatial task0-2 `0/5`, Spatial task3 `0/1`;
- interpretation: because train/test suite are intended to be consistent for LIBERO40, this early all-failure pattern suggests a training configuration, training-step budget, or mixed-suite training sufficiency problem;
- purpose after pause: failed compatibility/training-sufficiency evidence, not a statistical benchmark.

## 0.2 Expected Contributions

1. **Canonical raw-time lineage.**  
   Define supervision on the original episode timeline, then project it to each policy format:
   `episode -> raw timestep -> window/segment -> action target -> batch exposure`.

2. **Temporal supervision ledger.**  
   For every training run, automatically report independent raw time, independent action labels, total seen labels, replay multiplier, replay concentration, optimizer steps, GPU-hours, wall-clock, and peak memory.

3. **Matched-resource protocols.**  
   Separate common retained-ratio reporting from three interpretable interventions:
   fixed-step retained-ratio, label-repetition-matched, total-exposure-matched, and compute-matched.

4. **Closed-loop control-sufficiency curves.**  
   Plot closed-loop performance against `C_label`, `C_exposure`, and `C_compute`, and report confidence intervals for `B90/B95` rather than over-precise single points.

## 0.3 Current Checklist

- [x] Pause the LIBERO40 5-rollout eval after observed prefix `0/32`; use it only as failed compatibility/training-sufficiency evidence.
- [ ] Freeze Long full reference distribution: second seed, extra rollouts for weak tasks, confidence intervals, failure taxonomy.
- [ ] Make checkpoint selection and final test evaluation separable and written into a manifest.
- [ ] Implement the automated ledger before any compressed-policy training.
- [ ] Recompute ledger for completed Long and LIBERO40 full-data runs.
- [ ] Run the first accounting counterfactual on Long with Random-50:
  - Full reference;
  - Random-50 fixed-step;
  - Random-50 label-repetition-matched;
  - Random-50 total-exposure-matched;
  - Random-50 compute-matched.
- [ ] Add Uniform/FrameSkip/Motion/Event-stage only after Random-50 shows whether protocol definitions change the interpretation.

Forbidden until the above is satisfied:

- selector sweep;
- skeleton or memory baseline;
- Event-stage as the main method;
- claims based only on retained segment ratio;
- claims that `91/100` is the final full-data oracle;
- statistical LIBERO40 claims from `5 rollouts/task`.

## 1. 一句话

在闭环机器人模仿学习中，retained-data ratio 不能唯一确定策略实际消耗的独立时间经验、动作监督、重复 exposure 或训练计算；本文建立 unit-consistent temporal supervision accounting，并用 matched-resource interventions 测量达到 full-data 控制能力所需的真实 temporal supervision。

题目候选：

- **Temporal Supervision Accounting for Closed-Loop Robot Imitation**
- **Beyond Retained Data Ratio: Measuring Temporal Supervision Efficiency in Robot Imitation**
- **What Does “Using Less Robot Data” Mean? Temporal Supervision Accounting for Imitation Learning**
- **Measuring Control-Sufficient Temporal Supervision for Long-Horizon Robot Imitation**

## 2. 问题定义

当前机器人模仿学习通常把 expert episode 展开为密集 observation-action 序列，并默认 recording density 可以直接转成 supervision density。但固定频率记录是传感器和控制系统的工程选择，不代表每个时间步都提供等量新监督。长时程操作中同时存在新决策、接触、动作效果、阶段转换、稳定运输、保持动作、action chunk 重叠和完成尾帧。

给定专家轨迹：

$$
\tau=\{(o_t,a_t,s_t)\}_{t=1}^{T},
$$

一个数据压缩或采样协议会选择或重映射一部分原始时间监督：

$$
\mathcal I\subseteq\{1,\ldots,T\}.
$$

将它投影到某个 policy backbone 的训练格式：

$$
D_{\mathcal I}^{(b)}=P_b(\tau,\mathcal I),
\qquad
\pi_{\mathcal I}^{(b)}=\operatorname{Train}_b(D_{\mathcal I}^{(b)}).
$$

如果：

$$
J(\pi_{\mathcal I}^{(b)})
\geq
J(\pi_{\mathrm{full}}^{(b)})-\epsilon,
$$

则称 $\mathcal I$ 是相对于该任务、backbone 和训练协议的经验控制充分 temporal support。

三个限定必须写清楚：

- 这是经验充分，不是理论充分；
- 这是相对于给定 policy 和任务分布，不是对所有未来任务充分；
- “最小”来自 control-sufficiency curve，不是信息论全局最小。

主指标：

$$
B_p=\min\{B:S(B)\ge pS_{\mathrm{full}}\},\quad p\in\{0.90,0.95\}.
$$

## 3. 这不是 memory，也不是新的 FrameSkip

### 与 memory 的边界

memory 研究执行时该保留哪些历史信息；本阶段研究训练时完整专家轨迹中的监督如何被计入优化。只有当 temporal supervision accounting 做清楚后，才值得继续讲 selective memory、event memory 或 minimum sufficient representation。

### 与 demonstration curation 的边界

CUPID、DataMIL、ATHENA、DemInf、Re-Mix 等主要从多个 demonstrations、datasets 或 domains 中选择更有价值的经验来源。这里固定任务、episode 集合和策略接口，问题改为：

> 所谓“使用 20%/50% 数据”到底减少了多少 unique raw time、多少 unique action label、多少 seen label 和多少 compute？

因此重点不是提出另一个 heuristic selector，而是建立可复现的 temporal supervision accounting。

### 与 FrameSkip / SIEVE / SCIZOR / TGM-VLA 的关系

FrameSkip 已经覆盖“用 action variation、visual-action coherence、task progress 和 gripper transition 选择更 informative frames 训练 VLA”的方法型故事。SIEVE 已经占据 primitive / transition interface / structure-aware demonstration selection。SCIZOR 已经做 state-action pair 级 curation。TGM-VLA 也讨论 RLBench keyframe redundancy 和 temporal distribution imbalance。

CSS 的安全定位是：把 FrameSkip 作为第一强基线和审计对象，复现或等价实现其选择/重映射规则，并用四类成本和三套协议重新测量其真实压缩效果。主问题不是“FrameSkip 能不能提高 success”，而是“FrameSkip 式 20% retained frames 实际对应多少 unique labels、seen labels、replay multiplier 和 compute”。

## 4. 核心主张

第一篇只保留三条主张。

**Claim 1：retention ratio 不是 supervision cost。**  
保留 20% raw frames 不等于使用 20% action labels，更不等于 20% seen supervision 或 20% compute。action chunk、history window、index remapping、warmup 和 anchor batches 都会改变真实监督成本。

**Claim 2：控制充分密度是任务和策略相关的曲线，而不是单个比例。**  
主结果应是：

$$
S(B,R,C;b,\mathcal T),
$$

其中 $B$ 是 unique temporal budget，$R$ 是 replay/exposure，$C$ 是 compute，$b$ 是 backbone，$\mathcal T$ 是任务族。报告 $B_{90}$、$B_{95}$ 及其置信区间。

**Claim 3：不同任务/控制形式的 sufficiency density 应可解释。**  
pick/place 可能有较低 $B_{95}$；精细对齐可能需要事件附近局部密集窗口；连续接触可能没有稀疏 skeleton；recovery 任务可能需要保留看似“不推进任务”的修正片段。

必须报告四类成本：

$$
C_{\mathrm{unique\text{-}raw}},\quad
C_{\mathrm{unique\text{-}label}},\quad
C_{\mathrm{seen\text{-}label}},\quad
C_{\mathrm{compute}}.
$$

其中：

- $C_{\mathrm{unique\text{-}raw}}$：原始时间轴上真正保留或监督的 unique timesteps；
- $C_{\mathrm{unique\text{-}label}}$：投影到 policy 训练格式后的 unique conditioning-time / target-time labels；
- $C_{\mathrm{seen\text{-}label}}$：训练过程中 loss mask 实际覆盖的有效 action targets 次数；
- $C_{\mathrm{compute}}$：wall-clock、GPU-hours、显存和训练 step。

重复倍率：

$$
R_{\mathrm{replay}}=
\frac{C_{\mathrm{seen\text{-}label}}}{C_{\mathrm{unique\text{-}label}}}.
$$

## 5. 选择规则信息权限

后续实验可以包含 selector，但它们只是诊断规则或基线，不是主贡献。必须区分信息权限，避免把 oracle 当作可部署方法。

| Selector 类型 | 可用信息 | 角色 |
| --- | --- | --- |
| Oracle success-tail / phase | 完整 episode、成功标签或 simulator phase | 诊断上界 |
| FrameSkip-style | action variation、visual-action coherence、task progress、gripper transition | 第一强基线 |
| Offline RGB + action | 完整离线 expert episode | 诊断候选 |
| Sensor/action-only | proprio、gripper、action delta | 工程可部署候选 |
| Prefix-causal | 只看当前及过去 | 后续 memory / 在线扩展 |

Oracle 可以做，但主图中必须用上界标注，不能和部署型 selector 混在一起。

## 6. Baseline 与预算协议

### 必须审计的基线/规则

- Full-data；
- Random；
- Uniform；
- FrameSkip replication / equivalent；
- Motion / action-delta；
- Gripper event；
- Event-window / stage-window；
- Oracle phase/event upper bound。

### 四套训练/评估协议

**Fixed-step retained-ratio：常见报告协议 / 混淆基线。**  
固定 optimizer steps，并报告“保留了多少数据”。它复现社区常见写法，但必须明确这是 potentially confounded baseline，因为小数据会提高每个 label 的 replay multiplier。

**Label-repetition-matched：独立经验压缩协议。**  
固定每个 retained unique action label 的平均重复次数，最好同时检查 replay 分布。数据越少，总 seen labels 通常也越少。它回答：真正减少 independent supervision 后，闭环控制能否保持。不要用 epoch 数近似，因为 segment epoch 不等于 unique-label repetition。

**Total-label-exposure-matched：监督重分配协议。**  
固定总 $C_{\mathrm{seen\text{-}label}}$。小数据通常被重复更多次。它回答：在相同监督曝光下，如何分配 temporal supervision 更有效。

**Compute-matched：工程资源协议。**  
固定 GPU-hours、FLOPs 或 wall-clock。optimizer steps 可以作为辅助指标，但不能单独等价于 compute，因为不同输入长度、padding、模型路径和 batch 有效 token 数会改变实际计算量。

### 成功指标

必须区分：

- open-loop action / validation loss；
- decision success；
- stage progress；
- final closed-loop success；
- failure taxonomy。

Loss 只能做快速筛选，不能替代 closed-loop success。

## 7. 核心图片结果设计

第一篇的主图不应是普通 success-rate bar chart，而应是 accounting + sufficiency curves。

1. **Cost accounting Sankey / stacked bar**  
   展示从 raw frames 到 unique labels、seen labels、compute 的转换。重点暴露“20% retained frames”是否变成了更高 replay multiplier。

2. **Control-sufficiency curves**  
   横轴分别用 $C_{\mathrm{unique\text{-}raw}}$、$C_{\mathrm{unique\text{-}label}}$、$C_{\mathrm{seen\text{-}label}}$、$C_{\mathrm{compute}}$，纵轴用 validation loss / stage progress / closed-loop success。报告 $B_{90}$、$B_{95}$。

3. **Protocol comparison panel**  
   同一个 selector 在 repetition-matched、total-label-exposure-matched、compute-matched 下的曲线对比，用来区分真实压缩和 replay allocation。

4. **FrameSkip audit figure**  
   复现或等价实现 FrameSkip 后，显示其 retention ratio、full warmup / anchor exposure、seen labels、replay multiplier 和 compute。FrameSkip 是强基线，也是被审计对象。

5. **Task × policy sufficiency heatmap**  
   行是任务族，列是 backbone / cost definition，格子里是 $B_{90}$ 或 $B_{95}$。目标是证明控制充分密度具有任务和策略依赖性。

6. **Failure localization timeline**  
   在 episode 时间轴上标注 retained / removed / failure-sensitive windows，分析失败是否集中在接触、阶段切换、尾帧确认、recovery 或连续接触区间。

## 8. 当前工程路线

### Phase 0：科学合同

冻结任务、backbone、数据版本、raw-time 定义、budget 定义、checkpoint 选择、train/eval seed、success 统计、full baseline 参考方式和 selector 信息权限。

输出：

- `experiment_contract.yaml`
- `protocol.md`

### Phase 1：Gate A，full-data qualification

没有稳定 full-data baseline，不做 accounting comparison，也不做 selector / skeleton 对比。

Gate A 顺序：

1. A1 数据与 latent provenance 审计；
2. A2 单 episode lineage：raw timestep → parquet action → latent lookup → dataloader batch → model input；
3. A3 sanity：single batch、forward/backward、optimizer step、save/load、tiny overfit；
4. A4 full-data training qualification：short smoke → train50 → validation loss；
5. A5 closed-loop qualification；
6. 只有 A5 通过后，进入 accounting comparison / selector baseline。

Gate A 结果只能是：`PASS`、`PARTIAL_PASS`、`BLOCKED`。

### Phase 2：Accounting gate

先在低成本、稳定 backbone 上建立 accounting pipeline。FrameSkip-style / random / uniform / diagnostic event-window 都是被审计对象，不是主贡献。ACT 可作为 discovery backbone，LingBot-VA 作为 confirmation backbone。预算先只做：

$$
\rho\in\{0.25,0.50,1.00\}.
$$

所有规则必须先在 raw timeline 上产生 mask 或 remapping，再投影到具体 backbone 的训练格式。必须同时记录 supervised frames、context-only frames、unique labels、seen labels 和 compute，不能直接在 segment 空间偷换成本。

### Phase 3：任务族边界

至少增加一个不同任务族：连续接触、精细对齐、失败恢复或长时序顺序/计数。目标不是证明所有任务都可压缩，而是测量 temporal supervision compression 的适用边界。

### Phase 4：VLA confirmation

只迁移 Phase 2 中最有信息量的设置，例如 Full、Uniform-50、Random-50、FrameSkip-50、Best-Window-50；不要在 LingBot 阶段重新搜索 selector 权重。

### Phase 5：方法升级

只有现象成立后，再考虑 learned selector、future-control value、action-effect prediction、budget allocator 或 causal memory extension。

## 9. 当前 Gate A 状态

代码隔离 clone：

`/data/jiaoguanbo/skeletonmem/repos/lingbot-va-gatea-official`

官方 LIBERO-LONG 数据：

`/data/jiaoguanbo/LIBERO/libero-long-lerobot`

审计结果：

- official download complete；
- episodes: 500；
- frames: 138090；
- parquet files: 500；
- `action_config` segments: 500；
- agentview latent: 500 / 500；
- eye-in-hand latent: 500 / 500；
- missing latent: 0；
- videos 两路相机完整；
- action shape: 7；
- LingBot-VA `libero_train` 使用 action channels `[0..6]`；
- HF 官方文件列表和本地完整下载均确认没有 `empty_emb.pt`；
- 已按用户授权从 `/data/zouyude/data/lingbot-va/libero_all/empty_emb.pt` 复制到数据根目录；
- copied `empty_emb.pt` sha256: `99e5568f3f700b675315dd9999a392a1eee8428d7ab8f3cefd945b093ef6a9c7`；
- copied `empty_emb.pt` shape/dtype: `[512,4096]`, `torch.bfloat16`。

Gate A1 判定：

`PASS_CANDIDATE_WITH_LOCAL_EMPTY_EMB`

Gate A2 已完成：

- single-episode lineage：episode 0 通过；
- LingBot-VA dataloader：runtime patch 后 official direct path 通过；
- sample/batch tensor finite check：通过；
- 模型 import/config/input-contract sanity：通过；
- full transformer weight-load / forward：未执行，归入下一阶段 tiny sanity，不能视为 full training。

Gate A2 runtime patch：

- `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a2_runtime_compat.patch`

Gate A3 已完成：

- 两卡 FSDP 从 `/data/jiaoguanbo/models/lingbot-va-base` 加载；
- 使用 LIBERO-LONG latent dataloader；
- 完成 1 个 optimizer step；
- finite losses：latent `0.1609`，action `0.4017`；
- 未保存大 checkpoint。

Gate A3 报告：

- `/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a3_tiny_training_sanity.md`

当前判定：

`LONG_FULLDATA_4K_TWO_ROUND_EVAL_COMPLETE_LOCAL_ANCHOR_LIBERO40_4K_TRAIN_COMPLETE_QUICK_EVAL_RUNNING`

下一步进入：

`LONG_FULL_REFERENCE_SOLIDIFICATION_AND_AUTOMATED_TEMPORAL_LEDGER`

A4-short 已验证 10-step full-data 训练链路、日志、显存、checkpoint save 和 reload smoke。A4-medium-train50 已验证 50-step full-data train curve 和 checkpoint save。早期 held-out validation 已确认 step-50 checkpoint 能完成两卡 FSDP 加载和前向 loss 评估，但该 split 已不再作为活跃 LIBERO 协议。根据 `/data/jiaoguanbo/HANDOFF_RMBench.md` 的 NVIDIA EGL ICD 提示，A5 的 LIBERO env construct/reset/one-step probe 已通过。修复 websocket client 默认 host 后，released `lingbot-va-posttrain-libero-long` checkpoint 在同一 A5 contract 下跑出 task0 `1/1` 和 `3/3`，证明 eval contract 可信。local step500 在 Gate2A task0-2 上得到 `7/10`, `7/10`, `4/10`，说明本地 full-data 训练能产生非零 closed-loop control，但 500 steps 对 LIBERO-LONG 混训过短，只能作为 engineering anchor。

当前活跃 full-data baseline：

- run id：`full4k_gacc10_20260818T134819Z`；
- 初始 checkpoint：`/data/jiaoguanbo/models/lingbot-va-base`；
- dataset：`/data/jiaoguanbo/LIBERO/libero-long-lerobot`；
- source：`/data/jiaoguanbo/skeletonmem/repos/lingbot-va-gatea-official`；
- training steps：`4000`；
- save interval：`1000`；
- gradient accumulation：`10`；
- data protocol：`LIBERO_OFFICIAL_FULLDATA_NO_EPISODE_FILTER`；
- 20-step supervision：通过；step 20/21 已出现，loss 有效，GPU util 约 `99-100%`，显存约 `72GB/GPU`；
- saved checkpoints observed：`checkpoint_step_1000`, `checkpoint_step_2000`, `checkpoint_step_3000`, `checkpoint_step_4000`；
- final checkpoint：`checkpoint_step_4000` 保存成功，目录约 `9.5G`；
- final displayed loss：latent `0.0977`，action `0.0278`，total `0.1255`；
- post-run state：训练进程已退出，两张 A100 空闲，`/data` 约 `18T` 可用；
- run record：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_gacc10_20260818T134819Z_run_record.md`；
- train log：`/data/jiaoguanbo/skeletonmem/logs/stage2/gate_a/full4k_gacc10_20260818T134819Z_train.log`。

### Gate B0：LIBERO multi-suite data acquisition / audit

用户建议下一步补齐 LIBERO Object、Goal、Spatial、Long 的常见 benchmark 结果图。当前判断是 `CONDITIONAL GO`：

- 该图对论文可信度必要，但它是 benchmark coverage / reproducibility figure，不是核心 novelty；
- 核心 novelty 仍然是 temporal supervision ledger、matched accounting protocols 和 control-sufficiency curves；
- 当前本地只确认存在 `/data/jiaoguanbo/LIBERO/libero-long-lerobot`；
- Object/Goal/Spatial 数据需要先下载到 `/data/jiaoguanbo/LIBERO` 并完成数据/latent/empty embedding/task mapping 审计；
- 下载和文件校验必须不占用 GPU，不启动训练，不影响当前 `full4k_gacc10_20260818T134819Z`；
- 若下载到的是 raw LIBERO demos 而不是 LingBot-ready LeRobot+Wan latent 格式，则不得直接进入训练，必须先记录为 `DATA_PRESENT_BUT_TRAINING_CONTRACT_NOT_READY`。

当前 B0 状态：

- candidate repos：`lerobot/libero_object_image`, `lerobot/libero_goal_image`, `lerobot/libero_spatial_image`；
- local target dirs：`/data/jiaoguanbo/LIBERO/libero-object-image-lerobot`, `/data/jiaoguanbo/LIBERO/libero-goal-image-lerobot`, `/data/jiaoguanbo/LIBERO/libero-spatial-image-lerobot`；
- prepared script：`/data/jiaoguanbo/skeletonmem/scripts/stage2/download_libero_multisuite_cpu_only.py`；
- status report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero_multisuite_data_gate_b0_status_20260819.md`；
- dry-run summary：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero_multisuite_download_summary.json`；
- Object partial target：`/data/jiaoguanbo/LIBERO/libero-object-image-lerobot`，曾达到约 `793M`、`23` files、`8` parquet data files、`1` incomplete file，后因 `/data` 满盘已删除；
- Object download log：`/data/jiaoguanbo/skeletonmem/logs/stage2/gate_b0/libero_object_download_retry_no_xet_20260819.log`；
- Object download status：stopped by user request; no residual download process; partial local copy removed during disk-pressure cleanup；
- Goal/Spatial download status：not started。
- Zouyude source audit：`/data/zouyude/data/lingbot-va/libero_all` contains four suite-level LingBot latent/meta directories, but no `data/` parquet directories；
- Zouyude copy decision：`DATA_CONTRACT_PASS_WITH_EXTRACTED_COPY_KEPT`；
- Zouyude audit report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_source_audit.md`；
- Zouyude decision report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_copy_decision_20260819.md`。
- Zouyude zip source：`/data/zouyude/data/lingbot-va/libero_all.zip`；
- zip extraction target used for audit：`/data/jiaoguanbo/LIBERO/libero_all`，审计后保留为 B0 固定路径；
- symlink relink：`data`/`videos` 已从不可达的 `/data/shared/yaoyifei/...` 改到 `/data/yaoyifei/dataset/fastwam/libero_mujoco3.3.2/...`；
- zip extraction audit result：四套 suite 均包含 `data/chunk-000/episode_*.parquet`、videos、two-camera latents、meta、`empty_emb.pt`、`text_embeddings.pt`；
- zip totals：`1712` episodes、`1712` parquet、`3424` videos、`3424` latent files、`0` incomplete files；
- suite breakdown：Object `457` episodes，Long/libero_10 `388` episodes，Goal `433` episodes，Spatial `434` episodes；
- schema finding：zip parquet 包含额外 `observation.states.ee_state`、`joint_state`、`gripper_state` 列；这是旧 dataloader 的 explicit schema compatibility 问题，不是数据损坏；
- runtime fix：`LatentLeRobotDataset.load_hf_dataset()` 已改为从 `meta/info.json` 动态构建 non-video features；
- smoke result：zip multi-suite dataloader `len=1712` 通过；existing Long dataloader regression `len=500` 通过；
- relink report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_symlink_relink_20260819.md`；
- relink script：`/data/jiaoguanbo/skeletonmem/scripts/stage2/relink_lingbot_libero_all_symlinks.sh`；
- LIBERO40 train script：`/data/jiaoguanbo/skeletonmem/scripts/stage2/run_gate_b_libero40_full_train.sh`；
- LIBERO40 train protocol：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/libero40_full_data_training_protocol_20260819.md`；
- LIBERO40 naming rule：run names must start with `libero40_`; outputs go under `/data/jiaoguanbo/skeletonmem/results/stage2/gate_b/libero40_full_train/`；
- contract report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_zip_extracted_contract_report_20260819.md`；
- dataloader patch：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/lingbot_libero_all_dynamic_schema_dataloader.patch`；
- disk cleanup report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/disk_pressure_cleanup_20260819.md`。

Gate B0 通过条件：

- 每个 suite 的 source repo、local path、file count、episodes、parquet/raw demo format、latents availability、videos availability、task mapping、`empty_emb.pt` provenance 全部记录；
- 明确是否可直接被当前 LingBot `libero_train` dataloader 使用；
- 不产生 GPU 进程，不修改当前 full-data 4k training。

当前 accounting prototype 结果：

- historical held-out diagnostic split：曾使用每个 task `45 train / 5 val`，共 `450 / 50` episodes；相关 split/accounting 产物已删除，后续 LIBERO closed-loop 主协议不使用该 split；
- full raw frames：`138090`；
- nominal LingBot segments：`500`；
- full `C_unique-label` official scalar labels：`971656`；
- A4-medium train50 observed `C_seen-label`：`1943312`；
- segment replay multiplier：`2.0x`；
- scalar-label replay multiplier vs full unique labels：`2.0x`。

这说明当前 `50 optimizer steps` 在官方 dataloader/sampler/grad-accum 设置下已经约等于 full dataset 的两遍 label exposure。后续所有 “少用数据” 结果必须同时报告 unique labels、seen labels、replay multiplier 和 compute。

当前 validation smoke 结果：

- checkpoint：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/a4_medium_train_a4medium50_20260817T083843Z/checkpoints/checkpoint_step_50`；
- held-out validation segments：`50`；
- smoke 设置：每卡 `5` batches，world size `2`；
- global seen labels：`19908`；
- avg latent loss：`0.160899`；
- avg action loss：`0.148452`；
- avg total loss：`0.309351`；
- 注意：这是 smoke validation，不是最终 validation curve；diffusion loss 需要固定 seed 后再做全 held-out 评估。

当前 full held-out validation 结果：

- checkpoint：同 step-50 checkpoint；
- held-out validation segments：`50`；
- evaluated batches per rank：`25`，world size `2`；
- global seen labels：`96012`；
- seed：`0`；
- avg latent loss：`0.149725`；
- avg action loss：`0.140073`；
- avg total loss：`0.289798`；
- 注意：这是 open-loop loss qualification，不是 closed-loop success。

当前关键产物：

- held-out split artifact：已删除，不再作为活跃协议输入。
- accounting audit / overview 初版产物：已删除，不再作为活跃协议输入；相关数值只保留为历史记录。
- A4-medium train curve：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a4_medium_train_loss_curve.png`
- A4 validation smoke report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_validation_smoke_report.md`
- A4 full validation report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_validation_full_report.md`
- A4 train/validation overview：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/validation/gate_a4_train_val_loss_overview.png`
- A5 dependency probe：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_closed_loop_dependency_probe.md`
- A5 closed-loop smoke report：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_closed_loop_smoke_report.md`
- released checkpoint reference eval：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a5_released_checkpoint_reference_eval_report.md`
- Gate2A step500 official-init result：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/gate_a_step500_gate2a_result.md`
- 4k grad-accum-10 run record：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_gacc10_20260818T134819Z_run_record.md`
- aborted grad-accum-1 run record：`/data/jiaoguanbo/skeletonmem/results/stage2/gate_a/full4k_20260818T133130Z_run_record.md`

## 10. 主要不确定性

1. **4k full-data baseline 稳定性。** 如果 4k checkpoint trajectory 的 closed-loop success seed 方差大，$B_{90}$ / $B_{95}$ 没有可信解释。  
2. **`empty_emb.pt` provenance。** 官方 LIBERO-LONG 数据不包含该文件；当前使用的是用户授权的本地 copy，需要在论文/实验合同中如实记录。  
3. **LIBERO-LONG 是否适合两周主实验。** 它可能太难、closed-loop 方差大、stage progress 难拆。  
4. **LIBERO Object/Goal/Spatial 数据格式。** 当前 `libero_all` 已确认有 LingBot-ready parquet/videos/latents/meta，并通过 dataloader smoke；但正式训练仍需等待 Long 4k baseline eval 完成，并为 mixed-suite 重新计算 exposure/accounting。  
5. **FrameSkip novelty 压力。** 原始 selector-method 故事已被高度覆盖；必须坚持 accounting / measurement 定位。  
6. **真实压缩与 replay allocation 混淆。** 必须用 repetition-matched、total-label-exposure-matched 和 compute-matched 三套协议区分。  
7. **action chunk 隐性重复。** 删除 observation frame 不等于删除 action label；必须审计 unique labels 和 seen labels。  
8. **selector 信息泄漏。** Oracle phase、成功轨迹尾部、完整 episode 统计必须单列为诊断上界。  
9. **任务类型边界。** 离散 pick/place 可能适合稀疏 temporal support；连续接触和精细插入可能需要 dense supervision。  
10. **ACT 与 LingBot-VA 可能结论不同。** ACT 用于发现现象，LingBot 用于确认，不应简单平均。  
11. **低层控制噪声可能淹没 accounting 信号。** 必须分开 decision、stage 和 final success。

## 11. Go / No-go 标准

### Go：方法论文方向

- full-data baseline 稳定；
- accounting pipeline 能稳定给出四类成本和 replay multiplier；
- 至少一个任务上 control-sufficiency curve 有可解释的任务结构差异；
- 增益出现在 closed-loop success 或 stage progress；
- 结果在多个 seed 下方向一致；
- 至少一个额外任务族不完全失效；
- LingBot-VA confirmation 不完全反转。

### Conditional go：measurement / protocol 论文

- selector 差异不大，但不同任务的 control-sufficiency curves 明显不同；
- 真实成本审计揭示 retained segments、unique supervision 和 seen exposure 严重不一致；
- 能总结什么时候压缩有效、什么时候失败。

### No-go：停止 CSS 主线

- full-data baseline 不稳定；
- 数据 / latent / `empty_emb.pt` provenance 无法修复；
- apparent gain 只来自 replay allocation，且无法形成有价值的审计结论；
- 结果主要来自 oracle success-tail；
- 低层控制噪声大到无法区分方法。

## 12. 当前 checklist

- [x] 清理不完整 skeletonmem 数据下载残片到 trash。
- [x] 新建干净 LingBot-VA clone。
- [x] 将正式 LIBERO-LONG 数据路径切到 `/data/jiaoguanbo/LIBERO/libero-long-lerobot`。
- [x] 完成官方数据下载。
- [x] 跑 Gate A1 数据与 latent 审计。
- [x] 确认官方数据不包含 `empty_emb.pt`。
- [x] 按用户授权复制本地 `empty_emb.pt` 并记录 hash/shape/dtype。
- [x] 更新 Gate A1 报告与 manifest。
- [x] 进入 Gate A2 单 episode lineage / sanity。
- [x] Gate A2 lineage/dataloader sanity 通过。
- [x] Gate A2 模型 import/config/input-contract sanity 通过。
- [x] Gate A3 tiny training sanity：single batch forward/backward。
- [x] Gate A3 one optimizer step。
- [x] A4-short full-data training smoke：10 steps，只做 full-data。
- [x] A4-short checkpoint save/load smoke。
- [x] A4-medium train50：50 steps full-data train curve。
- [x] A4-medium validation definition：deterministic task-balanced held-out split。
- [x] A4-medium validation smoke：step-50 checkpoint held-out forward/loss sanity。
- [x] A4-medium full held-out validation：50 validation segments。
- [x] A5 render dependency probe：HANDOFF NVIDIA EGL ICD 下 LIBERO env construct/reset/one-step 通过。
- [x] A5 official 1-rollout smoke：closed-loop pipeline 跑通，success `0/1`。
- [x] 第一版 accounting audit：full data unique/seen label 和 replay multiplier。
- [x] 第一版 accounting overview 图。
- [x] released LIBERO-LONG checkpoint reference eval：task0 `1/1` 与 `3/3`，eval contract 可信。
- [x] step500 local engineering anchor：Gate2A task0/task1/task2 为 `7/10`, `7/10`, `4/10`。
- [x] 中止并删除 `gradient_accumulation_steps=1` 的错误 4k run 训练产物。
- [x] 启动 `gradient_accumulation_steps=10` 的 official-scale full-data 4k run。
- [x] `full4k_gacc10_20260818T134819Z` 20-step supervision pass。
- [x] 验证 `checkpoint_step_1000` 已保存。
- [x] 验证 `checkpoint_step_2000` 已保存。
- [x] Gate B0：识别 LIBERO Object/Goal/Spatial 候选 LeRobot image 数据源。
- [x] Gate B0：创建 CPU-only 下载脚本并完成 dry-run。
- [x] Gate B0：记录当前 shell 网络代理 403，实际下载未开始。
- [x] Gate B0：用户恢复代理后启动 Object no-Xet single-worker 下载。
- [x] Gate B0：按用户要求停止 Object 下载；后因 `/data` 满盘删除 partial local copy。
- [x] Gate B0：审计 `/data/zouyude/data/lingbot-va/libero_all`。
- [x] Gate B0：判定 Zouyude source 缺 `data/` parquet，不复制为训练数据。
- [x] Gate B0：解压并审计 `/data/zouyude/data/lingbot-va/libero_all.zip`。
- [x] Gate B0：确认 zip 解压数据为 LingBot-VA multi-suite 数据，包含 parquet/videos/latents/meta。
- [x] Gate B0：为 richer parquet schema 添加 dynamic schema dataloader patch，并通过 zip + Long smoke。
- [x] Gate B0：将 `/data/zouyude/data/lingbot-va/libero_all.zip` 解压到 `/data/jiaoguanbo/LIBERO/libero_all`，并保留为固定 staging copy。
- [x] Gate B0：修复 `libero_all` 的 `data`/`videos` 软链接到 `/data/yaoyifei/dataset/fastwam/libero_mujoco3.3.2/...`，并通过 CPU-only dataloader smoke。
- [x] Gate B：准备 LIBERO40 full-data 训练脚本，强制 `libero40_` 命名并隔离输出目录；完成 dry-run guardrail 检查。
- [x] Gate B：LIBERO40 full-data 4k training 完成，run `libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z`。
- [x] Gate B：LIBERO40 mixed-data ledger 复算完成，`C_unique-label=1961820`, `C_seen-label=91676228`, `R_replay=46.73`。
- [x] Gate B：暂停当前 LIBERO40 step4000 `5 rollouts/task` quick eval；observed prefix `0/32`，仅作为 failed compatibility / training-sufficiency evidence。
- [x] 等待并验证 `checkpoint_step_3000`。
- [x] 等待并验证 `checkpoint_step_4000`。
- [x] 确认 `full4k_gacc10_20260818T134819Z` 训练正常完成且 GPU 空闲。
- [x] no-training dataloader smoke 在 `/data/jiaoguanbo/LIBERO/libero_all` 上复核通过。
- [x] 对 1k/2k/3k/4k checkpoints 做 official-init closed-loop trajectory eval。
- [ ] 固化 paper-level Long full reference distribution：second seed、弱任务 extra rollout、confidence intervals、failure taxonomy。
- [ ] 实现 automated temporal ledger，并先复算已完成 full-data runs。
- [ ] Ledger 和 full reference manifest 完成前不启动 selector / skeleton / memory baseline。
