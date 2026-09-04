# LingBot-VA 评测指南

本文档是本机当前 LingBot-VA 评测入口和协议的统一说明，覆盖：

- LIBERO Standard
- LIBERO Plus
- LIBERO OOD
- RoboTwin 2.0（clean 与 randomized）

除非专门说明，命令都从仓库根目录执行：

```bash
cd /root/zouyude/lingbot-va
```

所有推荐 launcher 都显式调用对应 conda 环境中的 Python，因此不需要
`conda activate`。四卡入口默认占用 GPU 0、1、2、3；同一时间只运行一个四卡
benchmark，并在启动前确认没有训练或其他评测进程占用这些卡。

## 一、配置总表

| Benchmark | Suite / task 范围 | 默认 rollout 数 | 初始状态 | Settling | 上限 | 图像 | 视频 |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| LIBERO Standard | `libero_10`、`libero_goal`、`libero_spatial`、`libero_object`，各 10 task，共 40 | 每 task 50，共 2,000 | benchmark 固定 init states | 5 step | 800 env step，包含 settling | 双相机 128×128 | 不保存 |
| LIBERO Plus | 同名四个 suite，本地共 10,030 task；默认固定抽样 1,506 task | 每个抽中 task 1 次，共 1,506 | benchmark 固定 init states | 5 step | 800 env step，包含 settling | 双相机 128×128 | 不保存 |
| LIBERO OOD | `libero_spatial_ood`、`libero_object_ood`、`libero_goal_ood`，各 10 task，共 30 | 每 task 10 次，共 300 | fresh seeded reset | 10 step | spatial 300、object 280、goal 300 control step | 双相机 128×128 | 定量入口不保存 |
| RoboTwin clean | 官方 50 个唯一 task | 每 task 100 次，共 5,000 | expert-feasible seeded episodes | expert 先筛选 | 各 task 400–1,700 step | head + 双 wrist；原始 320×240，模型 320×256 | 默认保存；定量推荐关闭 |
| RoboTwin randomized | 与 clean 相同的 50 task | 每 task 100 次，共 5,000 | expert-feasible seeded episodes | expert 先筛选 | 与 clean 相同 | 与 clean 相同 | 默认保存；定量推荐关闭 |

这里的 rollout 数是成功率的分母。RoboTwin 在每个 policy rollout 前可能尝试多个
seed 来找到稳定且 expert 能完成的场景；这些 expert 尝试不计入分母，但会增加运行
时间。

所有 LIBERO 定量入口使用 `fastwam_lerobot` 动作协议。不要把同一 checkpoint
在不同协议下得到的结果直接比较。

## 二、LIBERO Standard

### 推荐入口

```bash
bash evaluation/libero/launch_trained_4gpu_eval.sh \
  {full-4gpu|lora-4gpu|lora-action-4gpu} standard STEP
```

先只做 checkpoint、CUDA 和 EGL 检查：

```bash
PREFLIGHT_ONLY=1 \
bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu standard 20000
```

正式运行示例：

```bash
bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu standard 20000
bash evaluation/libero/launch_trained_4gpu_eval.sh lora-action-4gpu standard 16000
```

模型映射如下：

| `VARIANT` | 训练目录 | 加载方式 | 当前状态 |
| --- | --- | --- | --- |
| `full-4gpu` | `libero-all-full-4gpu` | 完整 transformer | 可用，当前有 step 2,000–20,000 |
| `lora-4gpu` | `libero-all-lora-r64-4gpu` | base transformer + adapter | Selector 支持，但本机训练目录当前不存在 |
| `lora-action-4gpu` | `libero-all-lora-r64-action-4gpu` | base transformer + adapter | 可用，当前有 step 2,000–20,000 |

Launcher 会检查 checkpoint 是否属于指定训练目录；LoRA 还会检查 manifest 中的
训练模式、step、四卡拓扑、packing 和 base transformer，避免误加载相似路径下的
权重。

### Canonical 配置

- Suites：`libero_10`、`libero_goal`、`libero_spatial`、`libero_object`。
- 每 suite 10 task，共 40 task，不抽样。
- `TEST_NUM=50`，即每模型 40 × 50 = 2,000 rollouts。
- Seed：42。
- 每回合使用 LIBERO benchmark 提供的固定 init state。
- 先执行 5 个 settling steps。
- `max_env_steps=800`，包含 settling steps。
- agentview 与 eye-in-hand 两路 RGB，均为 128×128。
- 协议固定为 `EVAL_PROTOCOL=fastwam_lerobot`。
- 四个 client shard，每个 shard 负责一部分 task。
- 默认不编码评测视频。

可以通过 `TEST_NUM` 改变每 task 次数，但这会改变 benchmark 口径。正式横向比较时
保持默认值：

```bash
TEST_NUM=50 \
bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu standard 20000
```

### 输出与恢复

默认输出根目录：

```text
/root/shared/zouyude/eval/lingbot-va/
```

单次运行包含：

```text
<OUT_ROOT>/launch_manifest.json
<OUT_ROOT>/results/<suite>_<task_id>.json
<OUT_ROOT>/logs/
<OUT_ROOT>/COMPLETE
```

Launcher 会锁定 `OUT_ROOT`，并拒绝接管没有 manifest 的旧结果。重新运行同一命令时
会按每个 task 已完成的 rollout 数恢复；只有所有 task 达到目标次数后才写入
`COMPLETE`。

## 三、LIBERO Plus

### 推荐入口

```bash
PREFLIGHT_ONLY=1 \
bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu plus 20000
```

预检查通过后运行：

```bash
bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu plus 20000
bash evaluation/libero/launch_trained_4gpu_eval.sh lora-action-4gpu plus 16000
```

`lora-4gpu` 的 selector 虽然存在，但对应训练目录当前缺失；恢复该训练产物前不要把它
作为可运行示例。

### Canonical 配置

本地 pinned LIBERO Plus 的 task 规模为：

| Suite | 全量 task | `TASK_SAMPLE_RATIO=0.15` 后 |
| --- | ---: | ---: |
| `libero_10` | 2,519 | 378 |
| `libero_goal` | 2,591 | 389 |
| `libero_spatial` | 2,402 | 361 |
| `libero_object` | 2,518 | 378 |
| 合计 | 10,030 | 1,506 |

默认协议：

- `TASK_SAMPLE_RATIO=0.15`。
- `TASK_SAMPLE_SEED=42`，每个 suite 独立、确定性抽样并向上取整。
- `TEST_NUM=1`，即每模型共 1,506 rollouts。
- Seed 42、固定 init state、5 settling steps、800 env steps、双相机 128×128。
- `EVAL_PROTOCOL=fastwam_lerobot`，四 shard，默认不保存视频。

如需全量任务，可以使用 `TASK_SAMPLE_RATIO=1`；如需每 task 多次，可以修改
`TEST_NUM`。这两项都会改变评测口径，比较不同 checkpoint 时必须保持比例、抽样
seed 和次数完全一致：

```bash
TASK_SAMPLE_RATIO=0.15 TASK_SAMPLE_SEED=42 TEST_NUM=1 \
bash evaluation/libero/launch_trained_4gpu_eval.sh full-4gpu plus 20000
```

输出、锁、resume、manifest 和 `COMPLETE` 约定与 Standard 相同。输出目录名会用
`libero_plus` 标记，不应与 Standard 共用 `OUT_ROOT`。

## 四、LIBERO OOD

### 当前实现状态

严格的 300-rollout 定量入口目前位于独立 worktree：

```text
.claude/worktrees/mot-joint/evaluation/libero/launch_joint_ood_4gpu_eval.sh
.claude/worktrees/mot-joint/evaluation/libero/libero_ood_client.py
.claude/worktrees/mot-joint/evaluation/libero/libero_ood_protocol.py
.claude/worktrees/mot-joint/evaluation/libero/summarize_libero_ood.py
```

这些文件当前还是该 worktree 的未跟踪文件，并未进入主工作树提交。它们在本机可用，
但在正式提交或迁移前不要对该 worktree 执行 `git clean`。主工作树通用
`evaluation/libero/client.py` 不能视为这套 OOD 协议的严格等价替代。

### 推荐定量入口

只做 pinned source、checkpoint、CUDA 和 EGL 预检查：

```bash
PREFLIGHT_ONLY=1 \
bash .claude/worktrees/mot-joint/evaluation/libero/launch_joint_ood_4gpu_eval.sh \
  full 20000
```

正式运行 full、LoRA，或依次运行两者：

```bash
bash .claude/worktrees/mot-joint/evaluation/libero/launch_joint_ood_4gpu_eval.sh \
  full 20000

bash .claude/worktrees/mot-joint/evaluation/libero/launch_joint_ood_4gpu_eval.sh \
  lora 20000

bash .claude/worktrees/mot-joint/evaluation/libero/launch_joint_ood_4gpu_eval.sh \
  all 20000
```

该入口面向 MoT joint 训练产物：

| Target | 训练目录 / base |
| --- | --- |
| `full` | `libero-all-full-bs48-joint-2opt` |
| `lora` | `libero-all-lora-bs48-joint`，base 为 `lingbot-va-mot-libero-all/transformer` |

### Canonical 配置

- OOD source：`/root/zouyude/FastWAM/third_party/pi0-text-latent`。
- Source commit：`587a6cbf64f16c7b87fa5805dc0ed934192239a4`。
- Config：`/root/zouyude/FastWAM/data/libero_ood_config`。
- Suites：`libero_spatial_ood`、`libero_object_ood`、`libero_goal_ood`。
- 每 suite 10 task，共 30 task。
- `NUM_TRIALS=10`，即每模型 30 × 10 = 300 rollouts。
- `EVAL_SEED=42`；每个 task 调用一次 `env.seed(42)`，随后各 trial 依次执行
  `env.reset()`，不使用 Standard 的固定 init states。
- 10 个 settling steps。
- 最大 control steps：spatial 300、object 280、goal 300。
- agentview 与 eye-in-hand 均为 128×128。
- `fastwam_lerobot` 动作转换。
- 固定执行 5-action receding-horizon plan；每次 replan 都从最新观测开始并清空
  server cache。
- 30 个 task 全局分为 8/8/7/7 四个 shard。
- 默认不保存定量视频。

`NUM_TRIALS` 可以覆盖，但正式结果保持 10：

```bash
NUM_TRIALS=10 EVAL_SEED=42 \
bash .claude/worktrees/mot-joint/evaluation/libero/launch_joint_ood_4gpu_eval.sh \
  full 20000
```

### 输出与恢复

默认输出名包含模型、step、30 task、trial 数和 `replan5`。主要文件为：

```text
<OUT_ROOT>/evaluation_manifest.json
<OUT_ROOT>/results/<suite>_<task_id>.json
<OUT_ROOT>/summary.json
<OUT_ROOT>/summary.csv
<OUT_ROOT>/logs/
<OUT_ROOT>/COMPLETE
```

已完整完成的 task 会跳过；不完整 task 会从 trial 0 重新执行。`summary.json` 包含
各 suite 与 overall micro success rate。

主工作树的 `evaluation/libero/render_ood_qualitative.py` 是定性可视化工具：只覆盖
3 个 suite 的前 5 个 task、每 task 1 回合，并强制写视频。它不属于上述
300-rollout 定量 benchmark，不能用其结果报告 OOD 成功率。

## 五、RoboTwin 2.0

### 环境与 checkpoint 准备

推荐入口：

```bash
bash evaluation/robotwin/run_local_eval.sh TRANSFORMER_DIR SAVE_ROOT
```

Launcher 会分别使用：

- Server：`/data/shared/zouyude/conda/envs/lingbot-va/bin/python`。
- Client：`/data/shared/zouyude/conda/envs/robotwin-lingbot-va/bin/python`。
- RoboTwin：`/root/zouyude/RoboTwin-lingbot-va`，commit
  `2eeec322d95799f537cbfe5f291a8220d965ccb8`。
- 与 driver 595 匹配的 NVIDIA Vulkan runtime。

训练 checkpoint 的 `attn_mode=flex` 不能直接用于评测。不要改动续训原件；创建
权重软链接和 `attn_mode=torch` 配置组成的 eval view：

```bash
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  evaluation/robotwin/prepare_eval_transformer.py \
  SOURCE_CHECKPOINT/transformer \
  /root/shared/zouyude/train/lingbot-va/robotwin-eval-views/NAME/transformer \
  --attn-mode torch
```

当前已准备：

```text
/root/shared/zouyude/train/lingbot-va/robotwin-eval-views/bs16-step40000/transformer
/root/shared/zouyude/train/lingbot-va/robotwin-eval-views/bs64-step24000/transformer
```

### Clean 评测

```bash
TASK_CONFIG=demo_clean SAVE_VIDEOS=0 \
GPU_IDS="0 1 2 3" TEST_NUM=100 SEED=0 \
bash evaluation/robotwin/run_local_eval.sh \
  /root/shared/zouyude/train/lingbot-va/robotwin-eval-views/bs16-step40000/transformer \
  /root/shared/zouyude/train/lingbot-va/robotwin-eval-results/bs16-step40000-clean
```

### Randomized 评测

```bash
TASK_CONFIG=demo_randomized SAVE_VIDEOS=0 \
GPU_IDS="0 1 2 3" TEST_NUM=100 SEED=0 \
bash evaluation/robotwin/run_local_eval.sh \
  /root/shared/zouyude/train/lingbot-va/robotwin-eval-views/bs16-step40000/transformer \
  /root/shared/zouyude/train/lingbot-va/robotwin-eval-results/bs16-step40000-randomized
```

评测 bs64 时替换 transformer 和输出名即可。两个 checkpoint 都跑 clean 与
randomized 时，总计 2 × 2 × 5,000 = 20,000 policy rollouts。

### Canonical 配置

- 官方 50 个唯一 task；本地脚本不会为了均分而重复 padding task。
- 四个 worker 轮转分配为 13/13/12/12。
- `TEST_NUM=100`，每种 task config、每个模型共 5,000 policy rollouts。
- `SEED=0` 对应起始 seed 10,000。
- 每个 episode 先运行 expert plan，跳过不稳定或 expert 无法完成的 seed，再在同一
  seed 上评测 policy。
- 每 task step limit 来自 pinned RoboTwin 的
  `task_config/_eval_step_limit.yml`，范围 400–1,700；未列出的 task fallback 为
  1,000。
- D435 head camera 与双 wrist camera，原始 RGB 320×240；模型配置为高 256、宽
  320。
- 模型配置：`action_dim=30`、`action_per_frame=16`、`frame_chunk_size=2`、视频扩散
  25 steps、动作扩散 50 steps、video guidance 5、action guidance 1。
- 每 5 个有效 episode 清一次模型 cache。
- `render_freq=0`，不打开 viewer，但策略输入仍需要离屏渲染。
- 默认端口为 policy 29556–29559、distributed master 29661–29664。

Domain randomization 差异：

| 配置项 | `demo_clean` | `demo_randomized` |
| --- | --- | --- |
| `random_background` | `false` | `true` |
| `cluttered_table` | `false` | `true` |
| `clean_background_rate` | `1` | `0.02` |
| `random_head_camera_dis` | `0` | `0` |
| `random_table_height` | `0` | `0.03` |
| `random_light` | `false` | `true` |
| `crazy_random_light_rate` | `0` | `0.02` |

两份 YAML 中的 `episode_num: 50` 是数据采集参数，不控制 LingBot-VA 评测次数；
这里的次数由 `TEST_NUM` 决定。

### 视频、输出和恢复

`SAVE_VIDEOS` 默认是 1，会同时保存 RoboTwin 原生 head-camera 视频和 LingBot
三相机拼接视频。定量评测推荐设置 `SAVE_VIDEOS=0`；这只跳过 MP4 编码与写盘，
不会改变模型请求、帧历史或动作历史。

指标位于：

```text
<SAVE_ROOT>/stseed-10000/metrics/<task>/res.json
```

日志位于 `<SAVE_ROOT>/logs/`。指标路径本身不包含 checkpoint 名或 task config，
因此每个 checkpoint 和 `TASK_CONFIG` 必须使用不同的 `SAVE_ROOT`，否则会覆盖。

RoboTwin launcher 以每个 task 的 `res.json` 为完整性依据：启动 server 前会先扫描
全部选中 task，`total_num == TEST_NUM` 且计数字段、成功率都合法的 task 会直接
跳过。缺失、JSON 损坏、计数不合法或回合数不足的 task 会从该 task 的第一个
episode 重新执行；不会续接一个 task 内已经完成的部分回合。

每轮 worker 结束后 launcher 会再次检查，并只将仍不完整的 task 重新分片补跑。
单个 task 进程失败不会阻断同一 worker 后面的 task；model server 如果退出也会在
下一轮尝试重启。`MAX_RETRY_ROUNDS` 默认是 3，表示初始执行后最多再补跑 3 轮；
所有选中 task 完整时才返回 0，达到上限仍有缺失时返回非零。可按需调整，例如：

```bash
MAX_RETRY_ROUNDS=5 \
bash evaluation/robotwin/run_local_eval.sh TRANSFORMER_DIR SAVE_ROOT
```

每次检查都会原子更新 `<SAVE_ROOT>/completeness.json`，记录各 task 的结果路径、
完整状态及失败原因。中断后使用相同 transformer、`TASK_CONFIG`、`TEST_NUM`、
`SEED`、task 集和 `SAVE_ROOT` 重新执行原命令即可补缺；不要用恢复机制混合不同
评测协议。Launcher 仍会在退出时清理自己启动的 server/client。

小规模链路检查可以限制任务和次数：

```bash
TASK_CONFIG=demo_clean SAVE_VIDEOS=0 \
TASKS="handover_block" TEST_NUM=1 GPU_IDS="0" SEED=0 \
bash evaluation/robotwin/run_local_eval.sh \
  /root/shared/zouyude/train/lingbot-va/robotwin-eval-views/bs16-step40000/transformer \
  /root/shared/zouyude/train/lingbot-va/robotwin-eval-smoke/manual-handover-block
```

## 六、本机 pinned 环境

| 组件 | 路径 / 版本 |
| --- | --- |
| LingBot server | `/data/shared/zouyude/conda/envs/lingbot-va`；Python 3.10.16，torch 2.9.0+cu126，transformers 4.55.2，diffusers 0.36.0 |
| LIBERO Standard client | `/data/shared/zouyude/conda/envs/libero`；Python 3.10.20；source `/data/zouyude/LIBERO` commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO Plus client | `/data/shared/zouyude/conda/envs/libero_plus`；Python 3.10.20；source `/root/zouyude/LIBERO-plus` commit `4976dc30028e805ff8094b55501d532c48fec182` |
| LIBERO OOD source | `/root/zouyude/FastWAM/third_party/pi0-text-latent` commit `587a6cbf64f16c7b87fa5805dc0ed934192239a4`；验证环境 `/data/shared/zouyude/conda/envs/fastwam_eval` |
| RoboTwin client | `/data/shared/zouyude/conda/envs/robotwin-lingbot-va`；SAPIEN 3.0.0b1、scipy 1.10.1、mplib 0.2.1、transforms3d 0.4.2 |
| RoboTwin source | `/root/zouyude/RoboTwin-lingbot-va` commit `2eeec322d95799f537cbfe5f291a8220d965ccb8` |

不要将原来给 LaWAM 使用的 `/root/zouyude/RoboTwin` 替换为上述 pinned checkout。

## 七、结果可比性检查清单

报告成功率前至少确认：

- checkpoint 路径、step、full/LoRA 类型正确；
- benchmark、suite/task 集和抽样 seed 一致；
- 每 task rollout/trial 数一致；
- init-state 模式、settling steps 和 step limit 一致；
- `fastwam_lerobot` 等动作协议一致；
- RoboTwin 的 `TASK_CONFIG` 与 `SEED` 一致；
- 输出目录中的 manifest 或日志记录了本次实际配置；
- 只聚合完整 task，并区分 micro success rate 与 task-macro average；
- smoke test 结果只用于证明链路，不能作为正式成功率。
