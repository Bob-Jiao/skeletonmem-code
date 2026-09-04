# 阶段 1：LingBot-VA 两卡复现性审计

> 日期：2026-08-14  
> 官方源码：`robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`  
> 硬件：2 × NVIDIA A100-SXM4-80GB  
> 结论：released-checkpoint 双 benchmark 闭环通过；两卡 fine-tune 源码可运行，但 LIBERO 完整训练仍受官方数据缺失约束。

## 1. 阶段结论

| 验证项 | 结果 | 准确解释 |
|---|---:|---|
| LIBERO-10 released checkpoint 闭环 | 1/1，成功 | 干净官方源码加兼容适配后，任务 0 完成 |
| RoboTwin released checkpoint 闭环 | 1/1，成功 | `adjust_bottle`，seed 10000，124/400 steps 成功 |
| RoboTwin 两卡 fine-tune smoke | 通过 | 官方数据的 2 个完整 episode，完成 1 次 optimizer step |
| LIBERO 两卡 fine-tune smoke | 通过（合同级） | 完成 1 次 optimizer step；使用 LIBERO 形状合同 proxy，不代表官方数据复现 |
| LIBERO 官方完整数据 fine-tune | 尚未验证 | 本机无 `libero-long-lerobot`，当前网络访问 HF/ModelScope 返回 403 |

阶段判断：**工程链路 Go；完整数据复现 Conditional Go；研究题目 Conditional Go。**

## 2. 评估结果

### 2.1 LIBERO-10

- checkpoint：`lingbot-va-posttrain-libero-long`；
- task：`put both the alphabet soup and the tomato sauce in the basket`；
- rollout：1；
- 结果：1/1；
- 耗时：约 160 秒；
- 模型 server 显存：约 23.3GB；
- 结果：[libero_10_0.json](./results/stage1/libero_eval/libero_10_0.json)；
- 日志：[libero_eval_smoke_retry2.log](./logs/stage1/libero_eval_smoke_retry2.log)。

历史目录中另有 2026-07-10 的 10-task、500-rollout 结果：485/500，即 97.0%。该结果加载同一 released checkpoint，但当时源码有本地修改，因此只作为已有证据，不作为干净官方复现。当前 1/1 smoke 证明现机器上的干净适配链可重新接通。

### 2.2 RoboTwin

- checkpoint：`lingbot-va-posttrain-robotwin`；
- task：`adjust_bottle`；
- seed：10000；
- 结果：1/1，124/400 steps 成功；
- 结果：[res.json](./results/stage1/robotwin_eval/stseed-10000/metrics/adjust_bottle/res.json)；
- 日志：[robotwin_eval_smoke_retry4.log](./logs/stage1/robotwin_eval_smoke_retry4.log)。

旧的 RoboTwin 日志在 episode 初始化前因 `warp.torch` 缺失而失败，不能计作模型成绩。本轮已验证 Vulkan、SAPIEN、CuRobo、websocket、模型推理和闭环控制全链路。

## 3. 两卡 fine-tune 结果

| 配置 | 数据 | 输入合同 | 结果 |
|---|---|---|---|
| `robotwin_train` | 2 个官方 `adjust_bottle` episodes | latent `[48,9,24,20]`；action `[30,9,16,1]` | latent loss 0.3966；action loss 0.3104；grad norm 2.85；60.3s |
| `libero_train` | 2-episode LIBERO contract proxy | latent `[48,9,8,20]`；action `[30,9,4,1]` | latent loss 0.3383；action loss 0.6976；grad norm 14.71；34.7s |

两项均使用：

- 2 个 rank、2 张 A100；
- FSDP；
- LingBot-VA base 初始化；
- batch size 1/rank；
- gradient accumulation 1；
- 1 次真实 forward、backward 和 optimizer step；
- WandB 关闭；
- 不保存 checkpoint。

通过日志：

- [train_robotwin_2gpu_retry2.log](./logs/stage1/train_robotwin_2gpu_retry2.log)
- [train_libero_2gpu.log](./logs/stage1/train_libero_2gpu.log)

LIBERO proxy 只证明源码、FSDP、相机拼接、7D action 映射和 `action_per_frame=4` 合同可运行。它不证明在官方 LIBERO-LONG 数据上的 loss、收敛速度或最终性能。

## 4. 官方源码为何不能直接一条命令运行

干净源码保存在 [vendor/lingbot-va-official](./vendor/lingbot-va-official)，实验工作树在 [worktrees/lingbot-va-stage1](./worktrees/lingbot-va-stage1)。适配 diff 为 [lingbot_stage1_reproduction.patch](./results/stage1/lingbot_stage1_reproduction.patch)。

必要适配包括：

1. 官方模型与数据路径是 `/path/to/...` 占位符；改为环境变量覆盖。
2. 官方训练配置强制在线 WandB；增加可关闭开关。
3. 数据集初始化固定 128 个进程；增加可控 `init_worker`。
4. 未安装 `flash-attn` 时，官方代码即使使用 torch/flex attention 也会在 import 阶段退出；改为按模式检查。
5. LIBERO websocket 默认连接 `0.0.0.0`，会被本机代理截获；改为 `127.0.0.1`。
6. RoboTwin 源路径、运行目录、图形库和 ffmpeg 需要显式隔离。
7. PyTorch 2.9 的两个 rank 共享 Triton cache 会竞争删除 launcher；必须使用 rank-local compile cache。

这些均为路径、依赖或运行时兼容适配，没有改变网络、loss 或评估成功条件。

## 5. 失败复盘

保留失败日志是为了避免后续重复踩坑：

| 失败 | 根因 | 处理 |
|---|---|---|
| LIBERO 卡在 0/1 | `0.0.0.0` websocket 被代理接管 | 使用 `127.0.0.1` 和 `NO_PROXY` |
| LIBERO 缺少 `libGL.so.1` | 容器系统图形库不完整 | deb 仅解包到 `skeletonmem/runtime` |
| RoboTwin Vulkan 不兼容 | 缺少 Vulkan loader 与 `libXext` | 隔离补齐运行库和 NVIDIA ICD |
| RoboTwin 找不到 assets | 为避免外部输出而改变 cwd | 在隔离 runtime 中创建只读资源链接 |
| RoboTwin 找不到 ffmpeg | conda 环境无 `ffmpeg` 命令 | 链接 imageio-ffmpeg 到隔离 runtime/bin |
| 两卡训练找不到 cubin/launcher | 两 rank 共用编译缓存竞争 | 每 rank 独立 Triton/Inductor cache |

完整失败日志位于 `logs/stage1/`，不得把失败 rollout 计入模型成功率。

## 6. 可复现入口

- 模型 server：[run_server.sh](./scripts/stage1/run_server.sh)
- LIBERO client：[run_libero_client.sh](./scripts/stage1/run_libero_client.sh)
- RoboTwin client：[run_robotwin_client.sh](./scripts/stage1/run_robotwin_client.sh)
- 两卡训练：[run_train_smoke.sh](./scripts/stage1/run_train_smoke.sh)
- rank-local cache 入口：[ranked_train_entry.py](./scripts/stage1/ranked_train_entry.py)
- smoke 数据生成：[prepare_smoke_datasets.py](./scripts/stage1/prepare_smoke_datasets.py)

所有新增文件、结果、运行库和缓存均位于 `/data/jiaoguanbo/skeletonmem`。LIBERO 首次导入曾自动生成 `/root/.libero/config.yaml`，发现后已迁入隔离 HOME 并删除外部文件。

## 7. 对研究方向的影响

### 7.1 得到的正面结论

1. LingBot-VA 可以作为强 VLA baseline，不需要再怀疑最基本的工程可运行性。
2. 两卡足以完成小规模 source fine-tune；后续关键设置可在 ACT 扫描后迁移到 LingBot。
3. 训练数据合同可被严格审计，适合研究监督预算、样本选择或控制充分表示。

### 7.2 新的风险

LIBERO 普通任务已经非常接近天花板。若新 idea 只比较标准 LIBERO-10 平均成功率，很难产生有辨识度的结论。后续实验必须至少包含一种：

- 相同当前观测、不同历史、不同正确动作的 paired-history 任务；
- OOD、失败恢复或遮挡后的几何控制；
- success–data/compute curve，而不是单点 success；
- decision success 与 low-level execution success 分解。

因此，当前题目不应收缩成“给 LingBot 再加一个 memory module”。更稳健的研究对象仍是：**控制所需信息或监督的最小充分结构，以及它随任务查询和控制难度如何变化。**

## 8. 阶段 2 建议

在继续大规模训练前先做一个 2–3 天的研究判别实验：

1. 固定一个低成本 backbone（优先 ACT）和一个记忆/长程任务族。
2. 构造 Full、Random、Uniform、Motion、Event/Stage 五种选择器。
3. 只跑 25%、50%、100% 三个预算点和 1 个 seed。
4. 同时画 success–unique supervised timesteps 和 success–GPU-hours。
5. 若 Event/Stage 在相同预算下没有稳定优于 Random/Uniform，立即止损或改题；若有明显间隔，再补种子并迁移 2–3 个设置到 LingBot。

在正式启动阶段 2 前，还需决定是否重新引入现有 `pick_objects_in_order` 50 episodes / 850 overlapping segments；阶段 1 没有使用这 850 segments。
