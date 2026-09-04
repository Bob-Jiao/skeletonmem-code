# lingbot-va MoT + Joint 变体设计方案

> 目标:在 lingbot-va(单塔共享 Wan)基础上引入 FastWAM 式 **Mixture-of-Transformers (MoT)** 双塔架构与 **joint** 注意力变体,action 专家权重用 FastWAM 对 Wan 的**线性插值 + alpha 缩放**从现有 base ckpt 初始化,且保证原 base ckpt 在 `mot_enabled=False` 时完全兼容。

---

## 1. FastWAM 三个关键机制(参照实现)

来源 `/root/zouyude/FastWAM/src/fastwam/models/wan22/`。

### 1.1 MoT 容器(`mot.py`)
- `mixtures = {"video": video_expert, "action": action_expert}`(`fastwam.py:147`),每个模态一套**完整** DiT。
- **硬约束**:两专家 `num_heads` 与 `attn_head_dim` 必须相等(`mot.py:43-51`)—— 这样两塔的 Q/K/V 在注意力里维度对齐、可拼接。
- 每层流程(`mot.py:167-176, 323-340, 114-126`):
  1. 各专家用**自己的** norm1 + adaLN + Q/K/V + RoPE 投影**自己的** token;
  2. 各专家 Q/K/V 沿 **序列维拼接** → **一次** joint attention;
  3. 输出按模态切回,各专家用**自己的** O-proj / cross-attn(到 text)/ FFN。
- **每模态独立(复制)**:Q/K/V/O、norm_q/k、norm1/2/3、FFN、每层 adaLN modulation、cross-attn、以及各自的 time/text embedding。**唯一共享**:那一次 scaled-dot-product attention 的数学 + mask。

### 1.2 joint vs idm 变体(`fastwam_joint.py` / `fastwam_idm.py`)
- 变体差异**只在 attention mask 拓扑**(action token 能看什么):
  - **joint**(`fastwam_joint.py:50-53`):`action→action` 且 `action→整段 video`(与 video 一起 co-denoising 的 noisy video)。
  - **idm**(`fastwam_idm.py:52-55`):teacher-force 两条 video 流(noisy + clean cond),action **只看 clean cond video**。
- 都复用同一套 MoT joint-attention,只换 mask。

### 1.3 从 Wan 插值初始化 action 专家(`scripts/preprocess_action_dit_backbone.py`)
- action 专家残差流更窄(hidden **1024** / ffn **4096**),但 attn inner 与 video 一致(24×128=3072)。
- 逐权重把 Wan 权重塞进 action 形状,shape 不一致的维度做 **1-D 线性插值**(`_resize_tensor_to_shape`, `preprocess_action_dit_backbone.py:53-96`):
  ```python
  flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
  flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
  ```
- **alpha 缩放**(`:196-210`):当**最后一维(fan-in)**变化时,乘 `alpha = sqrt(d_src / d_tgt)` 补偿 fan-in 方差:
  ```python
  alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
  value = value.to(torch.float32) * alpha
  ```
- 只有纯 I/O 层(`action_encoder`、`head`)随机初始化,其余全部从 Wan 插值。

---

## 2. lingbot-va 现状 vs MoT 目标

| 维度 | 现状(单塔共享) | MoT 目标(双塔) |
|---|---|---|
| 残差流 | video+action 拼成**一条** `hidden_states`,过同一套 block(`model.py:1047, 957`) | video 塔(hidden 3072)+ action 塔(hidden 1024),**两条**独立残差张量 |
| 30 层 block | `self.blocks`(共享)`model.py:693` | `self.blocks`(video)+ `self.action_blocks`(action,窄) |
| 输入 embed | `patch_embedding_mlp`(video)/ `action_embedder`(action)**已分离** `model.py:680-683` | video `→3072`,action `→1024`(action_embedder 输出维改 1024) |
| 输出 head | `proj_out`(video)/ `action_proj_out`(action)**已分离** `model.py:703-705` | 同左,action_proj_out 输入维改 1024 |
| 时间步 embed | `condition_embedder` / `condition_embedder_action`**已分离** `model.py:684-691` | action 侧 time_projection 输出改 6×1024 |
| 最终 norm | `norm_out` + `scale_shift_table`**共享** `model.py:702-707` | 需拆成 per-modality(action 侧 1024) |
| 注意力 | FlexAttention over 合并序列,mask 用 `seq_ids/frame_ids/noise_ids`(`_get_mask_mod` `model.py:196-252`) | **不变**:两塔 Q/K/V 拼接后仍走这套 mask |

**关键洞察**:lingbot-va 的 clean/noisy 双流、packing、cross-chunk history 全部由 **per-token 的 `seq_ids/frame_ids/noise_ids` mask** 决定,与"哪套权重投影了这个 token"**正交**。因此只要拼接 Q/K/V 时保持**与 mask metadata 一致的 token 顺序**,现有 packing / clean-noisy / history 机制**原样复用,无需改动**。这是本方案能低风险落地的根本原因。

---

## 3. 核心改动:MoT 双塔 + joint attention

### 3.1 两条残差流(不再 concat)
现在 `forward_train` 在进入 block loop 前把四段拼成一条(`model.py:1047`)。MoT 下保持 **video 流** 与 **action 流** 各自独立:
- video 流 = `[latent_noisy, latent_clean]`(hidden 3072)
- action 流 = `[action_noisy, action_clean]`(hidden 1024)

同时构造**合并 mask metadata**,顺序为 `[video 全部 token; action 全部 token]`(FastWAM 风格 video-first / action-second)。metadata 数组(seq_ids/frame_ids/noise_ids/chunk_sizes/window_sizes)按同一顺序拼接即可 —— FlexAttention mask 对 token 排列等变,效果与现在完全一致。

### 3.2 每层 MoT block loop(伪代码)
```python
for i in range(num_layers):
    vblk, ablk = self.blocks[i], self.action_blocks[i]
    # 1) 各塔 pre-attn:norm1 + adaLN + QKV + RoPE(用各自 temb / rope)
    vq, vk, vv = vblk.attn_qkv(vblk.modulate_norm1(video_hidden, v_temb), rope)
    aq, ak, av = ablk.attn_qkv(ablk.modulate_norm1(action_hidden, a_temb), rope)
    # 2) 拼接 → 一次 joint FlexAttention(复用现有 mask,顺序 = [video; action])
    q = cat([vq, aq], seq); k = cat([vk, ak], seq); v = cat([vv, av], seq)
    out = FlexAttnFunc(q, k, v, block_mask=combined_mask)        # ← 不变
    vout, aout = split(out, [Lv, La], seq)
    # 3) 各塔 post-attn:O-proj + cross-attn(text)+ FFN(各自权重/gate)
    video_hidden  = vblk.post(video_hidden,  vout, text, v_temb)
    action_hidden = ablk.post(action_hidden, aout, text, a_temb)
# 4) 各塔独立 final norm + head
video_out  = video_head(video_final_norm(video_hidden, v_ss))
action_out = action_head(action_final_norm(action_hidden, a_ss))
```
- `WanAttention` 拆成 `attn_qkv`(投影+RoPE+norm_q/k)与外层拼接后的 `attn_op`,或新增一个 `MoTAttention` 封装。attn inner 两塔都是 3072,`FlexAttnFunc` 调用签名不变。
- RoPE 在 head_dim=128 上作用,两塔共享 head 结构,直接复用 `self.rope`。

### 3.3 joint 变体的 mask(新增一条规则)
lingbot-va 现有 `noise2clean_mask`(`model.py:226-229`)让 **noisy 只看 clean 分支** —— 这天然对应 FastWAM 的 **idm** 语义(action 看 clean cond video)。要得到 **joint**(action 看**同步去噪的 noisy video**),在 `_get_mask_mod` 增加一条:
```python
def action_noise2video_noise_mask(b, h, q_idx, kv_idx):
    # q 是 action-noisy,kv 是 video-noisy,同 chunk(同 frame 对齐)
    return is_action[q_idx] & (~is_action[kv_idx]) \
           & (noise_ids[q_idx]==0) & (noise_ids[kv_idx]==0)
# joint 变体额外 or 进去(与 block_causal 组合),idm 变体不加
```
用一个 `mot_attn_variant ∈ {"idm","joint"}` flag 控制是否 or 这条规则。需要新增 per-token `is_action` metadata(video-first/action-second 布局下就是位置阈值,零成本)。

---

## 4. 权重初始化 / 转换脚本(镜像 FastWAM)

新增 `script/preprocess_mot_action_expert.py`,输入现有 lingbot-va base ckpt,输出 MoT ckpt:
- **video 专家**:`blocks.{i}.*`、`patch_embedding_mlp`、`proj_out`、`condition_embedder`、`norm_out`、`scale_shift_table` —— **形状不变,直接拷贝**(video 质量零损失)。
- **action 专家**(从 base 的**同一批** `blocks.{i}.*` 插值,base 已学会联合处理 action token,是比 Wan 原始权重更好的暖启动):

| action 参数 | base 形状 | action 形状 | 处理 | alpha |
|---|---|---|---|---|
| `attn1.to_q/k/v` | [3072,3072] | [3072,**1024**] | 插值 in 维 | √(3072/1024)=√3 |
| `attn1.to_out.0` | [3072,3072] | [**1024**,3072] | 插值 out 维 | 无(fan-in 不变) |
| `attn2.to_q` | [3072,3072] | [3072,**1024**] | 插值 in 维 | √3 |
| `attn2.to_k/v` | [3072,4096] | [3072,4096] | **直接拷贝**(text 4096 不变) | — |
| `attn2.to_out.0` | [3072,3072] | [**1024**,3072] | 插值 out 维 | 无 |
| `norm_q/k` | [128] | [128] | 直接拷贝(head_dim 不变) | — |
| `ffn.net.0.proj` | [14336,3072] | [**4096**,**1024**] | 双维插值 | √3(in 维) |
| `ffn.net.2` | [3072,14336] | [**1024**,**4096**] | 双维插值 | √(14336/4096) |
| `norm2` / `scale_shift_table` | [·,3072] | [·,**1024**] | 插值最后一维 | — |
| `action_embedder` | [3072,30] | [**1024**,30] | 插值 out 维 | 无 |
| `action_proj_out` | [30,3072] | [30,**1024**] | 插值 in 维 | √3 |
| `condition_embedder_action.time_proj` | [6·3072,·] | [6·**1024**,·] | reshape [6,3072]→插值[6,1024] | 视情况 |
| action 侧 final `norm_out`/`scale_shift_table` | [·,3072] | [·,**1024**] | 插值 | — |

插值函数与 alpha 规则**逐字照搬** `preprocess_action_dit_backbone.py:53-96, 196-210`(fan-in 变化才乘 alpha)。

---

## 5. 向后兼容策略

- 新增 config flag `mot_enabled`(默认 `False`)。`False` → 走现有单塔 code path,base ckpt **原样加载,行为完全不变**。
- `mot_enabled=True` → 构建双塔;若给定 MoT ckpt 则加载,否则由转换脚本从 base ckpt 现场生成 action 塔。
- base ckpt 始终是**唯一真源**,MoT 只是从它派生初始化 —— 满足"原 pretrain base ckpt 可用"。
- LoRA:`lora.py` 的 target/save 列表需为 action 塔补 `action_blocks.*` 分支(`LORA_TARGET_PATTERN` / `LORA_ACTION_MODULES_TO_SAVE`)。

---

## 6. 逐文件改动清单

| 文件 | 改动 |
|---|---|
| `wan_va/configs/*_cfg.py` | 新增 `mot_enabled` / `action_hidden_dim=1024` / `action_ffn_dim=4096` / `mot_attn_variant` |
| `wan_va/modules/model.py` | ① `__init__`:`mot_enabled` 时建 `action_blocks`(窄)+ action 侧 final norm;② 新增 `ActionTransformerBlock`(QKV `1024→3072`,O `3072→1024`,FFN `1024→4096→1024`);③ `WanAttention` 拆 `attn_qkv` + 外层 joint `attn_op`(或新 `MoTAttention`);④ MoT block loop(§3.2);⑤ `_get_mask_mod` 加 joint 规则 + `is_action` metadata;⑥ `forward_train`/`_forward_train_packed` 保持两条流、构造 video-first 合并 metadata |
| `script/preprocess_mot_action_expert.py`(新) | §4 转换脚本 |
| `wan_va/modules/lora.py` | target/save 列表补 `action_blocks.*` |
| `wan_va/train.py` | 加载分支:`mot_enabled` 时走 MoT 构建/加载;`_add_noise_packed` 输出保持 video/action 分离(现已基本满足) |

---

## 7. 风险 / 待定

1. **block loop 重写**是最大改动面:必须保证 MoT 拼接顺序与 mask metadata 顺序严格一致,否则 packing/history 语义错乱。建议先写一个 `mot_enabled=True` 但 action 塔 = 3072(直接拷贝、无插值)的"等价性单测",对比单塔输出数值一致,再切到 1024 插值版。
2. **窄 action 塔容量**:1024 是否够表达 action?可作为 `action_hidden_dim` 超参扫。
3. **joint 是否引入训练不稳定**:action 看 noisy video(co-denoising)比 idm 更难;建议 flag 可切,先 idm 复现现状,再开 joint。
4. **final norm 共享→拆分**:注意 `norm_out` 现无 affine,拆分主要是 `scale_shift_table` 与作用维度。
5. **time_projection 插值**:6 段 chunk 要 reshape 后分别插值,别把 6×dim 当一维插。

---

## 8. 建议的推进顺序(含默认取值)

**默认取值(待确认)**:
- Action 塔宽度:分两阶段 —— **阶段0 用 3072 等宽直拷**做等价性验证,**阶段1+ 切 1024/4096**(对齐 FastWAM ActionDiT)正式训练。目的是把"block loop 重写"与"窄塔插值"两个风险源解耦。
- 变体优先级:**先 idm 复现现状,再开 joint**。

**步骤**:
1. 转换脚本 + `mot_enabled` 骨架(action 塔=3072 直拷)→ 数值等价性单测通过(MoT path 输出应与单塔逐元素一致)。
2. action 塔切 1024 + 插值/alpha → 小步 finetune,看 loss 是否平滑。
3. 加 `mot_attn_variant`:先 idm(复现),再 joint。
4. 补 LoRA 分支 + 配置扫参(`action_hidden_dim` 等)。
