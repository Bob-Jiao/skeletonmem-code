# EBench LingBot-VA preprocessing and training segmentation

`script/prepare_ebench_lingbot.py` derives audited, full-parent EBench
Generalist artifacts without modifying the source LeRobot dataset. Afterward,
`script/prepare_ebench_segments.py` creates the virtual segment overlay used
for training. The overlay references slices of the existing latent and action
files; it does not copy or modify them. The preprocessing defaults point to the
local Generalist source, LingBot-VA base checkpoint, and this output directory:

```text
/root/shared/yaoyifei/dataset/openpi_lerobot/ebench/generalist_lingbot_va
```

## 1. Setup full-parent actions, metadata, stats, and text

Run setup once on one process:

```bash
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  script/prepare_ebench_lingbot.py setup --device cuda:0
```

Setup adds exactly one full-episode `action_config` to the canonical metadata,
registers the missing top camera, corrects `total_videos`, links the immutable
source `data/` and `videos/`, writes one action sidecar per parent episode,
computes source-level train-only q01/q99, and creates the shared 130-task plus
empty-prompt text cache. Existing valid sidecars are reused only when their
schema and the previous
`action_stats -> data_manifest -> sidecar` SHA chain all match; otherwise they
are rebuilt from the source parquet. Text caches are likewise checked against
their recorded hashes during audit. Use `--overwrite-actions` or
`--overwrite-text` only when intentionally rebuilding them. Source and output
must be separate, non-nested directories.

The 19D sidecars live at:

```text
actions/chunk-XXX/episode_NNNNNN.npz
```

The canonical arrays are `action_19d` and `loss_mask_19d`. Their row count is
`16 * F`: one synthesized zero16 prefix for the full parent followed by
`16 * floor((L - 1) / 16)` retained real actions. The incomplete tail and
zero prefix are excluded from the source-level quantile statistics. Training
uses the segment-local prefixes and statistics described in section 4.

For the canonical 6,600-episode Generalist train split, setup records these
audit targets: 9,820,912 retained real actions, 54,082 discarded tail rows,
2,461,828 sampled RGB timestamps per camera, 620,407 latent frames per camera,
and 26,400 latent files. The raw tensor estimate is about 43.49 GiB; the
audited serialized files occupy 43.626 GiB including record overhead.

## 2. Encode four GPU shards

Start one process per GPU. Sharding is deterministic by the position in the
train episode manifest, and complete valid episodes are skipped by default:

```bash
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  /data/shared/zouyude/conda/envs/lingbot-va/bin/python \
    script/prepare_ebench_lingbot.py encode \
      --device cuda:0 --num-shards 4 --shard-index $rank \
      > "ebench_encode_${rank}.log" 2>&1 &
done
wait
```

`--num-shards N` supports any positive logical worker count, including eight
workers. Every worker in one run must use the same `N`, with one distinct
`--shard-index` in `[0,N)`; episode position `p` belongs only to shard
`p mod N`. The four-worker command above is the official one-worker-per-GPU
recipe. Running two workers per GPU (for example eight workers on four GPUs)
is also partition-safe, but should only be used after confirming that two VAE
processes fit the GPU and host-memory budget.

For a targeted smoke test, use for example:

```bash
CUDA_VISIBLE_DEVICES=0 \
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  script/prepare_ebench_lingbot.py encode \
    --device cuda:0 --episode-indices 0 1750
```

Each episode decodes the same raw IDs `0,4,8,...` for all cameras and crops to
`1 + 4k` sampled RGB frames. The four 224x224 streams enter the VAE as a batch;
RGB images are never mosaicked before encoding, and the VAE streaming cache is
cleared once per episode. Each latent record contains the task string for
shared-cache lookup but no repeated `text_emb` tensor.

The latent representation keeps the RoboTwin temporal protocol:
`frame_chunk_size=2`, `action_per_frame=16`, and a 32-action model chunk. The
canonical files remain full-parent records; the training loader later exposes
bounded virtual slices through the segment overlay.

## 3. Full audit

After all four shards finish:

```bash
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  script/prepare_ebench_lingbot.py audit
```

This source-artifact audit recomputes actions from the source parquet files,
verifies the single full-parent anchor/zero-prefix/tail rule, checks all four
latent schemas and synchronized frame IDs, scans actions/latents for
non-finite values, verifies metadata, action, stats, and shared-text SHA
chains, and hashes every derived latent. A successful complete audit writes
`latent_manifest.json`, updates
`action_stats_v1.json` with its SHA256, writes `audit_report.json`, and finally
publishes the atomic v2 `audit_success.json` training gate. The gate pins the
stats, data/latent/preprocess manifests, shared text cache, and empty embedding.
At load time, every selected latent is also checked against its audited path,
size, SHA256, frame count, bf16 dtype, and finite-value contract. Any later
setup, text-cache rebuild, or latent rebuild revokes the gate before writing,
so a new full audit is required before training can start again. A no-op encode
resume and a partial diagnostic audit leave an existing gate unchanged.

`--episode-indices` performs a partial diagnostic audit but deliberately does
not finalize the dataset-wide latent manifest or stats reference.

## 4. Build the virtual training segment overlay

Run this once after the complete source audit succeeds:

```bash
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  script/prepare_ebench_segments.py \
  --dataset-root /root/shared/yaoyifei/dataset/openpi_lerobot/ebench/generalist_lingbot_va
```

The default `max_self_tokens=81920` permits at most 193 latent frames per
virtual segment: one anchor frame plus at most 192 real-action blocks. In the
canonical 6,600-parent dataset, 520 parents exceed this bound. Each of those
parents is split into exactly two segments, producing 7,120 training segments
in total.

Segments are views, not new latent or action files. Two adjacent segments from
one parent share their boundary anchor latent, while their real action rows are
disjoint. Every segment receives its own zero16 prefix and segment-local joint
and cumulative-base anchor. This makes every virtual segment an independent
RoboTwin-style `action_config` without resetting an anchor in the middle of a
training sample.

The segment preparation pass recomputes per-channel q01/q99 after applying
those segment-local transforms. It uses all 9,820,912 retained real action
rows exactly once and excludes every synthesized segment prefix. It writes and
hash-chains these lightweight overlay artifacts in the dataset root:

```text
segment_manifest_81920_v1.json
action_stats_segment_81920_v1.json
segment_audit_success_81920.json
```

The loader requires both the original full audit and this segment audit, so a
stale manifest, source artifact, or segmented statistics file fails before
training begins.

## 5. Train on four GPUs

The launch command is unchanged after building the overlay:

```bash
NGPU=4 CONFIG_NAME=ebench_train bash script/run_va_posttrain.sh
```

`ebench_train` uses true variable-length packing with an effective global batch
of 128 virtual segments per optimizer update. A physical pack is limited to
81,920 self-attention tokens and at most four segments. Block-diagonal
attention keeps segments independent, and packed gradients are synchronized
at every microbatch. Here, “128 episodes” in the generic packing logs means
128 virtual segments; a split parent contributes two independently sampled
training items.

The launch helper defaults `TORCHINDUCTOR_COMPILE_THREADS` to 1 for
`ebench_train` (callers may override it) and retains the expandable-segments
allocator setting. The first occurrence of a new packed shape can be slower
while FlexAttention compiles its kernel.

Each segment-local zero16 leaves 16 executable actions in that segment's first
rollout and 32 in later rollouts. This repository still implements EBench
preprocessing and training only; an online EBench evaluation client and the
cumulative-base inverse transform are not included.

Neither preparation script launches the 50,000-step training job.
