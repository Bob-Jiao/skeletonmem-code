# M1-E005 inference-only first-frame anchor

This directory contains a no-training LingBot-VA intervention.  It is isolated
from all existing training and inference entry points and never modifies a
checkpoint or monkey-patches production modules.

The runner injects a timestep-matched first-frame anchor after each
flow-matching update.  Only the spatial low-frequency part of the difference is
blended.  The default `static_video` source first repeats the RGB observation
to 13 frames and re-encodes it with the causal temporal VAE, producing valid
future temporal latents.  The legacy `repeated_first_latent` source is retained
as a negative control.  The clean anchor is noised to the scheduler's next
sigma with the same initial noise as the sampled plan before blending, avoiding
a clean-latent injection into a noisy denoising state.

Supported masks:

- `global_lowfreq`: anchor both camera views;
- `agent_lowfreq`: anchor only the fixed external camera and leave wrist free;
- `motion_selective_lowfreq`: anchor the fixed camera except latent cells with
  high baseline temporal motion, leaving likely robot/target motion cells free.

The strength is a total cumulative blend across all denoising steps, rather
than a per-step coefficient.  The runner compares original and LIBERO Plus
`level5_sample2`, saves exact latents/videos, and reports both agent-view layout
retention and motion preservation relative to the M1-E003 baseline.

Example:

```text
CUDA_VISIBLE_DEVICES=0 \
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  isolated_tests/first_frame_anchor/run.py \
  --output-dir /root/shared/zouyude/eval/ideas/1-causal-video-planner-faithful-idm/M1-E005/global \
  --modes global_lowfreq \
  --strengths 0.25,0.50,0.75 \
  --anchor-source static_video \
  --verify-baseline
```

## Recorded experiment

M1-E005 was completed on 2026-08-07. The best valid condition, static-video
global anchoring at total strength 0.75, changed the hardest-layout last-frame
retained projection from `0.39361` to only `0.40936` while preserving motion.
It therefore failed the `0.70-0.75` expansion gate and was not expanded to all
four layouts. Full results and the negative-control analysis are in:

```text
/root/shared/zouyude/eval/ideas/1-causal-video-planner-faithful-idm/M1-E005_20260807_072136
```
