# Causal plan intervention test for LingBot-VA

This directory is intentionally isolated from the production training and
inference paths.  It imports the existing `VA_Server`, but it does not patch or
modify any source module or checkpoint.

The test fixes the observation, video/action seeds, scheduler settings, and
checkpoint, then measures two causal dependencies:

1. plan dependency: replace the generated visual plan with static, zero-future,
   or wrong-prompt plans while keeping the action-side text fixed;
2. text shortcut: keep the generated plan fixed while replacing only the text
   seen during action denoising.

The runner saves plan latents, decoded plan videos, every action tensor, exact
metadata, and metrics relative to `normal_original`.

Example:

```text
CUDA_VISIBLE_DEVICES=0 \
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  isolated_tests/causal_plan_interventions/run.py \
  --output-dir /root/shared/zouyude/eval/ideas/1-causal-video-planner-faithful-idm/M1-E001-E002/lingbot
```

The default checkpoint is the final step under the user-requested training
directory:

```text
/root/shared/zouyude/train/lingbot-va/libero-all-full/checkpoints/checkpoint_step_20000/transformer
```
