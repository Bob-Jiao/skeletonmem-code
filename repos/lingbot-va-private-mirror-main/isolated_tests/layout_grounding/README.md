# M1-E003 layout-grounding test for LingBot-VA

This directory is an inference-only diagnostic isolated from the repository's
existing training and evaluation features.  It reuses the model-loading and
single-chunk planner helper from the adjacent causal-intervention test, but it
does not edit or monkey-patch production code.

The test holds the instruction, checkpoint, sampling seed, and denoising
configuration fixed while replacing only the two initial camera observations:
the original LIBERO layout plus four existing LIBERO Plus level-5 layouts.

It saves exact plan latents, decoded videos, a labeled comparison video, and a
whole-scene response metric.  The response metric compares the cross-layout RGB
difference at every generated frame with the cross-layout difference at frame
zero.  A projection near one means the visible layout difference is retained;
a projection toward zero means the plans are collapsing toward each other.

Example:

```text
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/zouyude/lingbot-va \
/data/shared/zouyude/conda/envs/lingbot-va/bin/python \
  isolated_tests/layout_grounding/run.py \
  --output-dir /root/shared/zouyude/eval/ideas/1-causal-video-planner-faithful-idm/M1-E003/lingbot
```
