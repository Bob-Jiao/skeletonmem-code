# Temporal Curation Resource-Equivalent Counterfactual Audit

Active date: 2026-09-03 UTC

This document is the active paper-positioning contract. If it conflicts with
older `SkeletonMem`, `Temporal Supervision Accounting`, `Gate A`, or
`Skeleton Supervision Gate` notes, this document wins.

## Strict Positioning

The paper is a resource-equivalent counterfactual audit of temporal curation.

It is not a memory paper, not a new selector paper, and not a generic
data-compression paper. The main object of study is whether closed-loop gains
attributed to temporal curation are actually caused by selected temporal
support, dense-data access schedules, extra target exposure, replay
concentration, or training compute.

Canonical thesis:

> Nominal temporal retention does not identify the resources consumed by a
> sequential robot imitation learner. In windowed, chunked, and remapping-based
> training, the same retention ratio can induce different whole-run support,
> target exposure, replay concentration, dense-data access, and compute.
> Resource-specific counterfactual evaluation is therefore necessary to
> attribute closed-loop gains to data selection rather than training schedule.

Chinese version:

> 名义时间保留比例不能唯一决定时序机器人模仿学习实际消耗的资源。相同
> retention ratio 可能对应不同的训练可访问支持、实际监督曝光、重放集中度、
> 完整数据访问和计算成本。因此，只有在明确的等曝光、等平均重放或等训练计算
> 协议下，才能判断闭环收益究竟来自数据选择还是训练调度。

Formal framing:

```text
q_retain does not identify B

B = (
  U_ret,
  U_eligible,
  U_seen,
  E_A,
  {n_i},
  R_bar,
  K_train,
  K_curate
)
```

Definitions:

- `U_ret`: action-target support in the retained manifest.
- `U_eligible`: action-target support theoretically accessible during the whole
  run.
- `U_seen`: support that actually enters effective loss.
- `E_A`: total effective action-target exposure.
- `n_i`: exposure count for each action target.
- `R_bar = E_A / U_seen`: mean replay multiplier.
- `K_train`: training cost.
- `K_curate`: curation cost.

Use the phrase `resource-specific matched protocols`. Do not imply that all
resources can be matched simultaneously.

## Method Decomposition

Use this as the stable theoretical interface:

```text
Temporal Curation Method = S + C
```

where:

- `S`: which temporal support is selected. This includes selector logic,
  retained manifest, support size, and nominal retention ratio.
- `C`: how the training run consumes that support. This includes remapping,
  sampling probability, warmup, dense anchors, replay, optimizer steps, schedule,
  and any other training-time access rule.

The lineage operator maps the selection rule and consumption protocol into the
realized resource vector:

```text
B = L(S, C) = (
  U_ret,
  U_eligible,
  U_seen,
  E_A,
  {n_i},
  R_bar,
  K_train,
  K_curate
)
```

Closed-loop performance should be treated as:

```text
J = F(S, C, B, theta_0, E)
```

where `theta_0` is the initial policy checkpoint and `E` is the evaluation
environment/protocol.

The retention ratio `q` is only a coarse property of `S`. It does not identify
`C`, `B`, or final closed-loop performance.

This decomposition is the paper's identification discipline:

- do not choose resource definitions after seeing results;
- do not choose checkpoints after seeing FrameSkip-style performance;
- do not interpret higher success as selector quality without counterfactuals
  that separate `S` from `C`;
- do not claim a relation is matched unless the lineage/resource ledger verifies
  it.

## Collaboration Contract

Role split:

- Codex is responsible for code implementation, training, evaluation, raw result
  generation, logs, Resource Cards, and abnormal-run records.
- The researcher is responsible for the research system, theory, experimental
  design, controls, pre-registration rules, result interpretation, paper
  narrative, and reviewer defense.

Required handoff artifacts from Codex:

- after tracer implementation: field definitions, synthetic test results, and a
  100-200 step Resource Card;
- before formal training: Full learning curve and checkpoint/resource evidence
  needed to freeze the low-budget endpoint, retention ratio, seeds, main
  contrasts, and interpretation rules;
- after each experiment batch: per-seed Resource Cards, per-task rollout
  results, training cost, curation cost, and abnormal-result notes.

The researcher decides whether the artifacts support paper claims and which
next experiment has the highest information value.

## Claims

### Claim 1: Retention And Run-Level Usage Are Not Equivalent

Show that the same nominal retained ratio does not determine:

```text
U_eligible
U_seen
E_A
K_train
```

Evidence is deterministic measurement, not a closed-loop result:

- implement a canonical raw-time lineage tracer;
- generate Resource Cards for Full, hard-prune, remap, and FrameSkip-style runs;
- report retained, eligible, and seen support separately;
- report exposure totals, mean replay, and exposure distribution;
- report training and curation compute.

### Claim 2: Support Breadth And Exposure Depth Have Control Meaning

Use a minimal `U_A x E_A` mechanism matrix:

| ID | Support | Exposure | Purpose |
| --- | --- | --- | --- |
| `F-L` | full support | low | low-exposure full-support anchor |
| `F-H` | full support | high | high-exposure full reference |
| `R-L` | random hard-pruned support | low | low-exposure small-support anchor |
| `R-H` | random hard-pruned support | high | high-exposure small-support anchor |

Key contrasts:

```text
S(F-H) - S(F-L)
S(R-H) - S(R-L)
S(F-H) - S(R-H)
```

This calibrates whether accounting axes matter for closed-loop control. It is
not a hard go/no-go test for FrameSkip: no significant Random interaction does
not prove importance-selected support cannot interact with replay.

### Claim 3: FrameSkip-Style Gains Need Factor Decomposition

One counterfactual cannot separate multiple mechanisms. The minimum split is:

- `FS-O`: original FrameSkip-style schedule.
- `FS-S`: strict retained-support run, no dense full-data access.
- `FS-R`: same retained support, replay/mean-exposure reduced toward the full
  reference level.
- `RR-O`: random-retention version of the original schedule.

Selection quality:

```text
Delta_selection_orig = S(FS-O) - S(RR-O)
```

`RR-O` and `FS-O` must match retention ratio, remapping, warmup, full-frame
anchors, exposure, and optimizer steps. The main intended difference is random
selection versus importance selection.

Strict selection:

```text
Delta_selection_strict = S(FS-S) - S(Random-Hard-H)
```

Dense-data access schedule bundle:

```text
Delta_dense = S(FS-O) - S(FS-S)
```

Use the term `dense-access schedule bundle`, because this comparison does not
separate warmup from periodic anchors.

Additional exposure on retained support:

```text
Delta_exposure = S(FS-S) - S(FS-R)
```

Do not call this a pure replay causal effect. Training steps, compute, and
optimizer trajectory also change.

## Main Experiment Matrix

Default main budget:

```text
q = 20%
Full reference = 20k optimizer steps
```

| ID | Support | Dense access | Target exposure | Training implementation |
| --- | --- | --- | --- | --- |
| `F-L` | 100% | N/A | about 20% | `Full-20k` checkpoint at 4k |
| `F-H` | 100% | N/A | 100% | `Full-20k` final checkpoint |
| `R-L` | 20% random hard-pruned | no | about 20% | `Random-Hard-20k` checkpoint at 4k |
| `R-H` | 20% random hard-pruned | no | 100% | `Random-Hard-20k` final checkpoint |
| `RR-O` | nominal 20% random | yes | 100% | original remap schedule |
| `FS-O` | nominal 20% importance | yes | 100% | original FrameSkip-style schedule |
| `FS-S` | strict 20% importance | no | 100% | strict support trained to 20k |
| `FS-R` | strict 20% importance | no | about 20% | `FS-S` checkpoint at 4k |

Training reuse:

- `F-L` and `F-H` come from one Full training run.
- `R-L` and `R-H` come from one Random-Hard training run.
- `FS-R` and `FS-S` come from one strict FrameSkip training run.
- Per training seed, the required full training runs are `Full-H`,
  `Random-Hard-H`, `RR-O`, `FS-O`, and `FS-S`.

Budget switch rule must be pre-registered from the Full learning curve only:

- If `Full-4k` success is in `60%-85%`, use `q=20%` and `4k/20k`.
- If `Full-4k` is below `60%` or extremely unstable, switch to `q=50%` and
  `10k/20k`.
- Do not inspect FrameSkip results before choosing the main budget.

Existing `Random50-10k/20k` can remain as a separate `q=50%` mechanism
experiment, but the main matrix should use one consistent budget.

## Measurement Checklist

Lineage correctness:

- lineage specification complete;
- canonical action ID uses absolute target timestep;
- requested and remapped anchors are recorded;
- absolute context timestamps are recorded;
- absolute action-target timestamps are recorded;
- valid mask and loss weights are recorded;
- DDP aggregation is correct;
- warmup, full, and pruned batches are distinguishable;
- synthetic overlap, remap, and padding tests pass;
- tracer does not alter batch order or training results.

Existing Full audit:

- name `320k/640k` as global sample draws, not unique support or exposure;
- compute actual `E_A = sum(mask * weight)`;
- verify whether `Full-10k` is an intermediate checkpoint of `Full-20k`;
- freeze scheduler and warmup state;
- mark unrecoverable historical realized support as `unavailable` or
  `estimated`.

FrameSkip-style port:

- freeze reference code commit;
- write an Original-vs-Port comparison table;
- define exactly what is remapped;
- define warmup steps;
- define the 5:1 mixed schedule;
- freeze retained manifest and SHA256;
- reuse the same manifest across all deterministic FS counterfactuals;
- run a 100-200 step dry run;
- verify full/pruned batch ratio;
- verify action chunks are aligned;
- verify no cross-episode target leakage;
- call it `FrameSkip-style port` if any item cannot be reproduced exactly.

Experimental control:

- same base checkpoint hash;
- same optimizer;
- same 20k scheduler;
- low points are early checkpoints from the same schedule;
- same global batch;
- same precision;
- same evaluation initial states;
- no result-dependent checkpoint selection;
- Random uses multiple subset seeds;
- deterministic FrameSkip uses one manifest and multiple training seeds;
- core conditions use at least 2-3 training seeds.

Closed-loop statistics:

- use a non-ceiling endpoint for the main result;
- `10 episodes/task` is screening only;
- upgrade core conditions to `50 episodes/task`;
- report task-macro, per-suite, and per-task success;
- use paired task-stratified bootstrap;
- report training-seed variance;
- report failure-stage breakdown;
- report paired-difference 95% confidence intervals.

Resource accounting:

- `U_ret`;
- `U_eligible`;
- `U_seen`;
- `E_A`;
- mean replay;
- exposure p10/p50/p90/p99;
- zero-exposure target ratio;
- normalized effective support `N_eff / U_A`;
- training GPU-hours;
- curation GPU-hours;
- wall-clock;
- valid targets/s;
- peak VRAM;
- whether the run still depends on complete raw data access.

`U_W` may be retained as an auxiliary lineage metric, but it is not a main
conclusion axis.

## Contributions

1. Resource semantics: temporal retention is not a sufficient statistic for
   data or compute usage in sequential robot imitation learning.
2. Canonical temporal lineage: requested and remapped anchors are traced
   through observation contexts, absolute action targets, valid loss masks, and
   distributed batch sampling.
3. Resource-specific counterfactual protocols: exposure-matched,
   mean-replay-matched, and training-compute-controlled evaluations.
4. Closed-loop attribution: apparent FrameSkip-style benefits are decomposed
   into selection quality, dense-data access, and additional optimization
   exposure.
5. Reporting standard: future temporal curation work should report nominal
   retained ratio, retained support, whole-run eligible support, whole-run seen
   support, realized target exposure, mean replay and concentration, training
   compute, curation compute, and closed-loop uncertainty.

## Pre-Registered Interpretation

Strong result:

- `FS-O` beats Full or `RR-O`;
- `FS-S` is much weaker than `FS-O`;
- `FS-R` drops further;
- method ranking changes after resource matching.

Interpretation: original gains depend heavily on dense-data access and
concentrated replay; they cannot be attributed only to a better retained 20%
support.

Moderate result:

- `FS-O` is close to `FS-S`;
- `FS-S` beats `Random-Hard-H`;
- `FS-R` still beats `F-L` or `R-L`.

Interpretation: importance-selected support has independent value under strict
resource limits, while nominal retention still fails to describe actual
resource usage.

Interaction result:

- `FS-O` beats `RR-O`;
- `FS-S` is close to `Random-Hard-H`;
- selection helps only with full warmup or anchors.

Interpretation: selection quality and dense-data schedule interact strongly;
curation cannot be evaluated apart from its consumption protocol.

Weak result:

- Resource Cards differ clearly;
- all closed-loop differences are inside confidence intervals;
- method conclusions do not change.

Interpretation: only Claim 1 is supported. Under a single model and benchmark,
this is not enough for a strong ICRA submission without adding a second policy
or benchmark.

## Submission Success Standard

Before submission, the project should satisfy:

- clear, reproducible mismatch between retention and whole-run usage;
- at least one counterfactual changes closed-loop performance beyond the
  confidence interval;
- at least two of selection, dense access, and additional exposure can be
  distinguished;
- the main evaluation is not dominated by ceiling effects;
- core findings are consistent across training seeds;
- every matched relation is verified by the ledger, not inferred from steps;
- FrameSkip port deviations are fully disclosed;
- the paper claim still holds if the name `FrameSkip` is removed.

Core sentence:

> Temporal curation has two inseparable-looking but experimentally separable
> components: which temporal support is retained, and how that support is
> consumed during optimization. Retention-only evaluation conflates the two.
