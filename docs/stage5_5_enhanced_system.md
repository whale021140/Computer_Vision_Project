# Stage 5.5 Enhanced System

Stage 5.5 is a transparent post-hoc repair study motivated by the locked Stage
5 diagnosis. Stage 5 remains the primary untouched-test representation grid;
Stage 5.5 does not replace or hide it. The enhanced protocol below was fixed
before producing any `outputs/stage5_5/*test*` result.

## Motivation

The current gRefCOCO val has no single-target expressions and only 28 `3+`
expressions, while the locked test splits contain thousands of both groups. In
the 10% training splits, the manual count weights give `3+` only about 0.6% of
the total weighted cardinality mass. The Stage 5 head also mean-pools all
candidates and trains every IoU-positive duplicate proposal as positive.

## Locked data protocol

- Source: the pre-test Stage 5 multi-seed train union only.
- Shadow-dev: deterministic seed 55, selected by whole image, with minimum
  count-group coverage 1,000 / 1,000 / 1,000 / 120 for `0/1/2/3+`.
- The realized shadow-dev has 4,138 expressions over 915 images and count-group
  counts 1,000 / 1,913 / 1,081 / 144.
- All shadow-dev images are removed before enhanced training sampling.
- Each enhanced train split has exactly 20,934 expressions with fixed counts
  1,914 / 12,062 / 6,785 / 173 and seed 0, 1, or 2.
- Frozen detector proposals and encoder features are reused; no test record is
  used for training, checkpoint selection, calibration, or pilot selection.

## Locked pilot protocol

The five SigLIP 2, 10%, seed-0 pilots are:

1. `selection_only`: historical head/labels/manual weights, but corrected
   shadow-dev selection and 40-epoch cosine schedule;
2. `balanced`: effective-number cardinality weights (`beta=0.9999`);
3. `hierarchical`: balanced loss plus hierarchical presence/positive-count
   heads using mean, max, and membership-statistic pooling;
4. `one_to_one`: balanced loss plus one-to-one candidate/target positives;
5. `combined`: hierarchical pooling and one-to-one positives together.

All use batch size 16, hidden size 256, dropout 0.1, AdamW learning rate and
weight decay `1e-4`, cardinality loss coefficient 1, and 40 epochs. Every epoch
is retained. Epoch selection maximizes the equal-weight macro of shadow-dev
mean F1 over `0/1/2/3+`, then official F1, then prefers the earlier epoch.

After selecting an epoch, calibration searches only this fixed grid:

- class-0 logit bias: `[-1.0, -0.5, 0.0, 0.5, 1.0]`;
- class-3 logit bias: `[0.0, 0.5, 1.0, 1.5, 2.0]`;
- `3+` membership threshold: `[0.3, 0.4, 0.5, 0.6, 0.7]`.

The calibration objective and tie-breakers are the same count-macro mean F1,
official F1, smaller absolute biases, threshold closest to 0.5, then the
lexicographically smaller setting.

The pilot variant with the highest calibrated shadow-dev count-macro mean F1 is
the final recipe. Ties use official F1, then the earlier variant in the declared
list. Every pilot result is retained whether or not selected.

## Locked final protocol

The selected recipe is applied without changes to CLIP, CLIP+DINOv2, and SigLIP
2 at 10% for seeds 0/1/2. The exact same per-seed split is used across
representations. Each cell selects and calibrates on shadow-dev only. All nine
selected models are then evaluated once on current gRefCOCO val, RefCOCO UNC
auxiliary val, full testA, and full testB. Main enhanced tables report mean and
sample standard deviation across seeds plus same-seed representation deltas.

Because Stage 5 test results were already observed before this repair was
designed, Stage 5.5 is explicitly labeled post-hoc even though its new choices
are test-independent. A negative or mixed enhanced result will be reported.

## Completed pilot

| Variant | Selected epoch | Uncalibrated count-macro F1 | Calibrated count-macro F1 | Calibrated official F1 |
|---|---:|---:|---:|---:|
| selection-only | 31 | 0.612951 | 0.615107 | 0.514983 |
| balanced | 31 | 0.600758 | 0.606305 | 0.511841 |
| hierarchical | 35 | **0.639108** | **0.644855** | **0.562349** |
| one-to-one | 22 | 0.598266 | 0.603985 | 0.495408 |
| combined | 36 | 0.625589 | 0.630399 | 0.531899 |

The locked winner is `hierarchical`. The result isolates the useful change:
richer pooling plus hierarchical presence/positive-count prediction improves
the balanced count-group objective, while effective-number weights and
one-to-one positives do not help on their own. Combining one-to-one labels with
the hierarchical head also trails the pure hierarchical recipe.

## Final results

Values are mean ± sample standard deviation across seeds 0/1/2. Stage 5 deltas
compare aggregate means and are not strictly paired because shadow-dev images
are excluded from the Stage 5.5 training splits.

| Split | Representation | F1_score | T_acc | N_acc | Mean F1 | Cardinality accuracy | ΔF1 vs. Stage 5 |
|---|---|---:|---:|---:|---:|---:|---:|
| testA | CLIP | 0.357292 ± 0.005288 | 0.943126 ± 0.011470 | 0.578162 ± 0.026420 | 0.508296 ± 0.003987 | 0.716076 ± 0.002222 | +0.027257 |
| testA | CLIP+DINOv2 | 0.342639 ± 0.000688 | 0.967349 ± 0.000978 | 0.494155 ± 0.001921 | 0.497556 ± 0.000542 | 0.690920 ± 0.007025 | +0.024826 |
| testA | SigLIP 2 | **0.447882 ± 0.001261** | **0.975574 ± 0.004003** | **0.579886 ± 0.004739** | **0.598878 ± 0.001570** | **0.738524 ± 0.002218** | **+0.036788** |
| testB | CLIP | 0.335824 ± 0.014634 | 0.892362 ± 0.032015 | **0.635281 ± 0.052854** | 0.446621 ± 0.012206 | 0.697483 ± 0.002095 | **+0.041877** |
| testB | CLIP+DINOv2 | 0.306854 ± 0.002796 | 0.946708 ± 0.003280 | 0.482845 ± 0.004758 | 0.422452 ± 0.001020 | 0.662911 ± 0.006724 | +0.025483 |
| testB | SigLIP 2 | **0.395339 ± 0.002472** | **0.947088 ± 0.004987** | 0.600114 ± 0.002462 | **0.511189 ± 0.002792** | **0.730540 ± 0.001420** | +0.033887 |

The same-seed SigLIP 2 gain over CLIP is `+0.090590 ± 0.004542` on testA
and `+0.059516 ± 0.013122` on testB. CLIP still exceeds CLIP+DINOv2 by
`+0.014653 ± 0.005925` and `+0.028969 ± 0.012490`, respectively. The repair
therefore improves every representation but does not change their ordering.

The current gRefCOCO val has no count-1/single-target records, whereas RefCOCO
UNC auxiliary val contains only that group. The machine-readable summary marks
those structurally absent groups as unavailable instead of treating their
zero-valued evaluator placeholders as measured performance.

## Acceptance and artifacts

Stage 5.5 is complete: 9/9 selected checkpoints, 9/9 selection/calibration
records, and 36/36 external evaluations are present. An audit of 936 numeric
official/diagnostic values found every value finite. The primary aggregate
artifacts are:

- `outputs/stage5_5/pilot_selection.json`;
- `outputs/stage5_5/summary.json`;
- `outputs/stage5_5/summary.txt`;
- `outputs/stage5_5/protocol_lock.json`.
