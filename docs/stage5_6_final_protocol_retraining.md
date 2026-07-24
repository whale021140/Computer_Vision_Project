# Stage 5.6 Final Unified Protocol Retraining

Stage 5.6 is the final protocol used for the main few-shot representation
comparison. It replaces the Stage 5 and Stage 5.5 procedures in the final main
table while retaining those earlier artifacts as development history.

The purpose of this reset is to make the complete experiment internally
consistent: every architecture decision, checkpoint choice, and inference
threshold is made on one image-disjoint development set that contains
no-target, single-target, two-target, and three-or-more-target expressions.
Official testA and testB provide the final Stage 5.6 benchmark. An initial
narrow-calibration pass is archived, so the report does not claim that these
files were never previously inspected. The precise claim is that neither its
metrics nor any earlier test metric is used to choose a recipe, checkpoint,
calibration range, calibration value, or retraining decision.

## Locked data protocol

The source is the complete official gRefCOCO training partition:

| Group | Full-train expressions |
|---|---:|
| 0 targets | 19,140 |
| 1 target | 120,624 |
| 2 targets | 67,848 |
| 3+ targets | 1,732 |
| Total | 209,344 |

The development split is selected by whole image with minimum requested
coverage `0/1/2/3+ = 1500/1500/1500/300`. Selecting a development image moves
all of its expressions into development, preventing image leakage. The realized
split contains 12,249 expressions from 854 images:

| Group | Development expressions |
|---|---:|
| 0 targets | 1,500 |
| 1 target | 6,707 |
| 2 targets | 3,654 |
| 3+ targets | 388 |

The primary development criterion is the unweighted average of mean F1 over the
four target-count groups. Consequently, the natural excess of single-target
expressions does not give that group extra weight during model selection.
Official F1 is the first tie-break.

Development images are removed before sampling training data. Within each seed,
the splits are count-stratified and strictly nested (`1% ⊆ 5% ⊆ 10%`). Their
label budgets are computed from the full official training partition with
largest-remainder apportionment:

| Fraction | 0 | 1 | 2 | 3+ | Total |
|---|---:|---:|---:|---:|---:|
| 1% | 191 | 1,206 | 679 | 17 | 2,093 |
| 5% | 957 | 6,031 | 3,392 | 87 | 10,467 |
| 10% | 1,914 | 12,062 | 6,785 | 173 | 20,934 |

The same group budget is used for seeds 0, 1, and 2. Rare 3+ examples therefore
cannot disappear or fluctuate merely because of random subset construction.
The remaining low absolute count at 1% is a genuine consequence of the
few-shot budget and will be reported rather than hidden through unreported
oversampling.

Exact split hashes, integrity checks, and the locked policy are recorded in:

- `outputs/stage5_6/split_manifest.json`;
- `outputs/stage5_6/protocol_lock.json`.

## Locked model-selection protocol

Five SigLIP 2, 10%, seed-0 pilots are rerun from scratch on the new splits:

1. selection-only with the original head and manual count weights;
2. effective-number count balancing;
3. hierarchical presence/positive-count prediction with richer pooling;
4. one-to-one positive proposal labels;
5. the combined hierarchical, balancing, and one-to-one recipe.

The recipe with the best calibrated development count-macro mean F1 is selected.
The tie-breaks are official development F1 and then the earlier declared recipe
order. No official test result participates.

The winning recipe is trained from scratch for:

```text
representations = {CLIP, CLIP+DINOv2, SigLIP 2}
fractions       = {1%, 5%, 10%}
seeds           = {0, 1, 2}
```

Each of the 27 runs uses 40 epochs, cosine learning-rate decay, the same frozen
Faster R-CNN proposals, and the same optimization settings. Every epoch is
evaluated on the new development set. The selected epoch and the class-0/class-3
logit biases plus membership threshold are chosen only on that set.

The first calibration pass used an inadequately narrow grid. Before the
corrected test pass, a protocol revision expanded the development-only search
without changing any checkpoint or training weight. The original narrow test
metrics were not consulted when defining or selecting the wider grid. Boundary
checks were repeated before revised test access. The final common `v2_wide`
grid is:

```text
class-0 bias:  -1.0 to 16.0, step 0.5
class-3 bias:   0.0 to 32.0, step 0.5
membership threshold:
  0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
  0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0
total: 45,500 settings per checkpoint
```

The expansion was driven only by development-set boundary behavior. Runs were
stopped before revised test evaluation whenever a bias or a truncatable
threshold boundary was selected. The final audit validates 31/31 unique pilot
and final calibrations with no truncated optimum. The original narrow
calibrations and tests remain archived under `*_v1_narrow`; the corrected
`v2_wide` results are canonical.

The old gRefCOCO validation and RefCOCO UNC auxiliary validation files are no
longer needed to complete the model-selection logic. They may remain as
diagnostic artifacts, but neither is combined with the new development metric
or used as a second selection authority.

## Final evaluation gate

The test runner refuses to run until all 27 selected checkpoints and all
required pilot/final development calibrations exist, and until the 31-unique-
artifact boundary audit passes. It then evaluates exactly one selected policy
per cell on full official testA and testB, producing 54 evaluations. The main
report uses mean and sample standard deviation across the three paired seeds,
target-type and `0/1/2/3+` breakdowns, and same-seed paired representation
differences.

The gate passed with 31/31 unique calibrations and all 54 evaluations completed.
The final F1 results are:

| Split | Representation | 1% | 5% | 10% |
|---|---|---:|---:|---:|
| testA | CLIP | 0.2496 | 0.3268 | 0.3605 |
| testA | CLIP+DINOv2 | 0.2402 | 0.2911 | 0.3438 |
| testA | SigLIP 2 | **0.2769** | **0.4151** | **0.4625** |
| testB | CLIP | 0.2656 | 0.3039 | 0.3430 |
| testB | CLIP+DINOv2 | 0.2528 | 0.2839 | 0.3288 |
| testB | SigLIP 2 | **0.2749** | **0.3722** | **0.4180** |

Every F1 and mean-F1 curve increases from 1% to 5% to 10%.

## Reproduction

All commands run in `ece485`:

```bash
bash scripts/run_stage5_6_prepare.sh
bash scripts/run_stage5_6_pilots.sh
bash scripts/run_stage5_6_grid.sh
bash scripts/run_stage5_6_recalibrate_v2.sh
```

The revised wrapper performs final recalibration, the boundary audit, all 54
test evaluations, and aggregation. `scripts/run_stage5_6_test.sh` remains the
gated test-only entry point and refuses to run without a passing v2 audit.

The first three training phases can also be invoked with the resumable wrapper:

```bash
bash scripts/run_stage5_6.sh
```

Frozen image-region shards from Stage 5 are reused only after checking encoder
identity and candidate boxes. New expressions are re-encoded, and images not
covered by the old union are encoded normally. Old head checkpoints are never
reused.
