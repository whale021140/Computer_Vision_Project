# Stage 7: Final Deliverables and Reproducibility

> Status: complete  
> Environment: `ece485`  
> Frozen input commit: `bf72502`  
> Test policy: read existing Stage 5.6/6 results only; do not rerun or tune on
> testA/testB.

## 1. Purpose

Stage 7 converts the completed experiments into a compact, auditable final
submission. It does not introduce a new model family or reopen the Stage 6 test
gate.

The final narrative keeps two result layers separate:

1. **Stage 5.6 main controlled study** answers the proposal's representation
   and supervision-scaling question with
   `3 representations x 3 fractions x 3 seeds`.
2. **Stage 6 final enhanced system** reports the strongest accepted SigLIP 2
   configuration with `lambda_cardinality=0.10`, pre-selection NMS, and three
   seeds.

Stage 5/5.5 remain development history. They are not mixed into either final
aggregate.

## 2. Deliverables

- [x] Machine-readable and Markdown main-result tables.
- [x] Supervision-scaling plot for testA/testB.
- [x] Stage 6 final-system comparison plot.
- [x] Final-system `0/1/2/3+` breakdown plot.
- [x] Proposal-recall and exact-count repair diagnostic plot.
- [x] Final English report mapping each proposal commitment to evidence.
- [x] Reproduction guide distinguishing lightweight report regeneration from
  expensive feature extraction and training.
- [x] Stage 7 manifest with source paths, SHA-256 hashes, environment versions,
  commands, and generated-output hashes.
- [x] Root README, documentation index, completion plan, and Chinese guide
  synchronized to the final accepted results.

## 3. Source-of-truth inputs

- `outputs/stage5_6/test_summary.json`
- `outputs/stage5_6/proposal_recall_feature_union.json`
- `outputs/stage3/oracle_vs_detector_val.json`
- `outputs/stage6/stage6_final_summary.json`
- `outputs/stage6/stage6_1_three_seed_confirmation.json`
- `outputs/stage6/stage6_2_input_ablation_summary.json`
- `outputs/stage6/candidate_cap_audit_dev.json`
- `outputs/stage6/multitarget_failure_diagnosis_dev.json`
- `outputs/stage6/counterfactual_local_audit.json`
- `outputs/stage6/final_test_lock.json`

All Stage 7 tables and figures must be derivable from these frozen artifacts.
The build fails if expected cells are missing, non-finite, or inconsistent with
the locked Stage 6 configuration.

## 4. Reporting rules

- Stage 5.6 is the main representation comparison.
- Stage 6 is the final enhanced-system result.
- The Stage 6 gain over Stage 5.6 is a combined-system gain; it must not be
  attributed entirely to NMS or entirely to `lambda=0.10`.
- CLIP+DINOv2 is reported as a negative result for the tested concatenation,
  not as proof that DINOv2 can never help.
- `3/4/5/6+` is not described as solved.
- Missing local C-RefCOCO/FineCops-Ref annotations are reported as an
  availability limitation, not a negative benchmark result.
- RECANTFormer/HieA2G numbers may be discussed as background but not presented
  as a controlled ranking against this few-shot system.

## 5. Acceptance criteria

Stage 7 is complete when:

1. the deterministic Stage 7 build succeeds in `ece485`;
2. every generated numeric cell is traced to a hashed input artifact;
3. all figures can be regenerated without model checkpoints, feature caches,
   or test inference;
4. unit tests and `compileall` pass;
5. Markdown links and generated artifact paths pass a local audit;
6. the final report explicitly covers the proposal's required, conditional,
   optional, negative, and incomplete outcomes.

## 6. Main command

```bash
conda run --no-capture-output -n ece485 \
  python -m src.reporting.build_stage7_deliverables
```
