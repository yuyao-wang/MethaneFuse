# MethaneFuse pretraining research log — 2026-07-27

This directory contains the reproducible manifests, launchers, audits, and
results for the ten-hour pretraining study.

## Fixed protocol

- The four original held-out test CSVs are never used for checkpoint selection.
- Validation events are the most recent events carved from each original
  training split.
- Splits are grouped by a canonical Carbon Mapper event key.
- Exploration training is capped per `event × label`; validation and final test
  rows remain uncapped.
- Primary model-selection metrics are validation AP and positive-class F1.
  AUROC and recall at fixed FPR are reported as supporting metrics.
- Every joint-sensor experiment must use one global event assignment. Per-sensor
  temporal splits may only be used for independent sensor experiments.

## Storage

Large manifests, local image caches, checkpoints, and logs live under:

```text
/diniuvol/yuyao/methanefuse_research_20260727/
```

The repository only keeps code and compact summaries.

Prepared sensor manifests are in
`/diniuvol/yuyao/methanefuse_research_20260727/manifests` and their rewritten
local-cache copies are in `manifests_staged`.  The controlled train/validation
row counts are:

| Sensor | Event-capped train | Recent event-held-out validation |
|---|---:|---:|
| S2 | 18,747 | 16,546 |
| L89 | 10,049 | 9,614 |
| EMIT32 | 11,688 | 5,201 |
| S5P | 18,772 | 3,805 |

Every train/validation overlap audit is zero.  Original test manifests remain
separate and are used only after model and input selection.

The production unified source index is
`manifests/multisensor_6time_512_wide.csv` (SHA-256
`2ada7e41e397689710440d242a1975271e3bb0c5bd2cc260122170da12bf0703`).
It contains 19,834 plume rows and 7,108 canonical event groups, assigned once
to train/validation/test (12,131/1,895/5,808 rows; zero event overlap).  Only
27 rows contain all four sensors, so paired training must support partial
sensor sets rather than restricting the dataset to complete four-sensor rows.

## Canonical S2 masks

S2 v14 image TIFFs are array containers with identity raster metadata, so their
metadata must not be used as a geographic grid.  The canonical mask repair
uses each plume's validated 512 × 512 exact-v3 georeference sidecar, then
reprojects the local Carbon Mapper binary footprint to that 20 m grid.  The
production command is dry-run-first, atomic, no-overwrite by default, and
rejects empty outputs.

Nearest-neighbour reprojection succeeded for 4,090 of 4,097 S2 rows.  The
remaining seven were verified as genuine sub-pixel cases: their non-empty
0.6–1.2 m raw positive pixels all lay inside the target grid, but no 20 m
nearest-neighbour sample landed on a positive source pixel.  The explicit
opt-in `--subpixel-point-fallback` is used only when nearest reprojection is
empty.  It marks a target cell only when the centre of a real positive source
pixel falls in that cell.  It performs no dilation and no polygon/all-touched
area fabrication.  The seven cases map to 3–6 positive target cells each.

The final cache contains 4,097 masks (33 MiB):

```text
/diniuvol/yuyao/methanefuse_research_20260727/cache/s2_512_masks
```

Full readback QA passed 4,097/4,097 masks: one 512 × 512 `uint8` band, binary,
non-empty, exact expected CRS/transform, and positive bounds overlapping both
the raw Carbon Mapper mask and catalogue plume bounds.  Reproducible records:

```text
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_repair_production.csv
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_repair_production.audit.json
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_subpixel_fallback.csv
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_subpixel_fallback.audit.json
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_final_readback.csv
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_final_readback.audit.json
```

## Unified crop planning smoke

`unified_crop_orchestrator.py` is dry-run-first and currently materializes task
metadata only.  Each task inherits the one global event split, stores one
canonical WGS84 query centre plus local east/north metre offsets, supports
partial sensor sets in `S2,L89,EMIT,S5P` bit order, and carries all six source
timepoints (`t0,prev1,prev2,prev3,seasonal,year`).  The Carbon Mapper footprint
is the sole label authority.  Crops use the real t0 image grid when available;
identity-array S2 falls back to its repaired canonical mask grid.  Sensor masks
are mapped independently at the same longitude/latitude for diagnostics and do
not vote on labels.  This avoids the observed L89 mask/image half-pixel
(15 m) transform offset.

S5P execution never uses the source plume-centre row/column as the crop
location.  For every query and every timepoint it reads that NetCDF's own
latitude/longitude arrays and recomputes the nearest valid pixel.  The current
environment uses the `h5py` backend; the source QC indices remain in the task
row only for auditing.

The small smoke selected one `S2|L89` plume and one
`S2|L89|EMIT|S5P` plume.  Planning produced four deterministic tasks (two
positive and two negative), zero event leakage, and no dropped sensors.
In-memory readback checked one positive task per sensor combination across six
times without writing crop files:

- `S2|L89`: `ok`; both raster finite-data masks were `111111`.
- `S2|L89|EMIT|S5P`: explicit `warning`; S2 and L89 were `111111`, while
  EMIT was `010011` because `t0`, `prev2`, and `prev3` were all-NaN at this
  query.  All six S5P query pixels were independently recomputed.

Source-file presence and query-level finite-data presence are separate fields.
The warning is intentionally retained for the future executor/missing-time
policy; no full crop was launched.

The follow-up
[`MANIFEST_V2_PROVENANCE_PILOT.md`](MANIFEST_V2_PROVENANCE_PILOT.md) keeps
production crop explicitly blocked.  The current metadata is missing 301 of
410 required v2 fields, including verified observation keys, deterministic
alias/quarantine state, query-usability and unique-time masks, per-channel
validity, and materialized readback hashes.  It also detects a known EMIT
`prev3` embedded-product conflict.  Production materialization may begin only
after the v2 execution fields, provenance repair/quarantine, stratified
metadata/header pilot, small query-coverage pilot, and readback gates pass.

```text
/diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.csv
/diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.schema.json
/diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.audit.json
/diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.validation.csv
```

## Controlled shared-MAE results

Both leakage-safe, same-seed one-epoch MAE promotion screens are complete.
They use the globally purged four-sensor protocol and the same matched scratch
control; neither uses sealed-test data:

| Pretraining objective | Integrity checks | Macro AP, scratch -> finetune | Per-sensor AP delta, S2 / L89 / EMIT / S5P | Macro best-F1 delta | Decision |
|---|---:|---:|---:|---:|---|
| Unmasked residual MAE | 21/21 | 0.620981 -> 0.612899 (-0.008082) | -0.027336 / -0.027591 / +0.021344 / +0.001256 | +0.005533 | `do_not_promote` |
| Validity-masked residual MAE | 42/42 | 0.620981 -> 0.611177 (-0.009804) | -0.025246 / -0.033357 / +0.014451 / +0.004937 | +0.001394 | `do_not_promote` |

Thus validity masking did not rescue classification transfer: EMIT and S5P AP
rose slightly, while S2 and L89 fell enough to reduce macro AP.  These are
credible negative validation results, not evidence that shared MAE improves
four-sensor classification.

Primary result artifacts:

```text
/diniuvol/yuyao/methanefuse_research_20260727/results/mae_vs_scratch_gate.json
/diniuvol/yuyao/methanefuse_research_20260727/results/validitymasked_mae_vs_scratch_gate.json
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_validitymasked_e1_seed20260727/metrics_history.json
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_validitymasked_finetune_e1_seed20260727/metrics_history.json
```

The full tables, hashes, validity diagnostics, and gate interpretation are in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); claim boundaries are in
[`RESEARCH_STORY.md`](RESEARCH_STORY.md).

## Controlled supervised RankNet result

After both MAE promotion gates failed, the one authorized fallback changed
only the supervised objective from weighted BCE to weighted BCE plus
within-sensor all-pairs RankNet (`weight=0.5`, `temperature=1.0`). This was a
scratch-trained, one-epoch supervised experiment—not pretraining—and it did
not read sealed-test data. It used the same global-purge protocol,
normalization, initialization, architecture, seed, 157 balanced rounds, 628
updates, and full four-sensor validation as the matched BCE scratch control.
All 20/20 integrity checks passed, and all 628 sensor batches contained both
classes.

| Metric | BCE scratch | BCE + RankNet | Delta |
|---|---:|---:|---:|
| Macro AP | 0.620981 | 0.633489 | **+0.012508** |
| Macro AUROC | 0.646924 | 0.660072 | **+0.013147** |
| Macro best-F1 | 0.650817 | 0.659562 | **+0.008745** |
| Macro F1@0.5 | 0.482741 | 0.582795 | **+0.100053** |

Per-sensor AP/AUROC deltas were positive for all four sensors:

| Sensor | AP delta | AUROC delta | Full-temporal AP minus current-information-only AP |
|---|---:|---:|---:|
| S2 | +0.007199 | +0.008386 | +0.005282 |
| L89 | +0.006351 | +0.004033 | +0.007437 |
| EMIT | +0.026974 | +0.027486 | +0.011179 |
| S5P | +0.009509 | +0.012684 | +0.003385 |
| **Macro** | **+0.012508** | **+0.013147** | **+0.006821** |

This is a supervised ranking-aware **partial success**, not a promoted result.
The formal decision is `do_not_promote`: although S5P probabilities were
nonconstant, AP was above prevalence, and AUROC was above 0.5, its best-F1
threshold was `0.0` and predicted all `3,805/3,805` validation rows positive.
The preregistered S5P non-degeneracy criterion therefore failed. Do not tune
RankNet weight/temperature, add epochs or seeds, describe it as pretraining,
or claim that S5P classification has been solved.

Authoritative paths and SHA-256:

| Artifact | SHA-256 |
|---|---|
| `/diniuvol/yuyao/methanefuse_research_20260727/results/ranknet_vs_bce_gate.json` | `c36e007a6e03025b4bf54195143d0876e96a9e9376c75e7b53634d29552cb6fc` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_scratch_globalpurge_e1_seed20260727/metrics_history.json` | `f278d859eee6c8c36c56f295ab34f1245700f60530ef11d86fc104c27d77c5db` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/metrics_history.json` | `0923eb5525d3db5d840a8666465dfd2159797183614bd080fbb923b7980bf1a9` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/checkpoint_latest.pth` | `ef070e6f47abaa2a8401203da2222365c51a913fe97b310d59784bb9c648f683` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/multisensor_residual_runner.launch_4a2c54aebf2e.py` | `4a2c54aebf2ec957a4ef0b1aa85368092b50e79ddbc13adff3e3b1292caa9fed` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_full_temporal.json` | `ca3066b996ecef6cb17b37786624282f9e1b993069380f10cbf1f4637145c054` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_full_temporal.csv` | `26508dc2da884e3835df0f1e55f4f6a4e3d6582bcd2ffc6f8913a29a154baa92` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_current_only.json` | `e34ca0c047adac56c3b8b720369eb5229f8a54ba1f38c67416e3f0e1d1ada7bf` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_current_only.csv` | `bab1dd6ba196e931648944688e507db1e6cf1bc0eab40a613d1b228fc334b13b` |

## Latest temporal finding: current-conditioned history, not reconstruction

The full research interpretation and manuscript-ready abstract are in
[`JOURNAL_STORY_20260727.md`](JOURNAL_STORY_20260727.md).  The concise result
is:

- three controlled pretraining/readout routes were negative:
  shared MAE (`-0.0081` to `-0.0098` macro AP), L89 background-innovation
  readout (`-0.0534 AP` innovation-only), and a converged label-free temporal
  correspondence pretext (`-0.0229 AP` at 100% labels; `-0.0466` at 10%);
- a parameter-matched current-query head using unique, role-tagged matched
  history improved three-seed L89 AP by `+0.0409` and EMIT AP by `+0.0232`
  versus current-only;
- cross-event history shuffling reduced AP by `0.0469` on L89 and `0.0274`
  on EMIT, while continuous `Δdays` did not beat the discrete-role model;
- a native-grid S5P approximation reproduced only a directional `+0.0120 AP`
  result and is not a headline claim.

The supported story is an objective–target mismatch: reconstruction,
background prediction, and same-location correspondence favor persistent
scene content, whereas methane is a weak current-time transient.  The
provisional method, **TransientQuery**, preserves t0 as the only decision
query and uses de-duplicated historical acquisitions only as conditional
context.  This is currently a frozen-feature temporal-head result, not a
successful foundation-model pretraining or four-sensor SOTA claim.

Formal result roots:

```text
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260727
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260728
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260729
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260727
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260728
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260729
/diniuvol/yuyao/methanefuse_research_20260727/experiments/l89_innovation_v1b
/diniuvol/yuyao/methanefuse_research_20260727/experiments/l89_correspondence_v1_seed20260727
```

Conservatively, S2 is excluded from the latest temporal comparison.  An early
read-only search displayed rows from a different all-rows source containing
`test` markers; the target temporal test manifest was not opened and no value
was used, but clean 512-source recropping and a fresh split are required before
S2 re-enters model selection.

## Experiment stages

1. Establish short-run Panopticon and residual-input baselines on exact
   event-held-out validation sets.
2. Compare independent sensor training with a shared backbone plus
   sensor-specific normalization/adapters/heads.
3. Screen suitable public EO foundation checkpoints.
4. Once the aligned 512-source recrop is ready, test sensor/time dropout and
   full-to-subset distillation.

The shared official-backbone prototype, exact input table, parameter counts,
30-round gate, checkpoint semantics, and reproduction commands are documented
in
[`MULTISENSOR_PANOPTICON_README.md`](MULTISENSOR_PANOPTICON_README.md).

## Two-phase sealed-test gate

For L89, EMIT32, and S5P Panopticon runs, sealed evaluation uses
`evaluate_panopticon_sealed.py` as a mandatory two-phase gate:

1. `freeze-threshold` accepts one explicit checkpoint, verifies that its epoch
   is a validation-AP maximum in the adjacent `metrics_history.json`, verifies
   the checkpoint's saved train/validation manifests and zero canonical-event
   overlap, then re-infers the **complete validation set**. It freezes the
   positive-class-F1 threshold, normalization values, checkpoint SHA-256,
   trainer-code SHA-256, input mode, path/time columns, and model schema in a
   new JSON artifact. This handles L89 runs whose historical metric records
   did not contain an optimized threshold. For an existing S5P run whose
   training-time normalization was not saved, the deterministic train-only
   computation is reproduced from the checkpoint arguments and its exact
   values are frozen in the artifact.
2. `evaluate-sealed` requires that artifact and the same explicit checkpoint.
   It exposes no threshold argument or checkpoint search. Before inference it
   verifies all hashes/schema and rejects any canonical plume event shared
   with train or validation. AP, AUROC, fixed-threshold positive F1,
   macro-F1, recall, FPR, and recall at 1%/5% FPR are then reported.

Both phases bypass the temporal datasets' retry-and-random-replacement
`__getitem__` behavior: a missing or corrupt row aborts the run, so prediction
count and CSV row count remain exactly aligned. Outputs are atomic and never
overwritten. A non-writing local setup check is:

```bash
python research/pretraining_20260727/evaluate_panopticon_sealed.py smoke \
  --sensor l89 \
  --checkpoint /path/to/explicit/ckpt_best_val_ap.pth \
  --validation_csv /path/to/staged/val.csv \
  --rows 3
```

After final model selection, but before opening a sealed manifest:

```bash
python research/pretraining_20260727/evaluate_panopticon_sealed.py freeze-threshold \
  --sensor l89 \
  --checkpoint /path/to/explicit/ckpt_best_val_ap.pth \
  --validation_csv /path/to/staged/val.csv \
  --output_artifact /path/to/new/l89_frozen_threshold.json

python research/pretraining_20260727/evaluate_panopticon_sealed.py evaluate-sealed \
  --sensor l89 \
  --checkpoint /path/to/explicit/ckpt_best_val_ap.pth \
  --threshold_artifact /path/to/l89_frozen_threshold.json \
  --eval_csv /path/to/staged/sealed_test.csv \
  --output_json /path/to/new/l89_sealed_result.json \
  --predictions_csv /path/to/new/l89_sealed_predictions.csv
```

The same commands use `--sensor emit32` or `--sensor s5p`. Threshold artifacts
must be frozen only after the final validation-based checkpoint decision; the
original test CSVs remain unopened until that point.
