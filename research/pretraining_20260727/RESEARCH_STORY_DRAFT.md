# MethaneFuse Journal research story draft

**Working title**

> **Pretraining for What the Eye Cannot See: Sensor-Response
> Counterfactual Learning for Transient Methane Detection**

**Status:** mechanism-supported RCTP proposal with a completed legacy
downstream engineering control; not yet an RCTP-transfer, clean-split SOTA,
or encoder-pretraining result  
**Evidence cutoff:** 2026-07-28

Local evidence sources: `MethaneFuse_ICDM.pdf`,
`RCTP_PRETRAINING_PLAN.md`, `RCTP_MATCHED_EXPERIMENT_PROTOCOL.md`, and
`rctp_l89_screen_seed20260728/summary.json`, plus
`legacy360_dual_axis/DUAL_AXIS_EXPERIMENT_LOG.md` for the downstream
engineering control.

---

## 1. Central story

Generic EO pretraining is effective at representing what a scene persistently
looks like. Methane detection instead requires recognizing how an invisible,
weak, short-lived gas changes that same scene under each sensor's spectral,
spatial, and product response. RCTP therefore learns exact
reference–methane counterfactual differences instead of reconstructing the
scene itself.

The intended evidence chain is:

```text
correct CH4 response
  > equal-energy wrong response and generic scene objectives
  > transfer to real event classification and plume localization
  > retain the gain under missing visits and sensors
```

The current L89 result supports only part of the first line. The completed
legacy360 evaluation shows that the downstream implementation is competitive,
but it does not test any RCTP pretraining arm and therefore does not advance
this causal evidence chain.

---

## 2. How this reuses the MethaneFuse story

The ICDM paper begins with a real deployment constraint—sensor observations
are incomplete because emissions are transient and coverage is limited by
revisit, cloud, quality, and mission availability—rather than with a new
attention module. It then shows that S2-only learning discards most reported
cases, builds MethaneUnion to expose the partial-observation problem, designs
MethaneFuse around naturally available sensor subsets, and closes the loop
with coverage and downstream gains.

Its abstract reports a coverage expansion from 3,211 valid S2-matched plume
cases to 8,981 cases with multi-sensor observations. At 480 m, it reports
`84.87` F1 and `93.62` AUROC, improvements of `5.65` F1 and `8.30` AUROC
points over the strongest baseline, with false positives reduced by `8.19`
points. The Journal paper needs the same problem–method–measurable-outcome
closure; a plausible mechanism alone is not enough.

| Narrative role | MethaneFuse (ICDM) | Proposed Journal paper |
|---|---|---|
| Real constraint | A plume rarely has all four sensors | Methane evidence is weak, non-visible, sparse, and may occur in only one irregular visit |
| Existing mismatch | S2-centric models cannot use partial heterogeneous observations | Generic scene objectives do not specify which weak change follows a CH4-specific sensor response |
| Method principle | Preserve native sensors and learn from available subsets | Preserve native observation operators and learn exact methane-induced differences |
| Method | Sensor-native encoding and masked sensor-set fusion | Response-Conditioned Counterfactual Transient Pretraining (RCTP) |
| Required closure | Coverage plus real F1/AUROC gains | Response falsifier plus real classification/localization and missingness robustness |

The new paper must therefore not be sold as “another cross-attention
variant.” The scientific question is whether the training target writes
sensor-dependent methane response into a transferable encoder.

---

## 3. Problem and FM mismatch

Methane differs from persistent EO semantics:

- its evidence lies in SWIR absorption, dense hyperspectral methane windows,
  or a retrieved XCH4 product rather than a stable visible object;
- it occupies a small region and may appear in only one of six irregular,
  missing, or duplicated acquisitions;
- response direction and strength depend on SRF, units/product semantics,
  GSD, PSF/footprint, retrieval processing, validity, and acquisition quality;
- cloud, BRDF, surface change, striping, registration error, and brightness
  variation can carry more pixel/latent energy than the plume.

A reconstruction or consistency objective can therefore optimize successfully
while treating the desired transient as noise.

The defensible gap is **objective–target mismatch**, not “all EO foundation
models ignore invisible bands.” Panopticon is wavelength-aware and can ingest
SWIR. Wavelength support alone, however, does not teach which cross-band
change is consistent with methane or distinguish it from an equal-energy
nuisance. Likewise, the ICDM result that MethaneFuse beat generic EO transfer
on its protocol motivates this question but does not isolate the pretraining
objective as the cause.

Project evidence makes the mismatch concrete:

- globally purged shared pixel MAE and validity-masked MAE changed four-sensor
  macro AP by `-0.0081` and `-0.0098` versus matched scratch;
- a cross-fitted background predictor achieved high latent similarity, while
  innovation-only classification lost `0.0534 AP` versus t0;
- temporal correspondence reached `96.5%` pretext accuracy but reduced
  downstream AP by `0.0229` with full labels and `0.0466` with 10% labels;
- conversely, using matched history as context for the current observation
  improved frozen-feature AP by `0.0409` on L89 and `0.0232` on EMIT.

These results diagnose persistent-scene bias and motivate RCTP. They do not
show that RCTP improves a trainable encoder.

The narrow research question is:

> How should a heterogeneous EO encoder be trained to preserve a weak,
> one-visit methane transient when the physically correct representation
> change depends on the observing sensor?

---

## 4. Mechanism hypothesis and RCTP

For one reference scene and physical plume column field \(C(p)\):

```text
x0_s = O_s(background)
x+_s = O_s(background + C)
x-_s = O_s(background + matched nuisance)
```

\(O_s\) is the sensor-native observation operator: SRF/CH4 sensitivity,
unit/product semantics, GSD, PSF/footprint, validity, and acquisition
metadata.

RCTP is falsified or narrowed in three steps:

1. **Response selectivity:** `x+` must outrank the exact reference and
   equal-energy achromatic/wavelength-shuffled changes.
2. **Real transfer:** continue-pretraining must improve real event
   classification and real-mask localization over direct PTH transfer and
   generic objectives.
3. **Cross-sensor causality:** rendering the same column field through native
   sensor operators should align response differences; scrambling/removing
   response metadata should remove a measurable part of the gain.

Passing only step 1 shows a learnable synthetic capability. Passing steps
1–2 but not 3 supports per-sensor counterfactual augmentation, not
response-conditioned multi-sensor pretraining.

The proposed implementation:

- uses outer-train reference controls and exact same-scene positive/nuisance
  variants with matched support, strength, visit, augmentation, and validity;
- injects one unique visit, including history, while retaining real
  `delta_days` and masking duplicate acquisitions;
- renders S2/L8/9 through spectral response plus PSF/GSD, retains native
  32-band EMIT, and treats S5P as native 3×3 XCH4 cells rather than a 224 image;
- conditions a residual adapter on response/product/acquisition metadata;
- optimizes localization, counterfactual ordering, strength, visit,
  cross-sensor response-difference alignment, background stability, and a
  frozen-clean anchor.

The proposed loss is:

```text
L_RCTP =
    1.00 L_loc + 0.50 L_cf + 0.25 L_col + 0.25 L_visit
  + 0.15 L_xsensor + 0.10 L_bg + 0.05 L_anchor
```

These weights and operators are a protocol starting point, not validated
optima. Formal work requires versioned SRFs, CH4 cross-sections, product
operators, renderer hashes, and train-only calibration. Reference controls
must not be called confirmed methane-free scenes.

### Two-axis readout boundary

The time × sensor head tests `time→sensor` and `sensor→time` factorization
under missing visits/sensors. It is an evaluation/downstream architecture,
not RCTP's core novelty. It must first be compared with current-only under
matched development conditions and then locked. Every P0–P5 arm must use the
same readout, parameters, data order, and downstream budget.

If the gain appears only after adding the dual-axis head, it is an architecture
result—not evidence for RCTP.

### Completed legacy360 downstream control: competitive, but not RCTP

The complete legacy360 engineering campaign provides a strong downstream
control for the readout used in later RCTP comparisons. Its train-core feature
cache contained 113,843 rows, and the only model/epoch/threshold selection
used a separate 12,621-row development cache. Among four predeclared
short-run candidates, the promoted model was the compact scale-aware
time×sensor head (`d=64`, LR `1e-4`, epoch 3):

| Development metric | Value |
|---|---:|
| F1 at 0.5 | `0.90313` |
| Development-selected F1 | `0.90558` |
| Locked threshold | `0.3910266161` |
| Macro F1 at 0.5 | `0.89622` |
| AP / AUC | `0.96003 / 0.96038` |

After that choice was frozen, the 31,564-row old-legacy test was evaluated
exactly once, with no test-threshold search:

| Scope | Binary F1 | Macro F1 | AP | AUC |
|---|---:|---:|---:|---:|
| overall | **`0.86485`** | `0.85063` | `0.93313` | `0.93042` |
| rows containing S2 | **`0.90616`** | `0.90140` | `0.95842` | `0.96368` |

This meets the requested `0.90` guardrail for S2-containing rows, but not for
the overall heterogeneous endpoint. Relative to the audited historical
universal full-test F1 `0.82832`, the overall result is `+0.03653` (3.65
points); relative to the previous paper's 360 m F1 `0.8377`, it is `+0.02715`
(2.71 points). These are historical engineering comparisons, not
matched-clean-split improvements.

Most importantly, the immutable old source split has **706 canonical events
present in both source train and source test**. The new campaign did not use
test labels, predictions, or metrics for candidate, epoch, or threshold
selection, so its exactly-once procedure is a valid faithful evaluation of
that legacy protocol. The inherited event overlap and historically
test-selected warm-start PTH nevertheless prevent treating `0.86485` or
`0.90616` as a clean journal benchmark or SOTA result.

No RCTP objective, response-conditioned renderer, or P4/P5 comparison was
used in this campaign; the encoder features came from existing PTHs.
Consequently, these numbers establish only that the downstream pipeline and
scale-aware readout are strong enough to evaluate upstream pretraining.
They do not show that RCTP works, that time×sensor attention caused the
historical gain, or that the new paper has reached its promotion line.

---

## 5. Current RCTP evidence: positive capability screen only

The completed L89 screen used 4,096 train reference controls and 2,048
canonical-event-disjoint development controls. One usable visit was injected
per row. Panopticon ViT-B/14 remained frozen; only a zero-initialized
response adapter and lightweight type/visit/strength heads were trained for
three epochs. Methane-like, achromatic, and wavelength-shuffled deltas were
energy matched.

The response `[0, 0, 0, 0, 0, 0.15, 1.0]` is an explicit SWIR approximation,
not an SRF integral.

| Capability panel | AP | ROC-AUC | Paired win |
|---|---:|---:|---:|
| All strengths | `0.7476` | `0.8382` | `0.7607` |
| Weak, `<=2%` | `0.5472` | `0.7076` | `0.6029` |
| Mid, `2–4%` | `0.7618` | `0.8448` | `0.7814` |
| Strong, `>4%` | `0.8845` | `0.9321` | `0.9024` |
| History injection | `0.7400` | `0.8331` | `0.7521` |
| t0 injection | `0.7868` | `0.8632` | `0.8041` |

With one methane and two nuisance variants, chance AP and matched-group win
rate are both `1/3`. Methane versus achromatic reached AP/AUC
`0.8321/0.8194`; methane versus wavelength shuffle reached
`0.8543/0.8569`. Development AP progressed
`0.6964 → 0.7387 → 0.7476`. No test artifact was read
(`test_evaluations = 0`).

Authoritative summary:
`/diniuvol/yuyao/methanefuse_research_20260727/results/`
`rctp_l89_screen_seed20260728/summary.json`. Selected checkpoint SHA-256:
`0c4759202c3d426f47ddc7f3f9bfe2c25b3877c9321c819aa9fcc8e1153c6761`.

The supported statement is narrow: a frozen Panopticon representation plus a
small adapter can separate an approximate L89 methane-response direction from
two equal-energy nuisances, including at historical visits and with weaker but
above-chance performance below 2%.

The screen does **not** establish:

- `x+ > x0` ordering; it tests only methane-shaped versus matched-nuisance
  separation;
- encoder continue-pretraining or transfer to real methane labels;
- P5 superiority over independently retrained P0–P4;
- downstream classification F1—the reported `0.6677` selected F1 belongs only
  to the synthetic one-vs-two task and is unrelated to a `0.90` target;
- real localization/segmentation or physical renderer accuracy;
- multi-sensor response conditioning, because L89 metadata is constant;
- real visit discovery—the `1.0` visit accuracy is structurally easy because
  the paired delta occupies one temporal slot;
- replication or an unbiased outer-test result. This is one seed, and the
  current L89 hard-event test remains model-conditioned.

This is a reason to run the matched protocol, not a paper result by itself.

---

## 6. Evidence still required

### Matched pretraining arms

All arms require the same initialization, native inputs, deterministic
synthetic stream, trainable encoder parameters, updates, locked readout, and
measured compute tolerance:

| Arm | Objective |
|---|---|
| P0 | Direct Panopticon PTH transfer |
| P1 | Normalized native pixel/cell MAE |
| P2 | Masked EMA-latent prediction |
| P3 | Validity-masked temporal residual MAE |
| P4 | Response-scrambled RCTP with matched energy/supervision |
| P5 | Correctly conditioned full RCTP |

P1–P3 may see the same rendered inputs for data matching, but cannot receive
pair identity, response label, injected mask, strength, or visit supervision.
The current within-task nuisance classifier is not P5-versus-P4: P4 and P5
must be independently retrained and transferred downstream.

### Required gates and downstream results

1. **Renderer shortcut gate:** non-methane-band/mask-out synthetic
   discriminator AUROC `<=0.60`; overlap with real-train plume SNR and
   geometry; alternate renderers; matched energy/support/validity.
2. **Real transfer:** identical 1–3 epoch heads; event-balanced positive F1,
   F1@0.5, macro-F1, AP/AUROC, fixed-FPR recall; three seeds and event-cluster
   paired bootstrap.
3. **Localization:** identical frozen patch/cell heads on provenance-trusted
   real masks, followed by segmentation only if this proxy passes.
4. **Cross-sensor falsifiers:** `P5-L_xsensor`, sensor-ID-only,
   response-shuffled metadata, and held-response/held-sensor transfer.
5. **Robustness:** missing-visit/sensor curves and per-availability strata;
   the fused gain cannot be driven only by the strongest sensor.
6. **Clean split:** one global event/acquisition/source assignment before
   recropping all sensors from verified 512 sources. Current L89/S5P tests,
   legacy360 warm starts, and historical test-selected PTHs remain engineering
   references only.

The journal promotion line should remain material: approximately `+2.0`
absolute MacroSensorF1 over the strongest matched comparator, separation from
P4, and a real localization gain. An unstable `0.2–0.5` point change is not
enough. A `0.90` F1 or SOTA claim additionally requires a named clean,
directly comparable protocol and a once-only locked outer test.

---

## 7. Claim boundary

### Supported now

- Scene-focused objectives are a credible failure mode for weak methane
  transients in this project; successful pretext optimization has repeatedly
  failed to guarantee transfer.
- Matched same-location history helps when it conditions rather than replaces
  current evidence.
- The train/dev-only L89 screen separates methane-shaped SWIR perturbations
  from two equal-energy spectral nuisances, including an above-chance weak
  panel.
- This capability result justifies advancing RCTP to matched
  continue-pretraining.
- A development-selected, exactly-once legacy360 downstream control reaches
  `0.86485` overall F1 and `0.90616` F1 on S2-containing rows. This supports
  downstream implementation readiness only; the inherited 706-event overlap
  keeps it outside the clean RCTP evidence chain.

### Conditional on the clean, matched RCTP experiments

- RCTP transfers better than generic EO pretraining to real classification.
- Correct sensor response—not generic synthetic augmentation—causes the gain.
- RCTP learns localizable evidence and improves real segmentation.
- Cross-sensor response-difference alignment improves partial-sensor
  robustness.
- The system is SOTA or reaches about `0.90` F1 on the specified clean
  protocol.

### Do not claim

- Existing EO foundation models use only visible light or cannot ingest SWIR.
- RCTP is the first residual, anomaly-ranking, synthetic anomaly,
  multisensor-temporal MAE, cross-sensor prediction, or pairwise-ordering
  method.
- Time × sensor dual-axis attention is the core novelty.
- The L89 capability screen is real methane detection, encoder pretraining,
  multi-sensor fusion, segmentation, or downstream F1 evidence.
- A fixed single-sensor `kappa` validates response metadata or cross-sensor
  fusion.
- Historical `0.8955` S2 F1, the legacy360 `0.86485` overall F1, or its
  `0.90616` S2-containing F1 are formal evidence under the new RCTP protocol.
- The legacy360 historical improvement is caused by RCTP or by time×sensor
  attention; neither mechanism was isolated in that comparison.

If validated, the narrow contribution is the combination of versioned
sensor/product response conditioning, exact reference–methane counterfactual
pairs, irregular single-visit supervision, sensor-native heterogeneous
operators, and transfer to both real classification and localization under
leakage-free matched controls.

---

## 8. Current abstract skeleton

> Satellite methane detection depends on weak, spatially sparse evidence that
> appears in non-visible spectral bands or retrieved atmospheric products and
> may occur in only one irregular acquisition. Generic Earth-observation
> pretraining rewards reconstruction or consistency of persistent scene
> content, but does not specify how methane should change an observation under
> each sensor's response. We propose Response-Conditioned Counterfactual
> Transient Pretraining (RCTP), which constructs exact same-scene
> reference–methane pairs, renders a shared plume column field through
> sensor-native operators, and learns methane-induced representation
> differences against equal-energy nuisances. In an initial train/dev-only
> L8/9 screen with a frozen Panopticon encoder, a lightweight screening
> adapter reaches `0.748` AP and `0.761` matched win rate; the `<=2%` panel
> remains above chance at `0.547` AP and `0.603` paired win. These synthetic
> capability results motivate—but do not establish—real methane transfer.

The final abstract must replace the last sentence with matched P0–P5,
real-classification/localization, clean-split, three-seed results. Otherwise
RCTP remains a proposed direction or negative study, not the claimed Journal
contribution. The legacy360 `0.86485/0.90616` figures should not fill those
slots: they are downstream engineering controls from an inherited overlapping
split, not P4/P5 or clean RCTP-transfer results.
