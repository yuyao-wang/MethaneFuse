# RCTP L89 mechanism screen

## What this screen answers

This is the smallest executable check of the RCTP hypothesis on the existing
six-visit Landsat 8/9 train/dev split. It asks whether the existing Panopticon
representation contains enough spectral evidence for a small
response-conditioned adapter to separate methane-shaped SWIR attenuation from
matched non-methane changes.

It does **not** claim a downstream F1 improvement, a production radiative
transfer renderer, or a multi-sensor result. Those require a positive mechanism
screen followed by continue-pretraining and locked downstream experiments.

## Exact intervention

The default reference library uses only label-0 rows from outer train or dev.
These are “reference controls,” not guaranteed methane-free observations. For
each selected row:

1. choose one usable visit by a deterministic, approximately role-balanced
   schedule; the schedule covers `t0` and all available history roles;
2. render one deterministic soft plume field and sample one log-uniform peak
   absorption strength from 1–8%;
3. form four same-scene images:
   `clean`, `methane`, `achromatic`, and `wavelength_shuffled`;
4. match both nuisance deltas to the methane delta’s exact per-sample L2 energy;
5. encode the four images in one frozen Panopticon forward pass;
6. replace that visit in the existing six-visit clean CLS cache with the online
   clean CLS, and place each exact `z_variant - z_clean` at the injected visit.

The screening response vector is:

```text
B1 B2 B3 B4 B5 B6 B7 = 0 0 0 0 0 0.15 1.0
```

This is intentionally an explicit SWIR-sensitive approximation. It is not a
formal L8/L9 spectral response function (SRF) integration. A positive screen
justifies replacing it with the versioned SRF/cross-section renderer described
in `RCTP_PRETRAINING_PLAN.md`.

## Model and objectives

The Panopticon encoder is frozen. The trainable probe contains:

- a six-visit clean-context and paired-difference projection;
- real `delta_days` and visit-role encodings;
- a response metadata encoder;
- a zero-initialized residual adapter;
- zero-initialized methane/nuisance, injected-visit, and strength heads;
- one lightweight temporal encoder and query pooler.

The loss is:

```text
1.00 * methane-vs-nuisance BCE
+ 0.25 * injected-visit CE
+ 0.10 * methane log-strength SmoothL1
```

Validation average precision selects the checkpoint. Reported diagnostics
include paired win rate, methane-vs-achromatic and
methane-vs-wavelength-shuffled AUROC/AP, visit accuracy, and fixed strength
panels (`<=2%`, `2–4%`, `>4%`) plus `t0`/history panels.

Chance references are AP `1/3` and paired win rate `1/3`. A useful go/no-go
screen should not rely only on the aggregate: it should retain a clear paired
margin for history injections and the `<=2%` panel. A high aggregate driven
only by `>4%` perturbations is a failed weak-transient story.

## Leakage and checkpoint contract

- The CLI has train and dev arguments only. There is no test argument or test
  evaluation path.
- CSV, cache, and output paths with test/sealed-like components are rejected.
- Train/dev cache canonical events must not overlap.
- CSV SHA and row IDs must exactly match their feature cache.
- The online PTH SHA must match the PTH used to build both six-visit clean CLS
  caches. A mismatch fails closed because cached context would otherwise mix
  encoders.
- The saved checkpoint records PTH/cache hashes, renderer configuration, plan
  hashes, all arguments, epoch history, and `test_evaluations: 0`.

## Commands

CPU tests:

```bash
cd /home/yuyao/panopticon
/home/yuyao/miniconda3/envs/panopticon/bin/python -m unittest \
  research.pretraining_20260727.test_rctp_l89_screen_cpu -v
```

The launcher is dry-run by default:

```bash
cd /home/yuyao/panopticon
bash research/pretraining_20260727/run_rctp_l89_screen.sh
```

After the two-GPU legacy extraction is finished, launch on one selected GPU:

```bash
cd /home/yuyao/panopticon
RCTP_GPU=0 bash research/pretraining_20260727/run_rctp_l89_screen.sh --run
```

Fast 64-step smoke:

```bash
cd /home/yuyao/panopticon
RCTP_GPU=0 bash research/pretraining_20260727/run_rctp_l89_screen.sh --run \
  --max-train-rows 768 --max-dev-rows 384 --max-train-steps 64 \
  --output-dir /diniuvol/yuyao/methanefuse_research_20260727/results/rctp_l89_smoke64
```

The formal screening defaults are 4,096 train controls, 2,048 dev controls,
batch 12, and three epochs. Each row encodes four 224×224 images. From the
existing frozen-cache throughput, the expected A100 footprint is roughly
18–30 GiB and the end-to-end runtime is approximately 15–30 minutes, including
three repeated dev panels. The 64-step smoke should take roughly 2–5 minutes.
These are planning estimates; the first run should be polled and the observed
throughput/peak memory recorded.

## Outputs

```text
checkpoint_best_dev_ap.pt
best_dev_predictions.csv
summary.json
```

The adapter and probe can be inspected immediately. It should not be called a
pretrained methane encoder yet: the backbone remained frozen. If the screen is
positive, the next experiment is matched-compute continue-pretraining that
unfreezes only the last Panopticon blocks plus the response adapter, then
compares downstream L89 classification against:

1. the identical PTH without RCTP;
2. generic masked reconstruction;
3. synthetic augmentation without response conditioning;
4. wrong/shuffled response metadata.

Single-sensor L89 cannot establish that response conditioning solves
multi-sensor fusion because its response metadata is constant across rows.
That claim requires the same physical column field rendered through at least
two sensor operators and a held-sensor/held-response transfer test.
