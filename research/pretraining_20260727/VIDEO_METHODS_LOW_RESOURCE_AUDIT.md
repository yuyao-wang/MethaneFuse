# Low-resource video-method audit for MethaneFuse

Date: 2026-07-28

## 1. Decision

Do **not** train a video foundation model, VideoMAE, TimeSformer, ViViT, or
MViT from scratch.  The current data and compute do not support that story,
and the existing two-axis head already occupies much of the ordinary
space--time-attention design space.

The strongest low-resource direction is instead:

> **Track the background, detect what breaks the track.**  Freeze the
> Panopticon scene representation; build a narrow transient pathway that
> matches local background patches across irregular visits and measures the
> remaining feature difference; periodically exchange only compact evidence
> tokens across heterogeneous sensors.

This borrows three useful video/wearable priors without inheriting the
large-scale video-pretraining requirement:

1. an asymmetric scene/transient pathway inspired by SlowFast;
2. local correspondence followed by temporal-difference excitation, inspired
   by Motionformer, TEA, and TDN;
3. per-sensor independent encoding with periodic liaison-token fusion,
   inspired by NormWear.

The attractive task-level story is not “satellite data are videos.”  It is
the opposite:

> Dense videos track moving foregrounds across nearly identical frames.
> Sparse methane EO sequences contain irregular, heterogeneous acquisitions:
> the ground scene is the persistent object, while a weak plume is often
> visible only once.  We therefore reverse the usual video formulation:
> follow the stable background trajectory and identify the local evidence
> that cannot be explained by it.

Working method names include **Trajectory-Break Fusion**, **Background
Trajectory Break (BTB)**, or **Track-the-Background (TTB)**.  None is a
cleared novelty claim.

## 2. The three mechanisms worth testing

### 2.1 Asymmetric scene/transient pathways

Use the existing Panopticon PTH as a frozen “scene” pathway.  For each
sensor--visit image, cache spatial patch tokens rather than only the final
CLS token.  A much narrower transient pathway processes only temporally
matched feature differences.

For sensor \(s\), target visit \(0\), spatial patch \(p\):

\[
z^{scene}_{s,p} = z_{s,0,p}, \qquad
z^{trans}_{s,p} =
g_s\left(z_{s,0,p} - B_{s,p}\right),
\]

where \(B_{s,p}\) is a robust, reliability-weighted background estimate from
the available reference visits, and \(g_s\) is a small adapter.  A width ratio
of 1:8 or 1:4 is a reasonable first screen.  The scene path stays frozen; only
the transient adapter, local matcher, fusion tokens, and classifier train.

Why it fits:

- the existing frozen-sidecar experiments show that preserving the base is
  safer than in-place continuation;
- stable land-cover/context and weak methane evidence have strongly
  asymmetric information content;
- the fast branch can be small because it need not relearn the scene.

What comes from prior work:

- [SlowFast](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)
  already introduced a high-capacity slow semantic pathway and a
  low-capacity fast motion pathway.

Claim boundary:

- neither dual pathways nor “slow semantics / fast change” are novel;
- the defensible delta, if experimentally validated, is the asymmetric
  **frozen EO scene / irregular sensor-native transient** decomposition for
  weak methane evidence, combined with the two mechanisms below.

Estimated cost:

- one frozen feature-cache pass over at most six visits per row;
- after caching, roughly 0.5--4M trainable parameters depending on adapter
  width and number of fusion blocks;
- head training should be dominated by token I/O rather than backbone
  backpropagation and can be screened in 3--5 epochs.

### 2.2 Local background trajectories plus difference excitation

Do not subtract features at the same nominal pixel immediately.  A small
registration, footprint, viewing-angle, or PSF discrepancy can create a
larger “change” than the plume.  Instead, let every target patch search a
small local window in each reference visit:

\[
\tilde z_{s,t,p} =
\sum_{q\in {\cal N}(p)}
\operatorname{softmax}_{q}
\left(
\frac{W_q z_{s,0,p}\cdot W_k z_{s,t,q}}{\sqrt d}
 b(\Delta t, s, quality)
\right)
W_v z_{s,t,q}.
\]

The aligned background is a robust pool of \(\tilde z_{s,t,p}\) across
reference visits.  The transient token is the residual between the target
patch and this matched background.  A TEA/TDN-style excitation gate then
amplifies channels whose aligned multi-lag differences are consistent:

\[
e_{s,p} =
\sigma\left(\operatorname{MLP}
[\Delta^{short}_{s,p},\Delta^{season}_{s,p},
\Delta^{year}_{s,p},\Delta t,quality]\right),
\qquad
h_{s,p}=e_{s,p}\odot z^{trans}_{s,p}.
\]

Use a 3x3 or 5x5 patch-token window, not global space--time attention.  With
at most six visits, local complexity is
\(O(S\,T\,P\,k^2d)\), rather than quadratic in all
space--time--sensor tokens.

Why it fits:

- methane is local, so a final CLS token can erase it;
- the target/reference images are not dense video frames and cannot be
  assumed perfectly registered;
- multiple lags have different meanings; adjacent, seasonal, and yearly
  differences should not be averaged as interchangeable frames.

What comes from prior work:

- [Motionformer](https://arxiv.org/abs/2106.05392) introduced trajectory
  attention that aggregates along implicitly estimated motion paths;
- [TEA](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_TEA_Temporal_Excitation_and_Aggregation_for_Action_Recognition_CVPR_2020_paper.html)
  computes feature-level temporal differences to excite
  motion-sensitive channels;
- [TDN](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html)
  explicitly models short- and long-range temporal differences;
- remote-sensing change detection already integrates registration and
  difference modeling, for example
  [ChangeRD](https://doi.org/10.1016/j.isprsjprs.2024.11.019).

Claim boundary:

- trajectory attention, local matching, registration before change
  detection, feature differencing, and motion excitation are all occupied;
- “the plume breaks a background trajectory” is an intuitive formulation,
  not by itself a novelty claim;
- the possible contribution is an event-centric construction specialized to
  irregular six-visit, heterogeneous-sensor methane evidence, plus matched
  evidence that each component fixes a measured failure mode.

Estimated cost:

- two or three low-rank projections plus a small excitation MLP;
- a 3x3 local search over cached patch tokens is inexpensive enough for
  parallel ablations;
- no optical-flow network and no video checkpoint are needed.

### 2.3 NormWear-style sensor liaison, after transient extraction

Treat sensors as heterogeneous variables, not interchangeable RGB-like
channels.  Each sensor retains:

- its own input normalization and small adapter;
- its own spatial/temporal evidence tokens;
- a sensor identity, acquisition-time, quality, and availability embedding.

Share the transient block across sensors where dimensions permit, but only
every \(K\) blocks pass the per-sensor evidence/CLS tokens through a small
liaison transformer.  Patch grids and raw channels are not mixed globally.
Missing sensors use an attention mask, and sensor dropout is applied during
training.

This is a direct adaptation of the useful architectural pattern in
[NormWear](https://arxiv.org/abs/2412.09758): channel-wise encoding followed
by periodic fusion of special liaison/CLS tokens.  NormWear's CWT transform
and large-scale masked pretraining are not required.

Why it fits:

- S2, Landsat, EMIT, and S5P have different bands, footprints, spatial
  resolutions, and noise;
- early raw fusion asks one attention block to solve calibration,
  registration, and classification simultaneously;
- evidence-token fusion naturally supports missing sensors and avoids
  resampling every modality into a false common pixel space.

Claim boundary:

- channel-aware encoding, periodic CLS fusion, modality-specific adapters,
  and masked set attention already exist;
- novelty cannot be “NormWear for EO” or “cross-sensor attention”;
- the possible delta is to fuse **trajectory-break evidence**, rather than
  raw pixels or generic semantics, under a verified common
  event/location/plume split.

Estimated cost:

- at most 24 sensor--time summary tokens for four sensors and six visits;
- liaison attention is negligible relative to frozen image feature
  extraction;
- the expensive part is rebuilding the verified 512-source, common-FOV
  multisensor dataset, not model training.

## 3. Family-by-family audit

| Family | Fit under present constraints | Use |
|---|---|---|
| SlowFast | High as an asymmetric design prior; original dense frame-rate interpretation does not transfer | Main scene/transient decomposition, with frozen scene path |
| TSM / temporal shift | Very cheap but assumes ordered neighboring frames; shifting channels between irregular lags or sensors has no physical meaning | Include as a zero-parameter video baseline only |
| TEA / TDN | High if applied after local alignment and with explicit lag metadata | Main lightweight transient block; not a novelty claim |
| TimeSformer / ViViT | Factorized spatial/temporal attention is already close to the existing two-axis head and is data hungry | Baseline at most; do not build the story around it |
| VideoMAE | Contradicts the low-data/no-pretraining decision; tube masking assumes dense redundancy | Reject |
| MViT | Multiscale processing is useful for small plume versus scene context, but replacing Panopticon is expensive | Reuse frozen intermediate-resolution tokens or a small feature pyramid; do not train MViT |
| Motionformer | Its correspondence principle is useful; the official model was trained on large video benchmarks and the released setup used a large multi-GPU cluster | Borrow only local trajectory matching |
| Temporal Query Network | The query-response mechanism and feature bank are already published and collide with TransientQuery | Baseline/negative control, not mainline |
| Long-term feature bank | Cached context is useful, but with only six visits the “bank” is trivial | Use frozen token caching as engineering, not a claim |
| Video test-time adaptation | Uses unlabeled test samples and can complicate leakage, ordering, reproducibility, and sensor-wise calibration | Do not use for the primary result; at most a separately labeled transductive robustness study |

Primary references:

- [TSM paper/code](https://github.com/mit-han-lab/temporal-shift-module)
  describes a zero-parameter, zero-FLOP temporal channel shift.
- [TimeSformer](https://proceedings.mlr.press/v139/bertasius21a.html)
  established divided spatial and temporal attention.
- [ViViT](https://openaccess.thecvf.com/content/ICCV2021/html/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.html)
  factorized video-transformer variants and relied on image initialization
  and substantial regularization.
- [VideoMAE official code](https://github.com/MCG-NJU/VideoMAE) uses
  90--95% tube masking for video pretraining.
- [MViT](https://openaccess.thecvf.com/content/ICCV2021/html/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.html)
  builds a multiscale feature hierarchy for video/image recognition.
- [Temporal Query Networks](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Temporal_Query_Networks_for_Fine-Grained_Video_Understanding_CVPR_2021_paper.html)
  introduced learned temporal queries and a stochastic feature bank.
- [Long-Term Feature Banks](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Long-Term_Feature_Banks_for_Detailed_Video_Understanding_CVPR_2019_paper.html)
  augmented short clips with cached long-range context.
- [ViTTA](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Video_Test-Time_Adaptation_for_Action_Recognition_CVPR_2023_paper.html)
  aligns test features to training statistics and enforces consistency over
  temporal augmentations.

## 4. Minimal decisive experiment

The video hypothesis can be tested before the four-sensor joint recrop.
Start on the existing single-sensor event splits with frozen cached patch
tokens:

1. **P0:** frozen Panopticon plus current pooling/classifier;
2. **P1:** P0 plus narrow scene/transient dual path, same-pixel differences;
3. **P2:** P1 plus 3x3 local background correspondence;
4. **P3:** P2 plus multi-lag difference excitation;
5. **P4:** P3 with time-token liaison; the four-sensor liaison is deferred
   until the common cohort exists;
6. **video controls:** TSM and ordinary divided time attention under matched
   trainable-parameter and update budgets.

Run 3--5 epochs, three seeds for only the surviving pair, and select entirely
on event-disjoint development data.  Report:

- event-clustered AP and macro-F1;
- sensor-wise positive F1 and macro-F1;
- all-negative-event false-positive mass;
- performance by plume size and temporal availability;
- local-alignment ablation (same pixel versus 3x3/5x5);
- current-only and shuffled-history controls.

The central mechanism is supported only if:

- P2/P3 beats P0, P1, TSM, and divided attention;
- shuffling history removes the gain;
- local correspondence reduces error on the independently measured
  misregistration/FOV subset;
- the gain is not bought with more all-negative-event false positives.

After the verified common-cohort recrop, add:

- no fusion, raw-token fusion, and periodic evidence-liaison fusion;
- one-sensor, leave-one-sensor-out, and random sensor-dropout evaluation;
- classification and plume segmentation using the same transient map.

No locked test should be opened until this development gate passes.

## 5. Recommended paper framing

### Problem

Methane EO is commonly treated either as static image classification or as a
generic satellite time series.  Both views are mismatched: the discriminative
signal is a weak local departure from a stable scene, observed at sparse and
irregular times by incompatible sensors.

### Gap

Video models exploit dense motion continuity, while EO fusion models largely
aggregate temporally stable semantics.  Neither explicitly models a plume as
the local failure of a background correspondence across sensor-native
observations.

### Method

Freeze a strong EO scene encoder; follow local background patches across
irregular visits; excite the feature differences that break those
trajectories; periodically fuse only sensor-specific evidence tokens.

### Capability claim to test

The method should improve **weak-transient selectivity**: sensitivity to
small plume evidence without increased false alarms from registration,
seasonality, sensor resolution, or missing modalities.

### What must not be claimed

- a new video transformer;
- a new SlowFast architecture;
- a new temporal-difference or residual principle;
- a new trajectory-attention principle;
- the first per-channel or periodic-CLS fusion mechanism;
- a foundation model or pretraining contribution.

The credible novelty, if the experiments succeed, is the validated task-level
combination and the associated capability, not any individual block.
