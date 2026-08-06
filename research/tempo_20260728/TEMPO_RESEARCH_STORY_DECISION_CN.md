# TEMPO journal technical decision

Date: 2026-07-28 UTC  
Status: architecture decision supported by development experiments; clean
outer confirmation remains required

## 1. Technical thesis

The next paper is not about a new attention operator and not about training a
small methane foundation model. Its technical thesis is:

> Methane EO is a **doubly partial observation** problem. Sensors are missing
> across modalities, while visits are missing, duplicated, and irregular
> within each sensor. Reliable methane evidence should therefore be formed
> inside each sensor first, with appearance and transient evidence trained as
> independent experts, and only then communicated through fixed
> availability-aware decision fusion.

Formally, the input is

\[
\mathcal X =
\{x_{s,t},m_{s,t},\Delta_{s,t},q_{s,t},a_{s,t}\},
\]

where \(s\) is sensor, \(t\) is visit role, \(m\) is availability/uniqueness,
\(\Delta\) is the real acquisition gap, \(q\) is quality, and \(a\) is
acquisition identity. There are two distinct missingness processes:

\[
\exists s:m_{s,0}=0,
\qquad
\exists(s,t):m_{s,t}=0\ \text{or}\ a_{s,t}=a_{s,t'}.
\]

The model factorization is:

\[
\{x_{s,t}\}_{t=0}^{5}
\xrightarrow[\text{within sensor}]
{(\ell_s^{appearance},\ell_s^{transient})}
\xrightarrow[\text{fixed consensus}]
\ell_s
\xrightarrow[\text{sensor availability}]
\ell.
\]

## 2. Research gap

Three adjacent families each miss part of this structure:

1. General EO FMs and single-image classifiers emphasize appearance. They do
   not force a weak current plume to be represented separately from background
   and acquisition nuisance.
2. Video models provide a useful appearance/motion division of labor, but
   standard frame differencing, optical flow, and SlowFast assume dense,
   homogeneous, approximately regular frames.
3. MethaneFuse handles partial sensor sets, but its old temporal interface
   concatenates three visits without real acquisition gaps or duplicate-aware
   evidence formation.

The gap is therefore not “no one has used residuals, ranking, two-stream video,
or late fusion.” Those are occupied ideas. The defensible gap is the specific
factorization required by irregular multisensor methane observations and its
leakage-free, matched-control validation.

## 3. Architecture

```text
for each sensor s
    six irregular visits + real gaps + quality + duplicate mask
                         │
             frozen Panopticon per visit
                         │
          ┌──────────────┴──────────────┐
          │                             │
  appearance/response expert      transient expert D1
  preserve methane-sensitive      signed (z0-zh)
  current decision surface        + absolute |z0-zh|
                                  + continuous gap/quality gate
          │                             │
          └──── fixed mean-logit consensus ────┘
                         │
                one evidence logit per sensor
                         │
         availability mask / frozen sensor router
                         │
                  classification logit
```

D1 uses:

\[
d_h=f_\Delta([z_0-z_h,\ |z_0-z_h|]),
\]

\[
\alpha_h=
\operatorname{MaskedSoftmax}
f_g(role_h,\Delta_h,q_h,\cos(z_0,z_h),\|z_0-z_h\|_2),
\]

\[
v_T=\sum_h\alpha_h d_h,\qquad
\ell_{D1}=\ell_{P0}+r(v_T).
\]

The final residual layer is zero initialized, hence D1 starts at exact P0.
D1 and appearance are trained independently. The default decision is:

\[
\ell_s =
0.5\,\ell_s^{appearance}
+0.5\,\operatorname{mean}_{seed}\ell_s^{D1}.
\]

No learned liaison or development-fitted fusion weight is required.

## 4. What is borrowed from video and NormWear

The video contribution is a division of labor, not a copied backbone:

- retained: independent scene/appearance and change/transient pathways;
- rejected: fixed frame rate, optical flow, fixed fast/slow sampling, and
  learned lateral connections.

The NormWear contribution is axis-local evidence before communication:

- retained: heterogeneous channels/sensors first form native evidence;
- rejected: CWT, masked pretraining, periodic liaison tokens, and a learned
  cross-channel communication block.

The resulting structure is methane-specific because spectral response,
irregular acquisition, duplicate observations, coarse/fine sensor geometry,
and sensor availability enter different parts of the computation.

## 5. Experimental logic

The paper should test a chain of falsifiable statements rather than only a
final score.

### H1 — Time evidence must be explicit and acquisition-aware

- History shuffle must reduce the transient branch while keeping P0, labels,
  target `t0`, masks, gaps, and quality fixed.
- Generic `t0`-query attention is not sufficient because the existing head
  already uses it.
- Uniform delta, background-normalized onset, and hard SlowFast are matched
  controls.

Fresh L89 supports the narrow mechanism:

- one-seed default D1 history shuffle changes AP by `-0.016110` and macro-F1
  by `-0.021742`;
- hard SlowFast shuffle changes AP by `+0.003690`, so its ranking is not
  load-bearing even though its macro-F1 is reasonable;
- uniform delta, gated onset, and hard SlowFast all lose to fixed
  P0+three-seed D1 on macro-F1.

### H2 — Transient evidence complements but does not replace appearance

Fresh event-disjoint L89:

| System | AP | AUC | Macro-F1 | Positive-F1 |
|---|---:|---:|---:|---:|
| P0 appearance | 0.752329 | 0.841389 | 0.765197 | 0.686078 |
| three-seed mean D1 | 0.748303 | 0.841200 | 0.770990 | 0.702631 |
| fixed P0+D1 | **0.752429** | **0.842090** | **0.772695** | **0.699453** |

Versus P0, canonical-event bootstrap gives:

- macro-F1 `+0.007498`, 95% CI `[+0.001151,+0.014078]`;
- positive-F1 `+0.013375`, 95% CI `[+0.004366,+0.023155]`;
- AP `+0.000100`, CI crosses zero.

Therefore D1 is not claimed to be a stronger ranker. It supplies an
independent operating-point correction that improves F1 without materially
changing AP.

### H3 — Communication should be late and simple on this data scale

- OOF learned reliability liaison loses to fixed equal fusion.
- NormWear-style learned liaison loses to masked mean/frozen routing.
- A 21-point post-hoc curve is flat: best AP is only 0.00042 above fixed
  0.5, and best macro-F1 only 0.00108 above it.
- Null-event penalty variants do not improve the fixed candidate.

This supports fixed late consensus, not another trainable fusion module.

### H4 — Time helps sensor partiality after sensor-native evidence formation

Legacy360 event-disjoint development:

| System | Binary F1 | Macro-F1 | AP | AUC |
|---|---:|---:|---:|---:|
| P0 | 0.905424 | 0.894916 | 0.960035 | 0.960381 |
| R4 four-seed | 0.910235 | 0.900769 | **0.961690** | **0.962527** |
| R4 only for one-current-sensor rows | **0.910745** | **0.901134** | 0.961190 | 0.962070 |

The frozen router versus P0 has F1 delta `+0.005275`, bootstrap 95% CI
`[+0.002109,+0.008604]`. The benefit is concentrated when only one current
sensor is available, consistent with history acting as virtual supporting
evidence. It does not justify mixing raw heterogeneous sensor tokens.

## 6. Result relative to the previous paper

The old 360 m locked-test cohort gives:

- current compact two-axis checkpoint binary F1 `0.864846`;
- previous-paper 360 m F1 `0.8377`;
- engineering point difference `+0.0271`.

This meets the requirement of not being worse than the previous paper on the
old protocol. It is not a clean SOTA claim because 706 canonical events in the
immutable old test split overlap source train, although plume overlap is zero
and test threshold search was not performed.

The clean L89 result and the old 360 m result answer different questions and
must stay in separate tables.

## 7. Defensible contribution statement

The strongest claim currently supported is:

> We formulate multisensor methane monitoring as doubly partial observation
> learning and introduce a sensor-native two-axis evidence architecture:
> methane-sensitive appearance and irregular-acquisition transient evidence
> are learned independently within each sensor, combined by fixed decision
> consensus, and only then routed across available sensors. Matched temporal
> controls and branch-isolated history shuffles show that the gain cannot be
> explained by another current-query attention head, fixed-rate video
> decomposition, learned cross-sensor liaison, or added parameter capacity.

Do not claim a new residual principle, a new two-stream principle, a new
late-fusion principle, or a completed four-sensor/six-visit FM.

## 8. Remaining promotion gates

1. Finish the fresh correct-response versus scrambled-response sidecar control.
2. Run the supervised last-block matched control on GPU0 only; retain it only
   if temporal-current AP is at least `+0.005`, macro-F1 does not regress, and
   history shuffle is load-bearing.
3. Run the frozen formal clean replicate without reopening hyperparameters.
4. Execute a new R4/TEMPO sealed evaluation only after literal authorization
   of the exact payload and lock; never rerun after seeing the result.
5. A true `4 sensors × 6 visits` claim requires rebuilding common crops and a
   common event split from each sensor's 512-scale source images.
