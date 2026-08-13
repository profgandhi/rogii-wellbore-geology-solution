# Where is the drill bit in the rock?

**ROGII wellbore geology prediction. Public LB 5.21 ft.** One file, `5point21.py`. No GPU, no neural net.
Leak-free run over all 773 training wells: pooled RMSE **10.66 to 7.30 ft**.

Every number below is measured. Each training well was decoded as if it were hidden, through the real
pipeline, recording the prediction after every trick.

---

## The task in one table

| | |
|---|---|
| **Given** | trajectory `MD, X, Y, Z` for the whole well; gamma log `GR` for the whole well; `TVT` for the heel only; a typewell giving `GR = f(TVT)` |
| **Predict** | `TVT` for every hidden row: how deep the bit sits inside the rock column |
| **Metric** | pooled RMSE over all rows, so one bad well with 5,000 rows outweighs 200 good wells |
| **Core difficulty** | gamma is not unique (shales repeat), and there is only one anchor, at the heel |
| **Failure mode** | "cycle-skip": the path latches onto a wrong repeated marker and stays 10 to 50 ft off forever |

![The problem](assets/fig01_problem.png)

**Intuition for the whole approach:** you are reading a book in the dark with one page of it lit.
The typewell is the book (which gamma value belongs at which depth). The lateral's gamma log is the
sentence you are reading right now. You have to slide your sentence up and down the page until it
matches, and keep sliding as you walk forward. The catch is that the book repeats itself, so many
depths "match". So we add a second rule: the answer must also look like real geology, which means
long straight runs with gentle dips. The whole solution is that one sentence, plus twenty guards
against the ways it goes wrong.

---

## All the tricks at a glance

"True effect" is the order-free main effect from an exact 128-configuration factorial.
"Ablation" is what the raw decode loses if the term is removed. All in feet.

| # | Trick | One-line idea | Worth |
|---|---|---|---|
| 0 | [geo coordinate](#0-model-the-rock-not-the-well) | model `geo = TVT + Z`, the rock surface, not the well's position in it | foundational |
| 1 | [Cauchy emission](#1-robust-emission) | log-cost on the GR residual so one bad bed cannot steer the path | ablation +11.85 |
| 2 | [GR gradient term](#2-gr-gradient-term) | also match how GR *changes*, to separate equal-level markers | ablation +0.39 |
| 3 | [Tightened slope prior](#3-tightened-slope-prior) | over-tighten AR(1) so the prior regularises the ambiguous emission | ablation +3.94 |
| 4 | [Slope-magnitude anchor](#4-slope-magnitude-anchor) | penalise big dips separately from slope changes | pairs with 3 |
| 5 | [TW_BLEND](#5-tw_blend-splice-the-wells-own-rock) | splice the well's own heel GR into the typewell | ablation +1.45 |
| 6 | [Expanding funnel](#6-expanding-anti-drift-funnel) | deadzone penalty on drifting off the known trend, widening with MD | ablation +0.28 |
| 7 | [Drill-follow re-centre](#7-drill-follow-re-centre) | centre the funnel on the wellbore shape, because the driller tracked the rock | **-0.86 pooled** |
| 8 | [Follow-gate](#8-follow-gate) | decode both ways, keep the one that fits gamma better | insurance, 24 wells |
| 9 | [Consensus arbitration](#9-consensus-arbitration) | switch prior only if the neighbour depth surface AND gamma both agree | -0.29 seq |
| 10 | [Two rescues](#10-the-two-rescues) | bounded-slope and segment-shape decodes, same double gate | -0.26 true |
| 11 | [Decorrelated canceller](#11-decorrelated-canceller) | average 4 detuned decodes that fail differently | -0.40 true |
| 12 | [Trust regions and guards](#12-trust-regions-and-guards) | clamp how far any blend may move the path | -0.045 seq |
| 13 | [Robust projection](#13-robust-low-order-projection) | 50/50 blend with a robust degree-4 fit | -0.049 true |
| 14 | [Family typewell](#14-family-enhanced-typewell) | rebuild the reference by pooling GR-matched neighbour wells | **-0.81 true** |
| 15 | [diponly blend](#15-two-spatially-safe-blends) | decode again with the frame tilted by regional dip, blend 0.15 | -0.26 true |
| 16 | [znorm blend](#15-two-spatially-safe-blends) | decode again with both GR logs z-normalised, blend 0.10 | -0.40 true |
| 17 | [track8 + learned gate](#16-track8-and-the-learned-gate) | blend an independent GBM, weight set by self-measured confidence | -0.40 true |
| 18 | [Visible-well shortcut](#17-visible-well-shortcut) | 3 test wells are in train with marker picks, so answer exactly | free |

Two themes run through all of it:

1. **Combine, never select.** Averaging models that fail independently beats picking one.
2. **When you cannot predict *whether* a fix will hurt, bound *how much* it can.** Conditional-harm
   AUC here is 0.54 to 0.61 for every free feature, so learned gates lose to fixed bounds.

---
---

## 0. Model the rock, not the well

**Intuition:** TVT tells you where the *driver* is on the road. geo tells you where the *road* is.
Roads are straight for miles; drivers weave. Model the road.

`TVT` is the well's position inside the layer: half rock, half driller. Subtract the wellbore out.

$$\mathrm{geo}(t) = \mathrm{TVT}(t) + Z(t)$$

That is the **elevation of the rock surface itself**, a gently dipping geological object made of long
straight runs. We want a model whose hidden state is piecewise constant, and geology gives that.
Steering does not.

![geo vs TVT](assets/fig02_geo.png)

**Proof (panel c).** Fit the *same* 8-segment piecewise-linear model to either coordinate, convert back,
score in TVT. Making geo linear: median **0.86 ft**. Making TVT linear: **1.64 ft**. geo wins on **91%**
of wells. Slope persistence agrees: 0.40 in geo vs 0.31 in TVT.

### The decoder

The rock is a chain of straight runs. Segment `i` lasts `b_i` rows with constant slope `s_i`:

$$\mathrm{geo}(t) = \mathrm{geo}(R_i) + s_i (\mathrm{MD}(t) - \mathrm{MD}(R_i))$$

$$b_i \sim \mathrm{LogNormal}(\mu_b, \sigma_b), \qquad s_i \mid s_{i-1} \sim \mathcal{N}(\rho s_{i-1},\ \sigma(b_i)^2 (1-\rho^2))$$

An **explicit-duration hidden semi-Markov model**: segment length is modelled, not implicit. Real dip
panels run hundreds of feet, and a geometric length prior would shred them.

Score of a full path:

$$\mathcal{S} = \underset{\mathrm{emission}}{-\sum_{t} \left[ w_L \rho_C\!\left(\frac{GR_t - \hat{GR}_t}{\sigma}\right) + w_G \rho_C\!\left(\frac{\Delta GR_t - \Delta \hat{GR}_t}{\sigma_g}\right)\right]} + \underset{\mathrm{prior}}{w_P \sum_i \left[\log p(b_i) + \log p(s_i \mid s_{i-1})\right]} + \underset{\mathrm{funnel}}{w_D \sum_i \pi_f(\mathrm{geo}_i)}$$

`GR_hat = f_tw(geo - Z)` is the gamma the typewell predicts at the depth this path claims.
17 slope classes, 6 duration classes, segment starts every 15 rows, beam of 20, numba kernel.

![The decoder](assets/fig03_decoder.png)

---

## 1. Robust emission

$$\rho_C(r) = \frac{1}{2}\log(1 + r^2)$$

**Intuition:** one washed-out bed should not outvote a thousand good rows, but under least squares it
does. The Cauchy cost grows logarithmically, so a 6-sigma residual costs 1.8 instead of 18. Bad beds get
discounted instead of steering the path. It beat L2, centred L2 and pure derivative.

Sigma is not tuned. It is estimated **per well** from the heel as
`std(GR_known - typewell_GR(TVT_known))`, clipped to [19.785, 47.3] API: literally "how much does this
well agree with its typewell where we can check".

![The emission](assets/fig04_emission.png)

**Panel b is the whole competition.** Take a real well's *true* depth track, shift it by a constant,
plot the emission. If gamma identified depth there would be one sharp minimum at zero. Instead the curve
is bumpy, and for one of the two wells shown the **global minimum sits 61 ft from the truth**. The gamma
genuinely prefers a wrong answer. That is a cycle-skip, and six tricks below exist to fight it.

---

## 2. GR gradient term

**Intuition:** two shale beds can read the same number. But one is on the way up and the other is on
the way down. Matching the *direction* of change tells them apart when the value cannot. So a second
Cauchy term scores the windowed difference.

$$\Delta GR_t = GR_t - GR_{t-4}$$

Small in the mean (+0.39 ft ablation), concentrated entirely on the cycle-skip tail, which is where a
pooled metric is decided. Panel c above.

---

## 3. Tightened slope prior

The generative fit gives `rho = 0.82`. The shipped decoder uses `rho = 0.96`.

**Intuition:** if your evidence is ambiguous, lean harder on your assumptions. A stubborn prior refuses
to jump to a wrong marker just because the gamma slightly prefers it.

**Why deliberately wrong.** The optimal *decoding* prior is not the true generative prior. Because the
emission is non-unique, the prior does double duty as a regulariser against that ambiguity. Tightening
rho and rescaling sigma moved a 100-well holdout from 9.43 to 8.29 mean RMSE, p90 20.2 to 17.2.

![The priors](assets/fig05_prior.png)

**99.8% of true 200-ft geo slopes have |s| < 0.10.** The grid still runs to 0.22 for the few genuinely
steep wells, and that extra width is exactly the escape route a cycle-skip uses.

---

## 4. Slope-magnitude anchor

An extra penalty added straight into the transition prior, independent of rho:

$$\log p(s_i \mid s_{i-1})\ \mathrm{+=}\ -\frac{W_{SMAG}}{2}\left(\frac{s_i}{0.09}\right)^2$$

**Intuition:** continuity says "do not change slope". This says "and do not hold a huge slope either".
Two different statements. Together they let rho stay loose enough to follow real gamma while big
spurious dips, which are always cycle-skips, stay expensive.

---

## 5. TW_BLEND: splice the well's own rock

**Intuition:** the reference book was printed on a different press. Same story, slightly different ink.
We get one page of our own printing for free (the heel, where TVT is known), so we use it to recalibrate
the book before reading the rest.

The typewell is a *different hole*: different tool, calibration, borehole, mud, facies. Measured across
all 773 wells, the median GR offset between a well and its own typewell is **1.7 API** (p90 5.0).
Fix it with the one piece of ground truth we get free: the well's own heel.

$$GR^{ref}(\tau) = 0.5 \cdot \mathrm{median}\{GR_t : t \in \mathrm{known},\ \mathrm{TVT}_t \approx \tau\} + 0.5 \cdot GR^{tw}(\tau)$$

Applied only in the TVT bins the heel actually reaches. Leak-free, since the heel's TVT is given.

![TW_BLEND](assets/fig06_twblend.png)

**Worth +1.45 ft if removed:** the second-largest term inside the decoder after gamma level itself.

---

## 6. Expanding anti-drift funnel

$$\pi_f(\mathrm{geo}, d) = -\frac{1}{2}\left(\frac{\max(|\mathrm{geo} - c(d)| - a(d),\ 0)}{a(d)}\right)^{2}, \qquad a(d) = 5.0 + 0.008 d$$

**Intuition:** a leash, not a cage. The bit is free to move anywhere inside a corridor around where the
heel said the rock was going, and the corridor gets wider the further you drill, because uncertainty
grows honestly with distance. Only leaving the corridor costs anything.

Three properties, all needed:

| property | why |
|---|---|
| centred on a **sloped** trend, not a flat level | good wells legitimately drift along the dip |
| **deadzone**, zero penalty inside the funnel | only genuine jumps get punished |
| **widens** with distance drilled | real deep dips stay reachable far from the heel |

Plus or minus 5 ft at the heel, plus or minus 45 ft five thousand feet out.

![The funnel](assets/fig07_funnel.png)

---

## 7. Drill-follow re-centre

My favourite trick here.

$$c(d) = \mathrm{geo}_0 + (1-w_f)\cdot \underset{\mathrm{straight}}{a_k d} + w_f \cdot \underset{\mathrm{hold\ TVT}}{(Z_t - Z_0)}, \qquad w_f = 0.7$$

**Intuition:** the well was geosteered. A human was steering the bit to stay inside the layer, so
**when the rock went down, the driller went down.** `Z(MD)`, which we can see, is a recording of the
rock's shape made by a human sensor. Straight extrapolation throws that away.

Uses only the trajectory. No truth, no neighbours. Free information sitting in the survey file.

**Measured on all 773 wells** (panel c above): the straight centre is **36.4 ft** pooled from the true
geo and runs away; drill-follow is **15.6 ft** and stays bounded.

**Worth -0.86 ft pooled**, the largest single lever in the decoder, and it is one line of arithmetic.
It also wrecks the minority of wells that genuinely dip, which is why the next trick exists.

---

## 8. Follow-gate

Decode the well **both ways**, keep whichever fits the observed gamma better.

$$E(p) = \frac{1}{N}\sum_t \frac{1}{2}\log\!\left(1 + \left(\frac{GR_t - f_{tw}(p_t)}{\sigma}\right)^2\right), \qquad \mathrm{keep\ straight\ if}\ E_{off} < E_{on} - 0.05$$

**Intuition:** ask the well which story it prefers. If you force a well to hold depth while the rock is
really diving, the decode has to bend the gamma to fit, and that strain shows up as a worse gamma
score. So decode it both ways and keep the version that does not have to fight.

**The general pattern.** When a lever is right for the majority and catastrophic for a minority, do not
try to predict which case you have from side-information. Run both arms and let the data referee. A
wrong funnel centre forces the decode to fight the gamma, and that fight is measurable.

![The follow-gate](assets/fig08_followgate.png)

It flips **24 of 773 wells** and costs +0.069 ft against always-following, so it is roughly free
insurance. Honest accounting: 17 flips win, 7 lose, and one well (`f6d009f4`, 16.2 to 49.4 ft if
unflipped) is 84% of the net. An oracle gate would be worth -0.70 ft, so the idea is right and the
discriminator is what is missing. Cost: it doubles the decode budget per well.

---

## 9. Consensus arbitration

Decode under four slope priors, then switch away from baseline only if **two independent witnesses agree**.

$$S_1: \quad \overline{|\mathrm{geo}_m - \mathrm{anchor}|} < \overline{|\mathrm{geo}_{base} - \mathrm{anchor}|} - 5\ \mathrm{ft}$$
$$S_2: \quad E(m) \leq 1.02 \cdot E(base)$$

**Intuition:** the gamma can tell you the shape of where you are, never the absolute depth. The
neighbouring wells can, because we know exactly how deep the rock sits under them. So we get a second,
completely different witness, and only overrule the decode when both witnesses point the same way.

The **anchor** is the one channel carrying absolute depth. Training wells have known TVT, so
`geo = TVT + Z` is known at their coordinates, and the rock surface is smooth in (X, Y). A local
ridge-regularised plane fit over the 90 nearest training points predicts our geo, with no gamma involved.

![The structural anchor](assets/fig09_anchor.png)

**The elegance is in S2.** A well that is already correct fits *its own* gamma best by construction, so
nothing can pass S2. Good wells are protected for free, with no gate to tune and no classifier to
overfit. All the risk is one-sided. Fires on 9 of 773 wells.

---

## 10. The two rescues

**Intuition:** keep two specialist doctors on call. One of them refuses to let the path make a big jump
at all; the other checks whether the *shape* of the last few hundred feet really matches, not just the
numbers. Neither is a good family doctor, so they are only called in when two independent tests say the
normal decode is sick.

Two extra decode variants offered to the same S1 and S2 gate:

| variant | what it changes | what it kills |
|---|---|---|
| bounded-slope | slope grid clipped to abs(s) <= 0.08 | removes the escape route a cycle-skip needs |
| segment-NCC | rewards shape correlation of each segment's detrended gamma | a wrong marker matches level, not local shape |

![The rescues](assets/fig10_rescue.png)

**Why gated, not global.** Blanket-tightening the slope grid costs **+1.5 ft per good well**, because
good wells need transient slope freedom to course-correct. The blanket NCC term costs +0.14 ft pooled.
Both are strictly worse as defaults and strictly better as options.

**17 wells switched, zero regressions.** The example above goes 13.8 to 0.9 ft.

> Reusable pattern: build a variant that is worse on average but better in one failure mode, then admit
> it only when two independent witnesses agree. Averaging cannot do this, because a cycle-skip is a
> discrete mode choice and averaging two modes gives a depth that is neither.

---

## 11. Decorrelated canceller

**Intuition:** ask four colleagues who use different reasoning and average their answers. Because their
mistakes point in different directions, the mistakes partly cancel and the truth survives. It does not
matter that each colleague alone is worse than you are, only that they are wrong *differently*.

Average the consensus with four deliberately detuned decodes:

| member | weight | detuning |
|---|---|---|
| consensus | 0.46 | the path we already have |
| `gs45_gradhi` | 0.18 | shape-heavy emission on a fixed wide GR scale |
| `tight` | 0.18 | bounded-slope grid |
| `grad_only` | 0.08 | almost pure GR shape |
| `gradhi_tight` | 0.10 | shape-heavy emission on the bounded grid |

$$\mathrm{Var}(\hat p) = \sum_{j,k} w_j w_k \rho_{jk} \sigma_j \sigma_k$$

All the gain is in the off-diagonal, so members must fail **independently**. Each trusts a different
feature: level, shape, slope bound. Measured error correlations: **0.59 to 0.78**.

![The canceller](assets/fig11_canceller.png)

**Panel c is the point.** Alone the members score 9.72, 11.27, 9.79, 10.69, 9.64 ft pooled, every one
*worse* than the consensus. Their weighted average scores **9.01**. Two members are individually
terrible and still improve the mix.

> A member needs to be DECENT and DECORRELATED, not individually good. This recurs three more times.

---

## 12. Trust regions and guards

**Intuition:** you cannot tell in advance which correction will backfire, so you cap how loud any single
correction is allowed to shout. It is a seatbelt, not a driver.

$$p^{capped}_t = p^{pre}_t + \mathrm{clip}(p^{mix}_t - p^{pre}_t,\ -10,\ +10)$$

![The trust region](assets/fig12_canccap.png)

**Why a bound and not a learned gate.** Three independent signal families were tested for "will this
stage hurt this well": decode byproducts AUC 0.54 to 0.61, neighbour geo-anchor 0.540, within-well
heel-holdout rho about 0. All coin flips. **Conditional harm is not identifiable from anything free.**
So bound the move instead of predicting the outcome.

Two more guards ride on top:

| guard | what | effect |
|---|---|---|
| blend cap | clamp the net family+dip+znorm move to 30 ft from the raw decode | clips 6 of 773 wells |
| znorm skip-gate | 5-feature logistic reverts znorm where P(hurt) is high | moves 330 wells, -0.005 ft |

The skip-gate signal is interpretable: big MAX move plus small MEAN move means one segment jumped, so revert.

![The guards](assets/fig16_guards.png)

---

## 13. Robust low-order projection

Fit a robust degree-4 polynomial (IRLS, Tukey reweighting, 4 iterations) to the decoded geo against
normalised MD, anchored at the heel, then blend 50/50 with the raw path.

**Intuition:** the decode's fine wiggles are search noise, but its slow bend is real error. A stiff curve
keeps the bend and forgets the wiggle. The raw path carries beam jitter, which is pure noise, while the dominant residual is
slow whole-well drift, which a low-order polynomial captures exactly. Robust is essential, since a plain
polyfit gets dragged by a cycle-skipped section.

![The projection](assets/fig13_projection.png)

Tiny (-0.049 ft) but a Pareto win: it improves good, medium and bad strata, interacts with nothing, and
costs one polyfit.

---

## 14. Family-enhanced typewell

The biggest single trick, and the one that nearly sank the solution.

**Intuition:** we were handed one photocopy of the book. Dozens of other wells drilled the same rock and
recorded their own copies, and we know exactly what depth each of their pages came from. Stack the
copies, take the median page, and you get a cleaner book. Then only trust it where the copies agree, and
only if it explains our own heel better than the photocopy did.

The provided typewell is one offset well: noisy, sometimes short, sometimes wrong for our location. But
dozens of training laterals drilled the same rock and logged gamma at known TVT.

```python
# 1. MATCH   within 12,000 ft, GR-shape correlation > 0.90, leave-one-out
# 2. ALIGN   remove each member's own depth datum by cross-correlation x-shift
# 3. POOL    one median per 0.5 ft TVT bin
# 4. FILL    where NO member reached, fall back to the provided typewell
#            (otherwise np.interp clamps FLAT and the emission carries zero information)
# 5. DAMP    per-bin member DISAGREEMENT decides how much to trust the family
# 6. GATE    keep it only if it fits THIS well's known heel better than the provided curve
```

Then re-decode the well four ways on that reference and fold in by a robust spread route:

$$\hat p_t = (1-w_t)\cdot \mathrm{median}_k p^{(k)}_t + w_t \cdot \mathrm{blend}(p^{enh})_t, \qquad w_t = \mathrm{sigmoid}\!\left(\frac{\mathrm{std}_k p^{(k)}_t - 6}{3}\right)$$

Where the four decodes agree, take their **median**, a selector that returns a real member's depth and
never an average across two incompatible modes. Where they disagree, lean on the enhanced reference.

![The family typewell](assets/fig14_family.png)

**Worth -0.81 ft true effect, -2.18 ft on the hard tier.**

### The scar tissue

The ungated version **regressed the real leaderboard from 6.3 to 7.232**. CV loved it, the test set hated
it. The reason is structural:

> Test wells are held out **together**. A test well's true nearest neighbours are other test wells, which
> are absent from the training pool. So CV finds genuine neighbours and inference finds coincidental
> far-away GR matches.

The fix was not a better model, it was a **self-disabling gate**: pool only wells within 12,000 ft, and
if fewer than 3 qualify, switch the whole correction off for that well. It now applies to 565 wells and
drops itself on 208.

---

## 15. Two spatially-safe blends

**Intuition:** two more colleagues, both of whom only look at this one well, so nothing about the field
layout can betray them at test time. One tilts the whole picture by the regional dip before reading it.
The other throws away the brightness and contrast of both gamma logs and reads only their texture.

Same idea as the canceller, using only per-well information so transfer cannot fail.

![diponly and znorm](assets/fig15_dip_znorm.png)

| trick | what | weight | why it is safe |
|---|---|---|---|
| **diponly** | decode again with the frame tilted, `Z~ = Z - s(MD - MD_heel)`, `s` from a global regional-dip regression | 0.15 | a global constant transferred train to test, not a neighbour lookup |
| **znorm** | decode again with both GR logs z-normalised to mean 90, sd 18 | 0.10 | per-well, leak-free, GR is observed everywhere |

znorm is **weak alone** (13.3 ft) but its per-well errors correlate only **0.39** with the consensus,
against 0.75 for diponly. That decorrelation is the entire point.

Its true effect is **-0.40 ft, eight times what the waterfall credits it with (-0.048)**, because it runs
last and the tricks ahead of it had already banked the shared credit.

---

## 16. track8 and the learned gate

**Intuition:** get a second opinion from a doctor trained at a different school. Everything above is the
same decoder wearing different hats, so its mistakes rhyme. track8 does not, so it removes error the
internal averaging structurally cannot. And we let it speak loudest exactly where our own stages argued
with each other.

Blend in a completely different model: a LightGBM stack on particle-filter features that consumes none of
the decoder's output. Per-well RMSE correlation with the decoder is **0.57**, the most independent view
available, so it removes error the internal canceller structurally cannot.

![track8](assets/fig17_track8.png)

$$P_{hurt} = \mathrm{sigmoid}(0.7734 + 0.0776 \delta_{fam} - 0.3042 \delta_{dip} - 0.6026 \delta_{zn})$$
$$w_8 = \frac{0.45}{1 + \exp((P_{hurt} - 0.60)/0.03)}, \qquad \hat p^{final} = (1-w_8)\hat p + w_8 p_{track8}$$

The features are the pipeline's **own disagreement shifts**: how far each blend stage had to move the
path. A well where every stage agreed is confident; a well the stages fought over is contested. Four
hard-coded constants, fitted offline with spatial GroupKFold, no in-kernel training.

Realised `w8`: mean 0.19, median 0.12, cut below the 0.45 cap on **723 of 773 wells**.

Its interaction with znorm is **-0.49 ft**, the largest in the pipeline and bigger than either trick's
own main effect. Neither is worth much alone; together they are the best thing in the stack.

---

## 17. Visible-well shortcut

**Intuition:** check the answer sheet before sitting the exam.

Three test wells also sit in train with their marker picks intact. Compute TVT straight from the rock
contacts (RMSE about 0.007 ft) and skip the entire pipeline including the final blend. Without the
explicit skip, the learned gate leaked w8 about 0.026 onto wells whose answer was already exact.

Always check whether part of your test set is simply given to you.

---
---

## Results

![The waterfall](assets/fig18_waterfall.png)

| stage | pooled RMSE | delta |
|---|---|---|
| raw decode, straight centre | 10.660 | |
| + drill-follow re-centre | 9.988 | **-0.672** |
| + follow-gate | 10.048 | +0.060 |
| + consensus arbitration | 9.757 | -0.291 |
| + tight rescue | 9.110 | -0.560 |
| + canceller | 8.618 | **-0.492** |
| + canceller trust region | 8.573 | -0.045 |
| + projection | 8.533 | -0.040 |
| + family typewell | 7.699 | **-0.834** |
| + diponly | 7.622 | -0.077 |
| + znorm and its gate | 7.633 | +0.011 |
| + blend cap | 7.639 | +0.006 |
| + track8 (leak-free OOF) | **7.296** | -0.343 |

> Deployed scores better than this last row. The diagnostic uses an out-of-fold track8; production ships
> track8 trained on all wells. That gap, not a missing trick, is the difference between 7.30 and 5.21.

### Sequential deltas lie. The factorial does not.

![Synergy](assets/fig19_synergy.png)

| trick | true effect | sequential | ratio |
|---|---|---|---|
| family | **-0.811** | -0.820 | 1.0x |
| canceller | -0.404 | -0.665 | 1.6x |
| track8 | -0.402 | -0.455 | 1.1x |
| znorm | -0.399 | -0.048 | **0.12x** |
| diponly | -0.261 | -0.116 | 0.45x |
| rescue | -0.260 | -0.473 | 1.8x |
| projection | -0.049 | -0.044 | 0.9x |

1. **This is a pile of overlapping correctors, not a team.** 18 of 21 pairs are redundant. Main effects
   sum to -2.585 ft; the pipeline delivers -2.621. They largely fix the same wells the same way.
2. **znorm x track8 = -0.49 ft** is the only thing that truly compounds.
3. **family is redundant with everything it meets** (worst overlap +0.235 with track8). Judge any new
   view by its interaction with track8, and be sceptical of anything overlapping family's territory.

![Where](assets/fig20_where.png)

Error grows away from the heel, the only anchor, and the pipeline shaves a roughly constant 30% at every
distance. Across 773 wells it helps 511, hurts 227, leaves 35 unchanged. The wells it hurts are easy
wells taxed by a fraction of a foot; the wells it helps include a dozen catastrophes pulled from 30 to
50 ft down to single digits. Under a pooled metric that trade is strongly positive.

---

## The whole pipeline

```python
# ONE-TIME, FROM TRAIN
params     = fit_segment_priors(train)          # LogNormal duration + AR(1) slope
ens_priors = 4 slope-prior variants             # base / tight / vtight / loose
surf       = KDTree of (X,Y) -> geo=TVT+Z       # structural anchor
fam_data   = per-well binned GR(TVT) + (X,Y)    # family pool
dip_model  = global regional-dip regression

# PER TEST WELL
def process_well(wid):
    if wid in train and has marker picks:
        return tvt_from_contacts(...)           # exact, skip everything

    # 1. FOLLOW-GATE: decode both ways, keep the better gamma fit
    p_on  = decode_consensus(follow_w=0.7)
    p_off = decode_consensus(follow_w=0.0)
    cons  = p_off if emiss(p_off) < emiss(p_on) - 0.05 else p_on

    # 2. FAMILY STACK: 5 decodes on a neighbour-pooled reference, gated, spread-routed
    p = combine_stack(cons, fam_data, dip_model)

    # 3. TWO DECORRELATION BLENDS
    p = 0.85*p + 0.15*decode(Z tilted by regional dip)
    p = 0.90*p + 0.10*decode(both GR logs z-normalised)

    # 4. GUARDS
    p = soft_revert_znorm_if_logistic_says_hurt(p)
    p = cons + clip(p - cons, -30, +30)

    return p, {family, diponly, znorm disagreement shifts}

def decode_consensus(follow_w):                 # about 10 decodes
    preds  = {name: decode_well(prior) for prior in ens_priors}     # 4
    best   = argmin |geo_m - surface_anchor|
    result = preds[best] if (anchor prefers by >5 ft
                             and emiss <= 1.02*emiss(base)) else preds['base']
    result = tight_rescue(result)               # S1 and S2 and sanity     (+1)
    result = ncc_rescue(result)                 # S1 and S2 and sanity     (+1)
    result = canceller_blend(result)            # 4 detuned decodes        (+4)
    result = clip(result, +-10 ft from pre-canceller)
    return 0.5*result + 0.5*robust_deg4_fit(result)

# FINAL BLEND
P_hurt = sigmoid(0.7734 + 0.0776*fam - 0.3042*dip - 0.6026*zn)
w8     = 0.45 / (1 + exp((P_hurt - 0.60)/0.03))
final  = (1 - w8)*prediction + w8*track8
```

**Budget:** 2 x (4 + 1 + 1 + 4) + 5 family + 1 diponly + 1 znorm = about **27 beam decodes per well**.
A numba kernel plus a per-well decode cache make it fit; wells are independent so one runs per core.

---

## What did NOT work

| dead end | result |
|---|---|
| reference built from neighbouring **laterals** | even oracle-aligned pooling fits 7.9% *worse* than the provided typewell |
| better neighbour typewell selection | 773 typewell files are only about 61 distinct master curves, so selection is a no-op by data layout |
| optimising distance-to-oracle | reducing reference distance and improving the decode are **in opposition**, confirmed over 175 experiments |
| per-well routing and conditional gates | AUC 0.54 to 0.61 for every free feature, across three signal families |
| multi-way veto router | 29.1% accuracy against a 30.5% break-even bar, realised **+0.565 ft worse** |
| neural corrector (about 15 variants) | all land 10 to 27 ft against classical 7.2 |
| fan-attention slope net | -0.400 ft nested spatial OOF, then LB **5.21 to 6.00** |
| residual learner on decode byproducts | out-of-fold correlation 0.026: information absent, not capacity |
| per-well constant-offset debias | LB **5.3 to 5.5**; the "systematic undershoot" was a train-only artifact |
| `RESCUE_RELAX` | best CV profile of any unshipped lever (6/6 folds, LOFO, drop-best-5, P=0.987) and it **regressed anyway** |

**The ceiling.** A 12-piece per-well piecewise-linear oracle reaches 0.80 ft, and its knots are exactly
the cycle-skips. But the drift is not in the gamma (correlation about 0 with every free feature), the
neighbour surface collapses under spatial CV, and locating a knot does not tell you which way to move.
**97% of the remaining error is coherent per-well drift that decorrelated blending cannot touch.**

---

## Five sentences

1. Pick the coordinate the physics is simple in. `geo = TVT + Z` is worth more than any model choice after it.
2. Read the free channels. `Z(MD)` is a recording of a human tracking the rock, and using it as the funnel centre is one line of arithmetic and -0.86 ft.
3. When a lever helps the majority and destroys a minority, run both arms and let the data referee.
4. Combine configurations, never select one, but only if members fail independently, and never across a discrete mode choice.
5. When you cannot predict whether a correction will hurt, bound how much it can.

*Figures regenerate with `python3 writeup/figs_a.py && python3 writeup/figs_b.py`.*

---

**Note:** the image references above (`assets/fig01_problem.png` … `fig20_where.png`) are relative paths to figures that live alongside this file in the original project folder. Only `SOLUTION.md` and the final solution file (`5point21.py`) were pushed to this repo. The fully rendered writeup with all 20 figures inline is on the Kaggle solution writeup.
