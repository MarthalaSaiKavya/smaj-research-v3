# Selective Features and Behavioral Causality under Visual Occlusion in a VLA Policy

[Anonymous ICLR PDF](paper/main.pdf) | [LaTeX source](paper/main.tex)

# Introduction

Vision-language-action (VLA) policies turn multimodal observations and
instructions into robot actions. Their internal features offer potential
explanations of why visual changes alter control, but three questions
must be separated. Does a feature respond consistently to a condition?
Does changing that feature alter the action? Does the resulting
intervention improve task performance? A positive answer to the first
question does not imply positive answers to the other two.

We study this separation for target-linked visual occlusion in a
LIBERO-finetuned $\pi_{0.5}$ policy (Physical Intelligence et al. 2025;
Liu et al. 2023). The target is the object identified by the task
instruction. We construct rendered box and screen occluders, alongside a
pixel-paint control, and train sparse transcoders to approximate
language-prefix MLP computations. Feature sets are selected on discovery
demonstrations and patched on disjoint test demonstrations. We compare
selectivity-selected and gradient-attribution-selected features, include
random and full-MLP controls, and test restoration during execution.

The central result is a dissociation rather than a recovered circuit.
Selective features replicate their activation criterion, but the capped
selective set produces little action-gap closure. Attribution identifies
a small positive injection effect and a larger restoration effect.
Neither tested small set reliably restores task success under the box
occluder, while full-MLP restoration nearly recovers the clean baseline.
A second suite reproduces the small injection effects; it does not
provide a second-policy or closed-loop replication.

Our contributions are: (1) an empirical separation of selectivity,
held-out action effects and behavioral recovery under controlled visual
perturbations; (2) quantitative comparisons of feature selection,
intervention direction, budget and token scope; and (3) explicit
accounting for realized task coverage, reconstruction fidelity and
reproducibility limits. The results constrain the tested dictionary and
intervention procedures. They do not prove inherent distributedness or
establish an object-permanence mechanism.

# Related work

**Sparse representations and causal interventions.** Transcoders
approximate MLP input-output maps with sparse features (Dunefsky,
Chlenski, and Nanda 2024). Unlike reconstruction of a single activation
space, this formulation supplies decoded feature contributions to an MLP
output. We apply this established representation to a VLA prefix and
evaluate finite interventions. Activation-patching conclusions depend on
metrics, corruption choices and controls (Zhang and Nanda 2024); we
therefore report both action-gap closure and closed-loop success, with
uncertainty at the task level.

**VLA and visual-model interpretation.** Sparse-autoencoder
interpretation and steering have been studied in VLAs (Swann et al.
2026), including comparisons between feature types (Grant, Zhao, and
Wang 2026) and event-grounded feature learning with behavioral
evaluation (Jin et al. 2026). Transcoders have also been applied to
visual grounding in vision-language models (Damianos et al. 2026). Thus
neither sparse VLA features nor behavioral interventions are new in
isolation. Our specific contribution is the joint evaluation of
occlusion selectivity, discovery-ranked interventions, held-out transfer
and behavioral recovery. We make no general claim that related studies
lack held-out or causal evaluation.

# Method

## Policy, data and intervention site

The completed runs use the LIBERO-finetuned $\pi_{0.5}$ policy on
Spatial and Object suites (Physical Intelligence et al. 2025; Liu et al.
2023). The policy conditions an action-chunk sampler on camera images,
language and robot state. The implementation hooks MLP inputs and
outputs in the 18 PaliGemma language-prefix layers. These are *not*
action-expert MLPs. Transcoders do not have denoising-timestep
conditioning; attribution differentiates through the action sampler to
the prefix perturbations.

Spatial discovery uses demonstration IDs 0 and 1; test uses 2–4; extra
training uses 5–16. Object uses discovery IDs 0 and 1, test IDs 2 and 3,
and extra training IDs 4–11. Probe frames target phases 0.30, 0.55 and
0.80 of the approach before the first target lift exceeding 0.01 m, with
a three-step pre-lift margin. A local search within six steps enforces
visibility and edit-placement constraints, including a target mask of at
least 80 pixels.

The realized Spatial sample contains 46 discovery and 70 test frames
across eight tasks, despite ten configured tasks. Tasks 4 and 6 supply
no valid probes. Object supplies 60 discovery and 60 test frames across
ten tasks. Extra training uses 419 Spatial and 299 Object frames.
Dictionary training includes discovery data; test demonstrations are
excluded. The dictionary’s random token holdout is not an independent
episode holdout.

## Visual conditions

<span class="smallcaps">Box</span> inserts a render-only opaque box
around the target, visible to both cameras.
<span class="smallcaps">Screen</span> places a panel between the
third-person camera and target, leaving wrist-camera information
available. <span class="smallcaps">Paint</span> replaces target pixels
with gray. Each family has a segmentation-checked miss control. Box and
paint also have distractor controls; screen does not. Base, recolor and
absent conditions complete eleven observation conditions. Absence is
rendered by temporarily moving the target away and restoring simulator
state.

The box and screen are recomputed from live state and have no collision
dynamics. Their location reveals target location, and their extent can
hide contextual or robot pixels. The experiments therefore manipulate
visual evidence without testing physical obstacle interaction. They do
not directly test memory for a persistently hidden object.
Counterfactual clean renders used for restoration are privileged
diagnostics.

## Transcoders and feature selection

For each layer $\ell$, a TopK transcoder with 4,096 features and 64
active features per token approximates the MLP output:
$$\widehat y_\ell(x)=D_\ell f_\ell(x)+b_\ell.$$ Across 18 layers there
are 73,728 feature coordinates. Training uses normalized MSE plus a
dead-feature auxiliary loss, Adam at learning rate 0.002, batch size
4,096 and 8,000 steps, with a 12,000-step retry when needed. Token
holdout is 5%. Additional settings appear in
Appendix <a href="#app:config" data-reference-type="ref"
data-reference="app:config">7</a>.

To patch feature set $S$ from donor $d$ into recipient $r$, we add
$$\Delta y_{\ell,t}=\sum_{j\in S_\ell}D_{\ell,j}\big(f^d_{\ell,j,t}-f^r_{\ell,j,t}\big)$$
to the real recipient MLP output. This preserves the recipient
reconstruction residual. We separately patch all transcoder features,
the residual error only, and full MLP outputs. Their measured effects
are nonlinear and are not an additive causal decomposition.

**Selectivity.** Discovery-time selection sums activation over tokens. A
feature must increase on at least 90% of frames, rise by at least one
typical firing magnitude, and meet specificity 0.5 relative to recolor,
absent and miss responses. A target-specific variant also compares a
distractor. Capped sets retain at most eight features per layer. We
report capped and uncapped replication separately.

**Attribution.** For each discovery frame, decoded feature differences
are scored against gradients of a directional action-projection
objective. The injection contribution has the form
$$\alpha_{\ell,j}=\sum_t (f^o_{\ell,j,t}-f^b_{\ell,j,t})
\left\langle D_{\ell,j},\frac{\partial m}{\partial y_{\ell,t}}\right\rangle.$$
Injection and restoration endpoint scores are averaged, then aggregated
over discovery frames. The top 100 alive features are ranked in layers
0–13, selected by a discovery residual-swap criterion. Full-transcoder
patches span all 18 layers; this support mismatch limits a pure
feature-count interpretation of sparse-versus-full comparisons.

**Controls.** Primary random controls match per-layer counts and decoded
per-token edit norms, with repeated draws averaged within frame. Zero
random edits remain zero. Budget-curve random controls use raw edits.
Cross-task comparisons reuse controls without fold-specific rematching
and retain the overall dictionary, so they are not entirely
task-isolated training replications.

## Metrics and statistical analysis

For recipient action $a_s$, target action $a_t$ and patched action
$a_p$, we measure normalized action-chunk gap closure
$$C=100\left(1-\frac{\operatorname{RMSE}(a_p,a_t)}{\operatorname{RMSE}(a_s,a_t)}\right).
\label{eq:closure}$$ Negative values indicate increased distance. A
separate projection metric measures movement along the start-to-target
direction. Injection transfers occluded feature values into the clean
run; restoration transfers clean values into the occluded run. Neither
metric is a language-logit score. Rollouts execute postprocessed
actions.

Causal tests require an action gap of at least 0.1, distinct from the
0.02 validity threshold. Retained Spatial box/screen/paint frame counts
are 70/69/63; Object box/paint counts are 59/48. Confidence intervals
use 10,000 whole-task bootstrap resamples of a frame-weighted mean.
Paired one-sided sign-flip tests enumerate signs over task-mean
differences. The stored protocol sets a ten-point practical margin for
the primary attribution contrast; no external preregistration is
available. Exploratory tests have no multiplicity correction.

# Results

## Instrument fidelity and selective responses

Median held-out-token FVU is 0.204 for Spatial and 0.183 for Object; on
held-out test frames it is 0.294 and 0.334. FVU is variance
*unexplained*. The Spatial reconstruction criterion fails, and
nonblocking configuration permits continuation. The dictionary is
therefore an imperfect instrument, not a complete account of the policy
computation. The saved batching check tests row composition at a fixed
batch size, not universal single-item/batched equivalence.

Spatial discovery identifies 452 box-selective features (417
target-specific), 731 screen-selective features and 24 paint-selective
features. Object identifies 457 box-selective features (439
target-specific) and 12 paint-selective features (nine target-specific).
For Spatial box, 108/129 capped features reproduce selectivity (83.72%),
while the uncapped rate is 341/452 (75.44%). Object box rates are 75.19%
capped and 64.11% uncapped. These rates validate the operational
activation criterion; they do not by themselves identify causal
mediators.

## Small injection effects and asymmetric restoration

Table <a href="#tab:action" data-reference-type="ref"
data-reference="tab:action">1</a> reports held-out injection effects.
Spatial box attribution exceeds matched random by 1.45 points \[0.64,
2.47\], with positive task means on all eight tasks (one-sided sign-flip
$p=0.00390625$). This is detectable but falls below the ten-point
practical margin. The capped selective contrast is 0.12 \[$-0.14$,
0.45\]. Even all 452 selective features close only 0.46% \[$-0.25$,
1.13\].

<div id="tab:action">

| Intervention     |      Spatial box | Spatial screen |    Spatial paint |    Object box |
|:-----------------|-----------------:|---------------:|-----------------:|--------------:|
| Attribution, 100 |             1.49 |           2.81 |             5.25 |          0.90 |
| % CI             |    \[0.65,2.54\] |  \[0.44,5.01\] |    \[3.31,7.47\] | \[0.46,1.41\] |
| Capped selective |             0.16 |           0.45 |          $-0.01$ |          0.20 |
| % CI             | \[$-0.11$,0.50\] |  \[0.00,1.17\] | \[$-0.21$,0.15\] | \[0.04,0.38\] |
| All transcoder   |            56.85 |          55.10 |            59.02 |         47.70 |
| All MLP          |            73.26 |          78.54 |            75.80 |         74.10 |
| Test frames      |               70 |             69 |               63 |            59 |

Held-out injection: mean percent action-gap closure \[95% task-bootstrap
CI\]. Full controls are shown as means; complete intervals and all
controls are in the supplementary tables. Spatial uses eight task
clusters; Object uses ten.

</div>

All-transcoder Spatial box injection closes 56.85% \[49.40, 67.82\],
versus 73.26% \[67.46, 80.90\] for full MLP outputs. The error-only
patch closes 6.88%. These positive controls show that the intervention
site carries a substantial action effect, while preserving uncertainty
about information outside the learned dictionary.

Restoration is substantially stronger than injection. Spatial box
attribution restoration closes 12.52% \[3.07, 22.29\], exceeding random
by 12.79 points \[3.35, 22.49\]. This contrast meets the configured
ten-point margin with an interval above zero. Screen and paint
restoration close 7.29% and 18.73%; Object box closes 7.12% \[3.01,
12.41\]. A blanket claim that sparse interventions have no causal effect
would therefore be incorrect.

## Budget, ranking and localization

Discovery-ranked Spatial box injection grows from 0.19% at ten features
to 13.61% at 1,000 and 23.65% at 3,000
(Figure <a href="#fig:budget" data-reference-type="ref"
data-reference="fig:budget">1</a>). A test-frame-specific attribution
diagnostic reaches 16.7% at 100 features and 29.2% at 3,000. This
diagnostic uses the test frame and is not a held-out ranking or an upper
bound on all possible circuits. Its advantage is consistent with
context-dependent ranking quality as an alternative to intrinsic
distributedness.

Only eight of the top 100 attribution features overlap the 452 Spatial
box-selective features. Their selectivity-rise/attribution Spearman
correlation is approximately $-0.0194$; Object box has zero top-set
overlap and correlation $-0.0151$. Thus the operational rankings
dissociate despite reproducible selectivity.

Full-MLP patches restricted to occluder-overlapping tokens close 45.62%
\[38.53, 56.79\], versus 9.89% \[4.86, 15.31\] for other tokens.
Corresponding attribution-set effects are 1.13% and 0.25%. These
token-scope effects are not additive. Across pooled Spatial box
intervention observations, predicted and measured projection correlate
at $r=0.28$, slope 0.06; screen and paint correlations are approximately
0.74 and 0.77. Repeated frames and overlapping sets limit
independent-sample interpretation.

Attribution’s advantage over random is 5.17 points on discovery frames
and 1.45 on held-out frames. Cross-task attribution closure is 0.65%,
with a reused-control contrast of approximately 0.62 points. The
dictionary and control qualifications in Section 3 prevent treating this
as fully isolated cross-task training.

<figure id="fig:budget">
<embed src="paper/figures/budget.pdf" />
<figcaption>Spatial box injection versus feature budget. Bands:
task-bootstrap 95% intervals. The test-frame diagnostic uses each test
input for ranking and is not a held-out global method or optimal upper
bound. Random budget controls use raw edits.</figcaption>
</figure>

## Executed behavior and second-suite evidence

Closed-loop evaluation uses Spatial tasks 0, 2, 6, 7 and 9, twelve
episodes per task, and nine conditions: 540 episodes total. The code
requests matched initialization IDs and noise seeds; saved records do
not contain hashes of actual reset states. Box is selected for
restoration after its larger observed baseline drop relative to screen.
At each inference step, restoration obtains a clean counterfactual
render of the current state.

<div id="tab:behavior">

| Condition                      |                    Successes | Rate (%) |
|:-------------------------------|-----------------------------:|---------:|
| Clean                          |                        58/60 |    96.67 |
| Box                            |                        10/60 |    16.67 |
| Screen                         |                        33/60 |    55.00 |
| Paint                          |                        44/60 |    73.33 |
| Box + restore all MLP          |                        59/60 |    98.33 |
| Box + restore attribution      |                         8/60 |    13.33 |
| Box + restore random           |                         8/60 |    13.33 |
| Box + restore selective        |                        11/60 |    18.33 |
| Clean + inject attribution     |                        60/60 |   100.00 |
| Paired contrast                | Percentage points \[95% CI\] |          |
| Attribution restore $-$ box    |   $-3.33$ \[$-15.00$, 5.00\] |          |
| Attribution $-$ random restore |     0.00 \[$-11.67$, 13.33\] |          |
| Full-MLP restore $-$ box       |       81.67 \[68.33, 93.33\] |          |

Closed-loop success. Each condition contains 60 episodes across five
tasks. Paired differences use task-bootstrap intervals; counts are not
60 independent task replications.

</div>

Table <a href="#tab:behavior" data-reference-type="ref"
data-reference="tab:behavior">2</a> shows a large visual-occlusion
impairment and strong full-MLP recovery. Attribution restoration
provides no reliable improvement over either covered or
random-restoration baselines. Its normalized recovery fraction is
$-0.04$ \[$-0.24$, 0.06\], compared with 1.02 \[0.95, 1.16\] for
full-MLP restoration. Small action-level restoration therefore does not
establish behavioral recovery. Injecting the attribution set into clean
rollouts also does not measurably impair success in this sample.

Object-suite box injection reproduces the qualitative separation:
attribution 0.90%, capped selectivity 0.20%, all-transcoder 47.70% and
full-MLP 74.10%. There are no Object rollouts. An attempted
second-policy run failed during gated tokenizer access before producing
comparable results. Generalization is therefore across two suites for
one policy, with behavioral evidence confined to Spatial.

# Discussion and limitations

The findings support a narrow but useful conclusion: reproducible
activation selectivity does not identify the tested small feature sets
as sufficient behavioral mediators. Attribution finds more action
influence, especially in restoration, but does not repair execution.
Full-MLP restoration is a strong positive control and a privileged
intervention rather than a deployable occlusion-robust policy.

Several explanations remain unresolved. The sparse dictionary leaves
reconstruction error, with a failed Spatial fidelity gate. Global
rankings can miss context-specific features; finite patches need not
follow first-order gradients; nonlinear interactions may make isolated
contributions misleading. Ranked and full-feature interventions also
differ in layer support. These possibilities prevent inferring that no
compact circuit exists. A stronger identification study would compare
layer-matched supports, context-conditioned rankings, alternative
dictionaries and multiple seeds.

The occluders reveal target location and lack physics. The screen
preserves wrist information. Demonstration probes emphasize pre-lift
approach states; rollout distributions differ. Discovery overlaps
dictionary training, and the second-suite results do not establish
architecture-level generalization. There are only eight or ten causal
task clusters and five rollout clusters; the many diagnostics are
exploratory and uncorrected for multiplicity. Finally, missing replay
frames, captured arrays and checkpoints prevent full inference-level
reproduction from the uploaded package alone.

# Conclusion

In the tested VLA, selective responses, action-gap mediation and success
recovery are distinct empirical outcomes. Selective features reproduce
their activation criterion, attribution yields measurable but limited
held-out action effects, and tested small-set restoration does not
reliably recover behavior. Reporting these stages together provides a
more precise mechanistic claim than feature selectivity alone.

# Reproducibility statement

The accompanying audit inventories all 181 input files and verifies
14,370 intervention rows and 540 rollout episodes using the saved
records. Appendix <a href="#app:config" data-reference-type="ref"
data-reference="app:config">7</a> and the full supplementary tables
document configuration and results. This statistical verification does
not rerun training or inference. The artifact manifest has 36 missing
entries, including replay inputs and checkpoints; only seven Spatial
layer checkpoints and no Object layer checkpoints are uploaded. These
gaps limit complete reproduction.

# Ethics statement

The reported interventions are simulator-based diagnostics. No
real-robot deployment or human-subject experiment is reported.
Counterfactual access to clean observations is privileged, and the
results do not justify safety guarantees for physical robots.

# AI use statement

AI assistance was used to inspect archived artifacts, prepare numerical
verification code, organize the analysis, and draft and format this
manuscript. No new policy training, simulator intervention or rollout
was performed during manuscript preparation. Human authors must verify
the scientific claims, references and final submission materials and
retain responsibility for them.

# Configuration and complete-result guide

Training uses 396,551 Spatial and 386,859 Object tokens per layer, with
holdouts of 20,871 and 20,361. The auxiliary TopK is 256, coefficient
0.03125, inactivity window 100, warmup 5% and decay 20%; seeds are based
on zero plus layer index. Dictionary dimensions are 4,096 features and
TopK 64 for every layer. The primary attribution budget is 100;
injection budgets are 10, 30, 100, 300, 1,000 and 3,000. Restoration
budgets are 30, 100, 300 and 1,000. The primary random intervention uses
eight draws; secondary and restoration controls use three. Three
single-feature interventions are also evaluated.

| Ranking                |     10 |   30 |  100 |  300 | 1,000 | 3,000 |
|:-----------------------|-------:|-----:|-----:|-----:|------:|------:|
| Discovery attribution  |   0.19 | 0.40 | 1.49 | 4.65 | 13.61 | 23.65 |
| Test-frame attribution |    4.6 |  9.6 | 16.7 | 23.4 |  28.5 |  29.2 |
| Activation change      | $-0.1$ |  0.0 |  0.5 |  1.1 |   2.6 |   4.6 |
| Raw random             |    0.0 |  0.0 |  0.0 |  0.0 |   0.6 |   1.2 |

Spatial box injection, mean percent closure. Test-frame ranking is a
diagnostic; activation-change and raw random controls use the same
displayed budgets.

The full supplementary tables preserve training and validity outputs,
all family-specific method results, restoration, token scopes, budget
curves, paired contrasts, selectivity replication, first-order
diagnostics, in-sample comparisons and every rollout outcome. The file
audit also covers the notebook, logs, status records, environment
information, binary arrays, checkpoints, figures and all nine example
videos. The results archive duplicates extracted files and must not be
counted as independent replication. Videos show only task 0 episode 0;
the paint video does not display the final pixel-edited policy
observation.

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-damianos2026transcoders" class="csl-entry">

Damianos, Dimitrios, Leon Voukoutis, Georgios Skyrianos, Vassilis
Katsouros, and Georgios Paraskevopoulos. 2026. “Transcoders Trace Visual
Grounding and Hallucinations in Vision-Language Models.” *arXiv Preprint
arXiv:2605.22902*.

</div>

<div id="ref-dunefsky2024transcoders" class="csl-entry">

Dunefsky, Jacob, Philippe Chlenski, and Neel Nanda. 2024. “Transcoders
Find Interpretable LLM Feature Circuits.” In *Advances in Neural
Information Processing Systems*.

</div>

<div id="ref-grant2026features" class="csl-entry">

Grant, Bryce, Xijia Zhao, and Peng Wang. 2026. “Not All Features Are
Created Equal: A Mechanistic Study of Vision-Language-Action Models.”
*arXiv Preprint arXiv:2603.19233*.

</div>

<div id="ref-jin2026event" class="csl-entry">

Jin, Xinchen, Aditya Chatterjee, Pranav Kumar, and Rohan Paleja. 2026.
“Event-Grounded Sparse Autoencoders for Vision-Language-Action
Policies.” *arXiv Preprint arXiv:2605.17204*.

</div>

<div id="ref-liu2023libero" class="csl-entry">

Liu, Bo, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and
Peter Stone. 2023. “LIBERO: Benchmarking Knowledge Transfer for Lifelong
Robot Learning.” In *Advances in Neural Information Processing Systems,
Datasets and Benchmarks Track*.

</div>

<div id="ref-pi2025pi05" class="csl-entry">

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan
Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, et al. 2025.
“$\pi_{0.5}$: A Vision-Language-Action Model with Open-World
Generalization.” *arXiv Preprint arXiv:2504.16054*.

</div>

<div id="ref-swann2026sparse" class="csl-entry">

Swann, Aiden, Lachlain McGranahan, Hugo Buurmeijer, Monroe Kennedy III,
and Mac Schwager. 2026. “Sparse Autoencoders Reveal Interpretable and
Steerable Features in VLA Models.” *arXiv Preprint arXiv:2603.19183*.

</div>

<div id="ref-zhang2024towards" class="csl-entry">

Zhang, Fred, and Neel Nanda. 2024. “Towards Best Practices of Activation
Patching in Language Models: Metrics and Methods.” In *International
Conference on Learning Representations*.

</div>

</div>
