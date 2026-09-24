# Selectivity, Causal Influence, and Behavioral Recovery under Visual Occlusion

## Detailed research report

**Evidence base:** the complete uploaded repository at commit `26005f6b0145886436e43689c1dfbe233691f682`. This report describes a fresh empirical study. The companion manuscript is `paper/main.pdf`; complete statistical tables and source coverage are in `SUPPLEMENTARY_MATERIALS.md` and `reports/FILE_AUDIT.md`.

## 1. Main finding

The strongest supported contribution is a careful separation of three things that are easy to confuse: a feature responding selectively to occlusion, a feature intervention changing an action prediction, and an intervention improving executed behavior. The study finds reproducible selective features, small but positive held-out effects from attribution-selected features, and no reliable success recovery from restoring the tested small feature sets. Full-MLP restoration, in contrast, nearly restores clean success.

For the Spatial box condition, 452 of 73,728 transcoder features pass discovery selectivity. The capped 129-feature set reproduces selectivity at 83.72%, but injecting that set closes only 0.16% of the action gap. A discovery-ranked set of 100 attribution features closes 1.49% [0.65, 2.54], compared with 0.04% for its matched random control. All transcoder features close 56.85%, and all MLP outputs close 73.26%. In closed loop, clean success is 58/60 and covered success is 10/60. Restoring attribution features yields 8/60; restoring all MLP outputs yields 59/60.

These observations support a selectivity-causality-behavior dissociation for the tested interventions. They do not establish that all useful representations are inherently distributed, that no compact circuit exists, or that the policy lacks object permanence. The experiments use visual occlusion edits, a fixed sparse dictionary, and a particular ranking and patching procedure.

## 2. Problem statements and research questions

**Problem 1: concept association does not identify causal mediation.** Features can respond consistently when the target is hidden without accounting for the resulting action change. The study asks whether features selected for occlusion-specific activation reproduce their responses on held-out demonstrations and outperform suitable random controls when patched.

**Problem 2: attribution quality is distinct from intervention quality.** A gradient-based score estimates local sensitivity, but finite changes through a nonlinear action sampler can behave differently. The study asks whether discovery-time attribution identifies useful intervention sets and how measured patch effects scale with feature budget.

**Problem 3: action similarity is not task success.** Closing a counterfactual action gap on saved demonstration frames need not repair an executing policy. Closed-loop interventions test whether restoring clean internal values improves success under occlusion.

**Problem 4: evaluation and reproducibility can be overstated.** A configured ten-task benchmark need not produce usable probes for all ten tasks; a second-suite result is not a second-policy result; and saved summaries do not replace missing replay inputs. The report therefore distinguishes configured coverage, realized samples, statistical units and available artifacts.

## 3. Scope and evidence accounting

All 181 tracked input files were included in the audit: the notebook, experiment and reporting code, configurations, logs, statuses, numerical tables, arrays, checkpoints, images, videos, manuscript material and archive. The results ZIP has 159 members that duplicate extracted artifacts; they are not independent experiments. Every file has a SHA-256 entry in `reports/file_inventory.json`.

The manifest lists 201 artifacts. Of these, 165 can be located and match their recorded sizes and supplied MD5 checksums; 36 are missing: 29 tensor/checkpoint files, four array archives and three replay-frame pickle files. Only seven Spatial transcoder layer checkpoints are uploaded (04, 06, 07, 10, 11, 13, 16), and no Object transcoder layer checkpoints are available.

The numerical verification reads all 14,370 saved intervention rows and all 540 closed-loop episodes. It reproduces summary statistics using the reviewed aggregation definitions without importing or executing the policy or simulator. This is verification of archived results, not a fresh training or inference replication. Full replay requires the missing frames, captured activations and checkpoints, as well as the policy and demonstration dependencies.

The notebook contains 22 cells. Exported experiment/report scripts and saved outputs provide the operative implementation. Status files, logs and aggregate reports were compared so that a success label in a wrapper does not substitute for completion of every scientific stage. The attempted second-policy run stops at a gated tokenizer download with an HTTP 401 error; it supplies no completed transcoder or causal results.

## 4. Experimental methodology

### 4.1 Policy and intervention site

The completed experiments use the LIBERO-finetuned pi0.5 policy. Camera images, language and robot state condition an action-chunk sampler. The code locates the 18 PaliGemma language-prefix layers and hooks their MLP inputs and outputs. These are not action-expert MLPs. The transcoders are not conditioned on a denoising timestep; attribution propagates through the sampler to the prefix perturbations.

Each layer receives a 4,096-feature TopK transcoder with 64 active features per token, giving 73,728 feature coordinates across 18 layers. A transcoder maps the original MLP input to its output. This differs from an autoencoder that reconstructs its own input representation.

For layer input x, the approximation is y-hat = D f(x) + b, with f sparse. Patching a set S adds the decoded donor-minus-recipient feature difference to the real MLP output. The recipient reconstruction residual is retained. Separate controls patch all dictionary features, only the reconstruction error, or the entire MLP output. These interventions help bound the measured effect of the chosen representation and patch site, but their effects are nonlinear and must not be added as an exact decomposition.

### 4.2 Demonstrations and frame selection

Spatial discovery uses demonstration IDs 0 and 1; test uses 2, 3 and 4; additional training demonstrations use 5 through 16. Object discovery uses 0 and 1; test uses 2 and 3; extra training uses 4 through 11. Frames target approach phases 0.30, 0.55 and 0.80 before the first target height increase exceeding 0.01 m, with a three-step pre-lift margin and local search within six steps. Visibility and edit-placement checks require sufficient target pixels and valid controls.

| Quantity | Spatial | Object |
|---|---:|---:|
| Configured tasks | 10 | 10 |
| Tasks represented in causal test frames | 8 | 10 |
| Discovery probe frames | 46 | 60 |
| Test probe frames before family filtering | 70 | 60 |
| Total probes | 116 | 120 |
| Extra training frames | 419 | 299 |
| Training tokens per layer | 396,551 | 386,859 |
| Token holdout per layer | 20,871 | 20,361 |
| Box causal test frames | 70 | 59 |
| Screen causal test frames | 69 | Not run |
| Paint causal test frames | 63 | 48 |

Spatial tasks 4 and 6 provide no retained probe frames; task 7 provides 11 total probes. Other retained Spatial tasks provide 15 each. This does not mean task 6 is absent from the separate rollout experiment.

Feature sets are selected on discovery demonstrations and evaluated on disjoint test demonstrations. However, dictionary training includes discovery frames. The 5% token holdout is not an episode-level dictionary holdout. Consequently, the selection/test split is meaningful for the feature sets, but training and discovery are not mutually independent datasets. All conclusions also remain conditional on the frame inclusion rules: causal analysis excludes normalized action gaps below 0.1, whereas a validity check uses a separate 0.02 threshold.

### 4.3 Visual conditions and controls

The eleven observation conditions comprise base, recolor, absent, box, screen, paint, three miss controls and two distractor controls. The box is rendered around the target and appears in both policy cameras. The screen blocks the third-person camera's line of sight to the target; it does not remove wrist-camera information. Paint replaces target pixels with gray. Box and paint have distractor controls; screen does not. Miss controls place the edit elsewhere, subject to segmentation checks. Recolor tests appearance sensitivity; absent uses a counterfactual render with the target moved out of view and restores simulator state afterward.

The rendered occluders have no collision dynamics. They follow the target, so their position remains informative about target location. They can cover parts of the robot or surrounding context. Therefore these are controlled visual perturbations, not physical obstacle tests or direct demonstrations of persistent hidden-object tracking. The no-physics design isolates visual changes, but the interpretation must retain these limitations.

### 4.4 Training and instrument fidelity

Training uses normalized MSE and a dead-feature auxiliary loss, Adam with learning rate 0.002, batch size 4,096, 8,000 optimization steps, 5% warmup and 20% decay. Retry length is 12,000 steps if the reconstruction criterion fails. Auxiliary TopK is 256 with coefficient 0.03125 and inactivity window 100; layer seeds are based on seed zero plus layer index.

| Fidelity measure | Spatial | Object |
|---|---:|---:|
| Median held-out-token FVU | 0.204 | 0.183 |
| Median held-out-test-frame FVU | 0.294 | 0.334 |
| Reconstruction gate | Fails | Passes |

FVU means fraction of variance unexplained: lower is better. The Spatial reconstruction check fails and the configured nonblocking/force-continue behavior permits later stages. This limits a strong claim that the dictionary captures all causally relevant computations. The large all-feature effects nevertheless provide a useful positive control.

Reported deterministic and batch checks support the tested configurations. The batching check varies row composition at a fixed batch size; it does not establish exact invariance between single-item and every batched execution. These distinctions matter for faithful reproducibility descriptions.

### 4.5 Selectivity, attribution and controls

Selectivity uses token-summed activation changes. A feature must rise on at least 90% of discovery frames, by at least one typical firing magnitude, and satisfy specificity of at least 0.5 relative to recolor, absence and miss responses. A target-specific variant additionally compares the distractor. Capped selection retains up to eight features per layer. The capped Spatial box set has 129 features, while the uncapped set has 452; their replication rates must not be interchanged.

Attribution scores the decoded donor-minus-recipient feature change against the gradient of a directional action-projection objective. Scores from clean-to-occluded injection and occluded-to-clean restoration are averaged over discovery frames. It is a symmetric endpoint construction, not merely a clean-state gradient. Ranking uses alive features in layers 0-13, selected by the discovery residual-swap criterion; full-transcoder patches span all 18 layers. This unequal layer support is an additional reason not to interpret the sparse-versus-full comparison as a pure cardinality experiment.

Primary random controls match per-layer counts and decoded per-token edit norms. Zero random edits remain zero, so exact norm matching is not universal. Random draws are averaged within each frame before statistical aggregation. Budget-curve random controls use raw edits, and the cross-task comparisons reuse controls that are not rematched separately for each fold. The latter tests also retain a dictionary trained across the same overall task collection.

### 4.6 Metrics and uncertainty

For recipient action a-start, donor target action a-target and patched action a-patch, gap closure is:

$$C = 100\left(1-\frac{\operatorname{RMSE}(a_{patch},a_{target})}{\operatorname{RMSE}(a_{start},a_{target})}\right).$$

It is measured on normalized action chunks, not language logits. Zero means no reduction in distance; negative values mean increased distance; 100 is exact action matching. A separate projection metric measures movement along the start-to-target direction. Rollouts apply the policy's action postprocessing before executing commands.

Confidence intervals use 10,000 whole-task bootstrap resamples and a frame-weighted mean. This differs from an equally weighted average of task means when tasks contribute different frame counts. One-sided sign-flip tests use task-mean paired differences and exact enumeration for these small task counts. The stored protocol uses a practical margin of ten closure points for the primary attribution contrast. This is a configured criterion, not evidence of an external preregistration.

There are eight Spatial and ten Object causal task clusters, and five closed-loop task clusters. With so few clusters, intervals are descriptive and potentially unstable. No multiplicity correction is supplied across the many exploratory tests. Wilson success intervals and McNemar tests are also reported but do not model task clustering; task-bootstrap paired differences are more pertinent to the central behavioral comparisons.

## 5. Results

### 5.1 Selectivity reproduces without large patch effects

Spatial discovery yields 452 box-selective features, 417 of them target-specific, 731 screen-selective features and 24 paint-selective features. Object yields 457 box-selective features, 439 target-specific, and 12 paint-selective features, nine target-specific.

For Spatial box, capped replication is 108/129 = 83.72%; uncapped replication is 341/452 = 75.44%. Object box rates are 75.19% capped and 64.11% uncapped. Spatial screen rates are 87.50% and 72.91%; paint rates are 70.83% and 58.33%. These are replication of the operational activation criterion, not semantic precision or causal success.

### 5.2 Held-out action-gap injection

Values below are mean percent gap closure, with task-bootstrap 95% intervals where shown.

| Intervention | Spatial box | Spatial screen | Spatial paint | Object box |
|---|---:|---:|---:|---:|
| Attribution top 100 | 1.49 [0.65, 2.54] | 2.81 [0.44, 5.01] | 5.25 [3.31, 7.47] | 0.90 [0.46, 1.41] |
| Capped selective set | 0.16 [-0.11, 0.50] | 0.45 [0.00, 1.17] | -0.01 [-0.21, 0.15] | 0.20 [0.04, 0.38] |
| All transcoder features | 56.85 | 55.10 | 59.02 | 47.70 |
| All MLP outputs | 73.26 | 78.54 | 75.80 | 74.10 |

Spatial box attribution exceeds matched random by 1.45 points [0.64, 2.47], with a positive task mean on all eight tasks and one-sided sign-flip p = 0.00390625. Thus the effect is statistically detectable under the specified analysis, although well below the ten-point practical margin. Saying attribution has no effect would be wrong.

The selective-minus-random contrast is 0.12 [-0.14, 0.45]. Even the entire 452-feature selective set closes only 0.46% [-0.25, 1.13]. The all-transcoder interval is [49.40, 67.82], compared with [67.46, 80.90] for all MLP outputs. The reconstruction-error-only patch closes 6.88%; it is not an additive missing share of the full effect.

### 5.3 Restoration is asymmetric

Attribution restoration from occluded to clean actions closes 12.52% [3.07, 22.29] for Spatial box. Its contrast with matched random is 12.79 points [3.35, 22.49], meeting the configured ten-point effect-size criterion with an interval above zero. Screen restoration is 7.29%; paint restoration is 18.73%; Object box restoration is 7.12% [3.01, 12.41].

This is an important positive result. The manuscript distinguishes weak injection from stronger action-level restoration rather than presenting every sparse intervention as a failure. Nonetheless, action-level restoration does not translate into demonstrated rollout recovery.

### 5.4 Budget, localization, ranking and diagnostics

For Spatial box, discovery-ranked attribution injection closes approximately 0.19%, 0.40%, 1.49%, 4.65%, 13.61% and 23.65% at K = 10, 30, 100, 300, 1,000 and 3,000. Larger sets improve the result but remain below the full-transcoder and full-MLP controls.

A test-frame-specific attribution diagnostic reaches 16.7% at K = 100 and 29.2% at K = 3,000. Because it uses the test frame itself, it is not a held-out global ranking. Nor is it a mathematical upper bound on all possible rankings. Its advantage suggests that cross-frame heterogeneity or imperfect global selection can contribute to weak sparse-set transfer. It does not prove intrinsic distributedness.

Only eight of the top 100 attribution features overlap the 452 Spatial box-selective features. Across compared features the selectivity-rise/attribution Spearman correlation is approximately -0.0194; Object box has zero top-set overlap and correlation approximately -0.0151. These are dissociations between operational rankings.

Restricting full-MLP patches to occluder-overlapping tokens closes 45.62% [38.53, 56.79], versus 9.89% [4.86, 15.31] for other tokens. Corresponding attribution values are 1.13% and 0.25%. These effects indicate spatial concentration under the chosen token masks but cannot be added to recover a global causal percentage.

Across Spatial box intervention observations, first-order prediction and measured projection correlate at Pearson r = 0.28, slope 0.06; screen and paint correlations are about 0.74 and 0.77. These pooled observations include repeated frames and overlapping sets, so they are diagnostics, not independent-sample validation or evidence of a universal attribution error rate.

The attribution advantage over random is 5.17 points on discovery frames and 1.45 on held-out frames. Cross-task attribution closure is 0.65%, with a reused-control contrast of about 0.62 points. These results support reporting selection-time and held-out effects separately, while retaining the dictionary and control caveats above.

### 5.5 Closed-loop outcomes

Five Spatial tasks (0, 2, 6, 7, 9), twelve episodes each and nine conditions yield 540 episodes. The code requests matched initialization IDs and noise seeds. Saved records do not include hashes proving equality of actual post-reset simulator states.

| Condition | Successes / episodes | Success |
|---|---:|---:|
| Clean | 58/60 | 96.67% |
| Box | 10/60 | 16.67% |
| Screen | 33/60 | 55.00% |
| Paint | 44/60 | 73.33% |
| Box + restore all MLP outputs | 59/60 | 98.33% |
| Box + restore attribution set | 8/60 | 13.33% |
| Box + restore random set | 8/60 | 13.33% |
| Box + restore selective set | 11/60 | 18.33% |
| Clean + inject attribution set | 60/60 | 100.00% |

Attribution restoration minus covered success is -3.33 percentage points [-15.00, 5.00]. Attribution minus random restoration is 0.00 [-11.67, 13.33]. Full-MLP restoration minus covered is 81.67 [68.33, 93.33]. Attribution recovery normalized by the clean-to-covered drop is -0.04 [-0.24, 0.06]; full-MLP recovery is 1.02 [0.95, 1.16]. Recovery can exceed one when restored success exceeds the clean baseline.

The box is selected for restoration after producing the larger observed baseline drop among box and screen on the same rollout collection. Restoration obtains clean counterfactual renders at each current state. It is therefore a privileged diagnostic intervention, not a deployable solution available to a robot whose sensors remain occluded. Object-suite rollouts are not supplied.

Nine videos each show task 0, episode 0 for one condition. They illustrate individual behavior but do not validate all 540 episodes. The paint video renders the simulator scene rather than the final painted policy-input pixels, so it cannot independently show what that policy condition received.

## 6. Novelty and positioning

The defensible novelty is the combined empirical test of reproducible selectivity, held-out finite interventions, matched controls, feature-budget dependence and closed-loop behavior for target-linked visual occlusion in one VLA policy across two suites. The contribution is primarily methodological and empirical, not a new sparse-training algorithm or a recovered causal graph.

Transcoders already approximate MLP computations (Dunefsky, Chlenski and Nanda, 2024). Sparse feature interpretation and steering already exist for VLAs, including Swann et al. (2026), Grant et al. (2026) and Jin et al. (2026). Event-grounded sparse-autoencoder work includes behavioral evaluation; therefore a claim of the first closed-loop VLA interpretability study would be unsupported. Visual-grounding transcoders also have related precedent (Damianos et al., 2026).

The fresh framing is: **reproducible occlusion selectivity is insufficient to identify compact interventions that recover behavior, and held-out action effects must be distinguished from success recovery.** Broad claims that earlier literature always uses overlapping data, lacks causal tests or cannot handle visual control are not justified by this repository.

## 7. Limitations and paper-development priorities

The study is an exploratory negative/positive dissociation, with one policy, two completed suites and five rollout tasks. Its central weakness is identification: unsuccessful tested sets do not exhaust the space of compact circuits. Dictionary error, unequal layer support, contextual ranking heterogeneity and nonlinear interactions remain competing explanations.

To strengthen a paper, first restore the missing replay/checkpoint artifacts and complete an inference-level reproduction. Next evaluate layer-matched full-feature controls, equal-cardinality selectivity/attribution comparisons, adaptive or context-conditioned feature sets, and strictly rematched cross-task controls. Use dictionary training demonstrations disjoint from both discovery and test. Repeat dictionary training across seeds and quantify causal fidelity rather than only reconstruction error.

Extend rollout evaluation to more tasks and the Object suite, preserving actual reset-state hashes and pairing records. Evaluate occluders that do not reveal target location, and compare synthetic occlusion against temporally persistent natural occlusion if making an object-permanence claim. Complete a genuinely independent policy experiment before claiming architecture-level generalization. These are proposed experiments; they are not presented as completed results.

## 8. ICLR manuscript and submission status

The companion manuscript uses the official ICLR 2027 anonymous LaTeX style, with problem formulation, related work, methodology, results, limitations, reproducibility and AI-use statements. It contains no comparisons to earlier project iterations. Author identity and an identifying repository URL are omitted from the anonymous manuscript.

The official guidelines specify a maximum of nine main-text pages, an abstract deadline of September 18, 2026 AoE, and a full-paper deadline of September 25, 2026 at 23:59 AoE. An existing valid abstract registration is therefore necessary for this submission cycle. Preparing or pushing this package does not submit a paper to OpenReview. Authors must review the scientific claims, citations, AI-use statement, anonymity and eligibility before submission.

## References and source locations

- ICLR 2027 author guidelines: https://iclr.cc/Conferences/2027/AuthorGuidelines
- ICLR AI policy: https://iclr.cc/Conferences/2027/AIPolicyForAuthors
- Dunefsky et al., *Transcoders Find Interpretable LLM Feature Circuits*: https://arxiv.org/abs/2406.11944
- Liu et al., *LIBERO*: https://arxiv.org/abs/2306.03310
- Physical Intelligence et al., *pi0.5*: https://arxiv.org/abs/2504.16054
- Zhang and Nanda, *Towards Best Practices of Activation Patching*: https://arxiv.org/abs/2309.16042
- Swann et al., *Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models*: https://arxiv.org/abs/2603.19183
- Grant et al., *Not All Features Are Created Equal*: https://arxiv.org/abs/2603.19233
- Jin et al., *Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies*: https://arxiv.org/abs/2605.17204
- Damianos et al., *Transcoders Trace Visual Grounding and Hallucinations in Vision-Language Models*: https://arxiv.org/abs/2605.22902

Every numerical statement is traceable to the saved result tables, JSON or code through the complete supplementary tables and the file audit. The authored text and numerical checks were prepared with AI assistance; no policy training, simulator intervention or new rollout was performed for this report.
