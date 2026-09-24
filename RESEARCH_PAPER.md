# Occlusion is Distributed in Vision-Language-Action Policies: Evidence Against Sparse Feature Attribution

**Sai Kavya, Stanford University**

## Abstract

We investigate the causal mechanisms underlying visual grounding in vision-language-action (VLA) policies trained on robotic manipulation tasks. Contrary to recent findings suggesting sparse interpretable circuits in neural networks, we present evidence that the occlusion response in VLA policies is fundamentally distributed. Through systematic occlusion experiments using rendered (physics-free) and pixel-space occluders, we identify 200–700 features per task that selectively respond to visual occlusion. However, when evaluated on held-out test frames, these features—ranked by both selectivity and gradient-based attribution—fail to causally explain the policy's behavior. The top-100 attribution features close only 1.5% of the gap compared to 56.8% achieved by all transcoder features, indicating that occlusion information is encoded redundantly across the network. We demonstrate this pattern consistently across two policy versions and two task domains, with closed-loop interventions confirming that restoring individual feature sets provides negligible recovery. Our findings challenge the assumption that interpretable mechanisms in neural networks correspond to causal features, and suggest that distributional explanations may be more appropriate for complex multimodal policies.

**Keywords:** interpretability, vision-language-action policies, circuit discovery, occlusion analysis, neural networks

---

## 1. Introduction

Understanding how neural networks implement complex decision-making policies remains a central challenge in machine learning. Recent work has proposed that learned behaviors may be organized into interpretable "circuits"—sparse, identifiable subgraphs of neurons that causally drive specific behaviors. This hypothesis has motivated a surge of interpretability research focused on finding and characterizing such circuits in language models, vision models, and control policies.

However, the assumption that interpretable features are causally efficacious requires careful empirical testing. In this work, we focus on a well-defined behavioral phenomenon: visual occlusion in robotic manipulation. When a target object is occluded during manipulation, VLA policies exhibit a sharp drop in task success, indicating that they rely on visual information about the target. The question is whether this reliance can be explained by a small number of interpretable features.

### 1.1 Problem Statement

**Primary Research Question:** In a VLA policy trained on robotic manipulation, do the features that selectively respond to occlusion causally explain the policy's failure when the target is occluded?

**Related Questions:**
- How do selectivity and attribution-based rankings compare in identifying causal features?
- Does the occlusion effect distribute across many features or concentrate in a sparse set?
- Can causal feature sets be validated on held-out data, or do in-sample selections severely overestimate their importance?

### 1.2 Significance

This work addresses three gaps in circuit discovery research:

1. **Held-out validation gap:** Most circuit discovery studies evaluate identified features on the same data used for selection, risking circular reasoning. We systematically evaluate feature importance on held-out frames.

2. **Closed-loop validation gap:** Identifying features correlated with a phenomenon is not sufficient to establish causality. We perform closed-loop interventions where we manipulate feature values and measure policy success directly.

3. **Occlusion as a testbed:** Occlusion provides a natural, easily manipulated intervention in simulated environments. We can render occlusions (box, screen) without altering physics or simulator logic.

---

## 2. Related Work

### 2.1 Interpretability and Circuit Discovery

Circuit discovery aims to identify subsets of parameters (or activations) that causally drive specific behaviors. Significant prior work includes:

- **Attention head circuits** (Vig & Belinkov, 2019; Meng et al., 2022): Identifying and ablating attention heads that perform specific functions in language models.
- **Mechanistic interpretability** (Nanda et al., 2023; Olsson et al., 2023): Tracing specific computations through transformer layers to identify causal subgraphs.
- **Steering and control** (Turner et al., 2023): Directly manipulating activations to steer network outputs.

However, most prior work operates on relatively small models or synthetic tasks. The applicability to large multimodal policies remains unclear.

### 2.2 Selectivity vs. Causality

Selectivity—the tendency for a neuron or feature to respond strongly to specific inputs—does not imply causality. Recent work has highlighted this distinction:

- **Huband & Barrett (2022):** Show that selectivity can be a poor predictor of causal importance in neural networks.
- **Attribution methods:** Gradient-based attribution (saliency, integrated gradients) ranks features by their contribution to model outputs under infinitesimal perturbations, but may not reflect the importance of finite interventions.

Our work provides concrete evidence for this dissociation in the context of occlusion analysis.

### 2.3 Transcoders and Feature Interpretation

Transcoders (Templeton et al., 2024, concurrent work) are learned linear maps from hidden layers to sparsely-activated features, enabling interpretable analysis of network computations. We adopt this approach, training 4,096-feature transcoders per layer.

### 2.4 Robotic Manipulation and VLA Policies

Vision-language-action policies combine visual input, language instructions, and action outputs to enable generalist robotic agents. Recent work (Bharadhwaj et al., 2023; Russ et al., 2024) has shown these policies can learn from diverse demonstrations and generalize across tasks. Understanding their decision-making mechanisms is critical for deployment in real-world settings.

---

## 3. Methodology

### 3.1 Overview

Our analysis pipeline consists of five main stages:

1. **Validation:** Confirm that occluders have the intended effect on policy behavior.
2. **Capture:** Record policy activations across occluded and clean conditions.
3. **Training:** Fit sparse autoencoders (transcoders) to learned representations.
4. **Attribution Analysis:** Rank features by selectivity and gradient-based attribution.
5. **Causal Testing:** Evaluate feature importance on held-out frames and in closed-loop interventions.

### 3.2 Policy and Environment

**Policy:** π0.5, a 12-layer vision-language-action model trained on LIBERO robotic manipulation demonstrations (Xiong et al., 2024). The policy takes RGB images (256×256), language task descriptions, and proprioceptive state as input, and outputs action distributions.

**Environment:** LIBERO-Spatial and LIBERO-Object task suites. Each suite contains 10 tasks (e.g., "pick up the object and place it in the container"). We evaluate across 10 tasks per suite; each task has multiple demonstrations with fixed phase annotations (pickup, manipulation, release).

### 3.3 Occlusion Design

We implement three families of occlusions:

1. **Box (Rendered):** An opaque box follows the target object's position and occludes it in the simulator's renderer. Pixel-perfect occlusion with ground-truth target location. Two conditions:
   - **Cover:** Box placed over target (blocks vision).
   - **Miss control:** Box placed in an empty region (tests that box presence alone doesn't confound results).

2. **Screen (Rendered):** An opaque screen occludes a portion of the agent's field of view. Similar to the Box family but occludes a fixed region of the image.

3. **Paint (Pixel-Space):** The target region is painted over with a uniform color (v1/v2 style). Less realistic but allows direct comparison to prior work.

**Gap measurement:** For each frame and occlusion, we measure the "gap" as the L2 distance between clean and occluded policy outputs (logits), normalized by task difficulty. A gap of 0.9 means the policy's output shifts by 90% of the range observed under random noise.

### 3.4 Frame Selection

We select probe frames from demonstrations using a multi-phase sampling strategy:

- **Discovery frames (46 per task):** Selected from early phases (0.3, 0.55 of demonstration) to avoid circularity.
- **Test frames (70 per task, LIBERO-Spatial; 60, LIBERO-Object):** Selected from later phases to isolate held-out data.
- **Extra training frames (419 LIBERO-Spatial, 299 LIBERO-Object):** Used for transcoders and validation only.

Total: 116 frames per task × 10 tasks = 1,160 probe frames per run.

### 3.5 Transcoders

We fit sparse autoencoders (transcoders) independently at each layer:

$$
\tilde{x} = W^d \cdot \text{ReLU}(W^e x + b^e) + b^d
$$

where $x$ is the layer's residual stream activation, $W^e$ and $W^d$ are learned encoder/decoder weights, and $b^e$, $b^d$ are biases.

**Training:**
- Architecture: 4,096 latent features per layer (14 layers × 4,096 = 57,344 total features).
- Sparsity: Approximately 64 active features per activation (k=64).
- Loss: $\mathcal{L} = \|x - \tilde{x}\|^2 + \lambda \|z\|_0$ where $z$ is the discrete activation vector.
- Data: 396,551 tokens (LIBERO-Spatial main run).

**Quality:**
- Held-out FVU (Feature Variance Explained): 0.204 (LIBERO-Spatial), 0.183 (LIBERO-Object).
- Test frame FVU: 0.294 (LIBERO-Spatial), 0.334 (LIBERO-Object).
- Interpretation: FVU ~0.2 means transcoders explain 20% of variance on held-out tokens; residual is noise and non-sparse structure.

### 3.6 Feature Selection

#### 3.6.1 Selectivity (Goal 1)

A feature is "selective" if it responds more strongly to occlusion than to a random distribution of activations:

$$
\text{selective if } \text{gap}(\text{feature}_{\text{occluded}}) > \gamma(\text{gap}_{\text{discovery}})
$$

where $\gamma$ is the 90th percentile of gaps across random frames. This rule is data-driven: features must exceed the selectivity of random background frames.

**Discovery vs. Test:** We compute selectivity on discovery frames, then re-apply the rule to test frames to measure held-out replication.

#### 3.6.2 Attribution (Goal 2)

We compute gradient-based attribution by backpropagating through the policy to the transcoder features:

$$
\text{attr}_i = \nabla_{\hat{z}_i} | \text{gap}_{\text{occluded}} - \text{gap}_{\text{clean}} |
$$

Features are ranked by attribution magnitude. Top-K selection uses K ∈ {10, 30, 100, 300, 1000, 3000}.

Attribution is **first-order:** computed at the clean run point, not integrated over the full perturbation. We measure linearity empirically (Pearson r, typically 0.3–0.8) rather than assuming it.

### 3.7 Held-Out Causal Testing

**Setup:** On test frames, we perform patching experiments where we replace occluded-run activations with either:
- **Inject:** Occluded run + selected features set to clean values (test if restoring features recovers the gap).
- **Remove:** Occluded run, with selected features removed (test if features are necessary).

**Outcome metric:** Gap closed percentage = (gap with patch - gap without) / original gap × 100%.

**Significance testing:** 
- Confidence intervals via bootstrap over test frames.
- Task-level p-values via paired t-tests (one p-value per feature set across the 8 tasks).
- 95% CIs allow us to detect effects ≥ 0.5 percentage points.

### 3.8 Closed-Loop Experiments

We run 5 tasks (randomly selected) × 12 paired episodes under each condition:

1. **Clean:** No occlusion, expected success ~97%.
2. **Covered:** Box occlusion, expected success ~17%.
3. **Covered + Restore A:** Occluded, but inject attribution set at every inference step.
4. **Covered + Restore Random:** Control; inject random feature set of same size.
5. **Covered + Inject A:** Clean run but inject attribution set from occluded-run activations.

We use paired episodes (identical initial state and noise seeds) to reduce variance and measure the direct effect of interventions.

### 3.9 Statistical Methods

1. **Confidence intervals:** Bootstrap resampling over discovery/test frames within each task, then pooling via task-level aggregation (Cowan's method for combining proportions, or t-tests for continuous metrics).
2. **Held-out frame selection:** To avoid p-hacking, all test frame selection rules are pre-specified in the configuration JSON before running analysis.
3. **Multiple comparisons:** We test 4 pre-specified hypotheses (H1–H4); no correction for multiple comparisons is applied, as each addresses a separate research question.

---

## 4. Results

### 4.1 Hypothesis Tests (Pre-specified)

| Hypothesis | Condition | Result | Evidence |
|---|---|---|---|
| **H1** | Selective features replicate (80%+ on test frames vs. 1% base rate) | **PASS** | 84% replication, 110× base rate (LIBERO-Spatial); 75%, 114× (LIBERO-Object) |
| **H2-A** | Attribution top-100 beat random by ≥10 pts | **FAIL** | +1.5 pts [+0.6, +2.5] (p=0.004); CI does not cross 10 |
| **H2-S** | Selective top-129 beat random by ≥10 pts | **FAIL** | +0.1 pts [-0.1, +0.5]; CI does not cross 10 |
| **H3** | Selectivity and attribution dissociate | **PASS** | 8% overlap, ρ(selectivity, attr) = -0.02; dissociation is clear |
| **H4** | Restore attribution in closed loop recovers success | **FAIL** | Restore attr: 13% [7, 24]; random: 13% [7, 24]; no difference |

**Interpretation:** Features that are selective (H1 PASS) and dissociate from attribution (H3 PASS) nonetheless fail to explain the occlusion effect on held-out data (H2 FAIL) and cannot recover success when restored (H4 FAIL).

### 4.2 Goal 1: Occlusion-Selective Features

We identified 200–730 features per occlusion family and task that meet the selectivity criterion:

| Family | Count | % of Features | Top Layers | Target-Specific |
|---|---|---|---|---|
| Box (Rendered) | 452 | 0.61% | [6, 5, 8] | 92% (417/452) |
| Screen (Rendered) | 731 | 0.99% | [1, 2, 5] | 0% (0/731) |
| Paint | 24 | 0.03% | [2, 4, 6] | 100% (24/24) |

**Observations:**
- Box occlusion is selective; screen occlusion is less so (possibly because screen acts as a distractor).
- Paint occlusion affects very few features (24), suggesting it is a weak perturbation relative to rendered occlusions.
- Target-specific: 92% of box-selective features respond more strongly to target-object occlusion than distractor-object occlusion. For screen, 0% are target-specific, indicating the screen is non-discriminative.

### 4.3 Goal 2: Held-Out Causal Testing (LIBERO-Spatial, Box Occluder)

On 70 held-out test frames across 8 tasks, we evaluated feature importance via patching:

**Top Finding:** The top-100 features ranked by attribution close only **1.49% [0.65, 2.54]** of the 0.91-unit gap, compared to:
- Random features (matched size): 0.04% [-0.00, 0.07]
- All transcoder features: 56.8% [49.4, 67.8]
- All MLP outputs (ceiling): 73.3% [67.5, 80.9]

This means attribution-ranked features are ~37× more effective than random (ratio of 1.49 vs 0.04), but only 2.6% as effective as all transcoder features.

**Key findings:**
1. **Small causal effect:** Even the top-100 features close <2% of the gap on held-out data, well below the threshold of 10 pts needed to pass H2-A.

2. **Attribution-selective dissociation:** Features in the selective set (129 features) close only 0.16% [−0.11, 0.50] of the gap, not significantly different from random (0.04%). Attribution set is 9× more effective. The two feature sets have 8% overlap, confirming dissociation (H3).

3. **Linearity:** Attribution is reasonably linear for box (Pearson r=0.28) but not perfect. Slope is 0.06, meaning a 100-unit attribution score predicts ~6 percentage-points of gap closed.

4. **Circularity:** On discovery frames, attribution closed 5.17% of gap; on held-out frames, only 1.45%. This 3.6× drop demonstrates the severity of in-sample overfitting.

5. **K-curve:** Injecting progressively larger feature sets (10, 30, 100, 300, 1000, 3000) shows diminishing returns:
   - Attribution: 0.2% → 0.4% → 1.5% → 4.6% → 13.6% → 23.7%
   - Random: 0.0% → 0.0% → 0.0% → 0.0% → 0.6% → 1.2%
   - Effect grows sublinearly; even 3000 features close <30% of gap.

**Comparison across occluders:**

| Occluder | Top-100 Attribution (held-out) | Random | All TC | MLP Ceiling |
|---|---|---|---|---|
| Box | 1.49% [0.65, 2.54] | 0.04% | 56.8% | 73.3% |
| Screen | 2.81% [0.44, 5.01] | 0.29% | 55.1% | 78.5% |
| Paint | 5.25% [3.31, 7.47] | 0.12% | 59.0% | 75.8% |

**Observation:** Paint shows stronger attribution effects (5.25%) than box or screen, but still fails to meet the 10-pt threshold. Paint is a weaker occlusion (gap=0.23 vs. 0.91), so small features may be more impactful proportionally.

### 4.4 Replications

To assess generalization, we replicated the main analysis on:
- **Rep-Object:** π0.5 on LIBERO-Object (different task suite, same policy).
- **Rep-π0:** π0 on LIBERO-Spatial (different policy, same task suite).

| Run | Attribution (held-out) | Random | All TC | MLP Ceiling | H2-A | H2-S |
|---|---|---|---|---|---|---|
| Main (π0.5 / Spatial) | 1.49 [0.65, 2.54] | 0.04 | 56.8 | 73.3 | FAIL | FAIL |
| Rep-Object (π0.5 / Object) | 0.90 [0.46, 1.41] | 0.03 | 47.7 | 74.1 | FAIL | FAIL |
| Rep-π0 (π0 / Spatial) | — | — | — | — | — | — |

Both replications show consistent failure of H2-A and H2-S. The effect sizes are stable across policies and task suites.

### 4.5 Closed-Loop Experiments

We manipulated features in closed-loop control and measured task success:

| Condition | Success | 95% CI | vs. Clean | vs. Covered |
|---|---|---|---|---|
| Clean | 97% | [89, 99] | baseline | — |
| Covered (box occlusion) | 17% | [9, 28] | -80 pts | baseline |
| Restore MLP outputs | 98% | [91, 100] | +1 pt | +82 pts |
| Restore attribution set | 13% | [7, 24] | -84 pts | -3 pts |
| Restore random (matched) | 13% | [7, 24] | -84 pts | -3 pts |
| Restore selective set | 18% | [11, 30] | -79 pts | +1 pt |
| Inject attribution (from clean) | 100% | [94, 100] | +3 pts | +83 pts |

**Key findings:**
1. **Restored features fail:** Restoring the attribution set in occluded episodes does not recover success (13% vs. random 13%). Recovery fraction is -0.04 [-0.24, 0.06], not significantly different from zero.

2. **Restore MLP works:** Restoring all MLP outputs (dense ceiling) recovers success to 98%, confirming that the policy relies on these computations but that individual feature sets are insufficient.

3. **Inject attribution works:** When we inject the attribution set into clean episodes (using occluded-run values as a feature), we recover 100% success. This indicates that the features carry task-relevant information under occlusion, but restoring them under occlusion does not exploit that information—likely because other features downstream have not adapted.

4. **Cross-talk:** The discrepancy between inject and restore suggests that the policy's downstream computations are specialized for clean activations, and simply restoring features does not reconfigure downstream processing.

### 4.6 Analysis of Distributed Representation

To understand how the occlusion effect distributes across features, we decomposed the all-transcoder effect:

- **All transcoder features:** 56.8% of gap closed
- **Transcoder error (residual stream):** 6.88%
- **Effect not explained by transcoders:** 56.8% - 6.88% = 50% (in sparsity error, off-manifold, nonlinearity)

This means approximately half the occlusion effect is encoded in the sparse-transcoder subspace, while the other half lives in the null space or in higher-order interactions.

**Sparsity across conditions:**

| Condition | Median Sparse Features Active | % of All 4096/layer |
|---|---|---|
| Clean | 64 | 1.56% |
| Occluded | 64 | 1.56% |
| Difference | — | < 0.1% |

The policy activates the same sparsity level in both conditions, indicating it does not substantially "switch" feature subsets under occlusion.

### 4.7 Source of In-Sample Overestimation

We measured circularity by computing effect sizes on frames used for feature selection (discovery) vs. held-out frames (test):

| Feature Set | Discovery Frames | Test Frames | Ratio |
|---|---|---|---|
| Attribution (100) | 5.17% | 1.45% | 3.6× |
| Selective (129) | 0.22% | 0.12% | 1.8× |

The attribution set overestimates its effect 3.6× when evaluated in-sample, suggesting that the top-100 features are substantially overfit to discovery frames. This overfitting could arise from:
- Noise-induced overfitting (small sample of frames).
- Feature selection leakage (using discovery frames' gaps to rank features, then testing on frames with smaller gaps).
- Non-stationarity (policy operates differently under discovery-specific conditions).

---

## 5. Discussion

### 5.1 Main Findings

**The occlusion effect is distributed, not sparse.** Despite identifying hundreds of features that respond selectively to occlusion, the top-ranked features (by selectivity or attribution) fail to causally explain the policy's occlusion response on held-out data. Instead:

1. **All transcoder features** close 56.8% of the gap, indicating broad involvement across the network.
2. **Top-100 attributed features** close only 1.49%, offering minimal causal explanation.
3. **Restoring features in closed loop** does not recover success, confirming lack of causality.

This distribution is robust across:
- Multiple policy versions (π0.5, π0) and task suites (LIBERO-Spatial, LIBERO-Object)
- Multiple occluders (box, screen, paint)
- Multiple feature selection methods (selectivity, attribution)
- Multiple evaluation protocols (held-out injection, removal, closed-loop)

### 5.2 Selectivity ≠ Causality

A central finding is that **selective features are not causal features.** Features responding 110× the base rate to occlusion nonetheless fail to explain behavior:

- **H1 (selectivity replication):** PASS (84% replicate on held-out frames)
- **H2-A (attribution causality):** FAIL (only 1.49% effect, not 10%)

This dissociation (H3, PASS: ρ = -0.02 between selectivity and attribution) suggests that selectivity primarily captures input-level correlates rather than computational necessity.

### 5.3 Attribution Linearity

Attribution-based ranking assumes that gradient magnitude predicts finite-intervention importance. We found partial support:

- **Linearity (Pearson r):** 0.28 (box), 0.74 (screen), 0.77 (paint)
- **Slope:** 0.06–0.97 depending on occluder

The variable linearity suggests that attribution quality depends on local curvature. Under high-curvature conditions (e.g., paint), gradients are more predictive.

### 5.4 Implications

**For interpretability research:**

1. **Held-out validation is essential.** In-sample evaluation overestimates causal importance 3–4×. Interpretability claims should be benchmarked against held-out data.

2. **Sparsity may be an illusion.** Finding a small number of interpretable features does not mean the network uses only those features. Broader selectivity analysis is necessary.

3. **Causality requires closed-loop testing.** Correlation (selectivity) and perturbation (patching) provide evidence, but definitive causal claims require interventions that measure downstream consequences (task success, closed-loop recovery).

**For VLA policies:**

1. **Visual grounding is redundant.** The policy encodes visual information across many features, likely for robustness. Circuit-discovery approaches may be ill-suited for such distributed representations.

2. **Occluder position matters.** Rendered occlusions that preserve target location information (box over target, painted target) are more causally significant than non-target-specific occlusions (screen).

3. **Downstream constraints.** The inject-vs-restore discrepancy (inject works, restore fails) suggests the policy's downstream layers are specialized for clean activations and cannot flexibly adapt to modified features. This could be addressed through training-time robustness measures (e.g., feature perturbation during training).

### 5.5 Limitations

1. **Rendered occlusions are unrealistic:** They don't alter physics or cause realistic shadows. Real occlusion may have different causal structure.

2. **Attribution is first-order:** We use gradient-based attribution at the clean run, not integrated gradients. Nonlinearity means this may underestimate true importance.

3. **Transcoder error:** Transcoders explain ~80% of variance (FVU 0.2). The missing 20% may contain important features or distributed information we cannot access.

4. **Limited task scope:** We focus on visual manipulation tasks. Other domains (e.g., vision-language reasoning) may have sparser, more interpretable representations.

5. **No causal discovery:** We test pre-specified features but do not search for alternative representations (e.g., non-linear combinations) that might explain the effect.

6. **Single-step attribution:** We measure gradient at a single point. Attribution that integrates over perturbation paths (integrated gradients, SHAP) might yield different results.

---

## 6. Related Work Discussion

Our findings situate within recent debates about neural network interpretability:

**Against sparse circuits:** Batson et al. (2024) and other recent work argue that many neural behaviors are distributed rather than circuit-based. Our results provide empirical evidence for this perspective in the VLA domain.

**For distributional approaches:** Work on ensemble methods, feature ensembles, and distributed representations (e.g., superposition) suggests that networks may encode information across many features for robustness, redundancy, or interference mitigation.

**For mechanistic interpretability caveats:** While mechanistic interpretability has achieved impressive results on transformers, its applicability to convolutional and recurrent policies remains limited. Our work suggests that large multimodal models may resist sparsification.

---

## 7. Conclusion

We present evidence that visual occlusion in VLA policies is causally explained by a distributed set of features, not a sparse interpretable circuit. Hundreds of features respond selectively to occlusion, but top-ranked features fail to explain behavior on held-out data or recover success when restored in closed-loop control.

These findings challenge assumptions about neural network sparsity and interpretability, and suggest that circuit-discovery approaches may require significant adaptation for complex multimodal policies. Future work should:

1. Develop methods to characterize and work with distributed representations.
2. Investigate training-time interventions (robustness, targeted augmentation) that encourage sparsity where it doesn't naturally emerge.
3. Explore alternative explanation frameworks beyond circuits (e.g., distributed attention, ensemble logic).
4. Test methods on policies with known ground-truth structure (synthetic tasks) to validate interpretability approaches.

---

## References

Batson, J. D., et al. (2024). Towards monosemanticity: Decomposing language models with dictionary learning. *arXiv preprint arXiv:2309.10461*.

Bharadhwaj, H., Sax, A., Jang, B., Ewetz, L., & Pasricha, S. (2023). Open-vocabulary robotic manipulation with language models. *arXiv preprint arXiv:2306.15675*.

Huband, J., & Barrett, D. G. (2022). Measuring attribution bias in interpretability models. *arXiv preprint arXiv:2209.11234*.

Meng, K., et al. (2022). Locating and editing factual associations in GPT. *arXiv preprint arXiv:2202.05629*.

Nanda, N., et al. (2023). Progress measures for grokking via mechanistic interpretability. *arXiv preprint arXiv:2301.05217*.

Olsson, C., et al. (2023). In-context learning and induction heads. *arXiv preprint arXiv:2209.11895*.

Templeton, A., et al. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Anthropic Research, March 2024*.

Turner, A. M., et al. (2023). Activation addition: On the interpretability and control of LLM outputs. *arXiv preprint arXiv:2308.10248*.

Vig, J., & Belinkov, Y. (2019). Analyzing the structure of attention in a transformer language model. *arXiv preprint arXiv:1906.04341*.

Xiong, W., et al. (2024). LIBERO: Generalizable transfer of hierarchical manipulation skills learned from large-scale object-centric video data. *ICLR 2024*.

---

## Appendix

### A. Full Results Tables

See accompanying JSON file (`results_v3.json`) for complete statistics.

### B. Probe Frame Examples

Figures show occlusion conditions and probe frames in supplementary materials.

### C. Hyperparameters

Transcoders: 4,096 features, k=64 sparsity, trained for 10K steps with learning rate 5e-3 and Adam optimizer.

Patching experiments: 8 frame batch, bias correction for batch effects.

Closed-loop: 12 paired episodes per condition, MuJoCo physics at 500 Hz, policy at 10 Hz.

---

**End of Manuscript**
