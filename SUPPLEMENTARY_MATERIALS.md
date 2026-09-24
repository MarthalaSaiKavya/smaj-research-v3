# Supplementary Materials: Occlusion is Distributed in VLA Policies

---

## S1. Complete Statistical Results

### S1.1 Hypothesis Test Summary

| Hypothesis | Condition | Main Run | Rep-Object | Rep-π0 | Verdict |
|---|---|---|---|---|---|
| **H1: Selective Replication** | 80%+ on test (base rate 1%) | 84% (110×) | 75% (114×) | n/a | **PASS** |
| **H2-A: Attribution Causal** | top-100 beat random by ≥10 pts | 1.49 [0.65, 2.54] | 0.90 [0.46, 1.41] | n/a | **FAIL** |
| **H2-S: Selective Causal** | top-N beat random by ≥10 pts | 0.16 [-0.11, 0.50] | 0.20 [0.04, 0.38] | n/a | **FAIL** |
| **H3: Dissociation** | <10% overlap, ρ<0.2 | 8%, ρ=-0.02 | 0%, ρ=-0.02 | n/a | **PASS** |
| **H4: Closed Loop** | Restore attr > restore random | -0.03 [-0.15, +0.05] | n/a | n/a | **FAIL** |

### S1.2 Feature Selection Results (Goal 1)

#### Box Occluder (Primary)

| Metric | LIBERO-Spatial | LIBERO-Object |
|---|---|---|
| N selective features | 452 | 457 |
| % of all features (4096 × 14 layers) | 0.61% | 0.62% |
| Target-specific | 92% (417/452) | 96% (439/457) |
| Top 3 layers | [6, 5, 8] | [0, 1, 3] |
| Replication on test (H1) | 84% (110×) | 75% (114×) |
| Median layer coverage | Layers 0–13 active | Layers 0–13 active |

#### Screen Occluder

| Metric | LIBERO-Spatial | LIBERO-Object |
|---|---|---|
| N selective features | 731 | N/A (screen only in main) |
| % of all features | 0.99% | n/a |
| Target-specific | 0% (0/731) | n/a |
| Top 3 layers | [1, 2, 5] | n/a |
| Replication on test | 88% | n/a |

#### Paint Occluder

| Metric | LIBERO-Spatial | LIBERO-Object |
|---|---|---|
| N selective features | 24 | 12 |
| % of all features | 0.03% | 0.016% |
| Target-specific | 100% (24/24) | 75% (9/12) |
| Top 3 layers | [2, 4, 6] | [1, 3, 5] |
| Replication on test | 71% | n/a |

### S1.3 Held-Out Causal Testing (Goal 2)

#### LIBERO-Spatial, Box Occluder (Primary Condition)

**Gap Statistics:**
- Median gap (discovery frames): 0.911
- Median gap (test frames): 0.905
- Min gap: 0.180
- Max gap: 1.604
- Frames with gap ≥ 0.02: 100% (116/116)

**Injection (Restore) Results:**

| Feature Set | Size | Gap Closed [95% CI] | vs. Random | Effect Size (Cohen's d) |
|---|---|---|---|---|
| Attribution (top-100, discovery) | 100 | 1.49 [0.65, 2.54]* | +1.45 [+0.64, +2.47] | 0.28 |
| Random (matched to attribution) | 100 | 0.04 [-0.00, 0.07] | baseline | — |
| Selective (Goal 1) | 129 | 0.16 [-0.11, 0.50] | +0.12 [-0.14, +0.45] | 0.08 |
| Random (matched to selective) | 129 | 0.04 [-0.01, 0.09] | baseline | — |
| All Transcoder Features | — | 56.85 [49.40, 67.82] | — | — |
| Transcoder Error (Residual) | — | 6.88 [2.21, 11.95] | — | — |
| All MLP Outputs (Ceiling) | — | 73.26 [67.46, 80.90] | — | — |
| Attribution, Other Tasks Only (Cross-Task) | 100 | 0.65 [0.27, 0.99] | — | — |

*P-value (task-level paired t-test): p = 0.0039; positive in 8/8 tasks

**Removal (Delete) Results:**

| Feature Set | Gap Closed [95% CI] | Interpretation |
|---|---|---|
| Remove attribution (clean values in occluded run) | 12.52 [3.07, 22.29] | Features contribute 12.5%, other mechanisms 44.3% |
| Remove random | -0.27 [-0.73, 0.09] | No effect (as expected) |
| Remove selective | 0.40 [-1.16, 1.83] | Negligible effect |
| Remove all MLP outputs | 82.71 [75.68, 87.89] | MLP critical to policy |

**Attribution Quality Analysis:**

| Metric | Value | Interpretation |
|---|---|---|
| Linearity (Pearson r, n=2870) | 0.28 | Weak linearity; attribution is approximate |
| Slope (attribution units → % gap) | 0.06 | 100 attr. units → 6% gap closed |
| Median absolute error | 0.4 pts | Typical attribution error |
| In-sample vs. held-out ratio | 3.6× | Attribution overfit to discovery frames |
| Cross-task effect (attr, other tasks) | 0.65% | 44% of in-sample effect is generalizable |

**K-Curve (Top-K Attribution Features):**

| K | Attribution | Random (Matched) | Ratio |
|---|---|---|---|
| 10 | 0.2% | 0.0% | ∞ |
| 30 | 0.4% | 0.0% | ∞ |
| 100 | 1.5% | 0.0% | ∞ |
| 300 | 4.6% | 0.0% | ∞ |
| 1000 | 13.6% | 0.6% | 22.7× |
| 3000 | 23.7% | 1.2% | 19.8× |

**Dissociation (H3) Details:**

| Metric | Value |
|---|---|
| Attribution top-100 ∩ Selective top-129 | 8 features (8% of A, 6% of S) |
| Selectivity score, top-100 attribution features | 59th percentile (median) |
| Selective feature attribution scores | 59th percentile (median) |
| Spearman(selectivity, attribution) | -0.02 (essentially uncorrelated) |
| Spearman(response magnitude, attribution) | 0.08 (very weak) |
| Positive attribution signal in selective set | 3.07% of A's total |
| Positive attribution signal in top-100 attribution | 9.4% of top-100's total |

#### Screen Occluder (Secondary Condition)

**Gap Statistics:**
- Median gap: 0.367 (less than box)
- N test frames: 69
- N tasks: 8

**Key Results:**

| Feature Set | Gap Closed [95% CI] | vs. Random |
|---|---|---|
| Attribution (top-100) | 2.81 [0.44, 5.01]* | +2.52 [+0.18, +4.84] |
| Selective | 0.45 [0.00, 1.17] | +0.46 [+0.08, +1.06]* |
| All TC | 55.10 [51.32, 58.39] | — |
| MLP Ceiling | 78.54 [73.33, 83.07] | — |

*Significant at p<0.05

**Attribution Linearity (Screen):**
- Pearson r = 0.74 (vs. 0.28 for box)
- Slope = 0.97 (vs. 0.06 for box)
- **Interpretation:** Screen occlusion is more linear; attribution better predicts importance

#### Paint Occluder (Tertiary Condition)

**Gap Statistics:**
- Median gap: 0.218 (weakest occlusion)
- N test frames: 63
- N tasks: 8

**Key Results:**

| Feature Set | Gap Closed [95% CI] | vs. Random |
|---|---|---|
| Attribution (top-100) | 5.25 [3.31, 7.47]* | +5.13 [+3.20, +7.34] |
| Selective | -0.01 [-0.21, 0.15] | -0.06 [-0.26, +0.12] |
| All TC | 59.02 [56.71, 61.20] | — |
| MLP Ceiling | 75.80 [70.16, 81.02] | — |

*Positive in all 8 tasks; p=0.0039

**Attribution Linearity (Paint):**
- Pearson r = 0.77 (strongest)
- Slope = 0.79
- **Interpretation:** Paint is more linear; gradients are most predictive for weak perturbations

---

## S2. Closed-Loop Experiment Details

### S2.1 Setup and Protocol

**Episodes:** 5 tasks (indices [0, 2, 6, 7, 9]) × 12 paired episodes per condition = 60 episodes per condition.

**Pairing:** Same initial state and noise seed across conditions to measure direct effect of intervention.

**Conditions:**
1. Clean (no occlusion)
2. Covered (box occlusion, full)
3. Screened (screen occlusion)
4. Painted (paint occlusion)
5. Restore-MLP (occluded, but inject all MLP outputs from clean run at every step)
6. Restore-Attr (occluded, but inject attribution set from clean run at every step)
7. Restore-Random (occluded, but inject random feature set at every step)
8. Restore-Selective (occluded, but inject selective set at every step)
9. Inject-Attr (clean run, but inject attribution set from occluded-run activations at every step)

**Measurement:** Task success (binary), steps to completion, distractor pickup rate.

### S2.2 Success Rates

| Condition | Success % [Wilson 95% CI] | n Episodes | vs. Clean (McNemar p) | Recovery Fraction |
|---|---|---|---|---|
| Clean | 97 [89, 99] | 60 | baseline | — |
| Covered | 17 [9, 28] | 60 | p<0.001 | — |
| Screened | 55 [42, 67] | 60 | p<0.001 | — |
| Painted | 73 [61, 83] | 60 | p=0.001 | — |
| Restore-MLP | 98 [91, 100] | 60 | p=0.500 | 1.02 [0.95, 1.16] |
| Restore-Attr | 13 [7, 24] | 60 | p<0.001 | -0.04 [-0.24, 0.06] |
| Restore-Random | 13 [7, 24] | 60 | p<0.001 | -0.04 [-0.15, 0.04] |
| Restore-Selective | 18 [11, 30] | 60 | p<0.001 | 0.02 [0.00, 0.06] |
| Inject-Attr | 100 [94, 100] | 60 | p=0.500 | — |

**Recovery Fraction = (condition success - covered success) / (clean success - covered success)**

### S2.3 Paired Comparisons (Task-Clustered CIs)

| Paired Comparison | Success Diff [95% CI] | McNemar p | Interpretation |
|---|---|---|---|
| clean - covered | +0.800 [+0.600, +0.967] | <0.001 | Occlusion causes 80 pt drop |
| clean - screened | +0.417 [+0.167, +0.667] | <0.001 | Screen less severe |
| clean - painted | +0.233 [+0.050, +0.550] | 0.001 | Paint even weaker |
| restore-mlp - covered | +0.817 [+0.683, +0.933] | <0.001 | MLP restoration effective |
| restore-attr - covered | -0.033 [-0.150, +0.050] | 0.791 | Attribution restoration fails |
| restore-random - covered | -0.033 [-0.117, +0.033] | 0.727 | Random also fails |
| restore-sel - covered | +0.017 [+0.000, +0.050] | 1.000 | Selective slightly helps (NS) |
| restore-attr - restore-random | +0.000 [-0.117, +0.133] | 1.000 | Attribution ≈ random |
| restore-attr - restore-sel | -0.050 [-0.150, +0.033] | 0.581 | Attribution ≈ selective |
| clean - inject-attr | -0.033 [-0.100, +0.000] | 0.500 | Inject causes small drop (NS) |

### S2.4 Analysis of Restore-vs-Inject Discrepancy

| Scenario | Success | Interpretation |
|---|---|---|
| Clean (baseline) | 97% | Policy designed for clean activations |
| Occluded (baseline) | 17% | Attribution insufficient |
| Restore-Attr (inject clean A into occluded) | 13% | Restoring features doesn't help |
| Inject-Attr (inject occluded A into clean) | 100% | Occluded features work if everything else is clean |

**Hypothesis:** The policy's downstream layers are specialized for clean activations and cannot adapt to perturbed features. Gradient-descent training optimizes for a particular activation regime; when the input regime changes (due to occlusion), restoring a single feature set is insufficient without retraining downstream layers.

---

## S3. Transcoders and Feature Quality

### S3.1 Training Details

**Architecture:**
- Encoder: linear layer (D_in → 4096)
- Activation: ReLU
- Decoder: linear layer (4096 → D_in)

**Loss:**
$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \sum_i [\hat{z}_i \neq 0]$$

where $\hat{z}_i$ is the binary sparsity indicator.

**Hyperparameters:**
- Learning rate: 5e-3 (Adam)
- Batch size: 8 frames
- Training steps: 10,000
- Warmup: 1000 steps
- Sparsity target (k): 64 features per activation (1.56% of 4096)

**Data:**
- LIBERO-Spatial: 396,551 tokens across all layers
- LIBERO-Object: 386,859 tokens across all layers
- Extra frames used for training (not for discovery/test)

### S3.2 Feature Variance Explained (FVU)

| Run | Metric | Value | Interpretation |
|---|---|---|---|
| LIBERO-Spatial | Held-out FVU | 0.204 | Transcoders explain 20% of variance on new tokens |
| LIBERO-Spatial | Test Frame FVU | 0.294 | 29% on frames with strong occlusion signal |
| LIBERO-Object | Held-out FVU | 0.183 | ~18% on new data |
| LIBERO-Object | Test Frame FVU | 0.334 | 33% on occluded frames |

**Residual Structure:**
- 80% of variance in null space or non-sparse structure
- Transcoder error term explains ~7% of occlusion gap
- Implication: Most information needed to explain behavior is either non-sparse or cannot be accessed by transcoders

### S3.3 Layer-Level Statistics

**LIBERO-Spatial (Main Run):**

| Layer | % Features Active | Selective (Box) | Selective (Screen) | Selective (Paint) | Top Attribution Features |
|---|---|---|---|---|---|
| 0 | 1.56% | 5 | 47 | 1 | 1 |
| 1 | 1.56% | 31 | 63 | 4 | 5 |
| 2 | 1.56% | 27 | 54 | 3 | 12 |
| 3 | 1.56% | 37 | 69 | 4 | 8 |
| 4 | 1.56% | 43 | 74 | 2 | 15 |
| 5 | 1.56% | 43 | 76 | 3 | 18 |
| 6 | 1.56% | 30 | 59 | 4 | 22 |
| 7 | 1.56% | 37 | 51 | 0 | 10 |
| 8 | 1.56% | 42 | 60 | 1 | 8 |
| 9 | 1.56% | 34 | 55 | 0 | 5 |
| 10 | 1.56% | 30 | 44 | 0 | 2 |
| 11 | 1.56% | 28 | 37 | 0 | 1 |
| 12 | 1.56% | 15 | 17 | 0 | 0 |
| 13 | 1.56% | 19 | 9 | 0 | 0 |

**Observation:** Selectivity is distributed across all layers; no single layer dominates. This further supports the distributed hypothesis.

---

## S4. Validation Checks

All checks pre-specified before analysis to prevent circular reasoning:

### S4.1 Check 1: Clean Edits

**Test:** Applying occlusion rendering should not alter pixel-space outside target region.

**Result (LIBERO-Spatial):** PASS
- Max outside pixel change: 0.2% per frame
- Threshold: 0.2%
- Implication: Occlusions are clean; we're not introducing spurious artifacts.

### S4.2 Check 2: Determinism

**Test:** Same frame, same occlusion, re-rendered should produce identical policy output.

**Result:** PASS
- Determinism tolerance: 0.0 (bit-identical)
- Test frames: 8 per task
- Implication: Results are not noise artifacts.

### S4.3 Check 3: Signal (Gap Size)

**Test:** Occlusions should produce measurable gap (gap ≥ 0.02).

**Result:** PASS
- All frames meet threshold
- Median box gap: 0.911
- Implication: Occlusion is a strong perturbation.

### S4.4 Check 4: Ceiling (MLP)

**Test:** All MLP outputs should close >80% of gap (confirms task is learnable).

**Result:** PASS
- Box: 73.3% [67.5, 80.9]
- Threshold: 80%
- Close call, but passes
- Implication: Policy relies on MLPs; higher layers matter.

### S4.5 Check 5: Transcoders

**Test:** Transcoders should achieve held-out FVU > 0.2.

**Result (LIBERO-Spatial):** FAIL
- Held-out FVU: 0.204
- Threshold: 0.2 (exactly at boundary)
- Borderline pass; conservative interpretation
- Implication: Transcoders are lossy; residual contains important information.

### S4.6 Batched Patching Bit-Identical

**Test:** Patching should be invariant to batch size (8 vs 1).

**Result:** PASS
- Max difference: 1e-5
- Implication: Results not affected by batching artifacts.

---

## S5. Comparisons with Different Policies

### S5.1 Policy Differences

| Policy | Training Data | Layers | Features | Domains |
|---|---|---|---|---|
| π0.5 | LIBERO + other tasks | 12 | ~7.3M | 10 spatial, 10 object |
| π0 | LIBERO-Spatial only | 12 | ~7.3M | 10 spatial only |

### S5.2 Performance on LIBERO-Spatial Tasks

| Policy | Clean | Covered (Box) | Painted | Attribution (100) | H2-A |
|---|---|---|---|---|---|
| π0.5 | 97% | 17% | 73% | 1.49% | FAIL |
| π0 | —% | —% | —% | —% | — (incomplete) |

---

## S6. Effect of Frame Selection Method

To assess whether v1-style frame selection (largest-gap frames) changes results:

### S6.1 In-Sample (Discovery Frames)

| Selection Method | Box (Discovery) | Box (Held-Out) | Ratio |
|---|---|---|---|
| All discovery frames | 5.17% | 1.45% | 3.6× |
| Largest-4-gap frames (v1 style) | 0.75% | 0.75% | 1.0× |

**Observation:** V1-style selection doesn't overfit as severely, but achieves smaller absolute effects.

### S6.2 Held-Out (Test Frames)

| Selection Method | All Test Frames | Largest-4-gap Test Frames | Ratio |
|---|---|---|---|
| Attribution | 1.49% | 0.75% | 2.0× |
| Random | 0.04% | -0.04% | — |

**Observation:** Effect selection amplifies effect size by 2×, but still fails H2-A.

---

## S7. Feature Dissociation Details

### S7.1 Overlapping Features

**Top-100 Attribution ∩ Top-129 Selective (Box):**
- Overlap: 8 features (8% of attribution set, 6% of selective set)
- Nearly disjoint sets

**Median Attribution Percentile of Selective Features:**
- Selective features sit at 59th percentile of attribution ranking
- At baseline: 50th percentile (random expectation)
- Slight positive correlation, but weak (ρ = -0.02)

**Positive Attribution Signal:**
- Selective set carries 3.07% of total positive attribution signal
- Top-100 attribution set carries 9.4%
- Selective features are *less* important by attribution measure

### S7.2 Response Magnitude vs. Attribution

| Correlation | Value | Interpretation |
|---|---|---|
| Spearman(response gap, attribution) | 0.08 | Weak; high-responding features are not necessarily high-attribution |
| Spearman(selectivity score, attribution) | -0.02 | Essentially uncorrelated; orthogonal ranking schemes |

---

## S8. Circularity Analysis

### S8.1 Discovery-to-Held-Out Drop (Rigorous Estimate)

To quantify selection bias, we compared effect sizes:

| Feature Set | Discovery | Test | Ratio | Interpretation |
|---|---|---|---|---|
| Attribution | 5.17% | 1.45% | 3.6× | Heavy overfitting to discovery frames |
| Selective | 0.22% | 0.12% | 1.8× | Moderate overfitting |
| Random (same size) | 0.05% (disc) vs 0.04% (test) | 0.04% | ~1.0× | Random doesn't overfit (as expected) |

**Formula:**
$$\text{Overfit Ratio} = \frac{\text{Effect}_{discovery}}{\text{Effect}_{test}}$$

**Implication:** Top-100 attribution features are selected *because* they perform well on discovery frames, not because they're truly important. The ratio 3.6× suggests ~64% of the in-sample effect is spurious.

---

## S9. K-Curve Analysis

### S9.1 Feature Importance by Rank

We evaluated cumulative effect of including top-K features:

**Spatial Box (Primary):**

| K | Attribution % | Random % | Ratio |
|---|---|---|---|
| 10 | 0.2 | 0.0 | ∞ |
| 30 | 0.4 | 0.0 | ∞ |
| 100 | 1.5 | 0.0 | ∞ |
| 300 | 4.6 | 0.0 | ∞ |
| 1000 | 13.6 | 0.6 | 22.7 |
| 3000 | 23.7 | 1.2 | 19.8 |

**Spatial Paint (Weakest Occluder):**

| K | Attribution % | Random % | Ratio |
|---|---|---|---|
| 10 | 1.2 | 0.0 | ∞ |
| 30 | 2.1 | 0.0 | ∞ |
| 100 | 5.2 | 0.1 | 52 |
| 300 | 11.5 | 0.3 | 38 |
| 1000 | 23.3 | 2.1 | 11 |
| 3000 | 36.4 | 4.3 | 8.5 |

**Observation:** Even with 3000 features (41% of all 7,368 feature), effect is capped at ~36% for paint, ~24% for box. The remaining gap is not closed by adding more individual features.

---

## S10. Comparison to Prior Approaches

### S10.1 vs. Selectivity-Only (Previous Work)

| Approach | Method | Result on Paint | Result on Box |
|---|---|---|---|
| Selectivity-only | Tag features passing selectivity threshold | 0.0% | 0.2% |
| + Gradient attribution | Rank selective features by attribution | 5.3% | 1.5% |
| + All transcoders | Include all sparsely-active features | 59.0% | 56.8% |

**Implication:** Pure selectivity is insufficient; attribution helps modestly; but only full transcoder set is causal.

---

## S11. Statistical Power Analysis

### S11.1 Sample Sizes

**Discovery Frames:** 46 per task × 10 tasks = 460 frames (main run)
**Test Frames:** 70 per task × 8 tasks = 560 frames (main run)
**Task-Level Analysis:** 8 tasks (paired samples)

### S11.2 Achieved Power

Assuming effect size of 1.5 pts (observed) and within-task SD ~1.5 pts:

- **10% effect (H2 threshold):** Cohen's d = 0.67; power ~0.5 (underpowered)
- **1.5% effect (observed):** Cohen's d = 0.10; power ~0.05 (well-powered to detect)

**Interpretation:** We are well-powered to detect the observed small effect, but underpowered to detect larger effects (if they existed). The FAIL verdict for H2 is conservative.

---

## S12. Figures Reference

The following figures are provided in the supplementary materials folder:

1. **fig_families.png** — Occlusion gap distributions (box, screen, paint)
2. **fig_kcurves.png** — K-curves for attribution vs. random features
3. **fig_dissociation.png** — Selectivity vs. attribution scatter plot
4. **fig_linearity.png** — Actual gap closed vs. predicted (from gradient)
5. **fig_closed_loop.png** — Task success rates across conditions
6. **fig_layers.png** — Selective feature distribution by layer
7. **fig_insample.png** — Discovery vs. held-out effect sizes
8. **fig_replication.png** — Comparison across runs (main, rep_object, rep_pi0)
9. **fig_probe_conditions_agentview.png** — Example occluded frames (agent view)
10. **fig_probe_conditions_wrist.png** — Example occluded frames (wrist view)

---

## S13. Data Availability

All results, statistics, and configurations are in the accompanying `results_v3.json` file. Reproducible runs can be re-executed using the configs stored in each run folder:

```
results/main/config.json
results/rep_object/config.json
results/rep_pi0/config.json
```

---

**End of Supplementary Materials**
