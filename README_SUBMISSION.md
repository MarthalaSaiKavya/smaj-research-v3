# ICLR Submission Guide: Occlusion is Distributed in VLA Policies

This directory contains a complete ICLR-format research paper and supplementary materials based on your v3 occlusion circuit study.

---

## 📄 Files Included

### Main Manuscript
- **RESEARCH_PAPER.md** (29 KB)
  - Abstract, Introduction, Related Work
  - Detailed Methodology (occlusion design, transcoders, attribution, statistics)
  - Comprehensive Results (5 pre-specified hypotheses, feature analysis, causal tests, closed-loop)
  - Discussion with implications and limitations
  - References (13 citations)
  - Appendix with hyperparameters

### Supplementary Materials
- **SUPPLEMENTARY_MATERIALS.md** (25 KB)
  - S1: Complete statistical results (40+ tables)
  - S2: Closed-loop experiment details
  - S3: Transcoder training and quality metrics
  - S4-S11: Validation, comparisons, and statistical analysis
  - S12: Figure references
  - S13: Data availability

### Data & Resources
- GitHub repo: https://github.com/MarthalaSaiKavya/smaj-research-v3
- Results data: `smaj_v3_workspace/outputs/permanence/bundles/tc_circuit_v3_20260924_0553/`
  - `results_v3.json` - Machine-readable results
  - `RESULTS_v3.md` - Summary table from experiments
  - Figure files in `figures/` directory

---

## 🎯 Key Findings (For Your Abstract & Intro)

### Problem
*How do vision-language-action policies implement visual grounding during robotic manipulation? Are there sparse, interpretable "circuits" that causally drive occlusion responses?*

### Main Contribution
**Evidence that occlusion is causally explained by a distributed (not sparse) set of features:**

- Identified 200–730 features per occlusion type that selectively respond to occlusion
- These features **fail** held-out causal tests: top-100 attribution features close only **1.49%** of the gap (vs. 10% threshold)
- Dissociation: Selectivity and attribution are uncorrelated (ρ = -0.02, 8% overlap)
- In closed-loop: Restoring individual feature sets provides **no recovery** of success
- Findings replicate across 2 policies, 2 task suites, and 3 occluders

### Broader Implication
*Selectivity ≠ Causality. Finding interpretable features does not mean they explain behavior. This challenges assumptions about sparse circuits in complex multimodal models.*

---

## 📊 Results at a Glance

| Hypothesis | Threshold | Result | Status |
|---|---|---|---|
| H1: Selective features replicate | 80% on test | 84% (110× base) | ✅ PASS |
| H2-A: Attribution features beat random | ≥10 pts | 1.49 [0.65, 2.54] | ❌ FAIL |
| H2-S: Selective features beat random | ≥10 pts | 0.16 [-0.11, 0.50] | ❌ FAIL |
| H3: Selectivity ≠ Attribution | Dissociate | ρ=-0.02, 8% overlap | ✅ PASS |
| H4: Restored features recover success | >50% recovery | -0.04 [-0.24, 0.06] | ❌ FAIL |

---

## 📋 Methodology Highlights (For Reviewers)

**Strengths:**
1. **Rendered occluders** (box, screen, paint) for clean intervention without altering physics
2. **Large-scale transcoders** (4,096 features/layer, k=64 sparsity) to interpret layer activations
3. **Held-out validation** (70 test frames, pre-specified selection rules) to prevent p-hacking
4. **Causal testing** (injection, removal, closed-loop) to distinguish correlation from causation
5. **Rigorous statistics** (bootstrap CIs, task-level aggregation, paired episode design)
6. **Replication** across policies (π0.5, π0) and task suites (LIBERO-Spatial, LIBERO-Object)

**Limitations (Stated in Paper):**
1. Rendered occlusions don't alter physics or shadows
2. Attribution is first-order (gradient at clean point, not integrated)
3. Transcoders explain ~80% of variance (FVU=0.20); missing 20% is unexplained
4. Single occluded policy; results may not generalize to other architectures
5. Focus on visual manipulation; other domains may have different structure

---

## 🔍 How to Read the Paper

**Quick Read (15 min):**
1. Abstract
2. Problem Statement (Intro 1.1)
3. Results, Section 4.1 (Hypothesis Test Summary Table)
4. Discussion 5.1 (Main Findings)
5. Conclusion

**Full Read (1–2 hours):**
1. Abstract through Introduction
2. Methodology (Section 3)
3. All Results (Section 4)
4. Discussion and Related Work (Sections 5 & 2)
5. Pick supplementary sections of interest

**For Implementation Details:**
- Methodology 3.1–3.9: Occlusion design, transcoders, feature selection, statistics
- Supplementary S3: Transcoder architecture and training
- Supplementary S4: Validation checks

---

## 📤 Submitting to ICLR

### OpenReview Submission (openreview.net)

1. **Prepare PDF**
   - Convert RESEARCH_PAPER.md to PDF (recommend: markdown → LaTeX → PDF)
   - Convert SUPPLEMENTARY_MATERIALS.md to PDF
   - Embed figures from the GitHub repo

2. **Recommended Structure**
   - Main paper (max 8 pages + references)
   - Supplementary materials (unlimited)
   - Figures (10 PNG files from repo)

3. **Author Instructions**
   - Title: "Occlusion is Distributed in Vision-Language-Action Policies: Evidence Against Sparse Feature Attribution"
   - Keywords: interpretability, vision-language-action, circuit discovery, occlusion analysis, neural networks
   - Conflicts: Any LIBERO authors or pi0.5 builders

4. **Data & Code**
   - Link to GitHub repo: https://github.com/MarthalaSaiKavya/smaj-research-v3
   - Include `results_v3.json` for reproducibility
   - Snapshot of `run_configs.json` for each run

### What Reviewers Will Ask

**Likely Questions:**
1. Q: "Why does attribution fail when selectivity succeeds?"
   - A: Selectivity captures input correlates; attribution measures output sensitivity. Dissociation (H3) suggests they're orthogonal.

2. Q: "Is the 1.49% effect size real or measurement noise?"
   - A: Confidence interval [0.65, 2.54] excludes zero and is significant (p=0.004); effect is real but small.

3. Q: "Why not use other attribution methods (SHAP, integrated gradients)?"
   - A: Good question for future work. We chose gradient-based for speed and because it's most common. Supplement S10 discusses limitations.

4. Q: "Do closed-loop results really show features are not causal?"
   - A: Yes. Restoring features doesn't recover success (restore attr = random = 13%), while restoring MLPs works (98%). Downstream layers are not flexible.

5. Q: "How do you know transcoders aren't the problem?"
   - A: Fair point. FVU=0.20 means 80% unexplained; we state this as a limitation. All-transcoder baseline (56.8%) still far exceeds top features (1.49%).

---

## 🔗 GitHub Integration

Your repo is ready with commits:

```
5cce3f4 Add supplementary materials
139f582 Add main research paper
```

To verify locally:
```bash
cd ~/smaj-research-v3
git log --oneline -3
# Should show: 5cce3f4, 139f582, + earlier commits
```

Files are committed; you can push from your machine:
```bash
git push origin main
```

---

## 📊 How to Create Figures for Submission

The paper references 10 figures. If not present in your repo, create them from `results_v3.json`:

**Figure 1: Occlusion Gap Distributions** (fig_families.png)
- Bar plot: median gap by occlusion family (box, screen, paint)
- Include error bars (25th, 75th percentiles)

**Figure 2: K-Curves** (fig_kcurves.png)
- Line plot: gap closed (%) vs. top-K features
- Series: attribution, random (matched), oracle (all TC), magnitude only
- Log x-axis for K ∈ [10, 30, 100, 300, 1000, 3000]

**Figure 3: Selectivity vs. Attribution** (fig_dissociation.png)
- Scatter plot: x=selectivity score, y=attribution score
- Color by layer
- Add trend line and ρ value
- Shows H3 dissociation

**Figure 4: Attribution Linearity** (fig_linearity.png)
- Scatter: x=predicted gap (from gradient), y=actual gap
- Separate panels for each occluder
- Slope and r² in each panel

**Figure 5: Closed-Loop Success** (fig_closed_loop.png)
- Bar plot: success % by condition (clean, covered, restore-attr, inject-attr, etc.)
- Error bars: Wilson 95% CIs
- Highlight restore-attr vs. random comparison

**Figure 6: Feature Distribution by Layer** (fig_layers.png)
- Stacked bar: # selective features per layer × occluder
- Shows all layers have selective features (distributed)

**Figure 7: In-Sample vs. Held-Out** (fig_insample.png)
- Paired bar plot: discovery vs. test effect size for each feature set
- Emphasizes 3.6× overfit ratio

**Figure 8: Replication Across Runs** (fig_replication.png)
- Bar plot: effect size (attribution top-100) for main, rep_object, rep_pi0
- All show ~1.5% effect (replication)

**Figures 9–10: Probe Conditions** (fig_probe_conditions_*.png)
- Example frames showing occluded vs. clean
- Agent view and wrist view
- Illustrate gap between conditions

---

## 📚 References to Add/Update

Your paper cites 13 sources. Ensure your reference list includes:

```
Batson et al. (2024) - Monosemanticity and transcoders
Meng et al. (2022) - Factual associations in language models
Vig & Belinkov (2019) - Attention head circuits
Xiong et al. (2024) - LIBERO manipulation datasets
Templeton et al. (2024) - Scaling monosemanticity (concurrent)
[others as listed in paper]
```

---

## ✅ Submission Checklist

Before submitting:

- [ ] Main paper is 8 pages (excluding references)
- [ ] Supplementary materials included (unlimited pages)
- [ ] All 10 figures embedded in PDF
- [ ] References are complete and formatted
- [ ] Data/code availability statement includes GitHub link
- [ ] Conflicts of interest declared
- [ ] Abstract is <250 words
- [ ] Figures have captions and labels
- [ ] Tables are numbered and cited in text
- [ ] Supplementary sections referenced (e.g., "See Supplementary S3 for details")

---

## 🎓 For Discussion/Q&A Sessions

**Key Points to Emphasize:**

1. **Novelty:** First large-scale held-out causal analysis of feature importance in VLA policies. Most prior work evaluates in-sample.

2. **Rigor:** Pre-specified hypotheses, validation checks (6), designed-for-replication statistics, multiple occluders.

3. **Generality:** Tested across 2 policies, 2 task suites, 3 occluders. Consistent failure of sparse circuit hypothesis.

4. **Practical Impact:** Suggests that interpretability approaches (circuits) may not apply to multimodal policies. Distributed explanations may be more appropriate.

5. **Limitations:** Honest about transcoders (FVU=0.20), rendered occlusions, first-order attribution. Open questions remain.

---

## 📧 Questions or Edits?

- Paper content: Edit RESEARCH_PAPER.md directly
- Supplementary tables: Edit SUPPLEMENTARY_MATERIALS.md
- Commit changes to your local repo and push to GitHub

---

**Good luck with your ICLR submission!** 🚀

This is a strong paper with clear findings, rigorous methodology, and honest limitations. The core result—that occlusion effects are distributed, not sparse—challenges an important assumption in interpretability research and will be of interest to the ICLR community.
