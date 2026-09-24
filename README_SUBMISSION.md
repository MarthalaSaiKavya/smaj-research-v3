# Research report and anonymous ICLR manuscript

This package presents a fresh study of selective features, causal action effects and behavioral recovery under visual occlusion. It separates completed results from missing experiments and avoids claims of a recovered object-permanence circuit or two-policy replication.

## Deliverables

- [Detailed report](RESEARCH_REPORT.md) and [report PDF](reports/RESEARCH_REPORT.pdf): problem statements, methodology, novelty, results and limitations.
- [Anonymous ICLR manuscript PDF](paper/main.pdf), [LaTeX source](paper/main.tex) and [readable manuscript](RESEARCH_PAPER.md).
- [Complete supplementary results](SUPPLEMENTARY_MATERIALS.md): all saved statistical sections and all 540 episode outcomes.
- [File-by-file audit](reports/FILE_AUDIT.md) and [inventory](reports/file_inventory.json): all 181 tracked input files and missing-artifact list.
- [Numerical verification](reports/numerical_verification.json), [recomputed method table](reports/recomputed_interventions.csv) and [verification script](scripts/audit_results.py).

## Evidence and limits

Occlusion selectivity replicates, but the tested small feature sets do not reliably recover closed-loop success. Attribution produces a small positive injection effect and a larger action-level restoration effect. Full-MLP restoration nearly restores clean success. Completed evidence covers one policy on two suites; rollout evidence covers five Spatial tasks.

All 14,370 intervention rows and 540 episodes were processed. The audit reproduces 238 statistical groups plus rollout counts, paired intervals and exact McNemar calculations. The results ZIP has 159 duplicate members. Of 201 manifest entries, 165 are present and 36 are missing; supplied checksums match. Missing files comprise 29 checkpoint/tensor files, four array archives and three replay-frame pickle files. Only seven Spatial layer checkpoints and no Object layer checkpoints are uploaded. No policy inference was rerun during this audit.

## Build and verify

From the repository root:

```bash
python3 scripts/audit_results.py
```

Dependencies: Python, NumPy, pandas, Pillow, Poppler and FFmpeg. The verifier executes only reviewed numerical aggregation definitions from the experiment source; it does not import the policy/simulator or load unrestricted pickle objects.

From the paper directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The official ICLR 2027 styles are supplied. Original uploaded scientific artifacts remain intact for provenance; the root report, manuscript and submission guide are the reviewed entry points.

## Submission requirements

The manuscript is an anonymous draft, not an OpenReview submission. Authors must verify the final claims, references, author information, AI-use disclosure and anonymous supplementary package. The identifying repository must not be linked in anonymous submission material.

ICLR 2027 permits at most nine main-text pages. Its abstract deadline is September 18, 2026 AoE, and full-paper deadline September 25, 2026 at 23:59 AoE. A valid abstract registration must already exist for this cycle. See the [official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) and [AI-use policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

Restoring missing artifacts, expanding independent-policy and closed-loop replication, and testing layer-matched/context-specific selections would strengthen the contribution. These remain proposed work and are not described as completed experiments.
