#!/usr/bin/env python
"""v3 report + paper builder. CPU only. Reads the output folders of the main run and the replications and writes

    <out>/RESULTS_v3.md        narrative summary with every pre-specified test, and the v1 / v2 / v3 comparison
    <out>/results_v3.json      every number quoted in RESULTS_v3.md and the paper
    <out>/figures/*.pdf|png    paper figures
    <out>/paper_v3/            ICLR-style LaTeX draft (main.tex, tables, figures, bib, style) [+ main.pdf with --compile]

    python tc_v3_report.py --run main=DIR --run rep_object=DIR --run rep_pi0=DIR --out DIR [--compile]
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path

import numpy as np

FAM_LABEL = {"cover": "box (rendered)", "screen": "screen (rendered)", "paint": "paint (v1/v2)"}
FAM_TEX = {"cover": "\\textsc{box}", "screen": "\\textsc{screen}", "paint": "\\textsc{paint}"}
RUN_ORDER = ["main", "rep_object", "rep_pi0"]

# v1 / v2 numbers (from smaj-research and smaj-research-v2, derived_stats.json)
V1 = {"tasks": 1, "frames": "4 (selected by effect size)", "split": "none (same frames)", "occluder": "paint",
      "transcoder": "512 features, k=16", "fvu_heldout": 0.174, "fvu_test": None, "n_selective": "72 / 9,216",
      "sel": 3.92, "rand": 0.56, "ratio": 6.9, "all_tc": 40.1, "ceiling": 65.5, "attr": None, "closed_loop": "not run",
      "ci": "none", "verdict": "selective; 6.9x random (suggestive)"}
V2 = {"tasks": 10, "frames": "60 (20 discovery + 40 test, fixed phases)", "split": "disjoint demonstrations", "occluder": "paint",
      "transcoder": "2,048 features, k=32", "fvu_heldout": 0.185, "fvu_test": 0.400, "n_selective": "29 / 36,864",
      "sel": 0.27, "sel_ci": [-0.32, 0.92], "rand": 0.05, "ratio": 4.9, "all_tc": 42.8, "ceiling": 76.6, "attr": None,
      "closed_loop": "clean 0.9, painted 0.9 (task 0, n=10)", "ci": "task-clustered bootstrap",
      "verdict": "selective and partly replicable; causally negligible"}


# ----------------------------------------------------------------------------- helpers


def jload(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fin(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def f_(x, nd=1, sign=False, pct=False):
    if not fin(x):
        return "n/a"
    s = f"{float(x):+.{nd}f}" if sign else f"{float(x):.{nd}f}"
    return s + ("%" if pct else "")


def ci_(s, nd=1):
    if not s or not fin(s.get("mean")):
        return "n/a"
    lo, hi = (s.get("ci95") or [None, None])[:2]
    return f"{f_(s['mean'], nd)} [{f_(lo, nd)}, {f_(hi, nd)}]"


def dci_(p, nd=1):
    if not p or not fin(p.get("mean_diff", p.get("diff"))):
        return "n/a"
    m = p.get("mean_diff", p.get("diff"))
    lo, hi = (p.get("ci95") or [None, None])[:2]
    return f"{f_(m, nd, True)} [{f_(lo, nd, True)}, {f_(hi, nd, True)}]"


def tn(x, nd=1, sign=False):
    """LaTeX-safe number."""
    s = f_(x, nd, sign)
    return s.replace("-", "$-$") if s != "n/a" else "--"


def tci(s, nd=1):
    if not s or not fin(s.get("mean")):
        return "--"
    lo, hi = (s.get("ci95") or [None, None])[:2]
    return f"{tn(s['mean'], nd)} [{tn(lo, nd)}, {tn(hi, nd)}]"


def tdci(p, nd=1):
    if not p or not fin(p.get("mean_diff", p.get("diff"))):
        return "--"
    m = p.get("mean_diff", p.get("diff"))
    lo, hi = (p.get("ci95") or [None, None])[:2]
    return f"{tn(m, nd, True)} [{tn(lo, nd, True)}, {tn(hi, nd, True)}]"


def pf(x):
    return "PASS" if x is True else ("FAIL" if x is False else "n/a")


def tpf(x):
    return "\\cmark" if x is True else ("\\xmark" if x is False else "--")


def g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def model_label(cfg):
    p = str(cfg.get("policy_path", ""))
    if isinstance(cfg.get("policy_path"), list):
        p = str(cfg["policy_path"][0])
    return "pi0.5" if "pi05" in p else ("pi0" if "pi0" in p else p)


def model_tex(cfg):
    return "$\\pi_{0.5}$" if model_label(cfg) == "pi0.5" else "$\\pi_0$"


def suite_label(cfg):
    return {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object", "libero_goal": "LIBERO-Goal",
            "libero_10": "LIBERO-Long"}.get(cfg.get("suite"), str(cfg.get("suite")))


def read_rows(p: Path):
    if not p.exists():
        return []
    out = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            for k in ("closed_pct", "proj_pct", "pred_pct", "gap"):
                try:
                    r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
                except Exception:
                    r[k] = None
            for k in ("frame", "task_id"):
                try:
                    r[k] = int(float(r[k]))
                except Exception:
                    pass
            out.append(r)
    return out


def load_run(name: str, d: Path) -> dict:
    d = Path(d)
    st = {p.stem: jload(p) for p in (d / "status").glob("*.json")} if (d / "status").exists() else {}
    R = {"name": name, "dir": d, "cfg": jload(d / "config.json") or {}, "status": st, "validate": jload(d / "validate.json"),
         "train": jload(d / "train.json"), "goal1": jload(d / "goal1_selected.json"), "goal2": jload(d / "goal2_stats.json"),
         "goal3": jload(d / "goal3.json"), "rows": read_rows(d / "goal2_rows.csv"), "frames_info": jload(d / "goal2_frames.json")}
    R["model"], R["suite"] = model_label(R["cfg"]), suite_label(R["cfg"])
    R["label"] = f"{R['model']} / {R['suite']}"
    R["prim"] = g(R, "goal2", "protocol", "primary_family") or R["cfg"].get("primary_family", "cover")
    return R


def fam_stats(R, fam=None):
    return g(R, "goal2", "families", fam or R["prim"]) or {}


# ----------------------------------------------------------------------------- derived analyses


def v1_style_selection(R, fam):
    """Inside v3: what if we had chosen frames like v1 (4 largest-gap frames of one task)?"""
    rows = [r for r in R["rows"] if r.get("family") == fam and r.get("split") == "test" and r.get("scope") == "all"
            and r.get("direction") == "inject" and r.get("closed_pct") is not None]
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault((r["frame"], r["method"]), []).append(r["closed_pct"])
    gap = {r["frame"]: r["gap"] for r in rows}
    task = {r["frame"]: r["task_id"] for r in rows}

    def mean_over(frames, m):
        v = [np.mean(by[(f, m)]) for f in frames if (f, m) in by]
        return float(np.mean(v)) if v else None

    all_f = sorted(gap)
    t0 = min(task.values())
    sel4 = sorted([f for f in all_f if task[f] == t0], key=lambda f: -gap[f])[:4]
    out = {}
    for m, c in (("sel_capped", "rand_sel_same_norm"), ("attr", "rand_attr_same_norm")):
        a_all, c_all, a_4, c_4 = mean_over(all_f, m), mean_over(all_f, c), mean_over(sel4, m), mean_over(sel4, c)
        if a_all is None:
            continue
        out[m] = {"all_frames": a_all, "all_frames_control": c_all, "v1_style_4_frames": a_4, "v1_style_control": c_4,
                  "ratio_all": a_all / c_all if c_all and abs(c_all) > 1e-9 else None,
                  "ratio_v1_style": a_4 / c_4 if (a_4 is not None and c_4 and abs(c_4) > 1e-9) else None}
    out["n_frames_all"], out["v1_style_frames"], out["task"] = len(all_f), sel4, t0
    return out


def key_numbers(R) -> dict:
    """Flat dict of the numbers the text quotes (also dumped to results_v3.json)."""
    K = {"label": R["label"], "model": R["model"], "suite": R["suite"], "dir": str(R["dir"])}
    st = R["status"]
    fr, v, t = st.get("frames") or {}, st.get("validate") or {}, st.get("train") or {}
    K.update({"n_tasks": fr.get("n_tasks"), "n_probe": fr.get("n_probe_frames"), "n_discovery": fr.get("n_discovery_frames"),
              "n_test": fr.get("n_test_frames"), "n_extra": fr.get("n_extra_train_frames"),
              "frame_source": fr.get("frame_source"), "families": fr.get("families"), "primary": R["prim"],
              "check1": v.get("check1"), "check2": v.get("check2"), "check3": v.get("check3"), "check4": v.get("check4"),
              "check5": t.get("check5"), "batch_invariant": v.get("batch_invariant"), "rank_layers": v.get("rank_layers"),
              "ceiling_best_pct": v.get("ceiling_best_pct"), "ceiling_best_layer": v.get("ceiling_best_layer"),
              "fvu_heldout": t.get("median_fvu"), "fvu_test": t.get("median_fvu_test_frames"),
              "tc_features": t.get("tc_features"), "tc_k": t.get("tc_k"), "n_train_tokens": t.get("n_train_tokens"),
              "gaps": {f: g(v, "check3_by_family", f) for f in (fr.get("families") or [])}})
    g1 = st.get("goal1") or {}
    K["goal1"] = g1.get("families")
    K["n_features_total"] = g1.get("n_features_total")
    G2 = R.get("goal2") or {}
    K["goal2"] = {}
    for f, FS in (G2.get("families") or {}).items():
        V = FS.get("verdict") or {}
        P = FS.get("paired") or {}
        K["goal2"][f] = {"n_included": FS.get("n_included"), "n_tasks": len(FS.get("tasks") or []),
                         "gap_median": g(FS, "gaps", "median"), "n_attr": g(FS, "n_features", "attr"),
                         "n_sel": g(FS, "n_features", "sel_capped"), "verdict": V,
                         "attr": FS["methods"].get("attr"), "rand_attr": FS["methods"].get("rand_attr_same_norm"),
                         "sel": FS["methods"].get("sel_capped"), "rand_sel": FS["methods"].get("rand_sel_same_norm"),
                         "all_tc": FS["methods"].get("all_tc"), "ceiling": FS["methods"].get("mlp_ceiling"),
                         "tc_error": FS["methods"].get("tc_error"), "attr_xtask": FS["methods"].get("attr_xtask"),
                         "sel_xtask": FS["methods"].get("sel_xtask"), "color": FS["methods"].get("color"),
                         "attr_remove": g(FS, "remove", "attr"), "rand_attr_remove": g(FS, "remove", "rand_attr_same_norm"),
                         "ceiling_remove": g(FS, "remove", "mlp_ceiling"), "sel_remove": g(FS, "remove", "sel_capped"),
                         "p_attr": P.get("attr_vs_rand_attr"), "p_sel": P.get("sel_vs_rand_sel"), "p_attr_sel": P.get("attr_vs_sel"),
                         "p_attr_remove": P.get("attr_vs_rand_attr_remove"), "p_attr_xtask": P.get("attr_xtask_vs_rand_attr"),
                         "kcurve": FS.get("kcurve"), "kcurve_remove": FS.get("kcurve_remove"), "linearity": FS.get("linearity"),
                         "dissociation": FS.get("dissociation"), "heldout": FS.get("heldout_selectivity"),
                         "insample": FS.get("insample_vs_heldout"), "scopes": FS.get("scopes"),
                         "beats": FS.get("beats_every_draw"), "v1_style": v1_style_selection(R, f)}
    K["hypotheses"] = G2.get("hypotheses")
    G3 = R.get("goal3")
    if G3:
        K["goal3"] = {k: G3.get(k) for k in ("tasks", "episodes_per_task", "occluder", "occluder_family", "drops", "rates",
                                             "comparisons", "recovery", "check6_drop_measurable",
                                             "H4_restoring_attr_features_recovers_success", "set_sizes", "stage_A", "stage_B")}
    tf = jload(R["dir"] / "status" / "timings.json") or {}
    K["timings_min"] = {k: round(v / 60, 1) for k, v in (tf.get("steps") or {}).items()}
    return K


# ----------------------------------------------------------------------------- figures


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    return plt


COL = {"attr": "#2a78d6", "oracle": "#6a3d9a", "sel": "#eb6834", "magnitude": "#eb6834", "random": "#8a8a8a",
       "ceiling": "#1baf7a", "clean": "#1baf7a"}


def _save(fig, figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.png", dpi=160, bbox_inches="tight")


def _bar(ax, labels, stats, colors):
    m = [s["mean"] if s and fin(s.get("mean")) else np.nan for s in stats]
    lo = [max(0.0, s["mean"] - s["ci95"][0]) if s and fin(g(s, "ci95", default=[None])[0]) else 0 for s in stats]
    hi = [max(0.0, s["ci95"][1] - s["mean"]) if s and fin(g(s, "ci95", default=[None, None])[1]) else 0 for s in stats]
    ax.bar(range(len(labels)), m, yerr=[lo, hi], capsize=2, color=colors, width=0.75)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="k", lw=0.6)


def fig_families(R, figdir):
    plt = _plt()
    fams = [f for f in ("cover", "screen", "paint") if f in (g(R, "goal2", "families") or {})]
    if not fams:
        return None
    items = [("sel_capped", "selective", COL["sel"]), ("rand_sel_same_norm", "rand.", "#f3b58f"),
             ("attr", "attrib.", COL["attr"]), ("rand_attr_same_norm", "rand.", "#9cc2ee"),
             ("all_tc", "all TC", "#7fd1b3"), ("mlp_ceiling", "MLP", COL["ceiling"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.35 * len(fams), 2.2), squeeze=False, sharey=True)
    for ax, f in zip(axes[0], fams):
        M = fam_stats(R, f).get("methods", {})
        its = [it for it in items if it[0] in M]
        _bar(ax, [i[1] for i in its], [M[i[0]] for i in its], [i[2] for i in its])
        ax.set_yscale("symlog", linthresh=5)
        ax.set_title(f"{FAM_LABEL[f]} (n={fam_stats(R, f).get('n_included')})")
        ax.tick_params(axis="x", rotation=60)
    axes[0][0].set_ylabel("gap closed on held-out frames (%)")
    fig.tight_layout()
    _save(fig, figdir, "fig_families")
    plt.close(fig)
    return "fig_families"


def fig_kcurves(runs, figdir):
    plt = _plt()
    rs = [R for R in runs if fam_stats(R).get("kcurve")]
    if not rs:
        return None
    fig, axes = plt.subplots(1, len(rs), figsize=(2.9 * len(rs), 2.4), squeeze=False)
    for ax, R in zip(axes[0], rs):
        FS = fam_stats(R)
        for k, lab in (("oracle", "attribution, test frame (oracle)"), ("attr", "attribution, discovery"),
                       ("magnitude", "activation change"), ("random", "random")):
            pts = FS["kcurve"].get(k) or []
            if pts:
                x = [p["k"] for p in pts]
                ax.plot(x, [p["mean"] for p in pts], marker="o", ms=3, color=COL[k], label=lab)
                ax.fill_between(x, [p["ci95"][0] if fin(p["ci95"][0]) else p["mean"] for p in pts],
                                [p["ci95"][1] if fin(p["ci95"][1]) else p["mean"] for p in pts], color=COL[k], alpha=0.12)
        for m, ls, lab in (("all_tc", "--", "all TC features"), ("mlp_ceiling", ":", "MLP-output ceiling")):
            s = FS.get("methods", {}).get(m)
            if s and fin(s.get("mean")):
                ax.axhline(s["mean"], color="k", ls=ls, lw=0.8, label=lab)
        s = FS.get("methods", {}).get("sel_capped")
        if s and fin(s.get("mean")):
            ax.plot([max(1, g(FS, "n_features", "sel_capped", default=1))], [s["mean"]], "*", ms=9, color=COL["sel"], label="selective set")
        ax.set_xscale("log")
        ax.set_xlabel(f"# features patched (of {g(R, 'goal2', 'protocol', 'n_features_total', default=0):,})")
        ax.set_title(f"{R['label']} [{FAM_LABEL[R['prim']]}]")
    axes[0][0].set_ylabel("gap closed (%)")
    axes[0][0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, figdir, "fig_kcurves")
    plt.close(fig)
    return "fig_kcurves"


def fig_dissociation(R, figdir):
    import torch

    f = R["prim"]
    p_rank, p_att = R["dir"] / "goal1_rank.pt", R["dir"] / "attrib" / f"rank_{f}.npz"
    sets = jload(R["dir"] / "goal2_sets.json") or {}
    if not (p_rank.exists() and p_att.exists()):
        return None
    rk = torch.load(p_rank, weights_only=False)
    att = np.load(p_att)
    layers = list(rk["layers"])
    rise = rk["fam"][f]["rise"].numpy()
    sym = att["sym"]
    alive = rk["alive"].numpy().astype(bool)
    rl = g(R, "goal2", "protocol", "rank_layers") or layers
    m = alive & np.isin(np.arange(len(layers))[:, None], [layers.index(l) for l in rl if l in layers])
    x, y = rise[m], sym[m] * 100
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(np.clip(x, 1e-3, None), y, s=2, color="#bbbbbb", alpha=0.5, rasterized=True, label="all features (rank layers)")
    n = rise.shape[1]
    for key, col, lab in (("sel_uncapped", COL["sel"], "selective (Goal 1)"), ("attr", COL["attr"], "attribution top-K")):
        pts = [(layers.index(int(l)), j) for l, js in (g(sets, f, key) or {}).items() for j in js if int(l) in layers]
        if pts:
            ax.scatter([max(rise[k, j], 1e-3) for k, j in pts], [sym[k, j] * 100 for k, j in pts], s=10, color=col, label=lab,
                       edgecolor="k", linewidth=0.2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("selectivity: occlusion response / typical activation")
    ax.set_ylabel("attribution (% of gap, discovery mean)")
    D = g(fam_stats(R, f), "dissociation") or {}
    ax.set_title(f"Selective vs causal features [{FAM_LABEL[f]}], "
                 f"$\\rho$={f_(D.get('spearman_selectivity_rise_vs_attr'), 2)}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, figdir, "fig_dissociation")
    plt.close(fig)
    return "fig_dissociation"


def fig_linearity(R, figdir):
    rows = [r for r in R["rows"] if r.get("family") == R["prim"] and r.get("split") == "test" and r.get("pred_pct") is not None
            and r.get("scope") == "all" and r.get("proj_pct") is not None]
    if len(rows) < 3:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    kinds = (("k_attr", COL["attr"], "attribution top-K"), ("k_oracle", COL["oracle"], "oracle top-K"),
             ("k_magnitude", COL["magnitude"], "activation-change top-K"), ("k_random", COL["random"], "random K"),
             ("sel", COL["sel"], "selective"), ("all_tc", "k", "all features"))
    for pre, col, lab in kinds:
        rr = [r for r in rows if r["method"].startswith(pre)]
        if rr:
            ax.scatter([r["pred_pct"] for r in rr], [r["proj_pct"] for r in rr], s=4, color=col, alpha=0.5, label=lab, rasterized=True)
    v = np.array([[r["pred_pct"], r["proj_pct"]] for r in rows], dtype=float)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.6)
    L = g(fam_stats(R), "linearity") or {}
    ax.set_xlabel("attribution prediction (% of gap)")
    ax.set_ylabel("patched effect (% of gap, projection)")
    ax.set_title(f"First-order attribution vs patching, r={f_(L.get('pearson_pred_vs_proj'), 2)}")
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    _save(fig, figdir, "fig_linearity")
    plt.close(fig)
    return "fig_linearity"


def fig_closed_loop(R, figdir):
    G3 = R.get("goal3")
    if not G3 or not G3.get("rates"):
        return None
    conds = [c for c in (G3.get("stage_A") or []) + (G3.get("stage_B") or []) if c in G3["rates"]]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(0.55 * len(conds) + 1.2, 2.3))
    S = [{"mean": G3["rates"][c]["success"] * 100, "ci95": [G3["rates"][c]["wilson95"][0] * 100, G3["rates"][c]["wilson95"][1] * 100]}
         for c in conds]
    cols = [COL["clean"] if c == "clean" else "#8a8a8a" if c in G3["stage_A"] else COL["attr"] if "attr" in c
            else COL["sel"] if "sel" in c else "#bbbbbb" if "random" in c else COL["ceiling"] for c in conds]
    names = {"clean": "clean", "covered": "box", "screened": "screen", "painted": "paint", "restore_mlp": "restore\nall MLP",
             "restore_attr": "restore\nattrib.", "restore_sel": "restore\nselective", "restore_random": "restore\nrandom",
             "inject_attr": "clean +\nattrib. inj."}
    _bar(ax, [names.get(c, c) for c in conds], S, cols)
    ax.set_ylim(0, 105)
    ax.set_ylabel("success (%)")
    ax.axvline(len(G3["stage_A"]) - 0.5, color="k", lw=0.5, ls=":")
    ax.set_title(f"Closed loop: {len(G3['tasks'])} tasks x {G3['episodes_per_task']} paired episodes; "
                 f"restores under '{names.get(G3['occluder'], G3['occluder'])}'")
    fig.tight_layout()
    _save(fig, figdir, "fig_closed_loop")
    plt.close(fig)
    return "fig_closed_loop"


def fig_layers(R, figdir):
    V = R.get("validate") or {}
    fams4 = g(V, "check4", "families") or {}
    att = g(R, "status", "attrib", "families") or {}
    if not fams4 and not att:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.1))
    for f, P in fams4.items():
        x = range(len(P["resid_swap_pct_mean"]))
        axes[0].plot(x, P["resid_swap_pct_mean"], marker="o", ms=2.5, label=f"{FAM_LABEL[f]}: residual swap")
        axes[0].plot(x, P["mlp_swap_pct_mean"], ls="--", lw=0.9, label=f"{FAM_LABEL[f]}: MLP swap")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("gap closed (%)")
    axes[0].set_title("Single-layer swaps (discovery frames)")
    axes[0].legend(frameon=False, fontsize=5)
    for f, A in att.items():
        axes[1].plot(range(len(A["positive_mass_by_layer_pct"])), A["positive_mass_by_layer_pct"], marker="o", ms=2.5, label=FAM_LABEL[f])
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("% of positive attribution")
    axes[1].set_title("Where the attribution sits")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, figdir, "fig_layers")
    plt.close(fig)
    return "fig_layers"


def fig_insample(R, figdir):
    fams = [f for f in ("cover", "paint", "screen") if g(fam_stats(R, f), "insample_vs_heldout")]
    if not fams:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    labels, dvals, tvals = [], [], []
    for f in fams:
        for m, lab in (("sel_capped", "selective"), ("attr", "attribution")):
            IS = g(fam_stats(R, f), "insample_vs_heldout", m)
            if IS and fin(IS.get("margin_discovery")):
                labels.append(f"{lab}\n{FAM_LABEL[f].split()[0]}")
                dvals.append(IS["margin_discovery"])
                tvals.append(IS.get("margin_test"))
    if not labels:
        plt.close(fig)
        return None
    x = np.arange(len(labels))
    ax.bar(x - 0.18, dvals, 0.36, color="#bbbbbb", label="discovery frames (in-sample)")
    ax.bar(x + 0.18, [v if fin(v) else 0 for v in tvals], 0.36, color=COL["attr"], label="held-out test frames")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("set - matched random (pts)")
    ax.set_title("Circularity: the same sets in-sample vs held-out")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, figdir, "fig_insample")
    plt.close(fig)
    return "fig_insample"


def fig_replication(runs, figdir):
    rs = [R for R in runs if fam_stats(R).get("methods")]
    if len(rs) < 2:
        return None
    plt = _plt()
    items = [("attr", "attribution", COL["attr"]), ("rand_attr_same_norm", "random (matched)", "#9cc2ee"),
             ("sel_capped", "selective", COL["sel"]), ("rand_sel_same_norm", "random (matched)", "#f3b58f"),
             ("all_tc", "all TC", "#7fd1b3"), ("mlp_ceiling", "MLP ceiling", COL["ceiling"])]
    fig, ax = plt.subplots(figsize=(1.9 * len(rs) + 1, 2.3))
    w = 0.8 / len(items)
    for i, (m, lab, col) in enumerate(items):
        S = [fam_stats(R).get("methods", {}).get(m) for R in rs]
        mean = [s["mean"] if s else np.nan for s in S]
        lo = [max(0, s["mean"] - s["ci95"][0]) if s and fin(s["ci95"][0]) else 0 for s in S]
        hi = [max(0, s["ci95"][1] - s["mean"]) if s and fin(s["ci95"][1]) else 0 for s in S]
        ax.bar(np.arange(len(rs)) + (i - (len(items) - 1) / 2) * w, mean, w, yerr=[lo, hi], capsize=1.5, color=col,
               label=lab if i < 6 else None)
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels([R["label"] for R in rs])
    ax.set_yscale("symlog", linthresh=5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("gap closed (%)")
    ax.set_title("Replication across models and suites (primary occluder, held-out frames)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    _save(fig, figdir, "fig_replication")
    plt.close(fig)
    return "fig_replication"


def make_figures(runs, figdir):
    made = {}
    main = runs[0]
    for fn, args in ((fig_families, (main,)), (fig_kcurves, (runs,)), (fig_dissociation, (main,)), (fig_linearity, (main,)),
                     (fig_closed_loop, (main,)), (fig_layers, (main,)), (fig_insample, (main,)), (fig_replication, (runs,))):
        try:
            name = fn(*args, figdir)
            if name:
                made[name] = True
        except Exception as exc:
            print(f"[report] figure {fn.__name__} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    for src in ("fig_probe_conditions_agentview.png", "fig_probe_conditions_wrist.png"):
        p = main["dir"] / src
        if p.exists():
            shutil.copy(p, figdir / src)
            made[src.replace(".png", "")] = True
    return made


# ----------------------------------------------------------------------------- markdown report


def outcome(K):
    f = K["primary"]
    V = g(K, "goal2", f, "verdict") or {}
    D = g(K, "goal2", f, "dissociation") or {}
    G3 = K.get("goal3") or {}
    return {"h2a": V.get("H2A_attr_beats_random"), "h2a_rem": V.get("H2A_remove"), "h2s": V.get("H2S_selective_beats_controls"),
            "h3": D.get("H3_dissociated"), "h4": G3.get("H4_restoring_attr_features_recovers_success"),
            "c6": G3.get("check6_drop_measurable")}


def title_for(K):
    o = outcome(K)
    if not fin(g(K, "goal2", K["primary"], "verdict", "attr")):
        return "Held-Out Transcoder Analysis of Occlusion in a Vision-Language-Action Policy"
    if o["h2a"] and not o["h2s"]:
        return "Selective Is Not Causal: Held-Out Attribution Finds the Sparse Features That Carry Occlusion in Vision-Language-Action Policies"
    if o["h2a"] and o["h2s"]:
        return "Sparse, Selective and Causal: Held-Out Transcoder Circuits for Occlusion in Vision-Language-Action Policies"
    return "Occlusion Is Distributed in Vision-Language-Action Policies: Selective and Attributed Transcoder Features Fail Held-Out Causal Tests"


def md_report(runs, KS, figs) -> str:
    main, K = runs[0], KS[0]
    f = K["primary"]
    E = K["goal2"].get(f, {})
    o = outcome(K)
    L = [f"# v3 results: {title_for(K)}", "",
         "Auto-generated by `tc_v3_report.py` from the run folders. Every number is also in `results_v3.json`.", "",
         "Runs: " + "; ".join(f"**{R['name']}** = {R['label']} (`{R['dir']}`)" for R in runs), ""]
    # ---- 1. headline
    L += ["## 1. Pre-specified tests", "",
          "| test | rule (fixed before the run) | " + " | ".join(R["label"] for R in runs) + " |",
          "|---|---|" + "---|" * len(runs)]

    def cell(Kx, fn):
        try:
            return fn(Kx)
        except Exception:
            return "n/a"

    def h2a_cell(Kx):
        E_ = Kx["goal2"].get(Kx["primary"], {})
        return f"{pf(outcome(Kx)['h2a'])} ({dci_(E_.get('p_attr'))} pts)"

    def h2s_cell(Kx):
        E_ = Kx["goal2"].get(Kx["primary"], {})
        return f"{pf(outcome(Kx)['h2s'])} ({dci_(E_.get('p_sel'))} pts)"

    def h1_cell(Kx):
        H = g(Kx, "goal2", Kx["primary"], "heldout") or {}
        n = g(Kx, "goal1", Kx["primary"], "n_pass_total")
        return f"{n} features; {f_(100 * (H.get('replicate_frac_sel_capped') or 0), 0)}% replicate ({f_(H.get('enrichment_sel_capped'), 0)}x base rate)"

    def h3_cell(Kx):
        D = g(Kx, "goal2", Kx["primary"], "dissociation") or {}
        return (f"{pf(D.get('H3_dissociated'))} (overlap {D.get('overlap_attr_sel_uncapped', 'n/a')}, "
                f"{f_(100 * (D.get('frac_attr_passing_selectivity') or 0), 0)}% of A selective, rho={f_(D.get('spearman_selectivity_rise_vs_attr'), 2)})")

    def h4_cell(Kx):
        G3 = Kx.get("goal3")
        if not G3:
            return "not run"
        c = (G3.get("comparisons") or {}).get(f"restore_attr_minus_{G3.get('occluder')}")
        return f"{pf(G3.get('H4_restoring_attr_features_recovers_success'))} (drop measurable: {pf(G3.get('check6_drop_measurable'))}; restore {dci_(c, 2)})"

    L += ["| H1 selective features replicate | Goal-1 rule on discovery frames, re-applied on test frames | " + " | ".join(cell(k, h1_cell) for k in KS) + " |",
          f"| **H2-A attribution set is causal** | top-{g(main, 'goal2', 'protocol', 'attr_k')} attribution features beat norm-matched random by >= "
          f"{g(main, 'goal2', 'protocol', 'margin_pts')} pts, CI > 0 | " + " | ".join(cell(k, h2a_cell) for k in KS) + " |",
          f"| H2-S selective set is causal | v2 rule: selective beats random and colour by >= {g(main, 'goal2', 'protocol', 'margin_pts')} pts | "
          + " | ".join(cell(k, h2s_cell) for k in KS) + " |",
          "| H3 selectivity and causality dissociate | < 10% of A selective and rho(selectivity, attribution) < 0.2 | " + " | ".join(cell(k, h3_cell) for k in KS) + " |",
          "| H4 closed loop | restoring A's clean values under the occluder raises success vs no restore and vs random (CIs > 0) | "
          + " | ".join(cell(k, h4_cell) for k in KS) + " |", ""]
    # ---- 2. setup / validity
    L += ["## 2. Setup and validity", "", "| | " + " | ".join(R["label"] for R in runs) + " |", "|---|" + "---|" * len(runs)]
    rowsv = [("tasks / probe frames (discovery + test) / training frames",
              lambda k: f"{k.get('n_tasks')} / {k.get('n_probe')} ({k.get('n_discovery')} + {k.get('n_test')}) / {k.get('n_extra')}"),
             ("frame source", lambda k: ", ".join(k.get("frame_source") or [])),
             ("occluder families", lambda k: ", ".join(k.get("families") or [])),
             ("check 1 clean edits / 2 determinism / 3 signal / 4 ceiling / 5 transcoders",
              lambda k: " / ".join(pf(k.get(c)) for c in ("check1", "check2", "check3", "check4", "check5"))),
             ("batched patching bit-identical", lambda k: pf(k.get("batch_invariant"))),
             ("best single-layer residual swap", lambda k: f"{f_(k.get('ceiling_best_pct'))}% (L{k.get('ceiling_best_layer')})"),
             ("rank layers (residual swap >= 10%)", lambda k: str(k.get("rank_layers"))),
             ("transcoders", lambda k: f"{k.get('tc_features')} features, k={k.get('tc_k')}, {k.get('n_train_tokens')} tokens/layer"),
             ("median FVU held-out tokens / test frames", lambda k: f"{f_(k.get('fvu_heldout'), 3)} / {f_(k.get('fvu_test'), 3)}")]
    for fam in ("cover", "screen", "paint"):
        rowsv.append((f"{FAM_LABEL[fam]}: median gap, frames >= min, target vs distractor",
                      lambda k, fam=fam: (f"{f_(g(k, 'gaps', fam, 'median_gap'), 3)}, {f_(100 * (g(k, 'gaps', fam, 'frac_frames_gap_ge_min') or 0), 0)}%, "
                                          f"{f_(g(k, 'gaps', fam, 'distractor', 'median_target'), 3)} vs "
                                          f"{f_(g(k, 'gaps', fam, 'distractor', 'median_distractor'), 3)}") if g(k, "gaps", fam) else "n/a"))
    for name, fn in rowsv:
        L.append(f"| {name} | " + " | ".join(cell(k, fn) for k in KS) + " |")
    L.append("")
    # ---- 3. Goal 1
    L += ["## 3. Goal 1: occlusion-selective features (discovery frames)", "", "| run | family | selective | target-specific | top layers |",
          "|---|---|---|---|---|"]
    for R, k in zip(runs, KS):
        for fam, s in (k.get("goal1") or {}).items():
            L.append(f"| {R['label']} | {FAM_LABEL[fam]} | {s.get('n_pass_total')} ({f_(s.get('pass_frac_pct'), 3)}%) | "
                     f"{s.get('n_pass_target_total')} | {s.get('top_layers')} |")
    L.append("")
    # ---- 4. Goal 2, main run
    L += [f"## 4. Goal 2: held-out causal tests ({main['label']})", ""]
    for fam, E_ in K["goal2"].items():
        L += [f"### {FAM_LABEL[fam]}{' (primary)' if fam == f else ''}: {E_['n_included']} held-out frames, {E_['n_tasks']} tasks, "
              f"median gap {f_(E_['gap_median'], 3)}", "",
              "| patched set (inject: base run + occluded values) | # features | gap closed % [95% CI] |", "|---|---|---|"]
        for key, lab, nk in (("attr", "attribution top-K (discovery)", "n_attr"), ("rand_attr", "random, matched to attribution set", "n_attr"),
                             ("attr_xtask", "attribution top-K, other tasks only (cross-task)", "n_attr"),
                             ("sel", "selective set (Goal 1)", "n_sel"), ("rand_sel", "random, matched to selective set", "n_sel"),
                             ("sel_xtask", "selective set, other tasks only", None), ("color", "colour-selective control", None),
                             ("all_tc", "all transcoder features", None), ("tc_error", "transcoder error term only", None),
                             ("ceiling", "all MLP outputs (ceiling)", None)):
            if E_.get(key):
                L.append(f"| {lab} | {E_.get(nk) if nk else ''} | {ci_(E_[key], 2)} |")
        for key, lab in (("attr_remove", "remove: attribution set (occluded run + clean values)"),
                         ("rand_attr_remove", "remove: random matched"), ("sel_remove", "remove: selective set"),
                         ("ceiling_remove", "remove: all MLP outputs")):
            if E_.get(key):
                L.append(f"| {lab} | | {ci_(E_[key], 2)} |")
        L += ["", f"- attribution - random: {dci_(E_.get('p_attr'), 2)} pts, p(tasks) = {f_(g(E_, 'p_attr', 'p_signflip_tasks'), 4)}, "
              f"positive in {g(E_, 'p_attr', 'n_tasks_positive')}/{g(E_, 'p_attr', 'n_tasks')} tasks; remove direction {dci_(E_.get('p_attr_remove'), 2)}",
              f"- selective - random: {dci_(E_.get('p_sel'), 2)} pts, p(tasks) = {f_(g(E_, 'p_sel', 'p_signflip_tasks'), 4)}",
              f"- attribution - selective: {dci_(E_.get('p_attr_sel'), 2)} pts; cross-task attribution - random: {dci_(E_.get('p_attr_xtask'), 2)} pts"]
        kc = E_.get("kcurve") or {}
        if kc:
            L.append("- k-curve (features -> gap closed %): " + "; ".join(
                f"{nm}: " + ", ".join(f"{p['k']}: {f_(p['mean'])}" for p in pts) for nm, pts in kc.items()))
        lin = E_.get("linearity")
        if lin:
            L.append(f"- attribution linearity: Pearson r = {f_(lin.get('pearson_pred_vs_proj'), 2)} (n = {lin.get('n')}), slope "
                     f"{f_(lin.get('slope_proj_on_pred'), 2)}, median |error| {f_(lin.get('median_abs_error_pts'), 1)} pts")
        D = E_.get("dissociation") or {}
        if D:
            L.append(f"- dissociation: A ∩ S = {D.get('overlap_attr_sel_uncapped')} of {E_.get('n_attr')} / {D.get('n_selective_uncapped')}; "
                     f"{f_(100 * (D.get('frac_attr_passing_selectivity') or 0), 0)}% of A pass the selectivity rule; selective features sit at the "
                     f"{f_(D.get('selective_attr_percentile_median'), 0)}th attribution percentile (median) and carry "
                     f"{f_(D.get('selective_share_of_positive_attr_pct'), 2)}% of the positive attribution (A: {f_(D.get('attr_share_of_positive_attr_pct'), 1)}%); "
                     f"Spearman(selectivity, attribution) = {f_(D.get('spearman_selectivity_rise_vs_attr'), 2)}, "
                     f"(response magnitude, attribution) = {f_(D.get('spearman_response_magnitude_vs_attr'), 2)}")
        H = E_.get("heldout") or {}
        if H:
            L.append(f"- held-out selectivity: {f_(100 * (H.get('replicate_frac_sel_capped') or 0), 0)}% of the selective set pass again on test "
                     f"frames (random draws {f_(100 * (H.get('replicate_frac_random_draws_mean') or 0), 1)}%, base rate "
                     f"{f_(100 * (H.get('base_rate_all_alive') or 0), 2)}%, attribution set {f_(100 * (H.get('replicate_frac_attr') or 0), 1)}%); "
                     f"Spearman discovery vs test response {f_(H.get('spearman_discovery_vs_test_response'), 2)}")
        IS = E_.get("insample") or {}
        for m, lab in (("sel_capped", "selective"), ("attr", "attribution")):
            if IS.get(m):
                L.append(f"- circularity [{lab}]: set - random = {f_(IS[m].get('margin_discovery'), 2, True)} pts on the discovery frames it was "
                         f"chosen on vs {f_(IS[m].get('margin_test'), 2, True)} pts on held-out frames")
        V1s = E_.get("v1_style") or {}
        for m, lab in (("sel_capped", "selective"), ("attr", "attribution")):
            if V1s.get(m):
                x = V1s[m]
                L.append(f"- v1-style frame choice [{lab}] (4 largest-gap test frames of task {V1s.get('task')}): {f_(x['v1_style_4_frames'], 2)}% vs "
                         f"random {f_(x['v1_style_control'], 2)}% (ratio {f_(x['ratio_v1_style'], 1)}x) against all {V1s.get('n_frames_all')} frames: "
                         f"{f_(x['all_frames'], 2)}% vs {f_(x['all_frames_control'], 2)}% (ratio {f_(x['ratio_all'], 1)}x)")
        L.append("")
    # ---- 5. closed loop
    G3 = K.get("goal3")
    L += ["## 5. Closed loop", ""]
    if G3:
        L += [f"{len(G3['tasks'])} tasks ({G3['tasks']}) x {G3['episodes_per_task']} paired episodes per condition (same initial states and noise "
              f"seeds across conditions). Restores run under **{G3['occluder']}** ({FAM_LABEL.get(G3.get('occluder_family'), '')}).", "",
              "| condition | success [Wilson 95%] | distractor picked up | mean steps |", "|---|---|---|---|"]
        for c in (G3.get("stage_A") or []) + (G3.get("stage_B") or []):
            r = G3["rates"].get(c)
            if r:
                L.append(f"| {c} | {f_(100 * r['success'], 0)}% [{f_(100 * r['wilson95'][0], 0)}, {f_(100 * r['wilson95'][1], 0)}] (n={r['n']}) | "
                         f"{f_(100 * (r.get('distractor_lifted_frac') or 0), 0)}% | {f_(r.get('mean_steps'), 0)} |")
        L += ["", "| paired comparison | diff in success [task-clustered 95% CI] | McNemar p |", "|---|---|---|"]
        for k_, c in (G3.get("comparisons") or {}).items():
            L.append(f"| {k_} | {dci_(c, 3)} | {f_(c.get('mcnemar_p'), 4)} |")
        for k_, r in (G3.get("recovery") or {}).items():
            if r:
                L.append(f"| recovery fraction {k_} | {f_(r.get('fraction'), 2)} [{f_(g(r, 'ci95', default=[None])[0], 2)}, "
                         f"{f_(g(r, 'ci95', default=[None, None])[1], 2)}] | |")
    else:
        L.append("not run")
    L.append("")
    # ---- 6. replications
    if len(runs) > 1:
        L += ["## 6. Replications (primary occluder, held-out frames)", "",
              "| run | attribution | random (matched) | selective | random (matched) | all TC | MLP ceiling | H2-A | H2-S |",
              "|---|---|---|---|---|---|---|---|---|"]
        for R, k in zip(runs, KS):
            E_ = k["goal2"].get(k["primary"], {})
            L.append(f"| {R['label']} | {ci_(E_.get('attr'), 2)} | {ci_(E_.get('rand_attr'), 2)} | {ci_(E_.get('sel'), 2)} | "
                     f"{ci_(E_.get('rand_sel'), 2)} | {ci_(E_.get('all_tc'), 1)} | {ci_(E_.get('ceiling'), 1)} | {pf(outcome(k)['h2a'])} | {pf(outcome(k)['h2s'])} |")
        L.append("")
    # ---- 7. v1 / v2 / v3
    E_p = K["goal2"].get("paint", {})
    L += ["## 7. v1 vs v2 vs v3", "", "| | v1 | v2 | v3 (this run) |", "|---|---|---|---|",
          f"| tasks | {V1['tasks']} | {V2['tasks']} | {K.get('n_tasks')} (+ replications: {', '.join(R['label'] for R in runs[1:]) or 'none'}) |",
          f"| probe frames | {V1['frames']} | {V2['frames']} | {K.get('n_probe')} ({K.get('n_discovery')} discovery + {K.get('n_test')} test), fixed phases |",
          "| occluders | paint | paint | box, screen (rendered in the simulator) + paint |",
          f"| transcoders | {V1['transcoder']} | {V2['transcoder']} | {K.get('tc_features')} features, k={K.get('tc_k')} |",
          f"| median FVU held-out / test frames | {V1['fvu_heldout']} / n/a | {V2['fvu_heldout']} / {V2['fvu_test']} | "
          f"{f_(K.get('fvu_heldout'), 3)} / {f_(K.get('fvu_test'), 3)} |",
          "| feature selection | selectivity, in-sample | selectivity, held-out test | selectivity AND gradient attribution, held-out test |",
          f"| selective set, paint (gap closed %) | {V1['sel']} | {V2['sel']} [{V2['sel_ci'][0]}, {V2['sel_ci'][1]}] | {ci_(E_p.get('sel'), 2)} |",
          f"| random matched, paint | {V1['rand']} | {V2['rand']} | {ci_(E_p.get('rand_sel'), 2)} |",
          f"| selective set, primary occluder | n/a | n/a | {ci_(E.get('sel'), 2)} |",
          f"| attribution set, primary occluder | n/a | n/a | {ci_(E.get('attr'), 2)} (random {ci_(E.get('rand_attr'), 2)}) |",
          f"| all TC features / MLP ceiling (primary) | {V1['all_tc']} / {V1['ceiling']} | {V2['all_tc']} / {V2['ceiling']} | "
          f"{f_(g(E, 'all_tc', 'mean'))} / {f_(g(E, 'ceiling', 'mean'))} |",
          f"| closed loop | {V1['closed_loop']} | {V2['closed_loop']} | " +
          (f"{len(G3['tasks'])} tasks x {G3['episodes_per_task']} eps; clean {f_(100 * G3['rates']['clean']['success'], 0)}%, "
           + ", ".join(f"{c} {f_(100 * G3['rates'][c]['success'], 0)}%" for c in (G3.get('stage_A') or [])[1:] if c in G3['rates'])
           if G3 else "n/a") + " |",
          f"| verdict | {V1['verdict']} | {V2['verdict']} | {'H2-A ' + pf(o['h2a']) + ', H2-S ' + pf(o['h2s']) + ', H3 ' + pf(o['h3']) + ', H4 ' + pf(o['h4'])} |", ""]
    # ---- 8. limitations
    L += ["## 8. Limitations to state in the paper", "",
          "- The rendered occluders are render-only (no physics); they follow the target, so their position still reveals where the target is.",
          "- Attribution is first-order (gradient at the clean run); its linearity is measured above, not assumed.",
          "- Transcoders are trained per layer on a finite token sample; the error term's share of the effect is reported.",
          "- Ten tasks per suite are the unit of replication for CIs; replications add a second suite and a second policy, not new robots.",
          ""]
    L += ["## 9. Figures", ""] + [f"- `figures/{n}.png`" for n in figs] + [""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- LaTeX paper draft

STYLE_ZIP_B64 = "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"

REFS_EXTRA = r"""
@inproceedings{syed2024attribution,
  title     = {Attribution Patching Outperforms Automated Circuit Discovery},
  author    = {Syed, Aaquib and Rager, Can and Conmy, Arthur},
  booktitle = {Proceedings of the 7th BlackboxNLP Workshop},
  year      = {2024},
  note      = {arXiv:2310.10348}
}

@article{kramar2024atp,
  title   = {{AtP*}: An efficient and scalable method for localizing {LLM} behaviour to components},
  author  = {Kram{\'a}r, J{\'a}nos and Lieberum, Tom and Shah, Rohin and Nanda, Neel},
  journal = {arXiv preprint arXiv:2403.00745},
  year    = {2024}
}

@misc{nanda2023attribution,
  title        = {Attribution Patching: Activation Patching At Industrial Scale},
  author       = {Nanda, Neel},
  year         = {2023},
  howpublished = {\url{https://www.neelnanda.io/mechanistic-interpretability/attribution-patching}}
}

@inproceedings{conmy2023automated,
  title     = {Towards Automated Circuit Discovery for Mechanistic Interpretability},
  author    = {Conmy, Arthur and Mavor-Parker, Augustine N. and Lynch, Aengus and Heimersheim, Stefan and Garriga-Alonso, Adri{\`a}},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}

@inproceedings{sundararajan2017axiomatic,
  title     = {Axiomatic Attribution for Deep Networks},
  author    = {Sundararajan, Mukund and Taly, Ankur and Yan, Qiqi},
  booktitle = {International Conference on Machine Learning},
  year      = {2017}
}

@article{bolukbasi2021illusion,
  title   = {An Interpretability Illusion for {BERT}},
  author  = {Bolukbasi, Tolga and Pearce, Adam and Yuan, Ann and Coenen, Andy and Reif, Emily and Vi{\'e}gas, Fernanda and Wattenberg, Martin},
  journal = {arXiv preprint arXiv:2104.07143},
  year    = {2021}
}

@article{zhu2020robosuite,
  title   = {robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author  = {Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and Mart{\'i}n-Mart{\'i}n, Roberto and Joshi, Abhishek and Nasiriany, Soroush and Zhu, Yifeng},
  journal = {arXiv preprint arXiv:2009.12293},
  year    = {2020}
}

@inproceedings{todorov2012mujoco,
  title     = {{MuJoCo}: A physics engine for model-based control},
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year      = {2012}
}
"""


def _k_to_reach(pts, target):
    for p in pts or []:
        if fin(p.get("mean")) and fin(target) and p["mean"] >= target:
            return p["k"]
    return None


def _kval(pts, K):
    for p in pts or []:
        if p["k"] == K:
            return p
    return None


def latex_tables(runs, KS) -> dict:
    T = {}
    main, K = runs[0], KS[0]
    # validity
    rows = []
    for R, k in zip(runs, KS):
        f = k["primary"]
        gp = g(k, "gaps", f) or {}
        rows.append(f"{model_tex(R['cfg'])} / {R['suite']} & {k.get('n_tasks')} & {k.get('n_discovery')}+{k.get('n_test')} & "
                    f"{tpf(k.get('check1'))}{tpf(k.get('check2'))}{tpf(k.get('check3'))}{tpf(k.get('check4'))}{tpf(k.get('check5'))} & "
                    f"{tn(gp.get('median_gap'), 3)} & {tn(100 * (gp.get('frac_frames_gap_ge_min') or 0), 0)}\\% & "
                    f"{tn(k.get('ceiling_best_pct'), 1)} & {tn(k.get('fvu_heldout'), 3)} / {tn(k.get('fvu_test'), 3)} \\\\")
    T["validity"] = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Setup and validity battery. Checks: (1) edits change only the "
                     "intended pixels, (2) determinism and bit-identical batched patching, (3) the occlusion effect exceeds the "
                     "minimum gap, (4) a single-layer residual swap closes $\\geq$80\\% of it, (5) transcoders reconstruct $\\leq$0.2 FVU. "
                     "Gap: median action RMSE between base and occluded (primary occluder). FVU on held-out tokens / held-out test frames.}\n"
                     "\\label{tab:validity}\n\\begin{tabular}{lccccccc}\n\\toprule\n"
                     "run & tasks & frames (disc.+test) & checks 1--5 & gap & $\\geq$ min & best swap (\\%) & FVU \\\\\n\\midrule\n"
                     + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # main results for all families of the main run
    fams = [f for f in ("cover", "screen", "paint") if f in K["goal2"]]
    lines = []
    for key, lab in (("attr", "attribution top-$K$"), ("rand_attr", "\\quad random, matched"), ("attr_xtask", "attribution, other tasks"),
                     ("sel", "selective (Goal 1)"), ("rand_sel", "\\quad random, matched"), ("color", "colour-selective"),
                     ("all_tc", "all transcoder features"), ("tc_error", "transcoder error only"), ("ceiling", "all MLP outputs")):
        cells = [tci(K["goal2"][f].get(key), 1) for f in fams]
        if any(c != "--" for c in cells):
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for key, lab in (("p_attr", "attribution $-$ random"), ("p_sel", "selective $-$ random"), ("p_attr_sel", "attribution $-$ selective"),
                     ("p_attr_remove", "remove: attribution $-$ random")):
        cells = [tdci(K["goal2"][f].get(key), 1) for f in fams]
        if any(c != "--" for c in cells):
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("H2-A (attribution) & " + " & ".join(tpf(g(K, "goal2", f, "verdict", "H2A_attr_beats_random")) for f in fams) + " \\\\")
    lines.append("H2-S (selective) & " + " & ".join(tpf(g(K, "goal2", f, "verdict", "H2S_selective_beats_controls")) for f in fams) + " \\\\")
    lines.append("held-out frames (tasks) & " + " & ".join(f"{K['goal2'][f]['n_included']} ({K['goal2'][f]['n_tasks']})" for f in fams) + " \\\\")
    T["main"] = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Held-out causal tests on "
                 f"{model_tex(main['cfg'])} / {main['suite']}. Mean \\% of the base$\\rightarrow$occluded action gap closed by setting a feature "
                 "set to its occluded values in the base run (inject), with 95\\% task-clustered bootstrap CIs. Random sets match the "
                 "per-layer counts and per-token norm of the set they control for. Feature sets are fixed on discovery demonstrations.}\n"
                 "\\label{tab:main}\n\\begin{tabular}{l" + "c" * len(fams) + "}\n\\toprule\n & "
                 + " & ".join(FAM_TEX[f] for f in fams) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # replication
    if len(runs) > 1:
        rr = []
        for R, k in zip(runs, KS):
            E = k["goal2"].get(k["primary"], {})
            rr.append(f"{model_tex(R['cfg'])} / {R['suite']} & {tci(E.get('attr'), 1)} & {tci(E.get('rand_attr'), 1)} & {tci(E.get('sel'), 1)} & "
                      f"{tci(E.get('rand_sel'), 1)} & {tn(g(E, 'all_tc', 'mean'), 1)} & {tn(g(E, 'ceiling', 'mean'), 1)} & "
                      f"{tpf(outcome(k)['h2a'])}{tpf(outcome(k)['h2s'])} \\\\")
        T["replication"] = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Replication across policies and suites (box occluder, held-out "
                            "frames, \\% gap closed [95\\% CI]). Last column: H2-A, H2-S.}\n\\label{tab:replication}\n"
                            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccccc}\n\\toprule\nrun & attribution & random & selective & random & all TC & MLP & H2 \\\\\n"
                            "\\midrule\n" + "\n".join(rr) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    # closed loop
    G3 = K.get("goal3")
    if G3:
        names = {"clean": "clean", "covered": "box", "screened": "screen", "painted": "paint", "restore_mlp": "box + restore all MLP outputs",
                 "restore_attr": "restore attribution set", "restore_sel": "restore selective set", "restore_random": "restore random set (matched)",
                 "inject_attr": "clean + inject attribution set"}
        oc = names.get(G3["occluder"], G3["occluder"])
        cl = []
        for c in (G3.get("stage_A") or []) + (G3.get("stage_B") or []):
            r = G3["rates"].get(c)
            if r:
                nm = names.get(c, c).replace("box", oc) if c.startswith("restore") else names.get(c, c)
                cl.append(f"{nm} & {tn(100 * r['success'], 0)} [{tn(100 * r['wilson95'][0], 0)}, {tn(100 * r['wilson95'][1], 0)}] & "
                          f"{tn(100 * (r.get('distractor_lifted_frac') or 0), 0)} \\\\")
        cm = []
        for k_, lab in ((f"clean_minus_{G3['occluder']}", f"clean $-$ {oc}"), (f"restore_attr_minus_{G3['occluder']}", "restore attribution $-$ no restore"),
                        ("restore_attr_minus_restore_random", "restore attribution $-$ restore random"),
                        (f"restore_mlp_minus_{G3['occluder']}", "restore all MLP $-$ no restore"), ("clean_minus_inject_attr", "clean $-$ inject attribution")):
            c = (G3.get("comparisons") or {}).get(k_)
            if c:
                cm.append(f"{lab} & {tdci({'mean_diff': 100 * c['diff'], 'ci95': [100 * x if fin(x) else x for x in c['ci95']]}, 0)} & {tn(c['mcnemar_p'], 3)} \\\\")
        T["closed_loop"] = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Closed loop on " + f"{len(G3['tasks'])} tasks $\\times$ "
                            f"{G3['episodes_per_task']}" + " paired episodes (same initial states and noise seeds in every condition). Success "
                            "with Wilson 95\\% CIs; \\emph{distractor}: episodes in which the other object was lifted. Restores replace the "
                            "features' values in the occluded run with their values from a clean render of the same state, at every inference step.}\n"
                            "\\label{tab:closedloop}\n\\begin{tabular}{lcc}\n\\toprule\ncondition & success (\\%) & distractor (\\%) \\\\\n\\midrule\n"
                            + "\n".join(cl) + "\n\\midrule\npaired difference (pts) & [95\\% CI] & McNemar $p$ \\\\\n\\midrule\n" + "\n".join(cm)
                            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # v1/v2/v3
    E, Ep = K["goal2"].get(K["primary"], {}), K["goal2"].get("paint", {})
    T["versions"] = ("\\begin{table}[t]\n\\centering\\small\n\\caption{The same question under three protocols. v1: in-sample pilot; "
                     "v2: held-out selectivity study; v3: this paper.}\n\\label{tab:versions}\n\\resizebox{\\linewidth}{!}{"
                     "\\begin{tabular}{llll}\n\\toprule\n & v1 & v2 & v3 \\\\\n\\midrule\n"
                     f"tasks / probe frames & 1 / 4 (max effect) & 10 / 60 (fixed phases) & {K.get('n_tasks')} / {K.get('n_probe')} (fixed phases) + 2 replications \\\\\n"
                     f"selection vs.\\ test & same frames & disjoint demos & disjoint demos + cross-task \\\\\n"
                     f"occluder & paint & paint & rendered box, screen + paint \\\\\n"
                     f"transcoder & 512, $k$=16 & 2{{,}}048, $k$=32 & {K.get('tc_features')}, $k$={K.get('tc_k')} \\\\\n"
                     f"FVU held-out / test frames & 0.174 / -- & 0.185 / 0.400 & {tn(K.get('fvu_heldout'), 3)} / {tn(K.get('fvu_test'), 3)} \\\\\n"
                     f"features chosen by & selectivity & selectivity & selectivity and attribution \\\\\n"
                     f"selective set, paint (\\%) & 3.92 & 0.27 [$-$0.32, 0.92] & {tci(Ep.get('sel'), 2)} \\\\\n"
                     f"random matched, paint (\\%) & 0.56 & 0.05 & {tci(Ep.get('rand_sel'), 2)} \\\\\n"
                     f"attribution set, box (\\%) & -- & -- & {tci(E.get('attr'), 1)} \\\\\n"
                     f"all TC / MLP ceiling (\\%) & 40.1 / 65.5 & 42.8 / 76.6 & {tn(g(E, 'all_tc', 'mean'))} / {tn(g(E, 'ceiling', 'mean'))} \\\\\n"
                     "\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    # appendix: k-curves
    kc = E.get("kcurve") or {}
    if kc:
        Ks = sorted({p["k"] for pts in kc.values() for p in pts})
        rows = [f"{nm} & " + " & ".join(tn(g(_kval(pts, k_), "mean"), 1) for k_ in Ks) + " \\\\"
                for nm, pts in (("attribution (discovery)", kc.get("attr")), ("attribution (oracle)", kc.get("oracle")),
                                ("activation change", kc.get("magnitude")), ("random", kc.get("random"))) if pts]
        T["kcurve"] = ("\\begin{table}[h]\n\\centering\\small\n\\caption{Gap closed (\\%) by the top-$K$ features of each ranking, held-out "
                       "frames, primary occluder.}\n\\label{tab:kcurve}\n\\begin{tabular}{l" + "c" * len(Ks) + "}\n\\toprule\nranking & "
                       + " & ".join(f"$K$={k_}" for k_ in Ks) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return T


def latex_paper(runs, KS, figs) -> str:
    main, K = runs[0], KS[0]
    cfg = main["cfg"]
    f = K["primary"]
    E = K["goal2"].get(f, {})
    o = outcome(K)
    D = E.get("dissociation") or {}
    H = E.get("heldout") or {}
    G3 = K.get("goal3") or {}
    n_tot = K.get("n_features_total") or g(main, "goal2", "protocol", "n_features_total")
    KA = g(main, "goal2", "protocol", "attr_k") or cfg.get("attr_k", 100)
    ceil = g(E, "ceiling", "mean")
    k50 = _k_to_reach(g(E, "kcurve", "attr"), 0.5 * ceil if fin(ceil) else None)
    k50m = _k_to_reach(g(E, "kcurve", "magnitude"), 0.5 * ceil if fin(ceil) else None)
    g1 = g(K, "goal1", f) or {}
    reps = list(zip(runs[1:], KS[1:]))
    rep_h2a = [outcome(k)["h2a"] for _, k in reps]
    title = title_for(K)
    mdl = model_tex(cfg)
    oc_names = {"covered": "box", "screened": "screen", "painted": "paint"}
    occ = oc_names.get(G3.get("occluder"), G3.get("occluder") or "box")
    rates = G3.get("rates") or {}
    comps = G3.get("comparisons") or {}
    pct = lambda c: tn(100 * g(rates, c, "success", default=float("nan")), 0)
    dpt = lambda k_: tdci({"mean_diff": 100 * comps[k_]["diff"], "ci95": [100 * x if fin(x) else x for x in comps[k_]["ci95"]]}, 0) if k_ in comps else "--"

    # ---- abstract
    ab = [f"Mechanistic claims about vision-language-action (VLA) policies usually select internal features by how selectively they "
          f"respond to a concept and then show their causal role on the same inputs. We ask whether such claims survive a held-out, "
          f"causally grounded protocol, for one concrete question: how does a VLA register that the object named in its instruction "
          f"is hidden, and which internal features carry that into its actions? We train per-layer TopK transcoders on the Gemma "
          f"backbone of {mdl}, hide the target with simulator-rendered occluders (an opaque box and a camera-blocking screen, plus a "
          f"pixel-paint control) on all {K.get('n_tasks')} {main['suite']} tasks, fix every feature set on discovery demonstrations, and "
          f"test on disjoint ones with task-clustered confidence intervals."]
    ab.append(f"Occlusion-selective features exist ({g1.get('n_pass_total', '--')} of {n_tot:,}) and their selectivity replicates "
              f"({tn(H.get('enrichment_sel_capped'), 0)}$\\times$ the base rate), yet patching them closes {tn(g(E, 'sel', 'mean'), 1)}\\% of the "
              f"occlusion effect on the action, indistinguishable from norm-matched random features ({tn(g(E, 'rand_sel', 'mean'), 1)}\\%)."
              if not o["h2s"] else
              f"Occlusion-selective features exist ({g1.get('n_pass_total', '--')} of {n_tot:,}) and close {tn(g(E, 'sel', 'mean'), 1)}\\% "
              f"of the occlusion effect on the action (random: {tn(g(E, 'rand_sel', 'mean'), 1)}\\%).")
    if o["h2a"]:
        ab.append(f"In contrast, the top-{KA} features by gradient attribution ({tn(100 * KA / max(n_tot or 1, 1), 2)}\\% of the dictionary) "
                  f"close {tci(E.get('attr'), 1)}\\%, {tn(g(E, 'p_attr', 'mean_diff'), 1)} points above matched random features, and "
                  f"{tn(g(E, 'attr_xtask', 'mean'), 1)}\\% when chosen on other tasks only.")
    else:
        ab.append(f"Even the top-{KA} features by gradient attribution close only {tci(E.get('attr'), 1)}\\% (random: "
                  f"{tn(g(E, 'rand_attr', 'mean'), 1)}\\%), against {tn(ceil, 1)}\\% for all MLP outputs"
                  + (f"; reaching half of that takes {k50:,} features." if k50 else "."))
    if D:
        ab.append(("The two sets barely overlap" if o["h3"] else "The two sets overlap") +
                  f" ({D.get('overlap_attr_sel_uncapped', 0)} shared features; Spearman $\\rho$={tn(D.get('spearman_selectivity_rise_vs_attr'), 2)} "
                  "between selectivity and attribution).")
    if G3:
        if o["c6"]:
            ab.append(f"In closed loop the {occ} lowers success from {pct('clean')}\\% to {pct(G3.get('occluder'))}\\%; restoring the attributed "
                      f"features' clean values at every inference step " + ("recovers" if o["h4"] else "does not reliably recover") +
                      f" it ({dpt('restore_attr_minus_' + G3.get('occluder', ''))} points).")
        else:
            ab.append(f"In closed loop the occluders barely change success ({pct('clean')}\\% clean vs {pct(G3.get('occluder'))}\\% with the {occ}), "
                      "so the policy's behaviour is robust to hiding its target even though its actions shift.")
    if reps:
        ab.append("The pattern " + ("holds" if all(x == o["h2a"] for x in rep_h2a) else "partly holds") + " for "
                  + " and ".join(f"{model_tex(R['cfg'])} on {R['suite']}" for R, _ in reps) + ".")
    ab.append("Selectivity is not a reliable proxy for causal relevance in VLAs; held-out, attribution-based tests should be the default."
              if (o["h3"] or (o["h2a"] and not o["h2s"])) else
              "Held-out causal tests, not selectivity, should decide which features are called mechanisms.")
    abstract = " ".join(ab)

    contrib = [
        "\\textbf{A held-out, causally grounded circuit protocol for VLAs.} Feature sets are fixed on discovery demonstrations and tested on "
        "disjoint ones and on unseen tasks; frames sit at fixed phases of the grasp approach; every effect has a norm-matched random "
        "control and a task-clustered CI (\\S\\ref{sec:method}).",
        "\\textbf{Rendered occluders with matched controls.} An opaque box and a camera-blocking screen are drawn into the simulator's "
        "renderer, with an identical-size occluder placed elsewhere and an occluder on a distractor object as controls (\\S\\ref{sec:occluders}).",
        "\\textbf{Selectivity versus attribution.} " + ("Attribution-selected features carry the occlusion effect; selective features do not "
                                                        if o["h2a"] and not o["h2s"] else
                                                        "Neither selective nor attribution-selected sparse sets carry the effect, "
                                                        if not o["h2a"] else "Both selective and attributed sets carry the effect, ")
        + f"and we measure how far first-order attribution predicts patching (Pearson $r$={tn(g(E, 'linearity', 'pearson_pred_vs_proj'), 2)}; "
          "\\S\\ref{sec:q2}).",
        "\\textbf{Closed-loop tests} that restore or inject feature values at every inference step over paired episodes (\\S\\ref{sec:q3}).",
        "\\textbf{Replication} across a second suite and a second policy, and a within-study measurement of how in-sample selection "
        "inflates effects (\\S\\ref{sec:generalization}).",
    ]

    def fig(name, width, caption, label):
        if name not in figs:
            return ""
        ext = "png" if name.startswith("fig_probe") else "pdf"
        return (f"\\begin{{figure}}[t]\n\\centering\n\\includegraphics[width={width}]{{figures/{name}.{ext}}}\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n")

    T = latex_tables(runs, KS)
    IS = E.get("insample") or {}
    V1s = E.get("v1_style") or {}
    lin = E.get("linearity") or {}
    sc = E.get("scopes") or {}
    t = []
    t.append(r"""\documentclass{article}
\usepackage{iclr2027_conference,times}
\input{math_commands.tex}
\usepackage{hyperref}
\usepackage{url}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pifont}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{caption}
\captionsetup{font=small}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
""")
    t.append(f"\\title{{{title}}}\n\\author{{Anonymous authors\\\\Paper under double-blind review}}\n\\begin{{document}}\n\\maketitle\n")
    t.append(f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n")
    t.append(r"""\section{Introduction}
\label{sec:intro}
Vision-language-action (VLA) models map camera images and an instruction to robot actions by fine-tuning a vision-language backbone
with an action head \citep{brohan2023rt2,kim2024openvla,black2024pi0,pi2025pi05}. A growing literature opens them up with sparse
autoencoders, probes and steering \citep{haon2025mechanistic,buurmeijer2026observing,swann2026sparse,grant2026features,jin2026event,shi2026vlatrace}.
A recurring pattern, here and in circuit analysis of language models \citep{marks2025sparse,ameisen2025circuit}, is to select features
because they respond \emph{selectively} to a concept and then to demonstrate their causal role on the same inputs. Selection and test on
the same data inflate effects \citep{kriegeskorte2009circular}, and selective responses need not be the ones the network uses
\citep{bolukbasi2021illusion,makelov2024subspace}.

We study both risks on a question that matters for embodied agents: when the object named in the instruction is hidden, how does the
policy register it, and which internal features carry that into the action? Answering it requires three things that earlier VLA
analyses lack: occluders that remove the object from view (not just recolour its pixels), feature sets chosen on data disjoint from the
test, and a causal ranking of features rather than a selectivity score. Our contributions:
\begin{itemize}[leftmargin=*,itemsep=1pt,topsep=2pt]
""" + "\n".join(f"  \\item {c}" for c in contrib) + "\n\\end{itemize}\n")
    t.append(r"""\section{Related work}
\textbf{Interpreting VLAs.} Sparse autoencoders and probes have been used to find and steer interpretable directions in VLAs
\citep{haon2025mechanistic,buurmeijer2026observing,swann2026sparse,jin2026event}, to compare feature types \citep{grant2026features}
and to trace representations to behaviour \citep{shi2026vlatrace,zhang2026embodied}. These studies typically select features by
activation statistics and test on overlapping data. \textbf{Transcoders and circuits.} Transcoders replace an MLP with a sparse,
interpretable approximation that supports feature-level causal tracing \citep{dunefsky2024transcoders,ameisen2025circuit,lindsey2025biology},
including in VLMs \citep{damianos2026transcoders}; we use TopK sparsity \citep{gao2025scaling}. \textbf{Attribution patching}
approximates activation patching with one backward pass \citep{nanda2023attribution,syed2024attribution,kramar2024atp}; we use it to rank
transcoder features and measure where its first-order approximation holds. \textbf{Patching methodology.} We follow recommendations on
metrics and controls \citep{zhang2024towards,heimersheim2024patching} and add held-out selection and norm-matched random controls.
""")
    t.append(f"""\\section{{Method}}
\\label{{sec:method}}
\\textbf{{Policy and tasks.}} {mdl} \\citep{{pi2025pi05}} ({cfg.get('policy_path')}, LeRobot \\citep{{cadene2024lerobot}}) maps two
camera images, the instruction and the robot state to a 50-step action chunk with a flow-matching action expert \\citep{{lipman2023flow}}
that reads the key/value cache of an 18-layer Gemma backbone \\citep{{gemma2024,beyer2024paligemma}}. We use all {K.get('n_tasks')}
{main['suite']} tasks \\citep{{liu2023libero}} in MuJoCo/robosuite \\citep{{todorov2012mujoco,zhu2020robosuite}}; the target is the object
the instruction asks to pick up.

\\textbf{{Frames and split.}} For every task we replay demonstrations {cfg.get('discovery_demos')} (\\emph{{discovery}}) and
{cfg.get('test_demos')} (\\emph{{test}}) and take one frame at each of the fixed phases {cfg.get('frame_phases')} of the approach to the
target (nearest step where the target is visible and every occluder and control can be placed), giving {K.get('n_discovery')} discovery
and {K.get('n_test')} test frames. No frame is chosen by the size of its effect. Transcoders are trained on discovery frames and
{K.get('n_extra')} frames from further demonstrations; test frames never enter training or selection.

\\textbf{{Metric.}} For a frame, $a_{{b}}$ and $a_{{o}}$ are the action chunks predicted from the base and occluded images with identical
noise. A patch that turns $a_b$ into $a_p$ closes $100\\,(1-\\lVert a_p-a_o\\rVert/\\lVert a_b-a_o\\rVert)$ \\% of the gap (v1/v2 metric); we
also report the projection $100\\,\\langle a_p-a_b, a_o-a_b\\rangle/\\lVert a_o-a_b\\rVert^2$, which is linear in the patch. Frames with a gap
below {cfg.get('goal2_min_gap')} are excluded (pre-specified).

\\subsection{{Occluders and controls}}
\\label{{sec:occluders}}
We draw occluders into MuJoCo's scene at every render call (environment observations and segmentation alike); they have no physics and
are recomputed from the live state, so they follow the target (Figure~\\ref{{fig:conditions}}). \\textsc{{box}}: an opaque box around the
target's bounding spheres, seen by both cameras. \\textsc{{screen}}: a thin panel on the line from the third-person camera to the target,
sized to hide it from that camera only. \\textsc{{paint}}: the v1/v2 edit that paints the target's pixels gray. Each occluder has a
\\emph{{miss}} control (the same occluder placed where it hides neither the target, nor the distractor, nor other task objects, found by a
segmentation-checked search) and, for \\textsc{{box}} and \\textsc{{paint}}, a \\emph{{distractor}} control (the same occluder on the
nearest object of the same type). \\emph{{Recolour}} and \\emph{{absent}} (target moved out of view) complete the conditions.

\\subsection{{Transcoders, selective features and attribution}}
For every layer $\\ell$ we train a TopK transcoder \\citep{{dunefsky2024transcoders,gao2025scaling}} with {K.get('tc_features')} features and
$k$={K.get('tc_k')} that maps the MLP input to its output, $y_\\ell \\approx \\sum_j f_j(x_\\ell)\\,d_j + b$. Setting a set $S$ of features to
their values in another run adds $\\sum_{{j\\in S}}(f'_j-f_j)\\,d_j$ to the MLP output and keeps the transcoder's error term.

\\textbf{{Selective features (Goal 1)}} follow the v2 rule on discovery frames: a feature is selective for an occluder if its summed
activation rises on $\\geq${int(100 * cfg.get('min_pos_frac', 0.9))}\\% of frames, by at least its typical activation, and by at least
twice its largest response to recolour, absent and the miss control (specificity $\\geq${cfg.get('min_specificity')}); the capped set keeps
the {cfg.get('max_features_per_layer')} best per layer.

\\textbf{{Attribution.}} For each frame we backpropagate the projection metric through the full sampler (all denoising steps) to an
additive perturbation of every MLP output, and score feature $j$ at layer $\\ell$ by
$\\alpha_j=\\sum_t (f^{{o}}_{{j,t}}-f^{{b}}_{{j,t}})\\,\\langle d_j, \\partial m/\\partial y_{{\\ell,t}}\\rangle$ \\citep{{nanda2023attribution,syed2024attribution}},
the first-order effect of setting it to its occluded value in the base run; the remove direction (occluded run, base values) is scored
the same way and the two are averaged. The attribution set is the top-{KA} features by mean discovery-frame attribution among layers whose
residual stream carries the occlusion signal (a single-layer residual swap closes $\\geq${cfg.get('rank_min_resid_pct', 10)}\\% of the gap on
discovery frames: layers {K.get('rank_layers')}).

\\textbf{{Pre-specified tests.}} H1: selective features replicate on test frames. H2-A: the attribution set beats norm-matched random
features by $\\geq${cfg.get('goal2_margin_pts')} points with a 95\\% CI above 0 (primary: \\textsc{{box}}). H2-S: the v2 rule for the selective
set. H3: selectivity and attribution dissociate ($<$10\\% of the attribution set is selective and Spearman $\\rho<0.2$). H4: in closed loop,
restoring the attribution set's clean values under the occluder raises success relative to no restore and to a matched random set. CIs
resample the {K.get('n_tasks')} tasks with replacement; $p$-values are sign-flip tests over task means.
""")
    t.append(fig("fig_probe_conditions_agentview", "\\linewidth",
                 "The conditions on one frame per task (third-person camera, as the policy sees it): base, recolour, absent, the three "
                 "occluders (\\textsc{paint}, \\textsc{box}, \\textsc{screen}), their miss controls and the distractor controls.",
                 "fig:conditions"))
    t.append(f"""\\section{{Results}}
\\subsection{{The instrument works}}
\\label{{sec:validity}}
All occluders change the action: the median base-vs-occluded gap is {tn(g(K, 'gaps', 'cover', 'median_gap'), 3)} (\\textsc{{box}}),
{tn(g(K, 'gaps', 'screen', 'median_gap'), 3)} (\\textsc{{screen}}) and {tn(g(K, 'gaps', 'paint', 'median_gap'), 3)} (\\textsc{{paint}}), and occluding the
target moves the action more than occluding the distractor ({tn(g(K, 'gaps', 'cover', 'distractor', 'median_target'), 3)} vs
{tn(g(K, 'gaps', 'cover', 'distractor', 'median_distractor'), 3)} for \\textsc{{box}}). A single-layer residual swap closes up to
{tn(K.get('ceiling_best_pct'), 1)}\\% (layer {K.get('ceiling_best_layer')}); swaps after layer {max(K.get('rank_layers') or [0])} close little
(Figure~\\ref{{fig:layers}}). Transcoders reach a median FVU of {tn(K.get('fvu_heldout'), 3)} on held-out tokens and {tn(K.get('fvu_test'), 3)} on held-out
test frames (v2: 0.185 and 0.400). Batched patching is bit-identical to single-row patching, so all patches of a frame share one
forward pass. Table~\\ref{{tab:validity}} lists every check.
""")
    t.append(T.get("validity", ""))
    t.append(fig("fig_layers", "\\linewidth", "Left: single-layer swaps of the full residual stream (solid) or the MLP output (dashed) from the "
                 "occluded into the base run, discovery frames. Right: where the positive attribution mass sits.", "fig:layers"))
    t.append(f"""\\subsection{{Q1: selective features exist and replicate}}
\\label{{sec:q1}}
{g1.get('n_pass_total', '--')} of {n_tot:,} features pass the selectivity rule for \\textsc{{box}} ({g1.get('n_pass_target_total', '--')} also against the
distractor control; top layers {g1.get('top_layers')}). On held-out test frames {tn(100 * (H.get('replicate_frac_sel_capped') or 0), 0)}\\% of the
selected features pass the same rule again, against a base rate of {tn(100 * (H.get('base_rate_all_alive') or 0), 2)}\\%
({tn(H.get('enrichment_sel_capped'), 0)}$\\times$), and discovery and test responses correlate (Spearman
$\\rho$={tn(H.get('spearman_discovery_vs_test_response'), 2)}). Selectivity is therefore a real, reproducible property of these features.

\\subsection{{Q2: which features carry the effect into the action?}}
\\label{{sec:q2}}
Table~\\ref{{tab:main}} gives the held-out causal tests. """ +
             (f"The selective set closes {tci(E.get('sel'), 2)}\\% of the gap, no more than matched random features "
              f"({tdci(E.get('p_sel'), 2)} points). " if not o["h2s"] else
              f"The selective set closes {tci(E.get('sel'), 2)}\\% of the gap ({tdci(E.get('p_sel'), 2)} points above matched random features). ")
             + (f"The {KA} attribution-selected features close {tci(E.get('attr'), 1)}\\%, {tdci(E.get('p_attr'), 1)} points above matched random "
                f"features (H2-A {'passes' if o['h2a'] else 'fails'}), {tn(g(E, 'verdict', 'attr_share_of_ceiling_pct'), 0)}\\% of the MLP-output ceiling "
                f"({tn(ceil, 1)}\\%).")
             + (f" Chosen on the other half of the tasks only, they still close {tci(E.get('attr_xtask'), 1)}\\%." if E.get("attr_xtask") else "")
             + f"""

\\textbf{{How many features?}} Figure~\\ref{{fig:kcurves}} ranks features three ways. The attribution ranking reaches half of the MLP
ceiling with {f'{k50:,}' if k50 else 'more than the largest $K$ tested'} features; ranking by activation change needs
{f'{k50m:,}' if k50m else 'more than the largest $K$ tested'}, and random features barely move the action. The oracle curve (attribution
computed on the test frame itself) bounds what any ranking could achieve. All transcoder features together close
{tn(g(E, 'all_tc', 'mean'), 1)}\\%; the transcoders' error term alone closes {tn(g(E, 'tc_error', 'mean'), 1)}\\%.

\\textbf{{Is attribution trustworthy?}} Across every set we patched, the attribution prediction and the measured projection correlate
with Pearson $r$={tn(lin.get('pearson_pred_vs_proj'), 2)} (slope {tn(lin.get('slope_proj_on_pred'), 2)}, $n$={lin.get('n', '--')};
Figure~\\ref{{fig:linearity}}), so first-order attribution """ + ("is a usable proxy for patching here." if fin(lin.get('pearson_pred_vs_proj')) and lin.get('pearson_pred_vs_proj') > 0.7
                                                            else "is only a rough proxy here, which is why every claim rests on actual patches.") + f"""

\\textbf{{Selectivity and attribution dissociate.}} The attribution set and the selective set share {D.get('overlap_attr_sel_uncapped', '--')}
features; {tn(100 * (D.get('frac_attr_passing_selectivity') or 0), 0)}\\% of the attribution set passes the selectivity rule, selective features sit at the
{tn(D.get('selective_attr_percentile_median'), 0)}th attribution percentile (median) and carry {tn(D.get('selective_share_of_positive_attr_pct'), 2)}\\% of the
positive attribution, and across all features Spearman $\\rho$(selectivity, attribution)={tn(D.get('spearman_selectivity_rise_vs_attr'), 2)}
(Figure~\\ref{{fig:dissociation}}). H3 {'holds' if o['h3'] else 'does not hold'}.
""" + (f"""
\\textbf{{Where.}} Restricting the patch to tokens under the occluder closes {tci(g(sc, 'occluder_tokens', 'attr'), 1)}\\% (attribution set) and
{tci(g(sc, 'occluder_tokens', 'mlp_ceiling'), 1)}\\% (all MLP outputs); the remaining tokens close {tci(g(sc, 'other_tokens', 'attr'), 1)}\\% and
{tci(g(sc, 'other_tokens', 'mlp_ceiling'), 1)}\\%.
""" if sc.get("occluder_tokens") else ""))
    t.append(T.get("main", ""))
    n_kc = sum(1 for R in runs if fam_stats(R).get("kcurve"))
    t.append(fig("fig_kcurves", f"{min(1.0, 0.34 * n_kc + 0.16):.2f}\\linewidth", "Gap closed on held-out frames by the top-$K$ features of each ranking (shaded: 95\\% CI). Star: "
                 "the selective set. Lines: all transcoder features and all MLP outputs.", "fig:kcurves"))
    t.append("\\begin{figure}[t]\n\\centering\n" +
             ("\\includegraphics[width=0.49\\linewidth]{figures/fig_dissociation.pdf}\\hfill\n" if "fig_dissociation" in figs else "") +
             ("\\includegraphics[width=0.42\\linewidth]{figures/fig_linearity.pdf}\n" if "fig_linearity" in figs else "") +
             "\\caption{Left: selectivity vs attribution for every feature in the rank layers; orange: selective, blue: attribution set. "
             "Right: attribution prediction vs measured effect of every patched set.}\n\\label{fig:dissociation}\\label{fig:linearity}\n\\end{figure}\n")
    if G3:
        t.append(f"""\\subsection{{Q3: does it matter in closed loop?}}
\\label{{sec:q3}}
Table~\\ref{{tab:closedloop}} runs {len(G3.get('tasks', []))} tasks $\\times$ {G3.get('episodes_per_task')} paired episodes per condition. """ +
                 (f"The {occ} lowers success from {pct('clean')}\\% to {pct(G3.get('occluder'))}\\% ({dpt('clean_minus_' + str(G3.get('occluder')))} points). "
                  if o["c6"] else f"Success barely changes under the occluders ({pct('clean')}\\% clean, " +
                  ", ".join(f"{oc_names.get(c, c)} {pct(c)}\\%" for c in (G3.get('stage_A') or [])[1:]) + "), so H4 cannot be tested with power. ")
                 + f"Restoring all MLP outputs from a clean render at every inference step gives {pct('restore_mlp')}\\%, restoring the attribution set "
                 f"{pct('restore_attr')}\\% and a matched random set {pct('restore_random')}\\% (restore attribution $-$ random: "
                 f"{dpt('restore_attr_minus_restore_random')} points). Injecting the attribution set's occluded values into clean episodes gives "
                 f"{pct('inject_attr')}\\%. H4 {'holds' if o['h4'] else 'does not hold'}.\n")
        t.append(T.get("closed_loop", ""))
    t.append("\\subsection{Generalization and the cost of in-sample selection}\n\\label{sec:generalization}\n")
    if reps:
        t.append("Table~\\ref{tab:replication} repeats the core protocol on " + " and ".join(f"{model_tex(R['cfg'])} / {R['suite']}" for R, _ in reps)
                 + ". H2-A " + ", ".join(f"{'passes' if outcome(k)['h2a'] else 'fails'} for {model_tex(R['cfg'])} / {R['suite']}" for R, k in reps) + ". ")
        t.append(T.get("replication", ""))
    if IS:
        seg = []
        for m, lab in (("sel_capped", "selective"), ("attr", "attribution")):
            if IS.get(m) and fin(IS[m].get("margin_discovery")):
                seg.append(f"the {lab} set beats its matched random control by {tn(IS[m]['margin_discovery'], 2)} points on the discovery frames it was "
                           f"chosen on and by {tn(IS[m].get('margin_test'), 2)} points on held-out frames")
        if seg:
            t.append("Measured inside one study, in-sample evaluation inflates effects: " + "; ".join(seg) + ". ")
    if V1s.get("sel_capped"):
        x = V1s["sel_capped"]
        t.append(f"Choosing frames the v1 way (the four largest-gap frames of one task) turns the selective set's ratio to random from "
                 f"{tn(x.get('ratio_all'), 1)}$\\times$ on all frames into {tn(x.get('ratio_v1_style'), 1)}$\\times$. ")
    t.append("Table~\\ref{tab:versions} places the three protocols side by side.\n")
    t.append(T.get("versions", ""))
    t.append(r"""\section{Discussion and limitations}
""" + ("Selectivity and causal relevance come apart in this VLA: the features that respond most specifically to the hidden target are "
       "not the ones through which occlusion changes the action, while a small attribution-selected set is. "
       if o["h2a"] and not o["h2s"] else
       "The occlusion effect on the action is carried by many features at once: neither the most selective features nor the top "
       "attribution-ranked ones account for a large share of it. " if not o["h2a"] else
       "Both selective and attributed features carry the occlusion effect. ") +
             r"""Three limitations bound these claims. (i) The rendered occluders have no physics and follow the target, so their position still
reveals where the target is; they remove its appearance, not its location. (ii) Attribution is first-order at the clean run; we therefore
report its measured agreement with patching and base every claim on patches. (iii) Transcoders leave an error term whose share of the effect
we report; features outside the dictionary are invisible to both rankings. The unit of replication is the task; two suites and two policies
bound but do not settle generality across embodiments.

\section{Conclusion}
Held-out selection, rendered occluders with matched controls and attribution-based rankings change the answer to a simple mechanistic
question about a VLA. We recommend that VLA interpretability studies fix feature sets on data disjoint from the test, rank features by their
effect on the action rather than by selectivity, and report norm-matched random controls with task-level uncertainty.

\bibliography{references}
\bibliographystyle{iclr2027_conference}

\appendix
\section{Additional results}
""")
    t.append(T.get("kcurve", ""))
    t.append(fig("fig_families", "\\linewidth", "Held-out gap closed per occluder family and feature set (95\\% CIs).", "fig:families"))
    t.append(fig("fig_insample", "0.6\\linewidth", "The same feature sets evaluated on the discovery frames they were chosen on and on held-out frames.",
                 "fig:insample"))
    t.append(fig("fig_replication", "\\linewidth", "Replication across policies and suites.", "fig:replication"))
    t.append(fig("fig_closed_loop", "0.8\\linewidth", "Closed-loop success with Wilson 95\\% CIs.", "fig:closedloop"))
    cfg_txt = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()) if k not in ("libero_dataset_dir",) and not isinstance(v, dict))[:3000]
    cfg_txt = cfg_txt.replace("_", "\\_").replace("#", "\\#").replace("%", "\\%")
    t.append("\\section{Configuration}\n\\small All settings of the main run (\\texttt{config.json}): " + cfg_txt + ".\n\\end{document}\n")
    return "\n".join(t)


def write_paper(runs, KS, figs, outdir: Path, compile_pdf: bool, v2_bib: Path | None):
    pdir = outdir / "paper_v3"
    (pdir / "figures").mkdir(parents=True, exist_ok=True)
    if STYLE_ZIP_B64 and not STYLE_ZIP_B64.startswith("__"):
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(STYLE_ZIP_B64))) as z:
            z.extractall(pdir)
    for n in figs:
        for ext in ("pdf", "png"):
            p = outdir / "figures" / f"{n}.{ext}"
            if p.exists():
                shutil.copy(p, pdir / "figures" / p.name)
    bib = (pdir / "references.bib").read_text() if (pdir / "references.bib").exists() else ""
    if v2_bib and Path(v2_bib).exists() and not bib:
        bib = Path(v2_bib).read_text()
    (pdir / "references.bib").write_text(bib + "\n" + REFS_EXTRA)
    tex = latex_paper(runs, KS, figs)
    (pdir / "main.tex").write_text(tex)
    for name, body in latex_tables(runs, KS).items():
        (pdir / f"table_{name}.tex").write_text(body)
    if compile_pdf:
        try:
            for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                        ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
                subprocess.run(cmd, cwd=pdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            print("[report] paper compiled:", (pdir / "main.pdf").exists())
        except Exception as exc:
            print(f"[report] LaTeX compile skipped ({type(exc).__name__}: {exc}); upload paper_v3/ to Overleaf instead")
    return pdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="name=DIR (first = main run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--v2-bib", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for spec in args.run:
        name, d = spec.split("=", 1)
        if (Path(d) / "config.json").exists():
            runs.append(load_run(name, Path(d)))
        else:
            print(f"[report] skipping {name}: no config.json in {d}")
    if not runs:
        raise SystemExit("no runs to report")
    runs.sort(key=lambda R: RUN_ORDER.index(R["name"]) if R["name"] in RUN_ORDER else 99)
    KS = [key_numbers(R) for R in runs]
    figs = make_figures(runs, out / "figures")
    (out / "results_v3.json").write_text(json.dumps({"runs": KS, "v1": V1, "v2": V2, "title": title_for(KS[0]),
                                                     "outcome": outcome(KS[0])}, indent=2, default=str))
    (out / "RESULTS_v3.md").write_text(md_report(runs, KS, figs))
    pdir = write_paper(runs, KS, figs, out, args.compile, Path(args.v2_bib) if args.v2_bib else None)
    print(f"[report] wrote {out / 'RESULTS_v3.md'}, {out / 'results_v3.json'}, {len(figs)} figures, paper in {pdir}")


if __name__ == "__main__":
    main()
