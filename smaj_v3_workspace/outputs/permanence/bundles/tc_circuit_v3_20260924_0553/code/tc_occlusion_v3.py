#!/usr/bin/env python
"""Transcoder circuit analysis of occlusion in pi0.5 / pi0 (LIBERO), protocol v3.

Every subcommand reads <out_dir>/config.json (written by the notebook). Subcommands can be chained in ONE
process so the policy loads once, and `--resume` skips every step that already finished:

    python tc_occlusion_v3.py frames,validate,capture,train,goal1,attrib,goal2,goal3 --out DIR --resume

  frames    fixed-phase probe frames from demonstrations of every task (discovery / held-out test split) plus
            transcoder-training frames; 11 conditions in 3 occluder FAMILIES:
              cover  (primary) a rendered opaque box around the target, seen by both cameras
              screen           a rendered panel between the third-person camera and the target
              paint            target pixels painted gray (the v1/v2 occluder, kept for continuity)
            each family has its own "miss" control (same occluder elsewhere) and, where possible, a
            distractor control (the same occluder on another object)
  validate  checks 1-4 (clean edits, determinism + batch invariance, signal exists, ceiling) on the unpatched pipeline;
            also fixes the RANK LAYERS (layers whose residual stream carries the occlusion signal, discovery frames only)
  capture   MLP input/output of every LLM layer on discovery + training frames (never on test frames)
  train     one TopK transcoder per layer + check 5 (held-out FVU, test-frame FVU, splice)
  goal1     occlusion-SELECTIVE features per family (v2 rule) on discovery frames
  attrib    gradient ATTRIBUTION of every transcoder feature (attribution patching) on every probe frame and family
  goal2     held-out causal tests: selective vs attribution-selected vs random vs magnitude sets, k-curves,
            linearity of attribution, dissociation, in-sample vs held-out inflation, task-clustered CIs
  goal3     closed loop: success under each occluder, then restoring clean features at every inference step
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import inspect
import json
import math
import os
import pickle
import random
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import torch.nn as nn

PROTOCOL_VERSION = 3
CONDS_ALL = ["base", "recolor", "absent", "occluded", "slab_miss", "occluded_distractor",
             "covered", "cover_miss", "covered_distractor", "screened", "screen_miss"]
RENDERED = {"covered", "cover_miss", "covered_distractor", "screened", "screen_miss"}
FAMILIES = {
    "cover": {"target": "covered", "miss": "cover_miss", "distractor": "covered_distractor",
              "quiet": ["recolor", "absent", "cover_miss"], "closed_loop": "covered",
              "label": "rendered opaque box around the target (both cameras)"},
    "screen": {"target": "screened", "miss": "screen_miss", "distractor": None,
               "quiet": ["recolor", "absent", "screen_miss"], "closed_loop": "screened",
               "label": "rendered panel between the third-person camera and the target"},
    "paint": {"target": "occluded", "miss": "slab_miss", "distractor": "occluded_distractor",
              "quiet": ["recolor", "absent", "slab_miss"], "closed_loop": "painted",
              "label": "target pixels painted gray (v1/v2 occluder)"},
}
CAM_OF_KEY = {"image": "agentview", "image2": "robot0_eye_in_hand"}
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")

CFG: dict = {}
OUT: Path = Path(".")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE: dict = {}


# ----------------------------------------------------------------------------- config helpers


def families() -> list:
    fams = [f for f in CFG.get("families", ["cover", "screen", "paint"]) if f in FAMILIES]
    return fams or ["cover"]


def primary_family() -> str:
    p = CFG.get("primary_family", "cover")
    return p if p in families() else families()[0]


def active_conds() -> list:
    need = {"base", "recolor", "absent"}
    for f in families():
        F = FAMILIES[f]
        need |= {F["target"], F["miss"]}
        if F["distractor"]:
            need.add(F["distractor"])
    return [c for c in CONDS_ALL if c in need]


def quiet_for_color() -> list:
    F = FAMILIES[primary_family()]
    return [F["target"], "absent", F["miss"]]


def fam_target(fam: str) -> str:
    return FAMILIES[fam]["target"]


def parse_layers(spec, n_layers: int) -> list:
    if spec in (None, "", "all"):
        return list(range(n_layers))
    if isinstance(spec, (list, tuple)):
        return [int(v) for v in spec if 0 <= int(v) < n_layers]
    out = []
    for part in str(spec).replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [l for l in out if 0 <= l < n_layers]


# ----------------------------------------------------------------------------- utils


def log(*args):
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean_nan(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    return o


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean_nan(obj), f, indent=2, default=_json_default)
    tmp.replace(path)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_status(step: str, passed, summary: dict):
    save_json(OUT / "status" / f"{step}.json", {"step": step, "passed": passed, "protocol": PROTOCOL_VERSION,
                                                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": True, **summary})


def read_status(step: str):
    p = OUT / "status" / f"{step}.json"
    return load_json(p) if p.exists() else None


def require_gate(step: str, what: str):
    st = read_status(step)
    if st is None:
        raise SystemExit(f"[gate] {step} has not been run yet. Run the {step} step first.")
    if st.get("passed") is False and not CFG.get("force_continue", False):
        raise SystemExit(
            f"[gate] {step} did not pass, so {what} is stopped (pre-specified gate rule). "
            f"Reason: {st.get('reason', 'see status/' + step + '.json')}. Set FORCE_CONTINUE=True to run anyway."
        )
    return st


def setup_torch(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    warnings.filterwarnings("once")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


def closure_pct(a_patch, a_target, a_start) -> float:
    """v1/v2 metric: % of the RMSE gap between start and target closed by the patch."""
    gap = rmse(a_start, a_target)
    if gap <= 0:
        return float("nan")
    return 100.0 * (1.0 - rmse(a_patch, a_target) / gap)


def proj_pct(a_patch, a_target, a_start) -> float:
    """Linear metric: % of the start->target displacement covered by the patch, along that direction."""
    d = (a_target.float() - a_start.float()).flatten()
    n2 = float((d * d).sum())
    if n2 <= 0:
        return float("nan")
    return 100.0 * float(((a_patch.float() - a_start.float()).flatten() * d).sum()) / n2


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask.copy()
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def token_mask(region: np.ndarray, grid: int, min_frac: float) -> np.ndarray:
    """Region in raw (OpenGL) image coords -> (grid*grid,) bool over the model's image tokens.

    The LIBERO processor flips H and W before the policy sees the image, then the image is
    resized to 224 and cut into grid x grid SigLIP patches in row-major order.
    """
    r = region[::-1, ::-1].astype(np.float32)
    H, W = r.shape
    ys = (np.arange(grid + 1) * H / grid).astype(int)
    xs = (np.arange(grid + 1) * W / grid).astype(int)
    out = np.zeros((grid, grid), dtype=bool)
    for i in range(grid):
        for j in range(grid):
            blk = r[ys[i]: ys[i + 1], xs[j]: xs[j + 1]]
            out[i, j] = blk.size > 0 and blk.mean() >= min_frac
    return out.reshape(-1)


def model_view(img: np.ndarray) -> np.ndarray:
    """What the policy sees (after the LIBERO 180-degree flip)."""
    return np.ascontiguousarray(img[::-1, ::-1])


def _add_batch_axis(x):
    if isinstance(x, dict):
        return {k: _add_batch_axis(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x[None])
    return x


def _avail_ram():
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            return None
    return None


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- statistics


def cluster_bootstrap(values, clusters, n_boot: int, seed: int):
    """Mean and 95% CI, resampling whole clusters (tasks) with replacement."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return float("nan"), (float("nan"), float("nan"))
    uniq = sorted(set(c.tolist()))
    S = np.array([v[c == u].sum() for u in uniq])
    N = np.array([(c == u).sum() for u in uniq], dtype=float)
    if len(uniq) == 1:
        return float(v.mean()), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = S[idx].sum(1) / N[idx].sum(1)
    return float(v.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def sign_flip_p(d, n_mc: int, seed: int) -> float:
    """One-sided p for mean(d) > 0 under random sign flips (exact enumeration when len(d) <= 16)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = d.mean()
    if n <= 16:
        signs = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(float)
        perm = (signs * d[None]).mean(1)
        return float((perm >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < n_mc:
        b = min(20000, n_mc - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        hits += int(((signs * d[None]).mean(1) >= obs - 1e-12).sum())
        done += b
    return float((hits + 1) / (n_mc + 1))


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from discordant counts b (only A succeeds) and c (only B succeeds)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def spread_subset(frames, n: int):
    """Up to n frame ids, round-robin over tasks (deterministic)."""
    by = {}
    for fr in frames:
        by.setdefault(fr.get("task_id", 0), []).append(fr)
    out = []
    while len(out) < n and any(by.values()):
        for t in sorted(by):
            if by[t] and len(out) < n:
                out.append(by[t].pop(0))
    return [fr["frame_id"] for fr in out]


def cond_available(fr, cond: str) -> bool:
    return cond in fr["conds"] and bool((fr.get("available") or {}).get(cond, True))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ----------------------------------------------------------------------------- image edits (paint family, recolor)


def recolor(img: np.ndarray, mask: np.ndarray, rgb) -> np.ndarray:
    out = img.copy()
    if not mask.any():
        return out
    lum = img[mask].astype(np.float32).mean(axis=1)
    rel = lum / max(float(lum.max()), 1.0)
    shade = 0.35 + 0.65 * rel
    out[mask] = np.clip(shade[:, None] * np.asarray(rgb, np.float32)[None, :], 0, 255).astype(np.uint8)
    return out


def paint(img: np.ndarray, region: np.ndarray, gray: int) -> np.ndarray:
    out = img.copy()
    out[region] = np.uint8(gray)
    return out


def shift_region(region: np.ndarray, forbidden_strict: np.ndarray, forbidden_soft: np.ndarray):
    """Translate `region` to the nearest spot that avoids the objects (and, if possible, the robot)."""
    if not region.any():
        return region.copy(), (0, 0), "empty"
    H, W = region.shape
    ys, xs = np.nonzero(region)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    dirs = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for forbidden, tag in ((forbidden_strict | forbidden_soft, "clear"), (forbidden_strict, "overlaps_robot")):
        for mul in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            for dy, dx in dirs:
                sy, sx = int(round(dy * mul * h)), int(round(dx * mul * w))
                ny, nx = ys + sy, xs + sx
                if ny.min() < 0 or nx.min() < 0 or ny.max() >= H or nx.max() >= W:
                    continue
                new = np.zeros_like(region)
                new[ny, nx] = True
                if (new & forbidden).any():
                    continue
                return new, (sy, sx), tag
    return np.zeros_like(region), (0, 0), "no_room"


# ----------------------------------------------------------------------------- rendered occluders
#
# Occluders are extra geoms appended to the MuJoCo scene right before every mjr_render call made through
# robosuite's MjRenderContext (environment observations, direct renders and segmentation renders alike).
# They are render-only: no physics, so the policy's world is unchanged except for what the cameras see.
# Every occluder is recomputed from the live simulator state at each render, so it follows the target.


class _Occ:
    installed = False
    active: list = []
    ctx = None


def install_occluders():
    if _Occ.installed:
        return
    import mujoco
    import robosuite.utils.binding_utils as bu

    orig_render = bu.MjRenderContext.render

    def render(self, width, height, camera_id=None, segmentation=False):
        _Occ.ctx = self
        try:
            return orig_render(self, width, height, camera_id=camera_id, segmentation=segmentation)
        finally:
            _Occ.ctx = None

    bu.MjRenderContext.render = render
    orig_mjr = mujoco.mjr_render
    BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
    DECOR = int(mujoco.mjtCatBit.mjCAT_DECOR)
    UNKNOWN = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
    FIXED = int(mujoco.mjtCamera.mjCAMERA_FIXED)

    def mjr_render(viewport, scn, con):
        ctx = _Occ.ctx
        if ctx is not None and _Occ.active:
            m, d = ctx.model._model, ctx.data._data
            cam = int(ctx.cam.fixedcamid) if int(ctx.cam.type) == FIXED else -1
            for spec in _Occ.active:
                for g in spec(m, d, cam):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], BOX, np.asarray(g["size"], np.float64),
                                        np.asarray(g["pos"], np.float64), np.asarray(g["mat"], np.float64).reshape(9),
                                        np.asarray(g["rgba"], np.float32))
                    gg = scn.geoms[scn.ngeom]
                    gg.category = DECOR
                    gg.segid = scn.ngeom      # segmentation id = its index (a stale segid breaks seg renders)
                    gg.objtype = UNKNOWN      # (objtype 0, objid -1) marks occluder pixels in segmentation
                    gg.objid = -1
                    scn.ngeom += 1
        return orig_mjr(viewport=viewport, scn=scn, con=con)

    mujoco.mjr_render = mjr_render
    _Occ.installed = True


@contextlib.contextmanager
def occluders(*specs):
    install_occluders()
    prev = _Occ.active
    _Occ.active = [s for s in specs if s is not None]
    try:
        yield
    finally:
        _Occ.active = prev


class BodyGeoms:
    """Rendered geoms of a set of bodies (by name), resolved per MuJoCo model (hard resets rebuild the model)."""

    def __init__(self, body_names):
        self.body_names = sorted(set(body_names))
        self._key = None
        self._ids = np.zeros(0, dtype=np.int64)

    def ids(self, m):
        if self._key != id(m):
            import mujoco

            bids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.body_names}
            bids.discard(-1)
            ids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bids]
            vis = [g for g in ids if int(m.geom_group[g]) < 3 and not (m.geom_rgba[g, 3] <= 0 and int(m.geom_matid[g]) < 0)]
            self._ids = np.array(vis or ids, dtype=np.int64)
            self._key = id(m)
        return self._ids

    def aabb(self, m, d):
        ids = self.ids(m)
        if len(ids) == 0:
            return None
        c = np.asarray(d.geom_xpos[ids])
        r = np.asarray(m.geom_rbound[ids])
        lo = (c - r[:, None]).min(0)
        hi = (c + r[:, None]).max(0)
        return (lo + hi) / 2.0, (hi - lo) / 2.0


class Cover:
    """Opaque box around the object's bounding spheres; visible to every camera. `offset` moves it (miss control)."""

    def __init__(self, geoms: BodyGeoms, margin: float = 1.15, rgba=(0.5, 0.5, 0.5, 1.0), offset=None):
        self.geoms, self.margin, self.rgba = geoms, float(margin), list(rgba)
        self.offset = None if offset is None else np.asarray(offset, np.float64)

    def __call__(self, m, d, cam):
        b = self.geoms.aabb(m, d)
        if b is None:
            return []
        center, half = b
        if self.offset is not None:
            center = center + self.offset
        return [{"size": half * self.margin, "pos": center, "mat": np.eye(3), "rgba": self.rgba}]


class Screen:
    """Thin opaque panel on the line between one camera and the object, sized to hide it from that camera only.
    `shift` (in panel half-widths, panel x/y axes) moves it sideways (miss control)."""

    def __init__(self, geoms: BodyGeoms, cam_name: str = "agentview", frac: float = 0.35, scale: float = 1.35,
                 rgba=(0.5, 0.5, 0.5, 1.0), shift=(0.0, 0.0)):
        self.geoms, self.cam_name, self.frac, self.scale = geoms, cam_name, float(frac), float(scale)
        self.rgba, self.shift = list(rgba), (float(shift[0]), float(shift[1]))
        self._key, self._cid = None, -1

    def cam_id(self, m):
        if self._key != id(m):
            import mujoco

            self._cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
            self._key = id(m)
        return self._cid

    def __call__(self, m, d, cam):
        cid = self.cam_id(m)
        if cid < 0 or cam != cid:
            return []
        b = self.geoms.aabb(m, d)
        if b is None:
            return []
        center, half = b
        R = float(np.linalg.norm(half))
        cp = np.asarray(d.cam_xpos[cid], np.float64)
        v = cp - center
        D = float(np.linalg.norm(v))
        if D < 1e-6:
            return []
        u = v / D
        p = center + self.frac * v
        s = R * (1.0 - self.frac) * self.scale
        x = np.cross([0.0, 0.0, 1.0], u)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(u, x)
        p = p + self.shift[0] * 2.0 * s * x + self.shift[1] * 2.0 * s * y
        return [{"size": [s, s, 0.002], "pos": p, "mat": np.stack([x, y, u], 1), "rgba": self.rgba}]


# ----------------------------------------------------------------------------- simulator side


def libero_task(env):
    return env.task_description


def make_env(episode_index: int = 0, task_id: int | None = None):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    tid = int(CFG["task_id"] if task_id is None else task_id)
    suite = _get_suite(CFG["suite"])
    env = LiberoEnv(
        task_suite=suite,
        task_id=tid,
        task_suite_name=CFG["suite"],
        obs_type="pixels_agent_pos",
        observation_width=int(CFG["obs_size"]),
        observation_height=int(CFG["obs_size"]),
        init_states=True,
        episode_index=episode_index,
    )
    ensure_sim(env)
    return env


def env_task_id(env) -> int:
    return int(getattr(env, "task_id", CFG.get("task_id", 0)))


def ensure_sim(env):
    """Newer lerobot builds the LIBERO sim lazily (on first reset / _ensure_env); make sure it exists."""
    if getattr(env, "_env", None) is None:
        if callable(getattr(env, "_ensure_env", None)):
            env._ensure_env()
        else:
            env.reset(seed=int(CFG.get("seed", 0)))
    if getattr(env, "_env", None) is None:
        raise RuntimeError("LiberoEnv did not create its simulator (env._env is None).")
    return env


def _camera_obs_names(env):
    names = list(getattr(env, "camera_name", None) or [])
    return names or ["agentview_image", "robot0_eye_in_hand_image"]


def set_cams(env, on: bool) -> bool:
    """Enable / disable camera observables (rendering is ~95% of a LIBERO step; we only need images at inference)."""
    rs = getattr(env._env, "env", None)
    if rs is None or not hasattr(rs, "modify_observable"):
        return False
    try:
        for n in _camera_obs_names(env):
            rs.modify_observable(observable_name=n, attribute="enabled", modifier=bool(on))
        return True
    except Exception:
        return False


def env_obs(env):
    """Fresh observation of the current simulator state (renders with the active occluders)."""
    raw = env._env.env._get_observations(force_update=True)
    fmt = env._format_raw_obs(raw)
    return {"pixels": {k: np.ascontiguousarray(v).copy() for k, v in fmt["pixels"].items()},
            "robot_state": fmt["robot_state"]}


def _bddl_lists(env):
    """obj_of_interest / goal objects / declared objects, from every place LIBERO keeps them."""
    inner = getattr(env._env, "env", None)
    ooi, goal, objs = [], [], []
    try:
        ooi = [str(o) for o in (env._env.obj_of_interest or [])]
    except Exception:
        pass
    pp = getattr(inner, "parsed_problem", None) or {}
    if not ooi:
        ooi = [str(o) for o in (pp.get("obj_of_interest") or [])]
    for pred in pp.get("goal_state") or []:
        if isinstance(pred, (list, tuple)):
            goal += [str(t) for t in pred[1:] if isinstance(t, str)]
    try:
        objs = [str(k) for k in inner.objects_dict.keys()]
    except Exception:
        pass
    text = ""
    for cand in (getattr(inner, "bddl_file_name", None), getattr(env._env, "bddl_file_name", None),
                 getattr(env, "_task_bddl_file", None)):
        if cand and os.path.exists(str(cand)):
            text = Path(str(cand)).read_text()
            break
    if text:
        m = re.search(r"\(:obj_of_interest([^()]*)\)", text)
        if m and not ooi:
            ooi = m.group(1).split()
        g = re.search(r"\(:goal(.*)", text, re.S)
        if g and not goal:
            goal = re.findall(r"\(\s*\w+\s+([\w]+)(?:\s+([\w]+))?\s*\)", g.group(1))
            goal = [t for pair in goal for t in pair if t]
        if not objs:
            o = re.search(r"\(:objects(.*?)\)", text, re.S)
            if o:
                objs = [t for t in o.group(1).split() if t not in ("-",) and ":" not in t]
    return ooi, goal, objs


def _body_root(name: str) -> str:
    for suffix in ("_main", "_body", "_base"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _stem(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def infer_target(env):
    """TARGET_OBJECT, else the first obj_of_interest containing TARGET_KEYWORD ('' = the first obj_of_interest)."""
    kw = str(CFG.get("target_keyword", "")).strip()
    ooi, goal, objs = _bddl_lists(env)
    explicit = str(CFG.get("target_object", "auto")).strip()
    if explicit and explicit != "auto":
        return explicit, ooi, "TARGET_OBJECT"
    for name, cands in (("obj_of_interest", ooi), ("BDDL goal", goal), ("scene objects", sorted(objs))):
        hit = [o for o in cands if kw in o and (not objs or o in objs)]
        if hit:
            return hit[0], ooi, name
    import mujoco

    m = env._env.sim.model._model
    bodies = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)})
    hit = [b for b in bodies if kw and kw in b and not b.startswith(ROBOT_PREFIXES)]
    if hit:
        return _body_root(hit[0]), ooi, "MuJoCo body names"
    raise RuntimeError(
        f"Could not infer the target object for keyword {kw!r}. obj_of_interest={ooi} goal={goal} objects={objs}. "
        "Set TARGET_OBJECT in the plan controls (e.g. akita_black_bowl_1)."
    )


class Scene:
    """Object bookkeeping + rendering helpers on top of the LIBERO robosuite sim.

    robosuite's hard reset (LIBERO default) builds a new MjSim on every env.reset(), so every
    public method re-binds to the live sim first.
    """

    def __init__(self, env):
        import mujoco

        self.mj = mujoco
        self.env = ensure_sim(env)
        self._sim = None
        self.size = int(CFG["obs_size"])
        target, ooi, how = infer_target(env)
        self.target, self.obj_of_interest, self.distractor = target, ooi, None
        _, _, self.scene_objects = _bddl_lists(env)
        self._refresh()
        self.distractor, how_d = self._infer_distractor()
        self._sim = None
        self._refresh()
        self.target_geoms = BodyGeoms([self.body_names[b] for b in self.body_ids])
        self.dist_geoms = BodyGeoms([self.body_names[b] for b in self.dist_body_ids]) if self.dist_body_ids else None
        log(f"task {env_task_id(env)}: target {target!r} (via {how}) | distractor {self.distractor!r} (via {how_d}, "
            f"{len(self.dist_geom_ids)} geoms) | other objects of interest {self.other_names} | "
            f"geoms target={len(self.geom_ids)} | free_joints={len(self.free_joints)}")

    # --- structure
    def _subtree(self, pred):
        out = set()
        for b in range(self.m.nbody):
            a = b
            while a > 0:
                if pred(self.body_names[a]):
                    out.add(b)
                    break
                a = int(self.m.body_parentid[a])
        return out

    def _obj_bodies(self, name):
        return self._subtree(lambda n: n == name or n.startswith(name + "_"))

    def _geoms(self, bodies):
        return np.array([g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in bodies], dtype=np.int64)

    def _root(self, bodies):
        mj = self.mj
        for j in range(self.m.njnt):
            if int(self.m.jnt_bodyid[j]) in bodies and int(self.m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE):
                return int(self.m.jnt_bodyid[j])
        return min(bodies) if bodies else -1

    def _infer_distractor(self):
        explicit = str(CFG.get("distractor_object", "auto")).strip()
        if explicit and explicit != "auto":
            return (None, "DISTRACTOR_OBJECT=none") if explicit.lower() == "none" else (explicit, "DISTRACTOR_OBJECT")
        objs = [o for o in self.scene_objects if o != self.target]
        if not objs:  # no BDDL object list: every free-floating body is an object
            free = {_body_root(self.body_names[int(self.m.jnt_bodyid[j])]) for j in range(self.m.njnt) if int(self.m.jnt_type[j]) == 0}
            objs = sorted(o for o in free if o and o != self.target and not o.startswith(ROBOT_PREFIXES))
        same = sorted(o for o in objs if _stem(o) == _stem(self.target))
        if same:
            return same[0], "same object type"
        kw = str(CFG.get("target_keyword", "")).strip()
        if kw:
            hit = sorted(o for o in objs if kw in o)
            if hit:
                return hit[0], "keyword"
        # nearest movable object that is not an object of interest
        cands = []
        p0 = self.d.xpos[self.root_body][:2] if self.root_body >= 0 else None
        for o in objs:
            if o in self.obj_of_interest:
                continue
            bodies = self._obj_bodies(o)
            if not bodies or not any(int(self.m.jnt_bodyid[j]) in bodies and int(self.m.jnt_type[j]) == 0
                                     for j in range(self.m.njnt)):
                continue
            r = self._root(bodies)
            if p0 is not None and r >= 0:
                cands.append((float(np.linalg.norm(self.d.xpos[r][:2] - p0)), o))
        if cands:
            return sorted(cands)[0][1], "nearest movable non-target object"
        return None, "none found"

    def _refresh(self):
        ensure_sim(self.env)
        sim = self.env._env.sim
        if sim is self._sim:
            return
        mujoco = self.mj
        self._sim = sim
        self.sim = sim
        self.m = sim.model._model
        self.d = sim.data._data
        self.body_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.m.nbody)]
        self.body_ids = self._obj_bodies(self.target)
        if not self.body_ids:
            hint = sorted({n.rsplit("_", 1)[0] for n in self.body_names if n and not n.startswith(ROBOT_PREFIXES)})
            raise RuntimeError(f"No MuJoCo bodies found for target object {self.target!r}. Body-name stems in this scene: {hint}")
        self.geom_ids = self._geoms(self.body_ids)
        self.robot_geom_ids = self._geoms(self._subtree(lambda n: n.startswith(ROBOT_PREFIXES)))
        dname = self.distractor
        self.dist_body_ids = (self._obj_bodies(dname) - self.body_ids) if dname else set()
        self.dist_geom_ids = self._geoms(self.dist_body_ids)
        self.other_names = [o for o in self.obj_of_interest if o not in (self.target, dname)]
        ob = set()
        for o in self.other_names:
            ob |= self._obj_bodies(o)
        self.other_body_ids = ob - self.body_ids - self.dist_body_ids
        self.other_geom_ids = self._geoms(self.other_body_ids)
        self.free_joints = [
            (int(self.m.jnt_qposadr[j]), int(self.m.jnt_bodyid[j]))
            for j in range(self.m.njnt)
            if int(self.m.jnt_bodyid[j]) in self.body_ids and int(self.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        self.root_body = self.free_joints[0][1] if self.free_joints else min(self.body_ids)
        self.dist_root = self._root(self.dist_body_ids) if self.dist_body_ids else -1

    # --- rendering
    def rgb(self, cam: str) -> np.ndarray:
        self._refresh()
        return np.array(self.sim.render(camera_name=cam, width=self.size, height=self.size), copy=True)

    def seg(self, cam: str) -> np.ndarray:
        """(H, W, 2) [objtype, objid]. Own decoder: robosuite 1.4's overflows under numpy 2."""
        self._refresh()
        mj = self.mj
        ctx = self.sim._render_context_offscreen
        cid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_CAMERA, cam)
        W = H = self.size
        ctx.render(width=W, height=H, camera_id=cid, segmentation=True)
        buf = np.empty((H, W, 3), dtype=np.uint8)
        mj.mjr_readPixels(rgb=buf, depth=None, viewport=mj.MjrRect(0, 0, W, H), con=ctx.con)
        code = buf[..., 0].astype(np.int64) + (buf[..., 1].astype(np.int64) << 8) + (buf[..., 2].astype(np.int64) << 16)
        ng = ctx.scn.ngeom
        code[code >= ng + 1] = 0
        table = np.full((ng + 1, 2), -1, dtype=np.int64)
        for i in range(ng):
            g = ctx.scn.geoms[i]
            if g.segid != -1:
                table[g.segid + 1] = (g.objtype, g.objid)
        return table[code]

    def geom_mask(self, seg: np.ndarray, ids: np.ndarray) -> np.ndarray:
        if len(ids) == 0:
            return np.zeros(seg.shape[:2], dtype=bool)
        return (seg[..., 0] == int(self.mj.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 1], ids)

    @staticmethod
    def occ_mask(seg: np.ndarray) -> np.ndarray:
        return (seg[..., 0] == 0) & (seg[..., 1] == -1)

    def seg_masks(self, seg: np.ndarray) -> dict:
        return {"target": self.geom_mask(seg, self.geom_ids), "robot": self.geom_mask(seg, self.robot_geom_ids),
                "distractor": self.geom_mask(seg, self.dist_geom_ids), "others": self.geom_mask(seg, self.other_geom_ids),
                "occ": self.occ_mask(seg)}

    def counts(self, seg: np.ndarray) -> dict:
        return {k: int(v.sum()) for k, v in self.seg_masks(seg).items()}

    def masks(self, cams):
        """-> (target masks, robot masks, distractor masks), each {key: (H, W) bool}."""
        self._refresh()
        tgt, robot, dist = {}, {}, {}
        for key, cam in cams.items():
            s = self.seg_masks(self.seg(cam))
            tgt[key], robot[key], dist[key] = s["target"], s["robot"], s["distractor"]
        return tgt, robot, dist

    def render_absent(self, cams: dict):
        """Render with the target object moved out of view; restore the exact state afterwards."""
        self._refresh()
        mj = self.mj
        q = self.d.qpos.copy()
        v = self.d.qvel.copy()
        rgba = None
        if self.free_joints:
            for i, (adr, _) in enumerate(self.free_joints):
                self.d.qpos[adr: adr + 3] = np.array([60.0 + 5 * i, 60.0, -30.0])
        else:  # fixed object: make it transparent instead
            rgba = self.m.geom_rgba[self.geom_ids].copy()
            self.m.geom_rgba[self.geom_ids, 3] = 0.0
        mj.mj_forward(self.m, self.d)
        imgs = {key: self.rgb(cam) for key, cam in cams.items()}
        leftover = {key: int(self.geom_mask(self.seg(cam), self.geom_ids).sum()) for key, cam in cams.items()}
        self.d.qpos[:] = q
        self.d.qvel[:] = v
        if rgba is not None:
            self.m.geom_rgba[self.geom_ids] = rgba
        mj.mj_forward(self.m, self.d)
        assert np.array_equal(self.d.qpos, q), "state restore failed"
        return imgs, leftover

    def set_state_fast(self, state):
        self._refresh()
        self.env._env.sim.set_state_from_flattened(state)
        self.mj.mj_forward(self.m, self.d)

    def object_z(self) -> float:
        self._refresh()
        return float(self.d.xpos[self.root_body][2])

    def distractor_z(self) -> float:
        self._refresh()
        return float(self.d.xpos[self.dist_root][2]) if self.dist_root >= 0 else float("nan")

    # --- occluder specs
    def cover(self, offset=None):
        return Cover(self.target_geoms, CFG.get("cover_margin", 1.15), CFG.get("occluder_rgba", [0.5, 0.5, 0.5, 1.0]), offset)

    def screen(self, shift=(0.0, 0.0)):
        return Screen(self.target_geoms, CFG.get("screen_camera", "agentview"), CFG.get("screen_frac", 0.35),
                      CFG.get("screen_scale", 1.35), CFG.get("occluder_rgba", [0.5, 0.5, 0.5, 1.0]), shift)

    def cover_distractor(self):
        if self.dist_geoms is None:
            return None
        return Cover(self.dist_geoms, CFG.get("cover_margin", 1.15), CFG.get("occluder_rgba", [0.5, 0.5, 0.5, 1.0]))


def render_with(env, scene, cams, specs, want_pixels=True):
    """Pixels (via the env's own observation path), occluder masks and per-object pixel counts under `specs`."""
    with occluders(*specs):
        px = env_obs(env)["pixels"] if want_pixels else None
        segs = {k: scene.seg(cam) for k, cam in cams.items()}
    occm = {k: Scene.occ_mask(s) for k, s in segs.items()}
    cnt = {k: scene.counts(s) for k, s in segs.items()}
    return px, occm, cnt


def _miss_ok(cnt, base, ref, keys, strict_robot, cams_with_occluder):
    for k in keys:
        c, b = cnt[k], base[k]
        if c["target"] != b["target"] or c["distractor"] != b["distractor"] or c["others"] < 0.9 * b["others"]:
            return False
        if strict_robot and c["robot"] < 0.97 * b["robot"]:
            return False
        if k in cams_with_occluder and c["occ"] > max(2.0 * ref[k]["occ"], 400):
            return False
    return cnt["image"]["occ"] >= 0.5 * max(1, ref["image"]["occ"])


def search_cover_miss(env, scene, cams, base_cnt, cov_cnt):
    scene._refresh()
    b = scene.target_geoms.aabb(scene.m, scene.d)
    if b is None:
        return None
    ext = 2.0 * float(max(b[1][0], b[1][1])) * float(CFG.get("cover_margin", 1.15))
    keys = list(cams)
    for strict in (True, False):
        for mul in (1.25, 1.6, 2.0, 2.5, 3.0, 3.6):
            for ang in range(0, 360, 45):
                off = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0.0]) * mul * ext
                spec = scene.cover(offset=off)
                _, _, cnt = render_with(env, scene, cams, [spec], want_pixels=False)
                if _miss_ok(cnt, base_cnt, cov_cnt, keys, strict, keys):
                    px, occm, cnt = render_with(env, scene, cams, [spec])
                    return px, occm, cnt, {"offset_m": off.tolist(), "tag": "clear" if strict else "overlaps_robot"}
    return None


def search_screen_miss(env, scene, cams, base_cnt, scr_cnt):
    keys = list(cams)
    shifts = [(1.2, 0), (-1.2, 0), (0, 1.2), (0, -1.2), (1.2, 1.2), (-1.2, 1.2), (1.2, -1.2), (-1.2, -1.2),
              (1.7, 0), (-1.7, 0), (0, 1.7), (0, -1.7), (2.3, 0), (-2.3, 0)]
    for strict in (True, False):
        for sh in shifts:
            spec = scene.screen(shift=sh)
            _, _, cnt = render_with(env, scene, cams, [spec], want_pixels=False)
            if _miss_ok(cnt, base_cnt, scr_cnt, keys, strict, ["image"]):
                px, occm, cnt = render_with(env, scene, cams, [spec])
                return px, occm, cnt, {"shift": list(sh), "tag": "clear" if strict else "overlaps_robot"}
    return None


def render_frame(env, scene, state: np.ndarray, conds: list, search_controls: bool = True) -> dict:
    """All requested conditions of one simulator state. `available[cond]` says whether a condition is valid."""
    raw = env._env.set_init_state(state)
    fmt = env._format_raw_obs(raw)
    keys = list(fmt["pixels"].keys())
    cams = {k: CAM_OF_KEY[k] for k in keys}
    scene._refresh()
    base_px = {k: np.ascontiguousarray(fmt["pixels"][k]).copy() for k in keys}
    render_diff = {k: int(np.abs(scene.rgb(cams[k]).astype(int) - base_px[k].astype(int)).max()) for k in keys}
    segs = {k: scene.seg(cam) for k, cam in cams.items()}
    SM = {k: scene.seg_masks(segs[k]) for k in keys}
    base_cnt = {k: {kk: int(v.sum()) for kk, v in SM[k].items()} for k in keys}
    masks = {k: SM[k]["target"] for k in keys}
    robot = {k: SM[k]["robot"] for k in keys}
    dist = {k: SM[k]["distractor"] for k in keys}
    others = {k: SM[k]["others"] for k in keys}
    fr = {"state": np.asarray(state).copy(), "robot_state": fmt["robot_state"], "conds": {"base": base_px},
          "regions": {"base": {k: np.zeros_like(masks[k]) for k in keys}}, "masks": masks, "robot_masks": robot,
          "distractor_masks": dist, "available": {"base": True}, "counts": {"base": base_cnt},
          "render_diff": render_diff, "absent_leftover_px": {k: 0 for k in keys}, "edit_info": {}, "controls": {}}
    need_absent = any(c in conds for c in ("absent",))
    if need_absent:
        absent_px, leftover = scene.render_absent(cams)
        fr["absent_leftover_px"] = leftover
    g = int(CFG["paint_gray"])
    for key in keys:
        m, dm, om = masks[key], dist[key], others[key]
        occ_r = dilate(m, CFG["occ_pad_px"])
        abs_r = dilate(m, CFG["shadow_pad_px"])
        info = {"mask_px": int(m.sum()), "occ_px": int(occ_r.sum()), "distractor_px": int(dm.sum())}
        if "recolor" in conds:
            fr["conds"].setdefault("recolor", {})[key] = recolor(base_px[key], m, CFG["recolor_rgb"])
            fr["regions"].setdefault("recolor", {})[key] = m.copy()
        if need_absent:
            fr["conds"].setdefault("absent", {})[key] = np.where(abs_r[..., None], absent_px[key], base_px[key]).astype(np.uint8)
            fr["regions"].setdefault("absent", {})[key] = abs_r
        if "occluded" in conds:
            fr["conds"].setdefault("occluded", {})[key] = paint(base_px[key], occ_r, g)
            fr["regions"].setdefault("occluded", {})[key] = occ_r
        if "slab_miss" in conds:
            forbid = dilate(m, CFG["shadow_pad_px"] + 2) | dilate(dm, CFG["occ_pad_px"] + 2) | dilate(om, 2)
            slab_r, shift, tag = shift_region(occ_r, forbid, robot[key])
            fr["conds"].setdefault("slab_miss", {})[key] = paint(base_px[key], slab_r, g)
            fr["regions"].setdefault("slab_miss", {})[key] = slab_r
            info.update({"slab_shift": shift, "slab_tag": tag})
            if key == "image":
                fr["available"]["slab_miss"] = tag not in ("no_room", "empty")
        if "occluded_distractor" in conds:
            dist_r = dilate(dm, CFG["occ_pad_px"]) & ~dilate(m, CFG["occ_pad_px"])
            fr["conds"].setdefault("occluded_distractor", {})[key] = paint(base_px[key], dist_r, g)
            fr["regions"].setdefault("occluded_distractor", {})[key] = dist_r
            info["distractor_occ_px"] = int(dist_r.sum())
            if key == "image":
                fr["available"]["occluded_distractor"] = bool(dist_r.sum() > 0)
        fr["edit_info"][key] = info
    if "occluded" in conds:
        fr["available"]["occluded"] = bool(masks["image"].sum() > 0)
    pad = int(CFG.get("render_pad_px", 3))

    def put(cond, px, occm, cnt, ok):
        fr["conds"][cond] = px
        fr["regions"][cond] = {k: dilate(occm[k], pad) for k in keys}
        fr["counts"][cond] = cnt
        fr["available"][cond] = bool(ok)

    if any(c in conds for c in ("covered", "cover_miss")):
        px, occm, cnt = render_with(env, scene, cams, [scene.cover()])
        put("covered", px, occm, cnt, all(cnt[k]["target"] == 0 for k in keys))
        if "cover_miss" in conds:
            res = search_cover_miss(env, scene, cams, base_cnt, cnt) if search_controls else None
            if res is None:
                fr["available"]["cover_miss"] = False
            else:
                put("cover_miss", res[0], res[1], res[2], True)
                fr["controls"]["cover_miss"] = res[3]
    if "covered_distractor" in conds:
        spec = scene.cover_distractor()
        if spec is None or base_cnt["image"]["distractor"] == 0:
            fr["available"]["covered_distractor"] = False
        else:
            px, occm, cnt = render_with(env, scene, cams, [spec])
            ok = all(cnt[k]["target"] == base_cnt[k]["target"] for k in keys) and cnt["image"]["distractor"] == 0
            put("covered_distractor", px, occm, cnt, ok)
    if any(c in conds for c in ("screened", "screen_miss")):
        px, occm, cnt = render_with(env, scene, cams, [scene.screen()])
        put("screened", px, occm, cnt, cnt["image"]["target"] == 0)
        if "screen_miss" in conds:
            res = search_screen_miss(env, scene, cams, base_cnt, cnt) if search_controls else None
            if res is None:
                fr["available"]["screen_miss"] = False
            else:
                put("screen_miss", res[0], res[1], res[2], True)
                fr["controls"]["screen_miss"] = res[3]
    for c in conds:
        fr["available"].setdefault(c, c in fr["conds"])
    return fr


def fmt_obs(frame: dict, cond: str) -> dict:
    return {"pixels": frame["conds"][cond], "robot_state": frame["robot_state"]}


def _distractor_visible(fr) -> bool:
    return int(((fr.get("counts") or {}).get("base") or {}).get("image", {}).get("distractor", 0)) > 0


# ----------------------------------------------------------------------------- policy side


def disable_compile(model):
    """Drop instance-level torch.compile wrappers (sample_actions / forward) so the model runs eagerly."""
    removed = []
    for name in ("sample_actions", "forward", "denoise_step", "embed_prefix"):
        fn = model.__dict__.get(name)
        if fn is not None and (hasattr(fn, "_torchdynamo_orig_callable") or "compile" in type(fn).__name__.lower()
                               or "OptimizedModule" in type(fn).__name__):
            del model.__dict__[name]
            removed.append(name)
    if removed:
        log(f"removed torch.compile wrappers: {removed}")
    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass
    return removed


class Stack:
    def __init__(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        paths = CFG["policy_path"] if isinstance(CFG["policy_path"], list) else [CFG["policy_path"]]
        paths = paths + [p for p in CFG.get("policy_path_fallbacks", []) if p not in paths]
        pcfg, err, path = None, None, None
        for path in paths:
            try:
                pcfg = PreTrainedConfig.from_pretrained(path)
                break
            except Exception as exc:
                err = exc
                log(f"could not load a policy config from {path!r}: {type(exc).__name__}: {exc}")
        if pcfg is None:
            raise RuntimeError(f"no policy config could be loaded from {paths}") from err
        self.path = path
        pcfg.pretrained_path = path
        pcfg.device = "cuda"
        if getattr(pcfg, "compile_model", False):
            # torch.compile (max-autotune = CUDA graphs) reuses output buffers and bakes the graph,
            # which breaks forward hooks / patching. Interpretability needs eager mode.
            log("checkpoint config has compile_model=True -> disabling torch.compile (hooks need eager mode)")
            pcfg.compile_model = False
        ecfg = LiberoEnvConfig(task=CFG["suite"], task_ids=[int(CFG["task_id"])])
        t0 = time.time()
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.policy = make_policy(cfg=pcfg, env_cfg=ecfg)
        load_log = buf.getvalue()
        print(load_log, end="", flush=True)
        m = re.search(r"Missing keys when loading state dict: (\d+) keys", load_log)
        n_missing = int(m.group(1)) if m else 0
        if "Could not load" in load_log or "without loading pretrained weights" in load_log or \
                n_missing > int(CFG.get("max_missing_keys", 2)):
            raise RuntimeError(f"the pretrained weights of {path} did not load (missing keys: {n_missing}); refusing to "
                               "analyse a randomly initialised policy. See the loader output above.")
        self.load_log = {"missing_keys": n_missing, "all_keys_loaded": "All keys loaded successfully" in load_log}
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        disable_compile(self.policy.model)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=pcfg,
            pretrained_path=path,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=ecfg, policy_cfg=pcfg)
        self.config = self.policy.config
        try:
            self.act_dim = int(self.config.output_features["action"].shape[0])
        except Exception:
            self.act_dim = None
        fn = inspect.unwrap(type(self.policy.model).sample_actions)
        self.needs_state = "state" in inspect.signature(fn).parameters
        self.kind = type(self.policy).__name__
        log(f"policy {self.kind} loaded from {path} in {time.time() - t0:.0f}s | dtype={getattr(self.config, 'dtype', '?')} "
            f"chunk={self.config.chunk_size} n_action_steps={self.config.n_action_steps} "
            f"image_features={list(self.config.image_features)} | state input to sample_actions: {self.needs_state}")

    def batch(self, obs: dict, task: str) -> dict:
        from lerobot.envs.utils import preprocess_observation

        o = preprocess_observation(_add_batch_axis(obs))
        o["task"] = [task]
        o = self.env_pre(o)
        o = self.pre(o)
        return o

    def noise(self, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(seed))
        shape = (1, int(self.config.chunk_size), int(self.config.max_action_dim))
        return torch.randn(shape, generator=g, dtype=torch.float32).to(DEV)

    def img_keys(self, batch: dict):
        keys = [k for k in self.config.image_features if k in batch]
        keys += [k for k in self.config.image_features if k not in batch]
        return keys

    def to_env_actions(self, chunk: torch.Tensor) -> list:
        """(n, act_dim) normalized actions -> list of env actions (same post-processing as select_action)."""
        out = []
        for i in range(chunk.shape[0]):
            a = self.post(chunk[i: i + 1])
            a = self.env_post({"action": a})["action"]
            out.append(a.detach().to("cpu").float().numpy()[0])
        return out


def cat_batches(bs: list) -> dict:
    """Stack single-row lerobot batches into one batch (rows may come from different conditions)."""
    if len(bs) == 1:
        return bs[0]
    out = {}
    for k, v in bs[0].items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
            out[k] = torch.cat([b[k] for b in bs], 0)
        elif isinstance(v, list) and len(v) == 1:
            out[k] = [x for b in bs for x in b[k]]
        else:
            out[k] = v
    return out


def _find_lm_layers(model):
    pwe = model.paligemma_with_expert
    cands = []
    for getter in (
        lambda: pwe.paligemma.language_model.layers,
        lambda: pwe.paligemma.model.language_model.layers,
        lambda: pwe.paligemma.language_model.model.layers,
    ):
        try:
            layers = getter()
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
        except Exception:
            pass
    for name, mod in pwe.paligemma.named_modules():
        if name.endswith("language_model.layers") and isinstance(mod, nn.ModuleList):
            cands.append(mod)
    if cands:
        return cands[0]
    raise RuntimeError("Could not find the PaliGemma language-model decoder layers.")


class Runner:
    """Forward hooks on every LLM (PaliGemma/Gemma) decoder layer and its MLP.

    Hooks only fire during the prefix pass (images + prompt); the action expert has its own layers.
    Batched: every row of a batch can carry its own patches, so B patched variants run in one forward pass.
    """

    def __init__(self, stack: Stack):
        self.stack = stack
        self.policy = stack.policy
        self.model = stack.policy.model
        self.layers = _find_lm_layers(self.model)
        self.n_layers = len(self.layers)
        self.capture = None
        self.capture_layers: set = set()
        self.capture_rows = None
        self.capture_kinds = ("mlp_in", "mlp_out", "resid")
        self.row_mlp = None
        self.row_resid = None
        self.pair_fns = None
        self.grad_delta = None
        self.last_prefix: dict = {}
        self.d_model = None
        self.n_forward = 0
        orig = self.model.embed_prefix

        def wrapped(*args, **kwargs):
            embs, pad, att = orig(*args, **kwargs)
            images = args[0] if len(args) > 0 else kwargs["images"]
            tokens = args[2] if len(args) > 2 else kwargs.get("tokens", kwargs.get("lang_tokens"))
            self.last_prefix = {
                "pad": pad[0].detach().bool().clone(),
                "pad_rows": pad.detach().bool().clone(),
                "n_img": len(images),
                "lang_len": int(tokens.shape[1]),
                "T": int(embs.shape[1]),
                "tokens": tokens[0].detach().cpu().clone(),
            }
            return embs, pad, att

        self.model.embed_prefix = wrapped
        for i, layer in enumerate(self.layers):
            layer.mlp.register_forward_hook(self._mlp_hook(i), with_kwargs=True)
            layer.register_forward_hook(self._layer_hook(i))
        log(f"hooked {self.n_layers} LLM decoder layers (MLP + residual output)")

    def _store(self, kind, i, t):
        if self.capture is not None and i in self.capture_layers and kind in self.capture_kinds:
            rows = self.capture_rows
            self.capture[(kind, i)] = (t if rows is None else t[rows]).detach()

    def _mlp_hook(self, i):
        def hook(mod, args, kwargs, output):
            x = args[0] if args else next(iter(kwargs.values()))
            if self.d_model is None:
                self.d_model = int(output.shape[-1])
            self._store("mlp_in", i, x)
            self._store("mlp_out", i, output)
            out, changed = output, False
            rp = self.row_mlp
            if rp is not None and any(i in p for p in rp):
                parts = []
                for b in range(out.shape[0]):
                    fn = rp[b].get(i) if b < len(rp) else None
                    parts.append(fn(x[b: b + 1], out[b: b + 1]).to(out.dtype) if fn is not None else out[b: b + 1])
                out, changed = torch.cat(parts, 0), True
            pf = self.pair_fns
            if pf is not None and i in pf:
                out = out.clone()
                for s in range(0, out.shape[0] - 1, 2):
                    out[s + 1] = pf[i](x[s: s + 1], out[s: s + 1], x[s + 1: s + 2], out[s + 1: s + 2])[0].to(out.dtype)
                changed = True
            gd = self.grad_delta
            if gd is not None and i in gd:
                out, changed = out + gd[i].to(out.dtype), True
            return out if changed else None

        return hook

    def _layer_hook(self, i):
        def hook(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self._store("resid", i, h)
            rp = self.row_resid
            if rp is not None and any(i in p for p in rp):
                parts = []
                for b in range(h.shape[0]):
                    fn = rp[b].get(i) if b < len(rp) else None
                    parts.append(fn(h[b: b + 1]).to(h.dtype) if fn is not None else h[b: b + 1])
                h2 = torch.cat(parts, 0)
                return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
            return None

        return hook

    def _reset(self):
        self.capture = None
        self.capture_layers = set()
        self.capture_rows = None
        self.capture_kinds = ("mlp_in", "mlp_out", "resid")
        self.row_mlp = None
        self.row_resid = None
        self.pair_fns = None
        self.grad_delta = None

    @torch.no_grad()
    def run_rows(self, rows, noise, capture_layers=None, capture_rows=None, B=None, pair_fns=None,
                 capture_kinds=("mlp_in", "mlp_out", "resid")):
        """rows: list of (batch, mlp_patch | None, resid_patch | None). The batch is padded to B rows with
        unpatched copies of row 0, so every call has the same shape (bit-identical numerics across calls).
        Returns actions (n, chunk, act_dim) and captures {(kind, layer): (len(capture_rows), T, D)}."""
        n = len(rows)
        B = max(int(B or n), n)
        allrows = list(rows) + [(rows[0][0], None, None)] * (B - n)
        big = cat_batches([r[0] for r in allrows])
        mlp = [r[1] or {} for r in allrows]
        res = [r[2] or {} for r in allrows]
        self.row_mlp = mlp if any(mlp) else None
        self.row_resid = res if any(res) else None
        self.pair_fns = pair_fns
        self.capture = {} if capture_layers is not None else None
        self.capture_layers = set(capture_layers or [])
        self.capture_rows = None if capture_rows is None else torch.as_tensor(list(capture_rows), dtype=torch.long, device=DEV)
        self.capture_kinds = tuple(capture_kinds)
        nz = noise.expand(B, -1, -1).clone() if noise.shape[0] == 1 else noise.clone()
        try:
            actions = self.policy.predict_action_chunk(big, noise=nz)
            self.n_forward += 1
        finally:
            cap = self.capture
            self._reset()
        return actions.detach().float()[:n], cap

    def run(self, batch, noise, capture_layers=None, mlp_patch=None, resid_patch=None):
        a, cap = self.run_rows([(batch, mlp_patch, resid_patch)], noise, capture_layers=capture_layers)
        return a[0], cap

    def run_grad(self, batches: list, noise, U: torch.Tensor, layers: list):
        """Gradient of sum_r <a_r, U_r> w.r.t. an additive perturbation of every MLP output (all tokens) at `layers`.
        The policy's sampler is decorated with @torch.no_grad(); we call the undecorated function."""
        B = len(batches)
        big = cat_batches(batches)
        T = int(self.last_prefix.get("T"))
        D = int(self.d_model)
        deltas = {l: torch.zeros(B, T, D, device=DEV, dtype=torch.float32, requires_grad=True) for l in layers}
        pol = self.policy
        fn = inspect.unwrap(type(self.model).sample_actions)
        nz = noise.expand(B, -1, -1).clone() if noise.shape[0] == 1 else noise.clone()
        self.grad_delta = deltas
        try:
            with torch.enable_grad():
                images, img_masks = pol._preprocess_images(big)
                tokens = big["observation.language.tokens"]
                masks = big["observation.language.attention_mask"]
                if self.stack.needs_state:
                    a = fn(self.model, images, img_masks, tokens, masks, pol.prepare_state(big), noise=nz)
                else:
                    a = fn(self.model, images, img_masks, tokens, masks, noise=nz)
                ad = self.stack.act_dim or U.shape[-1]
                a = a[:, :, :ad].float()
                m = (a * U).sum()
                if not m.requires_grad:
                    raise RuntimeError("the action does not depend on the MLP perturbation (no gradient path)")
                m.backward()
            # the last layer's MLP output never reaches the action expert (the expert reads per-layer K/V, computed
            # before that layer's MLP), so its perturbation gets no gradient: its effect on the action is exactly zero
            grads = {l: (deltas[l].grad if deltas[l].grad is not None else torch.zeros_like(deltas[l])).detach() for l in layers}
        finally:
            self._reset()
        return a.detach(), grads

    # token layout ---------------------------------------------------------------
    def layout(self, batch=None):
        lp = self.last_prefix
        n_img, lang_len, T = lp["n_img"], lp["lang_len"], lp["T"]
        if (T - lang_len) % n_img:
            log(f"WARNING: prefix has {T - lang_len} non-language tokens, not divisible by {n_img} images")
        per_img = (T - lang_len) // n_img
        grid = int(round(math.sqrt(per_img)))
        keys = self.stack.img_keys(batch) if batch is not None else list(self.stack.config.image_features)
        suffix = [k.split(".")[-1] for k in keys]
        return {"n_img": n_img, "lang_len": lang_len, "T": T, "per_img": per_img, "grid": grid, "img_keys": keys,
                "img_suffix": suffix}

    def region_positions(self, layout, regions: dict) -> torch.Tensor:
        """Token positions overlapping an image region (e.g. where the occluder is), in any camera."""
        T = layout["T"]
        m = torch.zeros(T, dtype=torch.bool)
        for i, suf in enumerate(layout["img_suffix"]):
            if suf in regions:
                tm = token_mask(regions[suf], layout["grid"], CFG["token_min_frac"])
                idx = np.nonzero(tm)[0] + i * layout["per_img"]
                m[torch.as_tensor(idx, dtype=torch.long)] = True
        return m


def get_stack() -> Stack:
    """One policy per process (chained subcommands reuse it)."""
    if "stack" not in _CACHE:
        _CACHE["stack"] = Stack()
    return _CACHE["stack"]


def get_runner() -> Runner:
    """Hooks are registered once per model; never build a second Runner on the same policy."""
    if "runner" not in _CACHE:
        _CACHE["runner"] = Runner(get_stack())
    return _CACHE["runner"]


def patch_batch() -> int:
    """Rows per forward pass. Validation checks batch invariance; if it failed, everything runs with 1 row."""
    st = read_status("validate") or {}
    if st.get("batch_invariant") is False:
        return 1
    return max(1, int(CFG.get("patch_batch", 8)))


# ----------------------------------------------------------------------------- transcoder


class TopKTranscoder(nn.Module):
    """MLP-input -> MLP-output transcoder with TopK sparsity (inputs/outputs mean-centred, unit mean norm)."""

    def __init__(self, d_in: int, d_out: int, n_feat: int, k: int):
        super().__init__()
        self.d_in, self.d_out, self.n_feat, self.k = d_in, d_out, n_feat, k
        self.W_enc = nn.Parameter(torch.empty(d_in, n_feat))
        self.b_enc = nn.Parameter(torch.zeros(n_feat))
        self.W_dec = nn.Parameter(torch.empty(n_feat, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.register_buffer("y_mean", torch.zeros(d_out))
        self.register_buffer("x_scale", torch.ones(()))
        self.register_buffer("y_scale", torch.ones(()))

    def init_weights(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(self.n_feat, self.d_out, generator=g)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec.data.copy_(W)
        if self.d_in == self.d_out:
            self.W_enc.data.copy_(W.t())
        else:
            self.W_enc.data.copy_(torch.randn(self.d_in, self.n_feat, generator=g) / math.sqrt(self.d_in))

    def pre(self, x):
        xn = (x.float() - self.x_mean) / self.x_scale
        return xn @ self.W_enc + self.b_enc

    def encode(self, x):
        z = self.pre(x)
        v, i = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, i, torch.relu(v))

    def decode(self, f):
        return (f @ self.W_dec + self.b_dec) * self.y_scale + self.y_mean

    def delta(self, df, idx=None):
        W = self.W_dec if idx is None else self.W_dec[idx]
        return (df @ W) * self.y_scale

    def error(self, x, y):
        return y.float() - self.decode(self.encode(x))


def train_transcoder(x: torch.Tensor, y: torch.Tensor, seed: int, device=None, steps: int | None = None):
    """x, y: (N, d) bf16 on CPU. Data stays bf16 on the GPU; each batch is upcast to fp32."""
    device = device or DEV
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_hold = max(256, int(CFG["tc_holdout_frac"] * n))
    hold, train = perm[:n_hold], perm[n_hold:]
    xt = x[train].to(device)
    yt = y[train].to(device)
    xh = x[hold].to(device)
    yh = y[hold].to(device)
    tc = TopKTranscoder(x.shape[1], y.shape[1], int(CFG["tc_features"]), int(CFG["tc_k"])).to(device)
    tc.init_weights(seed)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(CFG.get("tc_tf32", True))
    try:
        with torch.no_grad():
            xs = xt[: min(200000, len(xt))].float()
            ys = yt[: min(200000, len(yt))].float()
            tc.x_mean.copy_(xs.mean(0))
            tc.y_mean.copy_(ys.mean(0))
            tc.x_scale.copy_(((xs - tc.x_mean) ** 2).sum(-1).mean().sqrt().clamp_min(1e-6))
            tc.y_scale.copy_(((ys - tc.y_mean) ** 2).sum(-1).mean().sqrt().clamp_min(1e-6))
            xs0 = xs[: min(4096, len(xs))]
            rec0 = (tc.encode(xs0) @ tc.W_dec).norm(dim=-1).mean()
            tc.W_enc.data /= rec0.clamp_min(1e-6)
            del xs, ys
        opt = torch.optim.Adam(tc.parameters(), lr=float(CFG["tc_lr"]), betas=(0.9, 0.999))
        steps, bs = int(steps or CFG["tc_steps"]), int(CFG["tc_batch"])
        since_fired = torch.zeros(tc.n_feat, device=device)
        gd = torch.Generator(device=device).manual_seed(seed)
        hist = []
        for step in range(steps):
            lr_mult = min(1.0, (step + 1) / max(1, steps // 20)) * (1.0 if step < 0.8 * steps else max(0.05, (steps - step) / (0.2 * steps)))
            for pg in opt.param_groups:
                pg["lr"] = float(CFG["tc_lr"]) * lr_mult
            idx = torch.randint(0, xt.shape[0], (bs,), generator=gd, device=device)
            xb = xt[idx].float()
            yb = (yt[idx].float() - tc.y_mean) / tc.y_scale
            z = tc.pre(xb)
            v, i = z.topk(tc.k, dim=-1)
            f = torch.zeros_like(z).scatter(-1, i, torch.relu(v))
            yhat = f @ tc.W_dec + tc.b_dec
            mse = ((yhat - yb) ** 2).sum(-1).mean()
            loss = mse
            fired = (f > 0).any(0)
            since_fired = torch.where(fired, torch.zeros_like(since_fired), since_fired + 1)
            dead = since_fired > int(CFG["tc_dead_steps"])
            n_dead = int(dead.sum())
            if n_dead > 0 and CFG["tc_aux_coef"] > 0:
                ka = min(int(CFG["tc_aux_k"]), n_dead)
                zd = z.masked_fill(~dead[None, :], float("-inf"))
                va, ia = zd.topk(ka, dim=-1)
                fa = torch.zeros_like(z).scatter(-1, ia, torch.relu(va))
                resid = (yb - yhat).detach()
                ehat = fa @ tc.W_dec
                aux = ((ehat - resid) ** 2).sum(-1).mean() / (resid ** 2).sum(-1).mean().clamp_min(1e-8)
                loss = loss + float(CFG["tc_aux_coef"]) * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tc.W_dec.data /= tc.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
            if step % 100 == 0 or step == steps - 1:
                hist.append({"step": step, "loss": float(mse.item()), "dead": n_dead})
        tc.eval()

        @torch.no_grad()
        def evaluate(xx, yy):
            sse, sq, s1, nn_ = 0.0, 0.0, None, 0
            nmse = 0.0
            for s in range(0, xx.shape[0], 8192):
                xb, yb = xx[s: s + 8192].float(), yy[s: s + 8192].double()
                yh_ = tc.decode(tc.encode(xb)).double()
                sse += float(((yh_ - yb) ** 2).sum())
                sq += float((yb ** 2).sum())
                s1 = yb.sum(0) if s1 is None else s1 + yb.sum(0)
                nn_ += yb.shape[0]
                nmse += float((((yh_ - yb) / float(tc.y_scale)) ** 2).sum())
            sst = sq - float((s1 ** 2).sum()) / max(nn_, 1)
            return sse / max(sst, 1e-12), nmse / max(nn_, 1)

        fvu_h, nmse_h = evaluate(xh, yh)
        fvu_t, nmse_t = evaluate(xt[: min(len(xt), 50000)], yt[: min(len(yt), 50000)])
        with torch.no_grad():
            fires = torch.zeros(tc.n_feat, device=device)
            for s in range(0, xt.shape[0], 8192):
                fires += (tc.encode(xt[s: s + 8192].float()) > 0).float().sum(0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    metrics = {
        "fvu_heldout": fvu_h, "loss_heldout": nmse_h, "fvu_train": fvu_t, "loss_train": nmse_t,
        "alive_features": int((fires > 0).sum()), "n_train_tokens": int(len(train)), "n_heldout_tokens": int(n_hold),
        "steps": steps, "history": hist,
    }
    del xt, yt, xh, yh
    return tc.cpu(), metrics


def load_tc(layer: int, device=None) -> TopKTranscoder:
    device = device or DEV
    ck = torch.load(OUT / "transcoders" / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    tc = TopKTranscoder(c["d_in"], c["d_out"], c["n_feat"], c["k"])
    tc.load_state_dict(ck["state_dict"])
    tc.requires_grad_(False)
    return tc.to(device).eval()


def load_all_tcs():
    layers = sorted(int(p.stem.split("_")[1]) for p in (OUT / "transcoders").glob("layer_*.pt"))
    if "tcs" not in _CACHE or _CACHE.get("tcs_layers") != layers:
        _CACHE["tcs"] = {l: load_tc(l) for l in layers}
        _CACHE["tcs_layers"] = layers
    return _CACHE["tcs"]


# ----------------------------------------------------------------------------- patch functions (one row)


def fp_set(tc, feats, target_f, token_mask=None, norm_match=None):
    """Set features `feats` to their values in `target_f` (T, n_feat); keeps the transcoder error term.
    norm_match: (T,) per-token norm the output change is rescaled to (same-norm random controls)."""
    idx = torch.as_tensor(list(feats), dtype=torch.long, device=target_f.device)

    def fn(x, y):
        f_live = tc.encode(x[0])
        d = tc.delta(target_f[:, idx] - f_live[:, idx], idx)
        if norm_match is not None:
            n = d.norm(dim=-1, keepdim=True)
            d = torch.where(n > 1e-8, d / n.clamp_min(1e-8) * norm_match[:, None], torch.zeros_like(d))
        if token_mask is not None:
            d = d * token_mask[:, None].to(d.dtype)
        return (y[0].float() + d)[None]

    return fn


def fp_multi(fns):
    """Compose several patch functions acting on the same layer output."""
    fns = [f for f in fns if f is not None]

    def fn(x, y):
        out = y
        for f in fns:
            out = f(x, out)
        return out

    return fn


def fp_swap(target_y, token_mask):
    def fn(x, y):
        m = token_mask[:, None]
        return torch.where(m, target_y.to(y.dtype), y[0]).float()[None]

    return fn


def fp_err_swap(tc, target_err, token_mask):
    """Replace only the transcoder's error term (y - TC(x)) by its value in the target run."""

    def fn(x, y):
        e_live = tc.error(x[0], y[0])
        return (y[0].float() + (target_err - e_live) * token_mask[:, None].float())[None]

    return fn


def rp_swap(target_h, token_mask):
    def fn(h):
        m = token_mask[:, None]
        return torch.where(m, target_h.to(h.dtype), h[0])[None]

    return fn


def fp_splice(tc, token_mask):
    """Replace the MLP output with the transcoder's reconstruction (no error term)."""

    def fn(x, y):
        rec = tc.decode(tc.encode(x[0]))
        return torch.where(token_mask[:, None], rec, y[0].float())[None]

    return fn


# ----------------------------------------------------------------------------- pair patches (closed loop)
# A pair = (source row 2i, destination row 2i+1) in the same forward pass; the hook edits the destination.


def _pad_now(runner, T):
    p = runner.last_prefix.get("pad")
    return p if p is not None and p.shape[0] == T else torch.ones(T, dtype=torch.bool, device=DEV)


def pf_set(tc, idx_list, runner):
    idx = torch.as_tensor(list(idx_list), dtype=torch.long, device=DEV)

    def fn(xs, ys, xd, yd):
        fs, fd = tc.encode(xs[0]), tc.encode(xd[0])
        d = tc.delta(fs[:, idx] - fd[:, idx], idx) * _pad_now(runner, xd.shape[1])[:, None].float()
        return (yd[0].float() + d)[None]

    return fn


def pf_set_normmatch(tc, idx_list, ref_list, runner):
    idx = torch.as_tensor(list(idx_list), dtype=torch.long, device=DEV)
    ref = torch.as_tensor(list(ref_list), dtype=torch.long, device=DEV)

    def fn(xs, ys, xd, yd):
        fs, fd = tc.encode(xs[0]), tc.encode(xd[0])
        pad = _pad_now(runner, xd.shape[1])[:, None].float()
        d = tc.delta(fs[:, idx] - fd[:, idx], idx)
        n_ref = tc.delta(fs[:, ref] - fd[:, ref], ref).norm(dim=-1, keepdim=True) if len(ref_list) else torch.zeros_like(d[:, :1])
        n = d.norm(dim=-1, keepdim=True)
        d = torch.where(n > 1e-8, d / n.clamp_min(1e-8) * n_ref, torch.zeros_like(d)) * pad
        return (yd[0].float() + d)[None]

    return fn


def pf_swap(runner):
    def fn(xs, ys, xd, yd):
        m = _pad_now(runner, xd.shape[1])[:, None]
        return torch.where(m, ys[0], yd[0]).float()[None]

    return fn


# ----------------------------------------------------------------------------- attribution


def feature_attr(tc, grad: torch.Tensor, df: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """First-order effect on the metric of moving every feature by df (T, n): y_scale * sum_t df_tj <d_j, g_t>."""
    G = grad.float() @ tc.W_dec.t()
    return ((df * G) * valid[:, None].float()).sum(0) * float(tc.y_scale)


# ----------------------------------------------------------------------------- figures shared by steps


def fig_conditions(frames, path: Path, key="image", conds=None):
    plt = _plt()
    n = len(frames)
    if n == 0:
        return
    conds = conds or [c for c in active_conds() if c in frames[0]["conds"]]
    fig, axes = plt.subplots(n, len(conds), figsize=(1.9 * len(conds), 2.0 * n), squeeze=False)
    for r, fr in enumerate(frames):
        for c, cond in enumerate(conds):
            ax = axes[r][c]
            if cond in fr["conds"] and key in fr["conds"][cond]:
                ax.imshow(model_view(fr["conds"][cond][key]))
                if not fr.get("available", {}).get(cond, True):
                    ax.text(5, 20, "n/a", color="red", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cond.replace("_", "\n"), fontsize=8)
            if c == 0:
                ax.set_ylabel(f"task {fr.get('task_id', '?')} {fr.get('split', '')}\nt={fr.get('t', '?')}", fontsize=7)
    fig.suptitle(f"Probe frames x conditions ({key}, as the policy sees it)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- data I/O


def load_frames():
    p = OUT / "frames.pkl"
    key = (str(p), p.stat().st_mtime_ns)
    if _CACHE.get("frames_key") != key:
        with open(p, "rb") as f:
            _CACHE["frames"] = pickle.load(f)
        _CACHE["frames_key"] = key
    return _CACHE["frames"]


def find_demo_file(task_name: str):
    roots = [Path(os.environ.get("LIBERO_DATASET_DIR", "")), Path(CFG.get("libero_dataset_dir", ""))]
    suite = str(CFG.get("suite", ""))
    for root in roots:
        if str(root) not in ("", ".") and root.exists():
            hits = sorted(root.rglob(f"{task_name}_demo.hdf5"))
            hits = sorted(hits, key=lambda p: (suite not in str(p), str(p)))
            if hits:
                return hits[0]
    return None


def probe_frames(split=None):
    fr = load_frames()["probe"]
    return [f for f in fr if split is None or f.get("split") == split]


# ============================================================================= frames


def _episode_roles():
    roles = [("discovery", int(d)) for d in CFG["discovery_demos"]]
    roles += [("test", int(d)) for d in CFG["test_demos"]]
    roles += [("extra", int(d)) for d in CFG.get("extra_demos", [])]
    return roles


def _task_episodes(env, scene, task_id: int):
    """Demo (or rollout) state sequences for this task, tagged discovery / test / extra."""
    roles = _episode_roles()
    src = CFG["frame_source"]
    eps = []
    if src in ("auto", "demo"):
        path = find_demo_file(env.task)
        if path is None:
            msg = f"task {task_id}: no demo file {env.task}_demo.hdf5 under LIBERO_DATASET_DIR"
            if src == "demo":
                raise SystemExit(msg)
            log(msg + " -> falling back to policy rollouts")
        else:
            try:
                import h5py

                with h5py.File(path, "r") as f:
                    names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
                    for role, di in roles:
                        if di >= len(names):
                            log(f"task {task_id}: demo index {di} out of range ({len(names)} demos); skipped")
                            continue
                        nm = names[di]
                        eps.append({"name": f"task{task_id}:demo:{nm}", "role": role, "index": di,
                                    "states": np.array(f["data"][nm]["states"])})
            except Exception as exc:
                if src == "demo":
                    raise
                log(f"task {task_id}: could not read demos ({type(exc).__name__}: {exc}) -> falling back to policy rollouts")
                eps = []
            ref = np.asarray(env._env.get_sim_state())
            if eps and eps[0]["states"].shape[1] != ref.shape[0]:
                log(f"task {task_id}: demo state dim {eps[0]['states'].shape[1]} != sim state dim {ref.shape[0]} -> using rollouts")
                eps = []
    if not eps:
        stack, runner = get_stack(), get_runner()
        for role, di in roles:
            res = run_episode(env, scene, stack, runner, task_id, di, "clean", {}, record_states=True,
                              seed=int(CFG["seed"]) + 101 * task_id + di)
            log(f"task {task_id}: rollout init_state={int(CFG.get('goal3_init_offset', 0)) + di} ({role}): {res['steps']} steps, "
                f"success={res['success']}")
            eps.append({"name": f"task{task_id}:rollout:init{di}", "role": role, "index": di,
                        "states": np.stack(res["states"]), "success": res["success"]})
    return eps


def cmd_frames():
    setup_torch(CFG["seed"])
    install_occluders()
    task_ids = [int(t) for t in CFG["task_ids"]]
    phases = [float(p) for p in CFG["frame_phases"]]
    search = int(CFG.get("frame_search", 6))
    n_extra = int(CFG.get("extra_frames_per_demo", 0))
    min_px = int(CFG["min_mask_px"])
    conds = active_conds()
    required = sorted({FAMILIES[f]["target"] for f in families()} | {FAMILIES[f]["miss"] for f in families()})
    extra_conds = [c for c in CFG.get("extra_conds", ["base", "covered", "screened", "occluded"]) if c in conds or c == "base"]
    probe, extra, cand_rows, episodes, tasks = [], [], [], [], {}
    t_all = time.time()
    for task_id in task_ids:
        env = make_env(0, task_id)
        set_cams(env, True)
        scene = Scene(env)
        task = libero_task(env)
        tasks[task_id] = {"task": task, "task_name": env.task, "target": scene.target, "distractor": scene.distractor,
                          "obj_of_interest": scene.obj_of_interest, "other_objects_of_interest": scene.other_names}
        log(f"task {task_id}: {task}")
        for ep in _task_episodes(env, scene, task_id):
            S = ep["states"]
            zs = []
            for s in S:
                scene.set_state_fast(s)
                zs.append(scene.object_z())
            zs = np.array(zs)
            z0 = float(np.median(zs[: max(3, min(10, len(zs)))]))
            lifted = np.nonzero(zs > z0 + float(CFG["lift_dz"]))[0]
            t_lift = int(lifted[0]) if len(lifted) else len(S)
            t_lo = int(CFG["min_t"])
            t_hi = max(t_lo + 1, t_lift - 3)
            used = set()
            n_before = len(probe)
            if ep["role"] in ("discovery", "test"):
                for ph in phases:
                    t_nom = int(round(t_lo + ph * (t_hi - t_lo)))
                    chosen = None
                    offsets = [0] + [s * k for k in range(1, search + 1) for s in (1, -1)]
                    for off in offsets:
                        t = t_nom + off
                        if t < 0 or t >= len(S) or t in used:
                            continue
                        fr = render_frame(env, scene, S[t], conds)
                        ei = fr["edit_info"]["image"]
                        missing = [c for c in required if not fr["available"].get(c, False)]
                        ok = ei["mask_px"] >= min_px and not missing
                        cand_rows.append({"task_id": task_id, "episode": ep["name"], "split": ep["role"], "phase": ph, "t": int(t),
                                          "t_nominal": t_nom, "mask_px_agentview": ei["mask_px"],
                                          "mask_px_wrist": fr["edit_info"].get("image2", {}).get("mask_px", 0),
                                          "distractor_px_agentview": ei["distractor_px"], "missing_conditions": missing,
                                          "accepted": ok})
                        if ok:
                            chosen = (t, fr)
                            break
                    if chosen is None:
                        log(f"WARNING {ep['name']}: no usable frame near phase {ph} (t={t_nom}±{search})")
                        continue
                    t, fr = chosen
                    used.add(t)
                    fid = len(probe)
                    fr.update({"frame_id": fid, "episode": ep["name"], "split": ep["role"], "t": int(t), "phase": ph,
                               "t_nominal": t_nom, "t_lift": t_lift, "task_id": task_id, "task": task, "task_name": env.task,
                               "target": scene.target, "distractor": scene.distractor,
                               "noise_seed": int(CFG["seed"]) + 7919 * (fid + 1)})
                    probe.append(fr)
            if ep["role"] in ("discovery", "extra") and n_extra > 0:
                ts = np.linspace(t_lo, len(S) - 1, n_extra + 2).round().astype(int)[1:-1]
                for t in ts:
                    if int(t) in used:
                        continue
                    fr = render_frame(env, scene, S[int(t)], extra_conds, search_controls=False)
                    extra.append({"episode": ep["name"], "t": int(t), "task_id": task_id, "task": task, "split": ep["role"],
                                  "robot_state": fr["robot_state"],
                                  "conds": {c: fr["conds"][c] for c in extra_conds if c in fr["conds"]}})
            got = [(p["t"], p["phase"]) for p in probe[n_before:]]
            log(f"  {ep['name']} ({ep['role']}): T={len(S)} lift at t={t_lift} -> probe frames (t, phase) {got}")
            episodes.append({"name": ep["name"], "role": ep["role"], "task_id": task_id, "T": int(len(S)),
                             "t_lift": t_lift, "success": ep.get("success")})
        try:
            env.close()
        except Exception:
            pass
        gc.collect()
        log(f"task {task_id} done ({time.time() - t_all:.0f}s so far): {sum(p['task_id'] == task_id for p in probe)} probe frames")

    data = {"version": PROTOCOL_VERSION, "tasks": tasks, "probe": probe, "extra": extra, "episodes": episodes,
            "candidates": cand_rows, "conds": conds}
    with open(OUT / "frames.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    _CACHE.pop("frames_key", None)
    save_json(OUT / "frames_candidates.json", cand_rows)
    show = []
    for tid in task_ids:
        cands = [p for p in probe if p["task_id"] == tid and p["split"] == "discovery"] or [p for p in probe if p["task_id"] == tid]
        if cands:
            show.append(cands[len(cands) // 2])
    fig_conditions(show, OUT / "fig_probe_conditions_agentview.png", "image")
    if show and "image2" in show[0]["conds"]["base"]:
        fig_conditions(show, OUT / "fig_probe_conditions_wrist.png", "image2")
    n_disc = sum(p["split"] == "discovery" for p in probe)
    n_test = sum(p["split"] == "test" for p in probe)
    avail = {c: float(np.mean([bool(p["available"].get(c, False)) for p in probe])) if probe else None for c in conds}
    summary = {
        "n_probe_frames": len(probe), "n_discovery_frames": n_disc, "n_test_frames": n_test,
        "n_tasks": len(task_ids), "frames_per_task": {t: sum(p["task_id"] == t for p in probe) for t in task_ids},
        "n_frames_distractor_visible": sum(_distractor_visible(p) for p in probe),
        "condition_availability": avail, "conds": conds, "families": families(), "primary_family": primary_family(),
        "frame_source": sorted({e["name"].split(":")[1] for e in episodes}),
        "control_tags": {c: {tag: sum(1 for p in probe if p["controls"].get(c, {}).get("tag") == tag)
                             for tag in ("clear", "overlaps_robot")} for c in ("cover_miss", "screen_miss")},
        "probe": [{"frame_id": p["frame_id"], "task_id": p["task_id"], "split": p["split"], "episode": p["episode"],
                   "t": p["t"], "phase": p["phase"], "edit_info": p["edit_info"], "available": p["available"],
                   "counts": p["counts"]} for p in probe],
        "n_extra_train_frames": len(extra), "extra_conds": extra_conds, "tasks": tasks,
        "target_object": sorted({v["target"] for v in tasks.values()}),
        "distractor_object": sorted({str(v["distractor"]) for v in tasks.values()}),
        "selection": f"fixed phases {phases} of the pre-grasp window (no selection on the action gap); nearest frame within "
                     f"±{search} steps where the target has >= {min_px} px in agentview and every family's occluder and "
                     f"miss control could be placed",
    }
    ok = n_disc >= 1 and n_test >= 1
    write_status("frames", ok, {**summary, "reason": "" if ok else "need at least one discovery and one test frame"})
    log(f"saved {len(probe)} probe frames ({n_disc} discovery, {n_test} test) from {len(task_ids)} tasks, "
        f"{summary['n_frames_distractor_visible']} with the distractor visible, {len(extra)} extra transcoder-training frames "
        f"({time.time() - t_all:.0f}s)")


# ============================================================================= validate


def frame_batches(stack, fr, conds):
    return {c: stack.batch(fmt_obs(fr, c), fr["task"]) for c in conds if cond_available(fr, c)}


def run_conds(runner, stack, fr, conds, B, capture_layers=None, capture_conds=(), capture_kinds=("mlp_in", "mlp_out", "resid")):
    """Unpatched actions for `conds` (available ones) in batches of B rows. Captures rows of `capture_conds`."""
    bs = frame_batches(stack, fr, conds)
    names = list(bs)
    noise = stack.noise(fr["noise_seed"])
    A, CAP = {}, {}
    for s in range(0, len(names), B):
        chunk = names[s: s + B]
        rows = [(bs[c], None, None) for c in chunk]
        crow = [i for i, c in enumerate(chunk) if c in capture_conds]
        a, cap = runner.run_rows(rows, noise, capture_layers=capture_layers if crow else None,
                                 capture_rows=crow if crow else None, B=B, capture_kinds=capture_kinds)
        for i, c in enumerate(chunk):
            A[c] = a[i]
        if crow and cap is not None:
            for j, i in enumerate(crow):
                CAP[chunk[i]] = {k: v[j] for k, v in cap.items()}
    return A, CAP, bs


def cmd_validate():
    setup_torch(CFG["seed"])
    frames = probe_frames()
    stack = get_stack()
    runner = get_runner()
    conds = active_conds()
    fams = families()
    prim = primary_family()
    B = max(1, int(CFG.get("patch_batch", 8)))
    res = {}
    det_ids = set(spread_subset(frames, int(CFG.get("determinism_frames", 8))))
    disc = [f for f in frames if f["split"] == "discovery"] or frames
    ceil_ids = spread_subset(disc, int(CFG.get("ceiling_frames", 8)))

    c1_rows, c1_ok, state_same, tokens_same = [], True, True, True
    rows3, c2, binv = [], [], []
    t0 = time.time()
    for fi, fr in enumerate(frames):
        # ---- check 1: edits are clean (pixels) and occluders hide what they should
        base = fr["conds"]["base"]
        for cond in conds[1:]:
            if not cond_available(fr, cond):
                continue
            lim = float(CFG["render_max_outside"]) if cond in RENDERED else float(CFG["clean_edit_max_outside"])
            for key in base:
                img = fr["conds"][cond][key]
                changed = np.abs(img.astype(int) - base[key].astype(int)).max(-1) > 0
                allowed = fr["regions"][cond][key]
                outside = float((changed & ~allowed).mean())
                inside = int((changed & allowed).sum())
                row = {"frame": fr["frame_id"], "cond": cond, "camera": key, "frac_changed_outside": outside,
                       "changed_inside_px": inside, "limit": lim}
                if outside >= lim:
                    c1_ok = False
                    row["warning"] = "changes outside the allowed region"
                if key == "image" and inside == 0:
                    row["warning"] = "edit changed nothing in agentview"
                    c1_ok = False
                c1_rows.append(row)
        for cond, cams_hidden in (("covered", list(base)), ("screened", ["image"])):
            if cond in fr["counts"]:
                for key in cams_hidden:
                    if fr["counts"][cond][key]["target"] != 0:
                        c1_ok = False
                        c1_rows.append({"frame": fr["frame_id"], "cond": cond, "camera": key,
                                        "warning": f"target still visible ({fr['counts'][cond][key]['target']} px)"})
        if any(fr["counts"]["base"][k]["occ"] for k in base):
            c1_ok = False
            c1_rows.append({"frame": fr["frame_id"], "cond": "base", "warning": "occluder pixels in the base render"})
        # ---- all conditions, one frame, batched
        A, _, bs = run_conds(runner, stack, fr, conds, B)
        ref = bs["base"]
        for c, b in bs.items():
            if not torch.equal(b["observation.state"], ref["observation.state"]):
                state_same = False
            for k in [k for k in ref if "tokens" in k]:
                if not torch.equal(b[k], ref[k]):
                    tokens_same = False
        noise = stack.noise(fr["noise_seed"])
        # ---- check 2: determinism (repeat) + batch invariance (row result independent of the other rows)
        if fr["frame_id"] in det_ids:
            tgt = fam_target(prim)
            rows_mixed = [(bs[c], None, None) for c in list(bs)[:B]]
            a1, _ = runner.run_rows(rows_mixed, noise, B=B)
            a2, _ = runner.run_rows(rows_mixed, noise, B=B)
            c2.append(float((a1 - a2).abs().max()))
            a_same, _ = runner.run_rows([(bs["base"], None, None)] * B, noise, B=B)
            binv.append(float((a_same[0] - a1[0]).abs().max()))
            if tgt in bs and B > 1:
                a_t, _ = runner.run_rows([(bs[tgt], None, None)] * B, noise, B=B)
                binv.append(float((a_t[0] - A[tgt]).abs().max()))
        # ---- check 3 inputs
        a_seed, _ = runner.run_rows([(bs["base"], None, None)], stack.noise(fr["noise_seed"] + 1), B=B)
        row = {"frame": fr["frame_id"], "task_id": fr["task_id"], "split": fr["split"], "t": fr["t"], "phase": fr["phase"],
               "seed_noise_rmse": rmse(A["base"], a_seed[0]), "distractor_visible": _distractor_visible(fr)}
        for c in conds[1:]:
            row[f"rmse_{c}"] = rmse(A["base"], A[c]) if c in A else None
        for f in fams:
            row[f"gap_{f}"] = row.get(f"rmse_{fam_target(f)}")
        rows3.append(row)
        del bs, A
        if fi % 10 == 0 or fi == len(frames) - 1:
            log(f"validated {fi + 1}/{len(frames)} frames ({time.time() - t0:.0f}s)")

    render_ok = all(max(fr["render_diff"].values()) == 0 for fr in frames)
    leftover = max(max(fr["absent_leftover_px"].values()) for fr in frames)
    c1_ok = c1_ok and state_same and tokens_same and leftover == 0
    outs = [r["frac_changed_outside"] for r in c1_rows if "frac_changed_outside" in r]
    res["check1"] = {
        "passed": c1_ok, "max_frac_changed_outside": max(outs) if outs else None,
        "max_frac_changed_outside_rendered": max([r["frac_changed_outside"] for r in c1_rows
                                                  if r.get("cond") in RENDERED and "frac_changed_outside" in r] or [0.0]),
        "robot_state_identical": state_same, "prompt_tokens_identical": tokens_same,
        "env_pixels_equal_direct_render": render_ok, "absent_render_leftover_px": leftover,
        "n_warnings": sum("warning" in r for r in c1_rows), "warnings": [r for r in c1_rows if "warning" in r][:50],
    }
    log(f"check 1 (edits clean): {'PASS' if c1_ok else 'FAIL'} | max outside={res['check1']['max_frac_changed_outside']} "
        f"(rendered {res['check1']['max_frac_changed_outside_rendered']:.4f}) | state identical={state_same} | "
        f"prompt identical={tokens_same} | absent leftover px={leftover} | warnings={res['check1']['n_warnings']}")

    tol = float(CFG.get("determinism_tol", 0.0))
    c2_ok = bool(c2) and max(c2) <= tol
    binv_tol = float(CFG.get("batch_invariance_tol", 1e-5))
    batch_invariant = (not binv) or max(binv) <= binv_tol
    res["check2"] = {"passed": c2_ok, "max_abs_diff_repeat": max(c2) if c2 else None, "frames": sorted(det_ids),
                     "batch_invariant": batch_invariant, "max_abs_diff_batch_composition": max(binv) if binv else None,
                     "patch_batch": B}
    log(f"check 2 (determinism, {len(c2)} frames): {'PASS' if c2_ok else 'FAIL'} | repeat max diff = "
        f"{max(c2) if c2 else float('nan'):.3e} | batch invariance (B={B}): max diff {max(binv) if binv else 0.0:.3e} -> "
        f"{'OK, batched patching enabled' if batch_invariant else 'NOT invariant -> every later step runs 1 row per pass'}")

    c3 = {}
    for f in fams:
        g = np.array([r[f"gap_{f}"] for r in rows3 if r.get(f"gap_{f}") is not None], dtype=float)
        noise_r = np.array([r["seed_noise_rmse"] for r in rows3 if r.get(f"gap_{f}") is not None])
        frac_ok = float((g >= float(CFG["min_gap"])).mean()) if len(g) else 0.0
        F = FAMILIES[f]
        dist_rows = [r for r in rows3 if F["distractor"] and r.get(f"rmse_{F['distractor']}") is not None]
        c3[f] = {"passed": frac_ok >= float(CFG.get("check3_min_frac", 0.9)), "n": int(len(g)),
                 "frac_frames_gap_ge_min": frac_ok, "median_gap": float(np.median(g)) if len(g) else None,
                 "min_gap": float(g.min()) if len(g) else None, "max_gap": float(g.max()) if len(g) else None,
                 "gap_over_seed_noise_median": float(np.median(g / np.maximum(noise_r, 1e-8))) if len(g) else None,
                 "median_rmse_miss": float(np.median([r[f"rmse_{F['miss']}"] for r in rows3 if r.get(f"rmse_{F['miss']}") is not None] or [np.nan])),
                 "distractor": {"n": len(dist_rows),
                                "median_target": float(np.median([r[f"gap_{f}"] for r in dist_rows])) if dist_rows else None,
                                "median_distractor": float(np.median([r[f"rmse_{F['distractor']}"] for r in dist_rows])) if dist_rows else None,
                                "frac_target_larger": float(np.mean([r[f"gap_{f}"] > r[f"rmse_{F['distractor']}"] for r in dist_rows])) if dist_rows else None}}
        log(f"check 3 [{f}]: {'PASS' if c3[f]['passed'] else 'FAIL'} | {100 * frac_ok:.0f}% of {len(g)} frames have gap >= "
            f"{CFG['min_gap']} | median gap {c3[f]['median_gap']} | miss control median {c3[f]['median_rmse_miss']:.3f} | "
            f"target vs distractor occluder {c3[f]['distractor']['median_target']} vs {c3[f]['distractor']['median_distractor']}")
    c3_ok = c3[prim]["passed"]
    med_rmse = {c: float(np.median([r[f"rmse_{c}"] for r in rows3 if r.get(f"rmse_{c}") is not None] or [np.nan])) for c in conds[1:]}
    res["check3"] = {"passed": c3_ok, "primary_family": prim, "families": c3, "median_rmse_by_condition": med_rmse, "rows": rows3}
    log("   median action RMSE vs base: " + " | ".join(f"{c} {v:.3f}" for c, v in med_rmse.items()))

    # ---- check 4: the ceiling works (full layer-output swap base -> occluded), discovery frames only; sets RANK LAYERS
    L = runner.n_layers
    sub = [fr for fr in frames if fr["frame_id"] in set(ceil_ids)]
    prof = {f: {"resid": np.full((len(sub), L), np.nan), "mlp": np.full((len(sub), L), np.nan)} for f in fams}
    for fi, fr in enumerate(sub):
        tconds = [fam_target(f) for f in fams if cond_available(fr, fam_target(f))]
        A, CAP, bs = run_conds(runner, stack, fr, ["base"] + tconds, B, capture_layers=list(range(L)),
                               capture_conds=tconds, capture_kinds=("mlp_out", "resid"))
        valid = runner.last_prefix["pad"]
        noise = stack.noise(fr["noise_seed"])
        jobs = []
        for f in fams:
            tc_ = fam_target(f)
            if tc_ not in CAP:
                continue
            for li in range(L):
                jobs.append((f, "resid", li, (bs["base"], None, {li: rp_swap(CAP[tc_][("resid", li)], valid)})))
                jobs.append((f, "mlp", li, (bs["base"], {li: fp_swap(CAP[tc_][("mlp_out", li)], valid)}, None)))
        for s in range(0, len(jobs), B):
            chunk = jobs[s: s + B]
            a, _ = runner.run_rows([j[3] for j in chunk], noise, B=B)
            for (f, kind, li, _), ai in zip(chunk, a):
                prof[f][kind][fi, li] = closure_pct(ai, A[fam_target(f)], A["base"])
        del CAP, bs
    res["check4"] = {"frames": ceil_ids, "families": {}}
    for f in fams:
        mr, mm = np.nanmean(prof[f]["resid"], 0), np.nanmean(prof[f]["mlp"], 0)
        res["check4"]["families"][f] = {"resid_swap_pct_mean": mr.tolist(), "mlp_swap_pct_mean": mm.tolist(),
                                        "best_layer": int(np.nanargmax(mr)), "best_pct": float(np.nanmax(mr)),
                                        "resid_swap_pct_per_frame": prof[f]["resid"].tolist(),
                                        "mlp_swap_pct_per_frame": prof[f]["mlp"].tolist()}
    P4 = res["check4"]["families"][prim]
    c4_ok = P4["best_pct"] >= float(CFG["ceiling_min_pct"])
    res["check4"]["passed"] = c4_ok
    thr = float(CFG.get("rank_min_resid_pct", 10.0))
    rank_layers = [l for l in range(L - 1) if P4["resid_swap_pct_mean"][l] >= thr]
    rank_rule = f"layers whose full residual swap closes >= {thr}% of the {prim} gap on {len(sub)} discovery frames (last layer excluded)"
    if not rank_layers:
        rank_layers = parse_layers(CFG.get("rank_layers_fallback", "0-13"), L)
        rank_rule = f"fallback {CFG.get('rank_layers_fallback', '0-13')} (no layer met the rule)"
    log(f"check 4 (ceiling, {len(sub)} discovery frames): {'PASS' if c4_ok else 'FAIL'} | best full-layer swap = "
        f"{P4['best_pct']:.1f}% at layer {P4['best_layer']} (need >= {CFG['ceiling_min_pct']}%) | RANK LAYERS {rank_layers}")
    log("   layer : " + " | ".join(f"{f} resid/mlp %" for f in fams))
    for li in range(L):
        log(f"   {li:5d} : " + " | ".join(f"{res['check4']['families'][f]['resid_swap_pct_mean'][li]:6.1f} / "
                                        f"{res['check4']['families'][f]['mlp_swap_pct_mean'][li]:5.1f}" for f in fams))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for f in fams:
        ax[0].plot(range(L), res["check4"]["families"][f]["resid_swap_pct_mean"], marker="o", label=f"{f}: residual swap")
        ax[0].plot(range(L), res["check4"]["families"][f]["mlp_swap_pct_mean"], marker="s", ls="--", label=f"{f}: MLP swap")
    ax[0].axhline(thr, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("LLM layer")
    ax[0].set_ylabel("gap closed (%)")
    ax[0].set_title(f"Single-layer swaps base -> occluded ({len(sub)} discovery frames)", fontsize=9)
    ax[0].legend(fontsize=6)
    for f in fams:
        ax[1].hist([r[f"gap_{f}"] for r in rows3 if r.get(f"gap_{f}") is not None], bins=20, alpha=0.5, label=f)
    ax[1].hist([r["seed_noise_rmse"] for r in rows3], bins=20, alpha=0.4, label="other noise seed", color="gray")
    ax[1].axvline(float(CFG["min_gap"]), color="gray", ls="--", lw=1)
    ax[1].set_xlabel("action RMSE vs base")
    ax[1].set_title(f"Check 3: occlusion gaps over {len(rows3)} frames", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_check4_ceiling.png", dpi=120)
    plt.close(fig)

    passed = c1_ok and c2_ok and c3_ok and c4_ok
    failed = [n for n, ok in (("1", c1_ok), ("2", c2_ok), ("3", c3_ok), ("4", c4_ok)) if not ok]
    save_json(OUT / "validate.json", res)
    write_status("validate", passed, {
        "check1": c1_ok, "check2": c2_ok, "check3": c3_ok, "check4": c4_ok, "batch_invariant": batch_invariant,
        "patch_batch": B, "n_layers": L, "rank_layers": rank_layers, "rank_rule": rank_rule,
        "ceiling_best_pct": P4["best_pct"], "ceiling_best_layer": P4["best_layer"],
        "check3_by_family": {f: {k: v for k, v in c3[f].items()} for f in fams},
        "gap_by_frame": {f: {r["frame"]: r.get(f"gap_{f}") for r in rows3} for f in fams},
        "median_gap": c3[prim]["median_gap"], "frac_frames_gap_ge_min": c3[prim]["frac_frames_gap_ge_min"],
        "n_frames": len(frames),
        "reason": "" if passed else f"check(s) {','.join(failed)} failed",
    })


def rank_layers_from_status(L: int) -> list:
    st = read_status("validate") or {}
    rl = st.get("rank_layers")
    if rl:
        return [int(l) for l in rl if int(l) < L]
    return parse_layers(CFG.get("rank_layers_fallback", "0-13"), L)


# ============================================================================= capture + train


def training_samples(data):
    """Transcoder training samples: DISCOVERY probe frames (all conditions) + extra frames (their conditions)."""
    samples = []
    for fr in data["probe"]:
        if fr.get("split") != "discovery":
            continue
        for c in active_conds():
            if cond_available(fr, c):
                samples.append({"kind": "probe", "frame_id": fr["frame_id"], "cond": c, "noise_seed": fr["noise_seed"], "ref": fr})
    for i, ex in enumerate(data["extra"]):
        for c in ex["conds"]:
            samples.append({"kind": "extra", "frame_id": -1 - i, "cond": c, "noise_seed": int(CFG["seed"]) + 31 * (i + 1), "ref": ex})
    return samples


def cmd_capture():
    setup_torch(CFG["seed"])
    data = load_frames()
    stack = get_stack()
    runner = get_runner()
    L = runner.n_layers
    samples = training_samples(data)
    B = max(1, int(CFG.get("capture_batch", CFG.get("patch_batch", 8))))
    xs = [[] for _ in range(L)]
    ys = [[] for _ in range(L)]
    sid, pos = [], []
    meta_samples, layout = [], None
    keep_frac, start = 1.0, 0
    t0 = time.time()
    # group samples that share a frame (same noise seed, same prompt) into one forward pass
    groups, cur = [], []
    for s in samples:
        if cur and (len(cur) >= B or cur[0]["ref"] is not s["ref"]):
            groups.append(cur)
            cur = []
        cur.append(s)
    if cur:
        groups.append(cur)
    si = 0
    for gi, grp in enumerate(groups):
        rows = [(stack.batch({"pixels": s["ref"]["conds"][s["cond"]], "robot_state": s["ref"]["robot_state"]}, s["ref"]["task"]),
                 None, None) for s in grp]
        _, cap = runner.run_rows(rows, stack.noise(grp[0]["noise_seed"]), capture_layers=list(range(L)),
                                 capture_rows=list(range(len(grp))), capture_kinds=("mlp_in", "mlp_out"))
        if layout is None:
            layout = runner.layout(rows[0][0])
        pad_rows = runner.last_prefix["pad_rows"]
        if gi == 0:
            n_valid = int(pad_rows[0].sum())
            per_tok = L * (cap[("mlp_in", 0)].shape[-1] + cap[("mlp_out", 0)].shape[-1]) * 2
            est = len(samples) * n_valid * per_tok
            avail = _avail_ram()
            import shutil as _sh

            disk_free = _sh.disk_usage(OUT).free
            budget = float(CFG.get("capture_max_ram_frac", 0.5)) * avail if avail else est
            budget = min(budget, float(CFG.get("max_train_tokens", 4e5)) * per_tok, float(CFG.get("capture_max_disk_frac", 0.4)) * disk_free)
            keep_frac = float(np.clip(budget / max(est, 1), 0.02, 1.0))
            log(f"capture: {len(samples)} samples ({sum(s['kind'] == 'probe' for s in samples)} discovery-probe, "
                f"{sum(s['kind'] == 'extra' for s in samples)} extra), {n_valid} valid tokens each -> ~{est / 2 ** 30:.1f} GB at full "
                f"density (available RAM {(avail or 0) / 2 ** 30:.1f} GB, free disk {disk_free / 2 ** 30:.0f} GB) -> token keep fraction "
                f"{keep_frac:.2f} (~{keep_frac * est / 2 ** 30:.1f} GB)")
        for r, s in enumerate(grp):
            valid = pad_rows[r].nonzero().flatten()
            if keep_frac < 1.0:
                g = torch.Generator().manual_seed(int(CFG["seed"]) + si)
                valid = valid[(torch.rand(len(valid), generator=g) < keep_frac).to(valid.device)]
            for li in range(L):
                xs[li].append(cap[("mlp_in", li)][r, valid].to(torch.bfloat16).cpu())
                ys[li].append(cap[("mlp_out", li)][r, valid].to(torch.bfloat16).cpu())
            sid.append(torch.full((len(valid),), si, dtype=torch.int32))
            pos.append(valid.cpu().to(torch.int32))
            meta_samples.append({"i": si, "kind": s["kind"], "frame_id": s["frame_id"], "cond": s["cond"], "start": start,
                                 "n_tokens": int(len(valid)), "task_id": s["ref"].get("task_id"), "split": s["ref"].get("split")})
            start += int(len(valid))
            si += 1
        del cap
        if gi % 25 == 0 or gi == len(groups) - 1:
            log(f"captured {si}/{len(samples)} samples ({time.time() - t0:.0f}s)")
    acts = OUT / "acts"
    acts.mkdir(parents=True, exist_ok=True)
    sid = torch.cat(sid)
    pos = torch.cat(pos)
    for li in range(L):
        x, y = torch.cat(xs[li]), torch.cat(ys[li])
        xs[li], ys[li] = None, None
        torch.save({"x": x, "y": y, "sample": sid, "pos": pos}, acts / f"layer_{li:02d}.pt")
        del x, y
    gc.collect()
    save_json(acts / "meta.json", {"samples": meta_samples, "layout": layout, "n_layers": L, "n_tokens": int(len(sid)),
                                   "token_keep_frac": keep_frac, "splits": ["discovery", "extra"]})
    n_tasks = len({m["task_id"] for m in meta_samples})
    log(f"saved activations for {L} layers, {len(samples)} samples from {n_tasks} tasks, {len(sid)} tokens -> {acts}")
    write_status("capture", True, {"n_layers": L, "n_samples": len(samples), "n_tokens": int(len(sid)), "layout": layout,
                                   "n_tasks": n_tasks, "token_keep_frac": keep_frac,
                                   "conds_by_kind": {k: sorted({m["cond"] for m in meta_samples if m["kind"] == k}) for k in ("probe", "extra")},
                                   "note": "discovery + extra (non-test) demonstrations only; test frames never touch training"})


def cmd_train():
    setup_torch(CFG["seed"])
    if not (OUT / "acts" / "meta.json").exists():
        raise SystemExit("[gate] activation cache (acts/) is missing, probably after a runtime reset. Re-run capture first.")
    meta = load_json(OUT / "acts" / "meta.json")
    L = meta["n_layers"]
    layers = list(range(L)) if CFG["tc_layers"] == "all" else parse_layers(CFG["tc_layers"], L)
    (OUT / "transcoders").mkdir(parents=True, exist_ok=True)
    metrics = {}
    retry = float(CFG.get("tc_retry_mult", 1.0))
    for li in layers:
        t0 = time.time()
        d = torch.load(OUT / "acts" / f"layer_{li:02d}.pt", weights_only=False)
        tc, m = train_transcoder(d["x"], d["y"], seed=int(CFG["seed"]) + li)
        if m["fvu_heldout"] > float(CFG["tc_max_fvu"]) and retry > 1.0:
            steps2 = int(int(CFG["tc_steps"]) * retry)
            tc2, m2 = train_transcoder(d["x"], d["y"], seed=int(CFG["seed"]) + li, steps=steps2)
            log(f"layer {li:2d}: FVU {m['fvu_heldout']:.4f} > {CFG['tc_max_fvu']} -> retrained with {steps2} steps: FVU {m2['fvu_heldout']:.4f}")
            if m2["fvu_heldout"] < m["fvu_heldout"]:
                tc, m = tc2, m2
        m["seconds"] = time.time() - t0
        metrics[li] = m
        torch.save({"state_dict": tc.state_dict(), "config": {"d_in": tc.d_in, "d_out": tc.d_out, "n_feat": tc.n_feat, "k": tc.k},
                    "layer": li, "metrics": m}, OUT / "transcoders" / f"layer_{li:02d}.pt")
        log(f"layer {li:2d}: held-out FVU={m['fvu_heldout']:.4f} alive={m['alive_features']}/{tc.n_feat} "
            f"tokens={m['n_train_tokens']} steps={m['steps']} ({m['seconds']:.0f}s)")
        del d, tc
        free_cuda()
    _CACHE.pop("tcs", None)

    # ---- check 5: faithful? (a) held-out FVU, (b) FVU and splice on held-out TEST frames (never seen in training)
    stack = get_stack()
    runner = get_runner()
    tcs = load_all_tcs()
    prim_t = fam_target(primary_family())
    test = probe_frames("test") or probe_frames()
    ids = set(spread_subset(test, int(CFG.get("splice_frames", 10))))
    sub = [fr for fr in test if fr["frame_id"] in ids]
    B = patch_batch()
    acc = {li: {"sse": 0.0, "sum": None, "sq": 0.0, "n": 0} for li in layers}
    splice = {li: [] for li in layers}
    for fr in sub:
        A, CAP, bs = run_conds(runner, stack, fr, ["base", prim_t], B, capture_layers=layers,
                               capture_conds=["base", prim_t], capture_kinds=("mlp_in", "mlp_out"))
        valid = runner.last_prefix["pad"]
        noise = stack.noise(fr["noise_seed"])
        if prim_t not in A:
            continue
        gap = max(rmse(A["base"], A[prim_t]), 1e-8)
        with torch.no_grad():
            for c in ("base", prim_t):
                for li in layers:
                    x = CAP[c][("mlp_in", li)][valid].float()
                    y = CAP[c][("mlp_out", li)][valid].double()
                    yh = tcs[li].decode(tcs[li].encode(x)).double()
                    a = acc[li]
                    a["sse"] += float(((yh - y) ** 2).sum())
                    a["sum"] = y.sum(0) if a["sum"] is None else a["sum"] + y.sum(0)
                    a["sq"] += float((y ** 2).sum())
                    a["n"] += int(y.shape[0])
        del CAP
        jobs = [(li, (bs["base"], {li: fp_splice(tcs[li], valid)}, None)) for li in layers]
        for s in range(0, len(jobs), B):
            chunk = jobs[s: s + B]
            a, _ = runner.run_rows([j[1] for j in chunk], noise, B=B)
            for (li, _), ai in zip(chunk, a):
                splice[li].append(rmse(ai, A["base"]) / gap)
    fvu_test = {}
    for li, a in acc.items():
        if a["n"]:
            sst = a["sq"] - float((a["sum"] ** 2).sum()) / a["n"]
            fvu_test[li] = a["sse"] / max(sst, 1e-12)
        else:
            fvu_test[li] = float("nan")
    fvus = [metrics[li]["fvu_heldout"] for li in layers]
    med = float(np.median(fvus))
    med_test = float(np.nanmedian([fvu_test[li] for li in layers]))
    c5_ok = med <= float(CFG["tc_max_fvu"])
    rows = [{"layer": li, **{k: v for k, v in metrics[li].items() if k != "history"},
             "fvu_test_frames": fvu_test[li], "splice_rmse_over_gap": float(np.mean(splice[li])) if splice[li] else float("nan")}
            for li in layers]
    save_json(OUT / "train.json", {"rows": rows, "histories": {li: metrics[li]["history"] for li in layers},
                                   "test_frames_for_splice": sorted(ids)})
    log(f"check 5 (transcoder faithful): {'PASS' if c5_ok else 'FAIL'} | median held-out FVU={med:.4f} (need <= {CFG['tc_max_fvu']}) "
        f"| median FVU on {len(sub)} held-out TEST frames={med_test:.4f}")
    log("   layer | FVU(held-out) | FVU(test frames) | splice RMSE / occlusion gap (test frames)")
    for r in rows:
        log(f"   {r['layer']:5d} | {r['fvu_heldout']:12.4f} | {r['fvu_test_frames']:15.4f} | {r['splice_rmse_over_gap']:8.3f}")
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    w = 0.4
    ax[0].bar(np.array(layers) - w / 2, fvus, w, label="held-out tokens (training distribution)")
    ax[0].bar(np.array(layers) + w / 2, [fvu_test[li] for li in layers], w, label="held-out test frames")
    ax[0].axhline(float(CFG["tc_max_fvu"]), color="gray", ls="--", lw=1)
    ax[0].set_title("FVU per layer")
    ax[0].set_xlabel("layer")
    ax[0].legend(fontsize=7)
    ax[1].bar(layers, [r["splice_rmse_over_gap"] for r in rows])
    ax[1].axhline(1.0, color="gray", ls="--", lw=1)
    ax[1].set_title("Splice error / occlusion gap on test frames (lower is better)", fontsize=9)
    ax[1].set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(OUT / "fig_check5_transcoders.png", dpi=120)
    plt.close(fig)
    acts_deleted = False
    if not CFG.get("keep_acts", False) and (OUT / "acts").exists():
        import shutil as _sh

        gb = sum(p.stat().st_size for p in (OUT / "acts").glob("*") if p.is_file()) / 2 ** 30
        _sh.rmtree(OUT / "acts", ignore_errors=True)
        acts_deleted = True
        log(f"deleted the activation cache (acts/, {gb:.1f} GB): only training needs it, and capture re-creates it if train is re-run")
    write_status("train", c5_ok, {"check5": c5_ok, "median_fvu": med, "activation_cache_deleted": acts_deleted, "median_fvu_test_frames": med_test, "fvu_per_layer": fvus,
                                  "fvu_test_frames_per_layer": [fvu_test[li] for li in layers],
                                  "splice_over_gap": [r["splice_rmse_over_gap"] for r in rows],
                                  "tc_features": int(CFG["tc_features"]), "tc_k": int(CFG["tc_k"]),
                                  "n_train_tokens": int(np.median([metrics[li]["n_train_tokens"] for li in layers])),
                                  "reason": "" if c5_ok else f"median held-out FVU {med:.3f} > {CFG['tc_max_fvu']}"})


# ============================================================================= Goal 1: selective features


def _family_stats(sums, avail, cidx, fam, frames_sel, typical, alive, min_pos, min_spec, min_rise):
    """sums (F, C, L, n) summed activations; returns per-feature stats of the v2 selectivity rule for one family."""
    F_ = FAMILIES[fam]
    T, base = cidx[F_["target"]], cidx["base"]
    quiet = [cidx[c] for c in F_["quiet"] if c in cidx]
    rows = [i for i in frames_sel if avail[i, T] and all(avail[i, q] for q in quiet)]
    if not rows:
        return None
    d_t = sums[rows, T] - sums[rows, base]                              # (F, L, n)
    t_mean = d_t.mean(0)
    pos_frac = (d_t > 0).mean(0)
    other = np.max(np.stack([np.abs(sums[rows, q] - sums[rows, base]).mean(0) for q in quiet]), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where(t_mean > 0, 1.0 - other / np.maximum(t_mean, 1e-12), -np.inf)
    rise = t_mean / np.maximum(typical, 1e-8)
    passed = (t_mean > 0) & (pos_frac >= min_pos) & (spec >= min_spec) & (rise >= min_rise) & alive
    dist_ratio = np.full(t_mean.shape, np.nan)
    pass_t = np.zeros_like(passed)
    spec_t = np.full(t_mean.shape, np.nan)
    if F_["distractor"] and F_["distractor"] in cidx:
        Dc = cidx[F_["distractor"]]
        drows = [i for i in rows if avail[i, Dc]]
        if drows:
            dist_abs = np.abs(sums[drows, Dc] - sums[drows, base]).mean(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                spec_t = np.where(t_mean > 0, 1.0 - np.maximum(other, dist_abs) / np.maximum(t_mean, 1e-12), -np.inf)
                dist_ratio = np.where(t_mean > 0, dist_abs / np.maximum(t_mean, 1e-12), np.nan)
            pass_t = passed & (spec_t >= min_spec)
    score = np.where(passed, np.where(passed, spec, 0.0) * rise, 0.0)
    return {"t_mean": t_mean, "pos_frac": pos_frac, "other": other, "spec": spec, "rise": rise, "pass": passed,
            "pass_t": pass_t, "spec_t": spec_t, "dist_ratio": dist_ratio, "score": score, "n_frames": len(rows)}


def _select(stats, all_L, cap_n):
    out = {}
    for k, l in enumerate(all_L):
        sc = stats["score"][k]
        order = [int(j) for j in np.argsort(-sc, kind="stable") if stats["pass"][k, j]]
        out[l] = {"occ_all": order, "occ": order[:cap_n],
                  "occ_target": [j for j in order if stats["pass_t"][k, j]][:cap_n]}
    return out


def cmd_goal1():
    require_gate("validate", "Goal 1")
    if CFG.get("check5_blocks", False):
        require_gate("train", "Goal 1")
    setup_torch(CFG["seed"])
    tcs = load_all_tcs()
    all_L = sorted(tcs)
    n = tcs[all_L[0]].n_feat
    stack, runner = get_stack(), get_runner()
    disc = probe_frames("discovery")
    conds = active_conds()
    cidx = {c: i for i, c in enumerate(conds)}
    fams = families()
    prim = primary_family()
    Bg = max(1, int(CFG.get("goal1_batch", 12)))
    NF, C, L = len(disc), len(conds), len(all_L)
    sums = np.zeros((NF, C, L, n), dtype=np.float32)
    avail = np.zeros((NF, C), dtype=bool)
    fire_cnt = torch.zeros(L, n, dtype=torch.float64, device=DEV)
    fire_sum = torch.zeros(L, n, dtype=torch.float64, device=DEV)
    edit = {f: torch.zeros(L, n, dtype=torch.float64, device=DEV) for f in fams}
    t0 = time.time()
    for fi, fr in enumerate(disc):
        cs = [c for c in conds if cond_available(fr, c)]
        noise = stack.noise(fr["noise_seed"])
        keep = {}
        for s in range(0, len(cs), Bg):
            chunk = cs[s: s + Bg]
            rows = [(stack.batch(fmt_obs(fr, c), fr["task"]), None, None) for c in chunk]
            _, cap = runner.run_rows(rows, noise, capture_layers=all_L, capture_rows=list(range(len(chunk))),
                                     capture_kinds=("mlp_in",))
            pad_rows = runner.last_prefix["pad_rows"]
            with torch.no_grad():
                for r, c in enumerate(chunk):
                    valid = pad_rows[r]
                    avail[fi, cidx[c]] = True
                    for k, l in enumerate(all_L):
                        Fm = tcs[l].encode(cap[("mlp_in", l)][r][valid])
                        sums[fi, cidx[c], k] = Fm.sum(0).float().cpu().numpy()
                        fire_cnt[k] += (Fm > 0).sum(0).double()
                        fire_sum[k] += Fm.sum(0).double()
                        if c == "base" or c in [fam_target(f) for f in fams]:
                            keep[(c, k)] = Fm
            del cap
        with torch.no_grad():
            for f in fams:
                t = fam_target(f)
                if ("base", 0) in keep and (t, 0) in keep:
                    for k, l in enumerate(all_L):
                        edit[f][k] += (keep[(t, k)] - keep[("base", k)]).abs().sum(0).double() * float(tcs[l].y_scale)
        del keep
        if fi % 10 == 0 or fi == NF - 1:
            log(f"goal1: encoded {fi + 1}/{NF} discovery frames x {len(cs)} conditions ({time.time() - t0:.0f}s)")
    typical = (fire_sum / fire_cnt.clamp_min(1)).cpu().numpy()
    alive = (fire_cnt > 0).cpu().numpy()
    min_pos, min_spec, min_rise = float(CFG["min_pos_frac"]), float(CFG["min_specificity"]), float(CFG["min_rise_tokens"])
    cap_n = int(CFG["max_features_per_layer"])
    all_frames = list(range(NF))
    folds = {str(p): [i for i, fr in enumerate(disc) if int(fr["task_id"]) % 2 == p] for p in (0, 1)}
    stats, sel, rank = {}, {}, {"layers": all_L, "typical": torch.as_tensor(typical, dtype=torch.float32),
                                "alive": torch.as_tensor(alive), "y_scale": [float(tcs[l].y_scale) for l in all_L], "fam": {}}
    for f in fams:
        st = _family_stats(sums, avail, cidx, f, all_frames, typical, alive, min_pos, min_spec, min_rise)
        if st is None:
            log(f"goal1 [{f}]: no discovery frame has every condition of this family; skipped")
            continue
        stats[f] = st
        chosen = _select(st, all_L, cap_n)
        fold_sel = {}
        for p, idx in folds.items():
            stf = _family_stats(sums, avail, cidx, f, idx, typical, alive, min_pos, min_spec, min_rise) if idx else None
            fold_sel[p] = {str(l): v["occ"] for l, v in _select(stf, all_L, cap_n).items()} if stf is not None else {}
        layer_score = {l: float(np.sort(st["score"][k])[::-1][:cap_n].sum()) for k, l in enumerate(all_L)}
        top = [l for l in sorted(all_L, key=lambda l: -layer_score[l]) if chosen[l]["occ"]][: int(CFG["top_layers"])]
        sel[f] = {"layers": {str(l): {**chosen[l], "n_pass": int(st["pass"][k].sum()), "n_pass_target": int(st["pass_t"][k].sum()),
                                      "score_sum": layer_score[l],
                                      "occ_stats": [{"feature": j, "score": float(st["score"][k, j]), "specificity": float(st["spec"][k, j]),
                                                     "rise_tokens": float(st["rise"][k, j]), "d_target": float(st["t_mean"][k, j]),
                                                     "frames_positive": float(st["pos_frac"][k, j]),
                                                     "distractor_over_target": float(st["dist_ratio"][k, j]) if np.isfinite(st["dist_ratio"][k, j]) else None}
                                                    for j in chosen[l]["occ_all"]]}
                            for k, l in enumerate(all_L)},
                  "top_layers": top, "folds": fold_sel, "n_frames": st["n_frames"],
                  "n_pass_total": int(st["pass"].sum()), "n_pass_target_total": int(st["pass_t"].sum()),
                  "n_selected_capped": int(sum(len(chosen[l]["occ"]) for l in all_L))}
        rank["fam"][f] = {"edit": (edit[f] / max(1, NF)).float().cpu(), "d_target": torch.as_tensor(st["t_mean"], dtype=torch.float32),
                          "score": torch.as_tensor(st["score"], dtype=torch.float32),
                          "spec": torch.as_tensor(np.nan_to_num(st["spec"], neginf=-10.0), dtype=torch.float32),
                          "rise": torch.as_tensor(st["rise"], dtype=torch.float32), "pass": torch.as_tensor(st["pass"]),
                          "pass_t": torch.as_tensor(st["pass_t"])}
        log(f"goal1 [{f}]: {sel[f]['n_pass_total']} passing features ({sel[f]['n_pass_target_total']} also target-specific) "
            f"of {n * L} | top layers {top} | capped set {sel[f]['n_selected_capped']} | fold sets "
            f"{[sum(len(v) for v in fold_sel[p].values()) for p in fold_sel]}")
    # colour features (primary family's quiet set; not passing the occlusion rule)
    color = {}
    if prim in stats:
        base_i, R = cidx["base"], cidx["recolor"]
        qc = [cidx[c] for c in quiet_for_color() if c in cidx]
        rows = [i for i in all_frames if avail[i, R] and all(avail[i, q] for q in qc)]
        d_c = sums[rows, R] - sums[rows, base_i]
        col_mean = d_c.mean(0)
        other = np.max(np.stack([np.abs(sums[rows, q] - sums[rows, base_i]).mean(0) for q in qc]), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            col_spec = np.where(col_mean > 0, 1.0 - other / np.maximum(col_mean, 1e-12), -np.inf)
        col_score = np.where((col_mean > 0) & alive, np.clip(col_spec, -10, 1) * col_mean / np.maximum(typical, 1e-8), -np.inf)
        for k, l in enumerate(all_L):
            nsel = max(1, len(sel[prim]["layers"][str(l)]["occ"]))
            order = [int(j) for j in np.argsort(-col_score[k], kind="stable")
                     if np.isfinite(col_score[k, j]) and not stats[prim]["pass"][k, j]]
            color[str(l)] = order[:nsel]
    rank_layers = rank_layers_from_status(max(all_L) + 1)
    save_json(OUT / "goal1_selected.json", {"families": sel, "color": color, "discovery_frames": [fr["frame_id"] for fr in disc],
                                            "rank_layers": rank_layers, "n_features_per_layer": n, "layers": all_L,
                                            "alive_per_layer": {str(l): int(alive[k].sum()) for k, l in enumerate(all_L)},
                                            "criteria": {"min_pos_frac": min_pos, "min_specificity": min_spec,
                                                         "min_rise_tokens": min_rise, "cap_per_layer": cap_n}})
    torch.save(rank, OUT / "goal1_rank.pt")
    for f, st in stats.items():
        with open(OUT / f"goal1_features_{f}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["layer", "feature", "pass", "pass_target_specific", "score", "specificity", "rise_tokens", "d_target",
                        "frames_positive", "distractor_over_target", "typical_act"])
            for k, l in enumerate(all_L):
                for j in np.nonzero(st["pass"][k] | (st["rise"][k] >= min_rise))[0][:2000]:
                    w.writerow([l, int(j), bool(st["pass"][k, j]), bool(st["pass_t"][k, j]), float(st["score"][k, j]),
                                float(st["spec"][k, j]), float(st["rise"][k, j]), float(st["t_mean"][k, j]),
                                float(st["pos_frac"][k, j]), float(st["dist_ratio"][k, j]), float(typical[k, j])])
    _goal1_figures(stats, sel, all_L, conds, cidx, sums, avail, typical, disc, tcs, stack, runner)
    ok = prim in sel and sel[prim]["n_pass_total"] > 0
    write_status("goal1", ok, {
        "families": {f: {"n_pass_total": s["n_pass_total"], "n_pass_target_total": s["n_pass_target_total"],
                         "top_layers": s["top_layers"], "n_selected_capped": s["n_selected_capped"],
                         "n_pass_per_layer": {l: s["layers"][str(l)]["n_pass"] for l in all_L},
                         "pass_frac_pct": 100.0 * s["n_pass_total"] / max(1, n * L)} for f, s in sel.items()},
        "n_features_total": n * L, "n_alive_total": int(alive.sum()), "n_discovery_frames": NF, "rank_layers": rank_layers,
        "n_color": sum(len(v) for v in color.values()),
        "reason": "" if ok else "no selective feature for the primary family (Goal 2 still runs: attribution sets do not depend on Goal 1)",
    })


def _goal1_figures(stats, sel, all_L, conds, cidx, sums, avail, typical, disc, tcs, stack, runner):
    plt = _plt()
    fams = list(stats)
    fig, ax = plt.subplots(figsize=(9, 3))
    w = 0.8 / max(1, len(fams))
    for i, f in enumerate(fams):
        ax.bar(np.array(all_L) + (i - (len(fams) - 1) / 2) * w, [sel[f]["layers"][str(l)]["n_pass"] for l in all_L], w, label=f)
    ax.set_xlabel("LLM layer")
    ax.set_ylabel("# selective features")
    ax.set_title(f"Goal 1: occlusion-selective features per layer ({len(disc)} discovery frames)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal1_counts.png", dpi=120)
    plt.close(fig)
    prim = primary_family()
    if prim not in stats:
        return
    st = stats[prim]
    k_of = {l: k for k, l in enumerate(all_L)}
    cand = [(st["score"][k, j], l, j) for k, l in enumerate(all_L) for j in np.nonzero(st["pass"][k])[0]]
    cand = sorted(cand, reverse=True)[:24]
    if cand:
        rows = [i for i in range(len(disc)) if all(avail[i, cidx[c]] for c in conds)] or list(range(len(disc)))
        M = np.array([[(sums[rows, cidx[c], k_of[l], j] - sums[rows, cidx["base"], k_of[l], j]).mean() / max(typical[k_of[l], j], 1e-8)
                       for c in conds[1:]] for _, l, j in cand])
        fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(cand) + 1.5))
        vmax = np.abs(M).max() or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(conds) - 1))
        ax.set_xticklabels(conds[1:], rotation=35, fontsize=7, ha="right")
        ax.set_yticks(range(len(cand)))
        ax.set_yticklabels([f"L{l} f{j}" for _, l, j in cand], fontsize=7)
        fig.colorbar(im, ax=ax, label="change vs base (token-firings)")
        ax.set_title(f"Goal 1 [{prim}]: top selective features, response per condition (discovery frames)", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig_goal1_features.png", dpi=120)
        plt.close(fig)
    top = sel[prim]["top_layers"]
    if not top or not disc:
        return
    l = top[0]
    feats = sel[prim]["layers"][str(l)]["occ"][:4]
    fr0 = disc[len(disc) // 2]
    cs = [c for c in conds if cond_available(fr0, c)]
    noise = stack.noise(fr0["noise_seed"])
    maps = {}
    Bg = max(1, int(CFG.get("goal1_batch", 12)))
    lay = None
    for s in range(0, len(cs), Bg):
        chunk = cs[s: s + Bg]
        rows = [(stack.batch(fmt_obs(fr0, c), fr0["task"]), None, None) for c in chunk]
        _, cap = runner.run_rows(rows, noise, capture_layers=[l], capture_rows=list(range(len(chunk))), capture_kinds=("mlp_in",))
        lay = runner.layout(rows[0][0])
        cam_i = lay["img_suffix"].index("image") if "image" in lay["img_suffix"] else 0
        lo, hi = cam_i * lay["per_img"], (cam_i + 1) * lay["per_img"]
        with torch.no_grad():
            for r, c in enumerate(chunk):
                Fm = tcs[l].encode(cap[("mlp_in", l)][r])[lo:hi][:, feats].t().float().cpu().numpy()
                maps[c] = Fm.reshape(len(feats), lay["grid"], lay["grid"])
    fig, axes = plt.subplots(len(feats), len(cs), figsize=(1.9 * len(cs), 2.0 * len(feats)), squeeze=False)
    for r in range(len(feats)):
        vmax = max(float(maps[c][r].max()) for c in cs) or 1.0
        for ci, c in enumerate(cs):
            ax = axes[r][ci]
            ax.imshow(model_view(fr0["conds"][c]["image"]))
            H = fr0["conds"][c]["image"].shape[0]
            mp = np.ma.masked_where(maps[c][r] <= 0, maps[c][r])
            ax.imshow(mp, cmap="autumn", alpha=0.6, vmin=0, vmax=vmax, extent=(0, H, H, 0), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(c.replace("_", "\n"), fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"L{l} f{feats[r]}", fontsize=8)
    fig.suptitle(f"Goal 1 contact sheet [{prim}]: selective-feature activation over agentview tokens", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal1_contact_sheet.png", dpi=90)
    plt.close(fig)


# ============================================================================= attribution


def cmd_attrib():
    require_gate("validate", "attribution")
    setup_torch(CFG["seed"])
    tcs = load_all_tcs()
    all_L = sorted(tcs)
    n = tcs[all_L[0]].n_feat
    L = len(all_L)
    stack, runner = get_stack(), get_runner()
    frames = probe_frames()
    fams = families()
    Ba = max(1, int(CFG.get("attr_batch", 3)))
    NF = len(frames)
    inj = {f: np.full((NF, L, n), np.nan, dtype=np.float32) for f in fams}
    rem = {f: np.full((NF, L, n), np.nan, dtype=np.float32) for f in fams}
    gaps = {f: np.full(NF, np.nan) for f in fams}
    fwd_check = []
    (OUT / "attrib").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = None
    for fi, fr in enumerate(frames):
        tconds = {f: fam_target(f) for f in fams if cond_available(fr, fam_target(f))}
        A, CAP, bs = run_conds(runner, stack, fr, ["base"] + sorted(set(tconds.values())), 1 + len(set(tconds.values())),
                               capture_layers=all_L, capture_conds=["base"] + sorted(set(tconds.values())),
                               capture_kinds=("mlp_in",))
        valid = runner.last_prefix["pad"]
        noise = stack.noise(fr["noise_seed"])
        with torch.no_grad():
            Fe = {c: {l: tcs[l].encode(CAP[c][("mlp_in", l)]) for l in all_L} for c in CAP}
        del CAP
        jobs = []
        for f, t in tconds.items():
            d = (A[t] - A["base"]).float()
            n2 = float((d * d).sum())
            gaps[f][fi] = rmse(A["base"], A[t])
            if n2 <= 0:
                continue
            jobs.append((f, "inject", bs["base"], d / n2, "base", t))
            jobs.append((f, "remove", bs[t], -d / n2, t, "base"))
        try:
            s = 0
            bsz = Ba
            while s < len(jobs):
                chunk = jobs[s: s + bsz]
                try:
                    a, grads = runner.run_grad([j[2] for j in chunk], noise, torch.stack([j[3] for j in chunk]), all_L)
                except torch.cuda.OutOfMemoryError:
                    free_cuda()
                    if bsz == 1:
                        raise
                    bsz = 1
                    log("attrib: out of GPU memory -> one row per backward pass")
                    continue
                with torch.no_grad():
                    for r, (f, direction, _, _, src, dst) in enumerate(chunk):
                        if direction == "inject" and r == 0:
                            fwd_check.append(float((a[r] - A["base"]).abs().max()))
                        tgt_arr = inj if direction == "inject" else rem
                        for k, l in enumerate(all_L):
                            df = Fe[dst][l] - Fe[src][l]
                            tgt_arr[f][fi, k] = feature_attr(tcs[l], grads[l][r], df, valid).cpu().numpy()
                del grads
                s += len(chunk)
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
            log(f"attrib FAILED on frame {fr['frame_id']}: {failed}")
            break
        del Fe
        free_cuda()
        if fi % 10 == 0 or fi == NF - 1:
            log(f"attrib: {fi + 1}/{NF} frames ({time.time() - t0:.0f}s) | forward consistency max diff "
                f"{max(fwd_check) if fwd_check else float('nan'):.2e}")
    if failed:
        write_status("attrib", False, {"reason": f"gradient attribution failed ({failed}); Goal 2 runs without attribution sets"})
        return
    split = np.array([fr["split"] for fr in frames])
    tids = np.array([int(fr["task_id"]) for fr in frames])
    fids = np.array([int(fr["frame_id"]) for fr in frames])
    disc = split == "discovery"
    summary = {}
    for f in fams:
        np.savez_compressed(OUT / "attrib" / f"attr_{f}.npz", frame_ids=fids, split=split, task_ids=tids,
                            inject=inj[f], remove=rem[f], gap=gaps[f], layers=np.array(all_L))
        sym = (inj[f] + rem[f]) / 2.0
        rk = {"inject": np.nanmean(inj[f][disc], 0), "remove": np.nanmean(rem[f][disc], 0), "sym": np.nanmean(sym[disc], 0)}
        for p in (0, 1):
            m = disc & (tids % 2 == p)
            rk[f"sym_fold{p}"] = np.nanmean(sym[m], 0) if m.any() else np.zeros((L, n), dtype=np.float32)
        np.savez_compressed(OUT / "attrib" / f"rank_{f}.npz", layers=np.array(all_L), **rk)
        v = np.nan_to_num(rk["sym"]).flatten()
        pos = np.sort(v[v > 0])[::-1]
        cs = np.cumsum(pos) / max(pos.sum(), 1e-12) if len(pos) else np.array([1.0])
        top = np.argsort(-v)[:100]
        summary[f] = {"n_frames": int(np.isfinite(inj[f][:, 0, 0]).sum()),
                      "predicted_total_all_features_pct": float(100 * np.nanmean(np.nansum(inj[f][disc], axis=(1, 2)))),
                      "predicted_top_k_pct": {K: float(100 * np.sort(v)[::-1][:K].sum()) for K in (10, 30, 100, 300, 1000, 3000)},
                      "n_features_for_50pct_positive_mass": int(np.searchsorted(cs, 0.5) + 1),
                      "n_features_for_80pct_positive_mass": int(np.searchsorted(cs, 0.8) + 1),
                      "top100_layer_histogram": {int(all_L[i // n]): int(c) for i, c in
                                                 zip(*np.unique(top // n * n, return_counts=True))},
                      "positive_mass_by_layer_pct": [float(100 * np.clip(rk["sym"][k], 0, None).sum() / max(pos.sum(), 1e-12)) for k in range(L)]}
        log(f"attrib [{f}]: predicted closure of the top-K discovery features: "
            + ", ".join(f"K={K}: {x:.1f}%" for K, x in summary[f]["predicted_top_k_pct"].items())
            + f" | all features {summary[f]['predicted_total_all_features_pct']:.1f}% | 50% of positive mass in "
              f"{summary[f]['n_features_for_50pct_positive_mass']} features")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3))
    for f in fams:
        ax.plot(all_L, summary[f]["positive_mass_by_layer_pct"], marker="o", label=f)
    ax.set_xlabel("LLM layer")
    ax.set_ylabel("% of positive attribution")
    ax.set_title("Where the attribution mass sits (discovery frames, symmetric attribution)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_attrib_layers.png", dpi=120)
    plt.close(fig)
    write_status("attrib", True, {"families": summary, "n_frames": NF, "n_discovery": int(disc.sum()),
                                  "forward_consistency_max_abs_diff": max(fwd_check) if fwd_check else None,
                                  "method": "attribution patching: gradient of the linear gap-projection metric w.r.t. every MLP output, "
                                            "contracted with the base->occluded (inject) or occluded->base (remove) feature change; "
                                            "symmetric = mean of both directions"})


# ============================================================================= Goal 2: held-out causal tests


def _sets_from_flat(flat, all_L, n):
    out = {}
    for i in np.asarray(flat, dtype=np.int64).tolist():
        out.setdefault(all_L[i // n], []).append(int(i % n))
    return out


def _top_flat(score, K, allowed):
    s = np.where(allowed, np.nan_to_num(np.asarray(score, dtype=np.float64), nan=-np.inf), -np.inf).flatten()
    order = np.argsort(-s, kind="stable")
    order = order[np.isfinite(s[order])]
    return order[: int(K)]


def _nfeat(sets):
    return int(sum(len(v) for v in sets.values()))


def _flat_of(sets, all_L, n):
    return {all_L.index(l) * n + j for l, fs in sets.items() for j in fs}


class G2Stats:
    """Frame-level aggregation with task-clustered bootstrap CIs and sign-flip tests."""

    def __init__(self, rows, nb, seed, n_mc):
        self.rows, self.nb, self.seed, self.n_mc = rows, nb, seed, n_mc
        self.index = {}
        for r in rows:
            v = r.get("closed_pct")
            if v is None or not np.isfinite(v):
                continue
            key = (r["family"], r["split"], r["method"], r["scope"], r["direction"])
            self.index.setdefault(key, {}).setdefault(r["frame"], []).append(r)

    def vals(self, fam, method, split="test", scope="all", direction="inject", metric="closed_pct"):
        by = self.index.get((fam, split, method, scope, direction), {})
        fids = sorted(by)
        v = np.array([np.nanmean([x[metric] if x.get(metric) is not None else np.nan for x in by[f]]) for f in fids], dtype=float)
        tasks = [by[f][0]["task_id"] for f in fids]
        return fids, v, tasks

    def summ(self, fam, method, split="test", scope="all", direction="inject", metric="closed_pct"):
        f, v, t = self.vals(fam, method, split, scope, direction, metric)
        ok = np.isfinite(v)
        if not ok.any():
            return None
        m, (lo, hi) = cluster_bootstrap(v, t, self.nb, self.seed)
        return {"mean": m, "ci95": [lo, hi], "sd": float(np.nanstd(v, ddof=1)) if ok.sum() > 1 else 0.0,
                "median": float(np.nanmedian(v)), "n_frames": int(ok.sum()), "n_tasks": len(set(np.asarray(t)[ok].tolist()))}

    def paired(self, fam, m1, m2, split="test", scope="all", direction="inject", metric="closed_pct"):
        f1, v1, t1 = self.vals(fam, m1, split, scope, direction, metric)
        f2, v2, _ = self.vals(fam, m2, split, scope, direction, metric)
        common = [x for x in f1 if x in set(f2)]
        if not common:
            return None
        i1, i2 = {x: i for i, x in enumerate(f1)}, {x: i for i, x in enumerate(f2)}
        d = np.array([v1[i1[x]] - v2[i2[x]] for x in common])
        tasks = np.array([t1[i1[x]] for x in common])
        ok = np.isfinite(d)
        d, tasks = d[ok], tasks[ok]
        if not len(d):
            return None
        m, (lo, hi) = cluster_bootstrap(d, tasks, self.nb, self.seed)
        ut = sorted(set(tasks.tolist()))
        tm = np.array([d[tasks == u].mean() for u in ut])
        return {"mean_diff": m, "ci95": [lo, hi], "p_signflip_tasks": sign_flip_p(tm, self.n_mc, self.seed),
                "p_signflip_frames": sign_flip_p(d, self.n_mc, self.seed), "frac_frames_positive": float((d > 0).mean()),
                "n_tasks_positive": int((tm > 0).sum()), "n_frames": int(len(d)), "n_tasks": len(ut),
                "per_task_mean_diff": {str(u): float(x) for u, x in zip(ut, tm)}}

    def beats_every_draw(self, fam, method, ctrl, split="test", direction="inject"):
        a = self.index.get((fam, split, method, "all", direction), {})
        b = self.index.get((fam, split, ctrl, "all", direction), {})
        fs = [f for f in a if f in b]
        win = sum(np.nanmean([x["closed_pct"] for x in a[f]]) > max(x["closed_pct"] for x in b[f]) for f in fs)
        lose = sum(np.nanmean([x["closed_pct"] for x in a[f]]) < min(x["closed_pct"] for x in b[f]) for f in fs)
        return {"beats_every_draw": int(win), "loses_to_every_draw": int(lose), "n": len(fs)}


def cmd_goal2():
    require_gate("validate", "Goal 2")
    if read_status("goal1") is None:
        raise SystemExit("[gate] goal1 has not been run yet. Run goal1 first.")
    setup_torch(CFG["seed"])
    sel = load_json(OUT / "goal1_selected.json")
    rk = torch.load(OUT / "goal1_rank.pt", weights_only=False)
    tcs = load_all_tcs()
    all_L = sorted(tcs)
    kof = {l: k for k, l in enumerate(all_L)}
    n = tcs[all_L[0]].n_feat
    L = len(all_L)
    alive = rk["alive"].numpy().astype(bool)
    rank_layers = [l for l in sel.get("rank_layers", all_L) if l in kof]
    rank_mask = np.zeros((L, n), dtype=bool)
    for l in rank_layers:
        rank_mask[kof[l]] = True
    allowed = alive & rank_mask
    fams = [f for f in families() if f in sel["families"]]
    prim = primary_family()
    st_attr = read_status("attrib")
    attr_ok = bool(st_attr and st_attr.get("passed"))
    ATT, RANK, att_row = {}, {}, {}
    if attr_ok:
        for f in fams:
            ATT[f] = dict(np.load(OUT / "attrib" / f"attr_{f}.npz"))
            RANK[f] = dict(np.load(OUT / "attrib" / f"rank_{f}.npz"))
            att_row[f] = {int(x): i for i, x in enumerate(ATT[f]["frame_ids"])}
    else:
        log("Goal 2: no attribution results (attrib failed or not run) -> attribution sets are skipped")
    K_A = int(CFG.get("attr_k", 100))
    Ks = [int(k) for k in CFG.get("kcurve_k", [10, 30, 100, 300, 1000, 3000])]
    Ks_rem = [int(k) for k in CFG.get("kcurve_k_remove", [30, 100, 300, 1000])]
    nd_p, nd_s = int(CFG.get("n_random_draws", 8)), int(CFG.get("n_random_draws_secondary", 3))
    nd_rem = int(CFG.get("n_random_draws_remove", 3))
    n_single = int(CFG.get("n_single_features", 3))
    min_gap = float(CFG.get("goal2_min_gap", 0.1))
    seed = int(CFG["seed"])
    C = {int(l): v for l, v in (sel.get("color") or {}).items() if v}

    # ---------------------------------------------------------------- feature sets (fixed before touching test frames)
    SETS = {}
    for f in fams:
        FL = sel["families"][f]["layers"]
        get = lambda key: {int(l): [int(j) for j in (FL[l].get(key) or [])] for l in FL if FL[l].get(key)}
        s = {"sel_capped": get("occ"), "sel_uncapped": get("occ_all"), "sel_target": get("occ_target"),
             "sel_xtask": {p: {int(l): v for l, v in d.items() if v} for p, d in sel["families"][f].get("folds", {}).items()}}
        if attr_ok:
            s["attr"] = _sets_from_flat(_top_flat(RANK[f]["sym"], K_A, allowed), all_L, n)
            s["attr_xtask"] = {str(p): _sets_from_flat(_top_flat(RANK[f][f"sym_fold{p}"], K_A, allowed), all_L, n) for p in (0, 1)}
            s["order_attr"] = _top_flat(RANK[f]["sym"], max(Ks + Ks_rem), allowed)
            s["order_attr_inject"] = _top_flat(RANK[f]["inject"], max(Ks), allowed)
        s["order_mag"] = _top_flat(rk["fam"][f]["edit"].numpy(), max(Ks), allowed)
        SETS[f] = s
    rng = np.random.default_rng(seed + 1000)

    def pools_excluding(excl_sets):
        ex = set()
        for es in excl_sets:
            ex |= _flat_of(es, all_L, n)
        return {l: [j for j in np.nonzero(alive[kof[l]])[0].tolist() if kof[l] * n + j not in ex] for l in all_L}

    def draw_like(sets, pools, r):
        return {l: sorted(r.choice(pools[l], size=min(len(v), len(pools[l])), replace=False).tolist())
                for l, v in sets.items() if v and pools.get(l)}

    RAND = {}
    alive_rank_flat = np.nonzero(allowed.flatten())[0]
    for f in fams:
        s = SETS[f]
        nd = nd_p if f == prim else nd_s
        pS = pools_excluding([s["sel_uncapped"], C])
        R = {"rand_sel": [draw_like(s["sel_capped"], pS, rng) for _ in range(nd)]}
        if "attr" in s:
            pA = pools_excluding([s["attr"]])
            R["rand_attr"] = [draw_like(s["attr"], pA, rng) for _ in range(max(nd, nd_rem))]
        R["rand_k"] = {K: _sets_from_flat(rng.choice(alive_rank_flat, size=min(K, len(alive_rank_flat)), replace=False), all_L, n)
                       for K in Ks}
        RAND[f] = R
    log("Goal 2 feature sets (fixed on discovery data): " + " | ".join(
        f"{f}: selective {_nfeat(SETS[f]['sel_capped'])} (uncapped {_nfeat(SETS[f]['sel_uncapped'])}), "
        f"attribution {_nfeat(SETS[f].get('attr', {}))}" for f in fams) + f" | rank layers {rank_layers} | K_A={K_A}")

    stack, runner = get_stack(), get_runner()
    B = patch_batch()
    conds = active_conds()
    cidx = {c: i for i, c in enumerate(conds)}
    test = probe_frames("test")
    if not test:
        raise SystemExit("[gate] no held-out test frames; re-run frames")
    disc = probe_frames("discovery") if CFG.get("goal2_insample", True) else []
    insample_fams = [f for f in fams if f in (prim, "paint")]
    sums = np.zeros((len(test), len(conds), L, n), dtype=np.float32)
    have = np.zeros((len(test), len(conds)), dtype=bool)
    rows, frames_info = [], []
    t0 = time.time()

    def pred_of(mat, sets):
        if mat is None or not sets:
            return None
        v = float(sum(np.nansum(mat[kof[l], fs]) for l, fs in sets.items() if fs))
        return 100.0 * v

    all_items = [(fr, "test", fi) for fi, fr in enumerate(test)] + [(fr, "discovery", -1) for fr in disc]
    for it, (fr, split, fi) in enumerate(all_items):
        fid, task = fr["frame_id"], fr["task"]
        tid = int(fr["task_id"])
        noise = stack.noise(fr["noise_seed"])
        fam_here = fams if split == "test" else insample_fams
        tconds = {f: fam_target(f) for f in fam_here if cond_available(fr, fam_target(f))}
        if split == "test":
            cs = [c for c in conds if cond_available(fr, c)]
        else:
            cs = ["base"] + sorted(set(tconds.values()))
        A_, CAP, bs = run_conds(runner, stack, fr, cs, B, capture_layers=all_L, capture_conds=cs,
                                capture_kinds=("mlp_in", "mlp_out"))
        valid = runner.last_prefix["pad"]
        lay = runner.layout(bs["base"])
        keepc = ["base"] + sorted(set(tconds.values()))
        with torch.no_grad():
            Fe = {c: {l: tcs[l].encode(CAP[c][("mlp_in", l)]) for l in all_L} for c in keepc}
            Yo = {c: {l: CAP[c][("mlp_out", l)] for l in all_L} for c in keepc}
            if split == "test":
                vf = valid.float()[:, None]
                for c in cs:
                    have[fi, cidx[c]] = True
                    for l in all_L:
                        Fm = Fe[c][l] if c in Fe else tcs[l].encode(CAP[c][("mlp_in", l)])
                        sums[fi, cidx[c], kof[l]] = (Fm * vf).sum(0).float().cpu().numpy()
            Er = {}
            if split == "test" and prim in tconds:
                for c in ("base", tconds[prim]):
                    Er[c] = {l: tcs[l].error(CAP[c][("mlp_in", l)], CAP[c][("mlp_out", l)]) for l in all_L}
        del CAP
        jobs = []

        def fset(sets, src, norms=None, mask=None):
            m = valid if mask is None else mask
            return {l: fp_set(tcs[l], fs, src[l], m, norm_match=(norms or {}).get(l)) for l, fs in sets.items() if fs}

        def norms_of(sets, to, frm):
            out = {}
            for l, fs in sets.items():
                if fs:
                    idx = torch.as_tensor(fs, dtype=torch.long, device=DEV)
                    out[l] = tcs[l].delta(to[l][:, idx] - frm[l][:, idx], idx).norm(dim=-1) * valid.float()
            return out

        for f, t in tconds.items():
            a_b, a_t = A_["base"], A_[t]
            gap = rmse(a_b, a_t)
            inc = gap >= min_gap
            frames_info.append({"frame": fid, "task_id": tid, "family": f, "split": split, "gap": gap, "included": bool(inc),
                                "distractor_visible": _distractor_visible(fr)})
            if not inc:
                continue
            s, R = SETS[f], RAND[f]
            arow = att_row.get(f, {}).get(fid)
            att_i = ATT[f]["inject"][arow] if arow is not None else None
            att_r = ATT[f]["remove"][arow] if arow is not None else None
            if att_i is not None and not np.isfinite(att_i[0, 0]):
                att_i = att_r = None
            Ft, Fb = Fe[t], Fe["base"]
            J = lambda method, direction, patch, extra=None, **kw: jobs.append(
                {"family": f, "method": method, "direction": direction, "patch": patch, "gap": gap,
                 "start": a_b if direction == "inject" else a_t, "target": a_t if direction == "inject" else a_b,
                 "batch": bs["base"] if direction == "inject" else bs[t], "scope": kw.get("scope", "all"),
                 "draw": kw.get("draw"), "k": kw.get("k"), "n_features": kw.get("nf"), "pred_pct": kw.get("pred")})
            full = split == "test"
            secondary = f != prim
            # --- selective sets (Goal 1)
            J("sel_capped", "inject", fset(s["sel_capped"], Ft), nf=_nfeat(s["sel_capped"]), pred=pred_of(att_i, s["sel_capped"]))
            nS = norms_of(s["sel_capped"], Ft, Fb)
            nd = (nd_p if not secondary else nd_s) if full else min(3, nd_s)
            for d_, Rs in enumerate(R["rand_sel"][:nd]):
                J("rand_sel_same_norm", "inject", fset(Rs, Ft, {l: nS[l] for l in Rs if l in nS}), draw=d_, nf=_nfeat(Rs))
            if full and not secondary:
                if s["sel_uncapped"] != s["sel_capped"]:
                    J("sel_uncapped", "inject", fset(s["sel_uncapped"], Ft), nf=_nfeat(s["sel_uncapped"]),
                      pred=pred_of(att_i, s["sel_uncapped"]))
                if s["sel_target"] and s["sel_target"] != s["sel_capped"]:
                    J("sel_target", "inject", fset(s["sel_target"], Ft), nf=_nfeat(s["sel_target"]))
                xs = s["sel_xtask"].get(str(1 - tid % 2), {})
                if xs:
                    J("sel_xtask", "inject", fset(xs, Ft), nf=_nfeat(xs), pred=pred_of(att_i, xs))
                if C:
                    J("color", "inject", fset(C, Ft), nf=_nfeat(C))
            if full:
                J("sel_capped", "remove", fset(s["sel_capped"], Fb), nf=_nfeat(s["sel_capped"]), pred=pred_of(att_r, s["sel_capped"]))
            # --- attribution sets
            if "attr" in s:
                J("attr", "inject", fset(s["attr"], Ft), nf=_nfeat(s["attr"]), pred=pred_of(att_i, s["attr"]))
                nA = norms_of(s["attr"], Ft, Fb)
                for d_, Ra in enumerate(R["rand_attr"][:nd]):
                    J("rand_attr_same_norm", "inject", fset(Ra, Ft, {l: nA[l] for l in Ra if l in nA}), draw=d_, nf=_nfeat(Ra))
                if full:
                    J("attr", "remove", fset(s["attr"], Fb), nf=_nfeat(s["attr"]), pred=pred_of(att_r, s["attr"]))
                    nAr = norms_of(s["attr"], Fb, Ft)
                    for d_, Ra in enumerate(R["rand_attr"][:nd_rem]):
                        J("rand_attr_same_norm", "remove", fset(Ra, Fb, {l: nAr[l] for l in Ra if l in nAr}), draw=d_, nf=_nfeat(Ra))
                if full and not secondary:
                    xa = s["attr_xtask"].get(str(1 - tid % 2), {})
                    if xa:
                        J("attr_xtask", "inject", fset(xa, Ft), nf=_nfeat(xa), pred=pred_of(att_i, xa))
                    for d_, Ra in enumerate(R["rand_attr"][:2]):
                        J("rand_attr_raw", "inject", fset(Ra, Ft), draw=d_, nf=_nfeat(Ra), pred=pred_of(att_i, Ra))
                    order_top = s["order_attr"][:n_single]
                    for rnk, flat in enumerate(order_top):
                        l, j = all_L[int(flat) // n], int(flat) % n
                        J(f"single_attr_{rnk}", "inject", fset({l: [j]}, Ft), nf=1, pred=pred_of(att_i, {l: [j]}))
            if not full:
                J("mlp_ceiling", "inject", {l: fp_swap(Yo[t][l], valid) for l in all_L})
                continue
            # --- ceilings and the transcoder's own limits
            J("mlp_ceiling", "inject", {l: fp_swap(Yo[t][l], valid) for l in all_L})
            J("mlp_ceiling", "remove", {l: fp_swap(Yo["base"][l], valid) for l in all_L})
            J("all_tc", "inject", {l: fp_set(tcs[l], range(n), Ft[l], valid) for l in all_L}, nf=n * L,
              pred=100.0 * float(np.nansum(att_i)) if att_i is not None else None)
            if not secondary and t in Er:
                J("tc_error", "inject", {l: fp_err_swap(tcs[l], Er[t][l], valid) for l in all_L})
            # --- k-curves
            for K in Ks:
                if "order_attr" in s:
                    sk = _sets_from_flat(s["order_attr"][:K], all_L, n)
                    J(f"k_attr_{K}", "inject", fset(sk, Ft), k=K, nf=_nfeat(sk), pred=pred_of(att_i, sk))
                J(f"k_random_{K}", "inject", fset(R["rand_k"][K], Ft), k=K, nf=_nfeat(R["rand_k"][K]),
                  pred=pred_of(att_i, R["rand_k"][K]))
                if not secondary:
                    sm = _sets_from_flat(s["order_mag"][:K], all_L, n)
                    J(f"k_magnitude_{K}", "inject", fset(sm, Ft), k=K, nf=_nfeat(sm), pred=pred_of(att_i, sm))
                    if att_i is not None:
                        so = _sets_from_flat(_top_flat(att_i, K, allowed), all_L, n)
                        J(f"k_oracle_{K}", "inject", fset(so, Ft), k=K, nf=_nfeat(so), pred=pred_of(att_i, so))
            if not secondary and "order_attr" in s:
                for K in Ks_rem:
                    sk = _sets_from_flat(s["order_attr"][:K], all_L, n)
                    J(f"k_attr_{K}", "remove", fset(sk, Fb), k=K, nf=_nfeat(sk), pred=pred_of(att_r, sk))
            # --- where: tokens under the occluder vs elsewhere
            if not secondary:
                reg = fr["regions"].get(t) or {}
                bowl = runner.region_positions(lay, reg).to(valid.device) & valid
                if bool(bowl.any()):
                    for sc_, m_ in (("occluder_tokens", bowl), ("other_tokens", valid & ~bowl)):
                        J("mlp_ceiling", "inject", {l: fp_swap(Yo[t][l], m_) for l in all_L}, scope=sc_)
                        if "attr" in s:
                            J("attr", "inject", fset(s["attr"], Ft, mask=m_), scope=sc_)
        # ---- run every job of this frame, B rows per forward pass
        for s0 in range(0, len(jobs), B):
            chunk = jobs[s0: s0 + B]
            a, _ = runner.run_rows([(j["batch"], j["patch"], None) for j in chunk], noise, B=B)
            for j, ai in zip(chunk, a):
                rows.append({"frame": fid, "task_id": tid, "family": j["family"], "split": split, "method": j["method"],
                             "scope": j["scope"], "direction": j["direction"], "draw": j["draw"], "k": j["k"], "gap": j["gap"],
                             "n_features": j["n_features"], "closed_pct": closure_pct(ai, j["target"], j["start"]),
                             "proj_pct": proj_pct(ai, j["target"], j["start"]), "pred_pct": j["pred_pct"]})
        jobs = Fe = Yo = Er = bs = None
        free_cuda()
        if it % 5 == 0 or it == len(all_items) - 1:
            last = {f: [r["closed_pct"] for r in rows if r["frame"] == fid and r["family"] == f and r["method"] in ("attr", "sel_capped", "mlp_ceiling")
                        and r["direction"] == "inject" and r["scope"] == "all"] for f in fam_here}
            log(f"goal2 {it + 1}/{len(all_items)} ({split} frame {fid}, task {tid}) {time.time() - t0:.0f}s | "
                + " | ".join(f"{f}: attr/sel/ceiling " + "/".join(f"{x:.1f}" for x in v) for f, v in last.items() if v))

    _goal2_aggregate(rows, frames_info, SETS, RAND, sel, rk, sums, have, test, cidx, conds, all_L, kof, n, allowed, alive,
                     RANK if attr_ok else None, fams, prim, rank_layers, K_A, Ks, Ks_rem, C)


def _goal2_aggregate(rows, frames_info, SETS, RAND, sel, rk, sums, have, test, cidx, conds, all_L, kof, n, allowed, alive,
                     RANK, fams, prim, rank_layers, K_A, Ks, Ks_rem, C):
    nb, seed = int(CFG["n_bootstrap"]), int(CFG["seed"])
    n_mc = int(CFG.get("n_permutations", 200000))
    margin = float(CFG["goal2_margin_pts"])
    G = G2Stats(rows, nb, seed, n_mc)
    L = len(all_L)
    out = {"protocol": {"version": PROTOCOL_VERSION, "families": fams, "primary_family": prim, "n_test_frames": len(test),
                        "min_gap": float(CFG.get("goal2_min_gap", 0.1)), "margin_pts": margin, "attr_k": K_A,
                        "rank_layers": rank_layers, "n_features_total": n * L, "n_bootstrap": nb,
                        "ci": "95% percentile bootstrap, resampling tasks (clusters) with replacement",
                        "p_values": "one-sided sign-flip permutation over task means (exact for <=16 tasks)",
                        "metric": "closed_pct = 100 * (1 - RMSE(patched, target) / RMSE(start, target)); proj_pct = projection on the gap"},
           "families": {}}
    methods_main = ["sel_capped", "sel_uncapped", "sel_target", "sel_xtask", "color", "rand_sel_same_norm", "attr", "attr_xtask",
                    "rand_attr_same_norm", "rand_attr_raw", "all_tc", "tc_error", "mlp_ceiling"]
    for f in fams:
        fi_inc = [x for x in frames_info if x["family"] == f and x["split"] == "test"]
        inc = [x for x in fi_inc if x["included"]]
        FS = {"n_frames": len(fi_inc), "n_included": len(inc), "tasks": sorted({x["task_id"] for x in inc}),
              "gaps": {"median": float(np.median([x["gap"] for x in fi_inc])) if fi_inc else None,
                       "min": float(min([x["gap"] for x in fi_inc] or [np.nan])), "max": float(max([x["gap"] for x in fi_inc] or [np.nan]))},
              "n_features": {k: _nfeat(v) for k, v in SETS[f].items() if isinstance(v, dict) and k in ("sel_capped", "sel_uncapped", "sel_target", "attr")},
              "methods": {}, "remove": {}, "scopes": {}, "kcurve": {}, "kcurve_remove": {}, "paired": {}}
        for m in methods_main + [f"single_attr_{i}" for i in range(int(CFG.get("n_single_features", 3)))]:
            s_ = G.summ(f, m)
            if s_:
                s_["proj"] = (G.summ(f, m, metric="proj_pct") or {}).get("mean")
                FS["methods"][m] = s_
        for m in ("sel_capped", "attr", "rand_attr_same_norm", "mlp_ceiling"):
            s_ = G.summ(f, m, direction="remove")
            if s_:
                FS["remove"][m] = s_
        for sc_ in ("occluder_tokens", "other_tokens"):
            FS["scopes"][sc_] = {m: G.summ(f, m, scope=sc_) for m in ("attr", "mlp_ceiling") if G.summ(f, m, scope=sc_)}
        for fam_k in ("attr", "oracle", "magnitude", "random"):
            pts = [dict(k=K, **G.summ(f, f"k_{fam_k}_{K}")) for K in Ks if G.summ(f, f"k_{fam_k}_{K}")]
            if pts:
                FS["kcurve"][fam_k] = pts
        pts = [dict(k=K, **G.summ(f, f"k_attr_{K}", direction="remove")) for K in Ks_rem if G.summ(f, f"k_attr_{K}", direction="remove")]
        if pts:
            FS["kcurve_remove"]["attr"] = pts
        P = FS["paired"]
        P["attr_vs_rand_attr"] = G.paired(f, "attr", "rand_attr_same_norm")
        P["attr_vs_rand_attr_remove"] = G.paired(f, "attr", "rand_attr_same_norm", direction="remove")
        P["sel_vs_rand_sel"] = G.paired(f, "sel_capped", "rand_sel_same_norm")
        P["attr_vs_sel"] = G.paired(f, "attr", "sel_capped")
        P["sel_vs_color"] = G.paired(f, "sel_capped", "color")
        P["attr_xtask_vs_rand_attr"] = G.paired(f, "attr_xtask", "rand_attr_same_norm")
        P["sel_xtask_vs_rand_sel"] = G.paired(f, "sel_xtask", "rand_sel_same_norm")
        for K in Ks:
            P[f"k_attr_{K}_vs_k_random_{K}"] = G.paired(f, f"k_attr_{K}", f"k_random_{K}")
            P[f"k_attr_{K}_vs_k_magnitude_{K}"] = G.paired(f, f"k_attr_{K}", f"k_magnitude_{K}")
        FS["paired"] = {k: v for k, v in P.items() if v}
        FS["beats_every_draw"] = {"attr": G.beats_every_draw(f, "attr", "rand_attr_same_norm"),
                                  "sel_capped": G.beats_every_draw(f, "sel_capped", "rand_sel_same_norm")}
        # --- verdicts (pre-specified)
        mean = lambda m, d="inject": ((FS["methods"] if d == "inject" else FS["remove"]).get(m) or {}).get("mean", float("nan"))
        pa, ps = P.get("attr_vs_rand_attr"), P.get("sel_vs_rand_sel")
        col = mean("color")
        FS["verdict"] = {
            "H2A_attr_beats_random": bool(pa and pa["mean_diff"] >= margin and pa["ci95"][0] > 0),
            "H2A_margin": pa["mean_diff"] if pa else None, "H2A_ci95": pa["ci95"] if pa else None,
            "H2A_remove": bool(P.get("attr_vs_rand_attr_remove") and P["attr_vs_rand_attr_remove"]["mean_diff"] >= margin
                               and P["attr_vs_rand_attr_remove"]["ci95"][0] > 0),
            "H2S_selective_beats_controls": bool(ps and ps["mean_diff"] >= margin and (not np.isfinite(col) or mean("sel_capped") - col >= margin)),
            "H2S_margin": ps["mean_diff"] if ps else None, "H2S_ci95": ps["ci95"] if ps else None,
            "attr": mean("attr"), "rand_attr": mean("rand_attr_same_norm"), "sel": mean("sel_capped"),
            "rand_sel": mean("rand_sel_same_norm"), "color": col, "all_tc": mean("all_tc"), "mlp_ceiling": mean("mlp_ceiling"),
            "tc_error": mean("tc_error"), "attr_share_of_ceiling_pct": 100 * mean("attr") / mean("mlp_ceiling") if mean("mlp_ceiling") > 0 else None,
            "attr_remove": mean("attr", "remove"), "rand_attr_remove": mean("rand_attr_same_norm", "remove"),
            "mlp_ceiling_remove": mean("mlp_ceiling", "remove")}
        # --- linearity of attribution (does the first-order prediction match the patch?)
        lr = [r for r in rows if r["family"] == f and r["split"] == "test" and r.get("pred_pct") is not None
              and r["scope"] == "all" and np.isfinite(r["proj_pct"])]
        if len(lr) >= 3:
            pr, ac, cl = (np.array([r[k] for r in lr], dtype=float) for k in ("pred_pct", "proj_pct", "closed_pct"))
            slope = float(np.polyfit(pr, ac, 1)[0]) if np.std(pr) > 0 else None
            FS["linearity"] = {"n": len(lr), "pearson_pred_vs_proj": pearson(pr, ac), "spearman_pred_vs_proj": spearman(pr, ac),
                               "pearson_pred_vs_closed": pearson(pr, cl), "slope_proj_on_pred": slope,
                               "median_abs_error_pts": float(np.median(np.abs(pr - ac)))}
        # --- dissociation: do selective features and causally attributed features coincide?
        R_f = rk["fam"][f]
        D = {"n_selective_uncapped": _nfeat(SETS[f]["sel_uncapped"])}
        if RANK is not None:
            sym = np.nan_to_num(RANK[f]["sym"])
            A_flat = _flat_of(SETS[f].get("attr", {}), all_L, n)
            S_flat = _flat_of(SETS[f]["sel_uncapped"], all_L, n)
            Sc_flat = _flat_of(SETS[f]["sel_capped"], all_L, n)
            D.update({"overlap_attr_sel_uncapped": len(A_flat & S_flat), "overlap_attr_sel_capped": len(A_flat & Sc_flat),
                      "jaccard_attr_sel_uncapped": len(A_flat & S_flat) / max(1, len(A_flat | S_flat))})
            v = np.where(allowed, sym, np.nan).flatten()
            okv = np.isfinite(v)
            order = np.argsort(-np.where(okv, v, -np.inf), kind="stable")
            pos = np.empty_like(order)
            pos[order] = np.arange(len(order))
            n_ok = int(okv.sum())
            srank = [100.0 * (1 - pos[i] / max(n_ok, 1)) for i in S_flat if okv[i]]
            D["selective_attr_percentile_median"] = float(np.median(srank)) if srank else None
            pm = np.clip(np.where(okv, v, 0), 0, None)
            D["selective_share_of_positive_attr_pct"] = float(100 * pm[list(S_flat)].sum() / max(pm.sum(), 1e-12)) if S_flat else 0.0
            D["attr_share_of_positive_attr_pct"] = float(100 * pm[list(A_flat)].sum() / max(pm.sum(), 1e-12)) if A_flat else 0.0
            passf = R_f["pass"].numpy().flatten()
            D["frac_attr_passing_selectivity"] = float(np.mean([passf[i] for i in A_flat])) if A_flat else None
            rise = R_f["rise"].numpy().flatten()
            dmag = np.abs(R_f["d_target"].numpy().flatten())
            edit = R_f["edit"].numpy().flatten()
            m_ = allowed.flatten()
            D["spearman_selectivity_rise_vs_attr"] = spearman(rise[m_], v[m_])
            D["spearman_response_magnitude_vs_attr"] = spearman(dmag[m_], v[m_])
            D["spearman_edit_norm_vs_attr"] = spearman(edit[m_], v[m_])
            D["n_features_compared"] = int(m_.sum())
            D["H3_dissociated"] = bool((D["frac_attr_passing_selectivity"] or 0) < 0.1 and
                                       (D["spearman_selectivity_rise_vs_attr"] if np.isfinite(D["spearman_selectivity_rise_vs_attr"]) else 0) < 0.2)
        FS["dissociation"] = D
        # --- held-out replication of Goal-1 selectivity (test frames only)
        F_ = FAMILIES[f]
        T_, B_ = cidx[F_["target"]], cidx["base"]
        qi = [cidx[c] for c in F_["quiet"] if c in cidx]
        rr = [i for i in range(len(test)) if have[i, T_] and all(have[i, q] for q in qi)]
        H = {"n_test_frames": len(rr)}
        if rr:
            d_t = sums[rr, T_] - sums[rr, B_]
            t_mean = d_t.mean(0)
            posf = (d_t > 0).mean(0)
            other = np.max(np.stack([np.abs(sums[rr, q] - sums[rr, B_]).mean(0) for q in qi]), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                spec = np.where(t_mean > 0, 1.0 - other / np.maximum(t_mean, 1e-12), -np.inf)
            typ = rk["typical"].numpy()
            rep = (t_mean > 0) & (posf >= float(CFG["min_pos_frac"])) & (spec >= float(CFG["min_specificity"])) \
                & (t_mean / np.maximum(typ, 1e-8) >= float(CFG["min_rise_tokens"])) & alive

            def rep_frac(sets):
                idx = [(kof[l], j) for l, fs in sets.items() for j in fs]
                return float(np.mean([rep[k, j] for k, j in idx])) if idx else None

            H.update({"replicate_frac_sel_capped": rep_frac(SETS[f]["sel_capped"]),
                      "replicate_frac_sel_uncapped": rep_frac(SETS[f]["sel_uncapped"]),
                      "replicate_frac_attr": rep_frac(SETS[f].get("attr", {})),
                      "replicate_frac_random_draws_mean": float(np.nanmean([np.nan if rep_frac(r) is None else rep_frac(r)
                                                                            for r in RAND[f]["rand_sel"]])) if RAND[f]["rand_sel"] else None,
                      "base_rate_all_alive": float(rep[alive].mean()) if alive.any() else None,
                      "spearman_discovery_vs_test_response": spearman((R_f["d_target"].numpy() / np.maximum(typ, 1e-8))[alive],
                                                                      (t_mean / np.maximum(typ, 1e-8))[alive])})
            br = H["base_rate_all_alive"] or 0
            H["enrichment_sel_capped"] = (H["replicate_frac_sel_capped"] / br) if br > 0 and H["replicate_frac_sel_capped"] is not None else None
        FS["heldout_selectivity"] = H
        # --- in-sample (discovery frames) vs held-out: the circularity inflation
        IS = {}
        for m, ctrl in (("sel_capped", "rand_sel_same_norm"), ("attr", "rand_attr_same_norm"), ("mlp_ceiling", None)):
            a_d, a_t = G.summ(f, m, split="discovery"), G.summ(f, m, split="test")
            if a_d and a_t:
                IS[m] = {"discovery": a_d["mean"], "test": a_t["mean"], "discovery_ci95": a_d["ci95"], "test_ci95": a_t["ci95"]}
                if ctrl:
                    c_d, c_t = G.summ(f, ctrl, split="discovery"), G.summ(f, ctrl, split="test")
                    if c_d and c_t:
                        IS[m].update({"control_discovery": c_d["mean"], "control_test": c_t["mean"],
                                      "margin_discovery": a_d["mean"] - c_d["mean"], "margin_test": a_t["mean"] - c_t["mean"]})
        FS["insample_vs_heldout"] = IS
        out["families"][f] = FS
        V = FS["verdict"]
        log(f"[{f}] attribution set ({FS['n_features'].get('attr', 0)}) {V['attr']:.2f}% vs random same-norm {V['rand_attr']:.2f}% "
            f"-> H2-A {'PASS' if V['H2A_attr_beats_random'] else 'fail'} | selective ({FS['n_features'].get('sel_capped', 0)}) "
            f"{V['sel']:.2f}% vs random {V['rand_sel']:.2f}% -> H2-S {'PASS' if V['H2S_selective_beats_controls'] else 'fail'} | "
            f"all TC {V['all_tc']:.1f}% | MLP ceiling {V['mlp_ceiling']:.1f}% | error term {V['tc_error']:.1f}%")
        for fam_k, pts in FS["kcurve"].items():
            log(f"   k-curve [{fam_k}]: " + " | ".join(f"K={p['k']}: {p['mean']:.1f}%" for p in pts))
        if "linearity" in FS:
            log(f"   attribution linearity: r={FS['linearity']['pearson_pred_vs_proj']:.2f} (n={FS['linearity']['n']}), "
                f"slope {FS['linearity']['slope_proj_on_pred']}")
        if D.get("spearman_selectivity_rise_vs_attr") is not None:
            log(f"   dissociation: |A ∩ S| = {D.get('overlap_attr_sel_uncapped')} | {100 * (D.get('frac_attr_passing_selectivity') or 0):.0f}% of A "
                f"selective | rho(selectivity, attribution) = {D['spearman_selectivity_rise_vs_attr']:.2f}")
    P0 = out["families"].get(prim, {})
    V0 = P0.get("verdict", {})
    out["hypotheses"] = {
        "H1_selective_features_replicate": {"primary": prim, "n_selective": (sel["families"].get(prim) or {}).get("n_pass_total"),
                                            "replicate_frac": (P0.get("heldout_selectivity") or {}).get("replicate_frac_sel_capped"),
                                            "enrichment": (P0.get("heldout_selectivity") or {}).get("enrichment_sel_capped")},
        "H2A_attribution_set_causal": V0.get("H2A_attr_beats_random"),
        "H2S_selective_set_causal": V0.get("H2S_selective_beats_controls"),
        "H3_selectivity_attribution_dissociated": (P0.get("dissociation") or {}).get("H3_dissociated"),
        "H2A_by_family": {f: out["families"][f]["verdict"]["H2A_attr_beats_random"] for f in out["families"]},
        "H2S_by_family": {f: out["families"][f]["verdict"]["H2S_selective_beats_controls"] for f in out["families"]},
    }
    save_json(OUT / "goal2_stats.json", out)
    save_json(OUT / "goal2_frames.json", {"frames": frames_info})
    save_json(OUT / "goal2_sets.json", {f: {k: {str(l): v for l, v in SETS[f][k].items()}
                                            for k in ("sel_capped", "sel_uncapped", "attr") if k in SETS[f]} for f in fams})
    with open(OUT / "goal2_rows.csv", "w", newline="") as fh:
        keys = ["frame", "task_id", "family", "split", "method", "scope", "direction", "draw", "k", "gap", "n_features",
                "closed_pct", "proj_pct", "pred_pct"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    _goal2_figures(out, rows)
    ok = bool(V0.get("H2A_attr_beats_random"))
    write_status("goal2", ok, {
        "primary_family": prim, "hypotheses": out["hypotheses"],
        "verdict": {f: out["families"][f]["verdict"] for f in out["families"]},
        "n_frames_included": {f: out["families"][f]["n_included"] for f in out["families"]},
        "kcurve": {f: {k: [(p["k"], p["mean"]) for p in v] for k, v in out["families"][f]["kcurve"].items()} for f in out["families"]},
        "reason": "" if ok else "the attribution-selected set does not beat norm-matched random features by the pre-specified margin "
                                "on held-out frames (primary family)",
    })


def _goal2_figures(stats, rows):
    plt = _plt()
    fams = list(stats["families"])
    labels = {"sel_capped": "selective\n(Goal 1)", "rand_sel_same_norm": "random\n(matched)", "color": "colour",
              "attr": f"attribution\ntop-{stats['protocol']['attr_k']}", "attr_xtask": "attribution\ncross-task",
              "rand_attr_same_norm": "random\n(matched)", "all_tc": "all TC\nfeatures", "tc_error": "TC error\nterm",
              "mlp_ceiling": "MLP swap\nceiling"}
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 3.6), squeeze=False)
    for ax, f in zip(axes[0], fams):
        M = stats["families"][f]["methods"]
        ms = [m for m in labels if m in M]
        mean = [M[m]["mean"] for m in ms]
        lo = [max(0.0, M[m]["mean"] - M[m]["ci95"][0]) if M[m]["ci95"][0] is not None and np.isfinite(M[m]["ci95"][0]) else 0 for m in ms]
        hi = [max(0.0, M[m]["ci95"][1] - M[m]["mean"]) if M[m]["ci95"][1] is not None and np.isfinite(M[m]["ci95"][1]) else 0 for m in ms]
        cols = ["#2a78d6" if m.startswith("attr") else "#eb6834" if m.startswith("sel") else "#1baf7a" if m in ("all_tc", "mlp_ceiling", "tc_error")
                else "#8a8a8a" for m in ms]
        ax.bar(range(len(ms)), mean, yerr=[lo, hi], capsize=3, color=cols)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([labels[m] for m in ms], fontsize=7)
        ax.set_yscale("symlog", linthresh=5)
        ax.set_ylabel("gap closed (%)")
        ax.set_title(f"{f}: {stats['families'][f]['n_included']} held-out frames", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal2_main.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 3.8), squeeze=False)
    for ax, f in zip(axes[0], fams):
        FS = stats["families"][f]
        for fam_k, col, lab in (("oracle", "#6a3d9a", "attribution (oracle, test frame)"), ("attr", "#2a78d6", "attribution (discovery)"),
                                ("magnitude", "#eb6834", "activation change (discovery)"), ("random", "#8a8a8a", "random")):
            pts = FS["kcurve"].get(fam_k) or []
            if pts:
                k = [p["k"] for p in pts]
                ax.plot(k, [p["mean"] for p in pts], marker="o", color=col, label=lab)
                ax.fill_between(k, [p["ci95"][0] for p in pts], [p["ci95"][1] for p in pts], color=col, alpha=0.12)
        for m, ls, lab in (("all_tc", "--", "all TC features"), ("mlp_ceiling", ":", "MLP ceiling")):
            if m in FS["methods"]:
                ax.axhline(FS["methods"][m]["mean"], color="k", ls=ls, lw=1, label=lab)
        if "sel_capped" in FS["methods"]:
            ax.plot([max(1, FS["n_features"].get("sel_capped", 1))], [FS["methods"]["sel_capped"]["mean"]], "*", ms=12, color="#1baf7a",
                    label="selective set")
        ax.set_xscale("log")
        ax.set_xlabel(f"features patched (of {stats['protocol']['n_features_total']:,})")
        ax.set_ylabel("gap closed (%)")
        ax.set_title(f"{f}: how many features carry the occlusion effect?", fontsize=9)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal2_kcurve.png", dpi=130)
    plt.close(fig)
    prim = stats["protocol"]["primary_family"]
    lr = [r for r in rows if r["family"] == prim and r["split"] == "test" and r.get("pred_pct") is not None and r["scope"] == "all"]
    if lr:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        pr = np.array([r["pred_pct"] for r in lr])
        ac = np.array([r["proj_pct"] for r in lr])
        ax.scatter(pr, ac, s=6, alpha=0.4)
        lim = [float(np.nanmin([pr.min(), ac.min(), 0])), float(np.nanmax([pr.max(), ac.max(), 1]))]
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.set_xlabel("attribution prediction (% of gap)")
        ax.set_ylabel("actual patch effect (projection, %)")
        ax.set_title(f"Linearity of attribution [{prim}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig_goal2_linearity.png", dpi=130)
        plt.close(fig)


# ============================================================================= closed loop


def _max_steps(env) -> int:
    """Episode length lerobot uses for this suite; 280 if the attribute is missing."""
    return int(getattr(env, "_max_episode_steps", None) or getattr(env, "max_episode_steps", None) or 280)


def observe(env, scene, specs=(), do_paint=False):
    """Observation of the current state as the policy would see it under `specs` (and/or the paint edit)."""
    lazy = _CACHE.get("lazy_cams", False)
    if lazy:
        set_cams(env, True)
    try:
        with occluders(*specs):
            obs = env_obs(env)
    finally:
        if lazy:
            set_cams(env, False)
    if do_paint:
        cams = {k: CAM_OF_KEY[k] for k in obs["pixels"]}
        masks, _, _ = scene.masks(cams)
        obs["pixels"] = {k: paint(img, dilate(masks[k], CFG["occ_pad_px"]), CFG["paint_gray"]) for k, img in obs["pixels"].items()}
    return obs


def policy_chunk(stack, runner, env, scene, spec, noise):
    task = libero_task(env)
    pair = spec.get("pair")
    if pair is None:
        obs = observe(env, scene, spec.get("occ", ()), spec.get("paint", False))
        a, _ = runner.run_rows([(stack.batch(obs, task), None, None)], noise, B=1)
        return a[0], obs
    clean = observe(env, scene)
    occ = observe(env, scene, pair["occ"], pair["paint"])
    src, dst = (clean, occ) if pair["direction"] == "restore" else (occ, clean)
    rows = [(stack.batch(src, task), None, None), (stack.batch(dst, task), None, None)]
    a, _ = runner.run_rows(rows, noise, B=2, pair_fns=pair["fns"])
    return a[1], dst


def run_episode(env, scene, stack, runner, task_id, ep, cond, spec, record_states=False, seed=None, video_path=None):
    import cv2

    seed = int(CFG["seed"]) + 100000 + 1000 * int(task_id) + int(ep) if seed is None else int(seed)
    env.init_state_id = int(CFG.get("goal3_init_offset", 0)) + int(ep)
    set_cams(env, True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    lazy = bool(CFG.get("lazy_cameras", True)) and set_cams(env, False)
    _CACHE["lazy_cams"] = lazy
    scene._refresh()
    max_steps = int(CFG.get("goal3_max_steps", 0) or 0) or _max_steps(env)
    n_act = int(CFG.get("goal3_n_action_steps", 0) or 0) or int(stack.config.n_action_steps)
    queue, states = [], []
    success, steps, n_inf = False, 0, 0
    z0, zd0 = scene.object_z(), scene.distractor_z()
    dz_t = dz_d = 0.0
    writer = None
    vspecs = list((spec.get("pair") or {}).get("occ", ()) if spec.get("pair") and spec["pair"]["direction"] == "restore"
                  else spec.get("occ", ()))
    try:
        for t in range(max_steps):
            if record_states:
                states.append(np.asarray(env._env.get_sim_state()).copy())
            if not queue:
                noise = stack.noise(seed * 997 + n_inf)
                act, _ = policy_chunk(stack, runner, env, scene, spec, noise)
                queue = stack.to_env_actions(act[:n_act])
                n_inf += 1
            if video_path is not None:
                with occluders(*vspecs):
                    frame = model_view(scene.rgb("agentview"))
                if writer is None:
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (frame.shape[1], frame.shape[0]))
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            a = queue.pop(0)
            _, _, done, _ = env._env.step(a)
            succ = bool(env._env.check_success())
            steps = t + 1
            dz_t = max(dz_t, scene.object_z() - z0)
            if np.isfinite(zd0):
                dz_d = max(dz_d, scene.distractor_z() - zd0)
            if done or succ:
                success = succ
                break
    finally:
        _CACHE["lazy_cams"] = False
        set_cams(env, True)
        if writer is not None:
            writer.release()
    out = {"task_id": int(task_id), "episode": int(ep), "cond": cond, "success": bool(success), "steps": int(steps),
           "n_inferences": int(n_inf), "target_lift_m": float(dz_t), "distractor_lift_m": float(dz_d),
           "distractor_lifted": bool(dz_d > float(CFG.get("lift_detect_m", 0.05))), "seed": seed}
    if record_states:
        out["states"] = states
    return out


def _goal3_sets(fam):
    """Attribution / selective / random sets of one family, identical to Goal 2's."""
    p = OUT / "goal2_sets.json"
    sets = load_json(p).get(fam, {}) if p.exists() else {}
    to_int = lambda d: {int(l): [int(j) for j in v] for l, v in (d or {}).items() if v}
    A, S = to_int(sets.get("attr")), to_int(sets.get("sel_capped"))
    rk = torch.load(OUT / "goal1_rank.pt", weights_only=False)
    alive = rk["alive"].numpy().astype(bool)
    layers = rk["layers"]
    kof = {l: k for k, l in enumerate(layers)}
    rng = np.random.default_rng(int(CFG["seed"]) + 4242)
    R = {}
    for l, fs in A.items():
        pool = [j for j in np.nonzero(alive[kof[l]])[0].tolist() if j not in set(fs)]
        if pool:
            R[l] = sorted(rng.choice(pool, size=min(len(fs), len(pool)), replace=False).tolist())
    return A, S, R


def cmd_goal3():
    require_gate("validate", "Goal 3")
    setup_torch(CFG["seed"])
    install_occluders()
    stack, runner = get_stack(), get_runner()
    tasks = [int(t) for t in CFG.get("goal3_tasks", [CFG.get("task_id", 0)])]
    n_ep = int(CFG["goal3_episodes"])
    fams = families()
    stageA = ["clean"] + [FAMILIES[f]["closed_loop"] for f in fams]
    occ_family = {FAMILIES[f]["closed_loop"]: f for f in fams}
    part_p = OUT / "goal3_partial.json"
    partial = load_json(part_p) if part_p.exists() else {}
    meta = {"episodes": n_ep, "init_offset": int(CFG.get("goal3_init_offset", 0)), "max_steps": int(CFG.get("goal3_max_steps", 0) or 0),
            "seed": int(CFG["seed"]), "occluder": str(CFG.get("goal3_occluder", "auto"))}
    if partial.get("_meta") != meta:
        if partial:
            log("goal3: settings changed since the saved episodes -> starting the closed loop from scratch")
        partial = {"_meta": meta}
    have_tcs = (OUT / "transcoders").exists() and any((OUT / "transcoders").glob("layer_*.pt"))
    tcs = load_all_tcs() if have_tcs else {}
    t0 = time.time()

    def spec_for(cond, scene, ostar, sets):
        occ_of = {"covered": ([scene.cover()], False), "screened": ([scene.screen()], False), "painted": ([], True)}
        if cond == "clean":
            return {}
        if cond in occ_of:
            o, p = occ_of[cond]
            return {"occ": o, "paint": p}
        o, p = occ_of[ostar]
        A, S, R = sets
        if cond == "restore_mlp":
            fns, direction = {l: pf_swap(runner) for l in sorted(tcs)}, "restore"
        elif cond == "restore_attr":
            fns, direction = {l: pf_set(tcs[l], A[l], runner) for l in A}, "restore"
        elif cond == "restore_sel":
            fns, direction = {l: pf_set(tcs[l], S[l], runner) for l in S}, "restore"
        elif cond == "restore_random":
            fns, direction = {l: pf_set_normmatch(tcs[l], R[l], A.get(l, []), runner) for l in R}, "restore"
        elif cond == "inject_attr":
            fns, direction = {l: pf_set(tcs[l], A[l], runner) for l in A}, "inject"
        else:
            raise ValueError(cond)
        return {"pair": {"fns": fns, "direction": direction, "occ": o, "paint": p}}

    def run_blocks(conds, ostar=None, sets=None, stage="A"):
        for task_id in tasks:
            todo = [c for c in conds if f"{task_id}|{c}" not in partial]
            if not todo:
                continue
            env = make_env(0, task_id)
            set_cams(env, True)
            scene = Scene(env)
            for cond in todo:
                spec = spec_for(cond, scene, ostar, sets)
                eps = []
                for ep in range(n_ep):
                    vp = OUT / "videos" / f"task{task_id}_{cond}_ep{ep}.mp4" if (CFG.get("save_videos") and ep == 0 and task_id == tasks[0]) else None
                    res = run_episode(env, scene, stack, runner, task_id, ep, cond, spec, video_path=vp)
                    eps.append(res)
                partial[f"{task_id}|{cond}"] = eps
                save_json(part_p, partial)
                log(f"goal3 stage {stage} | task {task_id} | {cond:15s}: success {sum(e['success'] for e in eps)}/{n_ep} | "
                    f"mean steps {np.mean([e['steps'] for e in eps]):.0f} | distractor lifted {sum(e['distractor_lifted'] for e in eps)} "
                    f"({time.time() - t0:.0f}s)")
            try:
                env.close()
            except Exception:
                pass
            gc.collect()

    run_blocks(stageA, stage="A")
    succ = lambda c: float(np.mean([e["success"] for t in tasks for e in partial.get(f"{t}|{c}", [])] or [np.nan]))
    drops = {c: succ("clean") - succ(c) for c in stageA[1:] if c in ("covered", "screened")}
    choice = str(CFG.get("goal3_occluder", "auto"))
    if choice in drops:
        ostar, why = choice, "GOAL3_OCCLUDER"
    elif drops:
        ostar = max(drops, key=lambda c: (drops[c], c == "covered"))
        why = "rendered occluder with the larger success drop in stage A (pre-specified rule)"
    else:
        ostar, why = stageA[-1], "only available occluder"
    fam_star = occ_family[ostar]
    log("goal3: stage A success " + ", ".join(f"{c} {succ(c):.2f}" for c in stageA) + f" -> stage B under {ostar!r} ({why})")
    stageB = []
    sets = None
    if tcs and (OUT / "goal2_sets.json").exists():
        sets = _goal3_sets(fam_star)
        stageB = ["restore_mlp"] + (["restore_attr", "restore_random", "inject_attr"] if sets[0] else []) + (["restore_sel"] if sets[1] else [])
        run_blocks(stageB, ostar=ostar, sets=sets, stage="B")
    else:
        log("goal3: no transcoders / Goal-2 sets -> stage B (restoring features) skipped")
    # ---------------------------------------------------------------- statistics
    nb, seed = int(CFG["n_bootstrap"]), int(CFG["seed"])
    all_conds = stageA + stageB
    per = {c: {(e["task_id"], e["episode"]): e for t in tasks for e in partial.get(f"{t}|{c}", [])} for c in all_conds}
    rates = {}
    for c in all_conds:
        v = [e["success"] for e in per[c].values()]
        p, (lo, hi) = wilson(sum(v), len(v))
        rates[c] = {"success": p, "wilson95": [lo, hi], "n": len(v), "k": int(sum(v)),
                    "per_task": {str(t): float(np.mean([e["success"] for (tt, _), e in per[c].items() if tt == t] or [np.nan])) for t in tasks},
                    "distractor_lifted_frac": float(np.mean([e["distractor_lifted"] for e in per[c].values()])) if per[c] else None,
                    "mean_steps": float(np.mean([e["steps"] for e in per[c].values()])) if per[c] else None}

    def paired(c1, c2):
        keys = sorted(set(per.get(c1, {})) & set(per.get(c2, {})))
        if not keys:
            return None
        a = np.array([per[c1][k]["success"] for k in keys], dtype=float)
        b = np.array([per[c2][k]["success"] for k in keys], dtype=float)
        d = a - b
        tk = [k[0] for k in keys]
        m, (lo, hi) = cluster_bootstrap(d, tk, nb, seed)
        bb, cc = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        return {"diff": m, "ci95": [lo, hi], "mcnemar_p": mcnemar_exact(bb, cc), "only_first": bb, "only_second": cc, "n": len(keys)}

    comps = {f"clean_minus_{c}": paired("clean", c) for c in stageA[1:]}
    for c in stageB:
        if c.startswith("restore"):
            comps[f"{c}_minus_{ostar}"] = paired(c, ostar)
    comps["restore_attr_minus_restore_random"] = paired("restore_attr", "restore_random")
    comps["restore_attr_minus_restore_sel"] = paired("restore_attr", "restore_sel")
    comps["clean_minus_inject_attr"] = paired("clean", "inject_attr")
    comps = {k: v for k, v in comps.items() if v}

    def recovery(c):
        keys = sorted(set(per.get(c, {})) & set(per.get(ostar, {})) & set(per.get("clean", {})))
        if not keys:
            return None
        tk = np.array([k[0] for k in keys])
        x = {nm: np.array([per[nm][k]["success"] for k in keys], dtype=float) for nm in (c, ostar, "clean")}
        ut = sorted(set(tk.tolist()))
        rng = np.random.default_rng(seed)
        def stat(idx):
            sel_ = np.concatenate([np.nonzero(tk == u)[0] for u in idx])
            den = x["clean"][sel_].mean() - x[ostar][sel_].mean()
            return (x[c][sel_].mean() - x[ostar][sel_].mean()) / den if abs(den) > 1e-9 else np.nan
        obs = stat(ut)
        boots = [stat(list(rng.choice(ut, size=len(ut), replace=True))) for _ in range(min(nb, 5000))]
        boots = np.array([b_ for b_ in boots if np.isfinite(b_)])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) > 10 else [None, None]
        return {"fraction": float(obs) if np.isfinite(obs) else None, "ci95": ci}

    recov = {c: recovery(c) for c in stageB if c.startswith("restore")}
    drop = comps.get(f"clean_minus_{ostar}")
    c6 = bool(drop and drop["ci95"][0] is not None and drop["ci95"][0] > 0)
    ra = comps.get(f"restore_attr_minus_{ostar}")
    rr = comps.get("restore_attr_minus_restore_random")
    H4 = bool(c6 and ra and rr and ra["ci95"][0] > 0 and rr["ci95"][0] > 0)
    out = {"tasks": tasks, "episodes_per_task": n_ep, "stage_A": stageA, "stage_B": stageB, "occluder": ostar,
           "occluder_family": fam_star, "occluder_rule": why, "drops": drops, "rates": rates, "comparisons": comps,
           "recovery": recov, "check6_drop_measurable": c6, "H4_restoring_attr_features_recovers_success": H4,
           "set_sizes": {"attr": _nfeat(sets[0]) if sets else 0, "sel": _nfeat(sets[1]) if sets else 0,
                         "random": _nfeat(sets[2]) if sets else 0},
           "episodes": {k: [{kk: vv for kk, vv in e.items() if kk != "states"} for e in v] for k, v in partial.items() if k != "_meta"}}
    save_json(OUT / "goal3.json", out)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.1 * len(all_conds) + 2, 3.4))
    x = np.arange(len(all_conds))
    m = [rates[c]["success"] for c in all_conds]
    lo = [max(0.0, rates[c]["success"] - rates[c]["wilson95"][0]) for c in all_conds]
    hi = [max(0.0, rates[c]["wilson95"][1] - rates[c]["success"]) for c in all_conds]
    cols = ["#1baf7a" if c == "clean" else "#8a8a8a" if c in stageA else "#2a78d6" if "attr" in c else "#eb6834" for c in all_conds]
    ax.bar(x, m, yerr=[lo, hi], capsize=3, color=cols)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in all_conds], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(f"Closed loop: {len(tasks)} tasks x {n_ep} paired episodes; stage B under '{ostar}'", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal3_success.png", dpi=130)
    plt.close(fig)
    for k, v in comps.items():
        log(f"goal3 {k}: {v['diff']:+.3f} [{v['ci95'][0]}, {v['ci95'][1]}] McNemar p={v['mcnemar_p']:.3g} (n={v['n']})")
    write_status("goal3", H4, {
        "check6": c6, "occluder": ostar, "success": {c: rates[c]["success"] for c in all_conds},
        "comparisons": comps, "recovery": recov, "tasks": tasks, "n_episodes_per_task": n_ep,
        "reason": "" if H4 else ("no measurable success drop under the occluder, so restoring cannot be tested" if not c6 else
                                 "restoring the attribution-selected features does not recover success beyond the random control"),
    })


# ============================================================================= main

COMMANDS = {"frames": cmd_frames, "validate": cmd_validate, "capture": cmd_capture, "train": cmd_train,
            "goal1": cmd_goal1, "attrib": cmd_attrib, "goal2": cmd_goal2, "goal3": cmd_goal3}
ALL_STEPS = list(COMMANDS)


def step_done(c: str) -> bool:
    st = read_status(c)
    if st is None or st.get("protocol") != PROTOCOL_VERSION or not st.get("complete"):
        return False
    if c == "frames":
        return (OUT / "frames.pkl").exists()
    if c == "capture":
        return read_status("train") is not None or (OUT / "acts" / "meta.json").exists()
    if c == "train":
        return any((OUT / "transcoders").glob("layer_*.pt"))
    if c == "attrib":
        return st.get("passed") is not False  # a failed attribution step is retried on resume
    return True


def main():
    global CFG, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("command", help="one command, a comma-separated chain, or 'all' (" + ",".join(ALL_STEPS) + ")")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="skip steps that already finished (a re-run step re-runs everything after it)")
    ap.add_argument("--redo", default="", help="comma-separated steps to re-run even with --resume")
    args = ap.parse_args()
    OUT = Path(args.out)
    CFG = load_json(OUT / "config.json")
    cmds = ALL_STEPS if args.command.strip() == "all" else [c.strip() for c in args.command.split(",") if c.strip()]
    bad = [c for c in cmds if c not in COMMANDS]
    if bad:
        raise SystemExit(f"unknown command(s) {bad}; choose from {ALL_STEPS}")
    redo = {c.strip() for c in args.redo.split(",") if c.strip()}
    if "train" in redo and not (OUT / "acts" / "meta.json").exists():
        redo.add("capture")  # the activation cache is deleted after training; re-create it
    tf = OUT / "status" / "timings.json"
    timings = (load_json(tf).get("steps", {}) if tf.exists() else {})
    t_all = time.time()
    rerun_rest = False
    for c in cmds:
        if args.resume and not rerun_rest and c not in redo and step_done(c):
            log(f"== {c}: already finished (status/{c}.json) -> skipped (--resume)")
            continue
        rerun_rest = True
        if c == "goal3" and c in redo and (OUT / "goal3_partial.json").exists():
            (OUT / "goal3_partial.json").unlink()
        log(f"== {c} | out={OUT}")
        t0 = time.time()
        try:
            COMMANDS[c]()
        except SystemExit as exc:
            msg = exc.code if isinstance(exc.code, str) else ""
            if msg.startswith("[gate]"):
                print(msg, flush=True)
                log(f"== chain stopped at {c} by the gate rule (an outcome, not a crash)")
                timings[c] = time.time() - t0
                save_json(tf, {"steps": timings, "stopped_at": c, "total_s": time.time() - t_all})
                sys.exit(3)
            raise
        timings[c] = time.time() - t0
        save_json(tf, {"steps": timings, "stopped_at": None, "total_s": time.time() - t_all})
        log(f"== {c} finished in {timings[c]:.0f}s")
        free_cuda()
    if len(cmds) > 1:
        log(f"== chain {','.join(cmds)} finished in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
