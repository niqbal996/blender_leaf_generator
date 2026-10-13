#!/usr/bin/env python
"""Apex leaves fitted to the photos: every leaf shaped until, drawn into all views, it looks like them.

    python scripts/apex_fit.py --workdir <specimen>/plant --init <plant>/debug/apex_select \
        --regions <plant>/debug/apex_labels/apex_regions.npz --zoom <plant>/debug/apex_zoom_fair \
        [--evidence <evidence_2d.json> --candidates <skeleton dir> ...] \
        --out <plant>/debug/apex_fit_v3 [--train even]

Option 3, step 3 of docs/apex-plan.md. Diagnostic -- writes into --out only.

Starts from apex_select.py's skeleton (its apex leaves are the tracks
"sel_apex_<X>_..."). Per apex, all its leaves are fitted at once.

The apex (section 3.3 of the plan): its tip O and its shoot axis a, both
fitted and shared by its leaves. O starts at the selection's node (the median
of its leaves' bases); a starts as the axis the leaves radiate around -- the
direction their tips (seen from O) vary least along, the axis of the cone
they lie on -- pointing away from the stem.

The leaf, in its shoot's frame (section 3.2 of the plan), 12 numbers:
  attach     how far below O it joins the axis, within the apex region (1;
             a share of the region's reach, so it slides freely -- clamped at
             0, as in v2, a leaf that touched the tip stuck there)
  azimuth    which way round the axis it points (1)
  elevation  the angle its base makes with the axis, 0 = along the shoot (1)
  incurve    how far the midrib turns back towards the axis from base to
             tip, 0 (straight) to 180 deg (1)
  legs       the midrib is a quadratic Bezier: base, base + leg 1 along the
             base direction, + leg 2 along the tip direction (2)
  blade      where the blade starts along the curve (1); half-width at its
             widest as a share of the blade's length, at most 1/2 (1); the
             outline t^a (1-t)^b along the blade, a the taper at the base, b
             at the tip, each from 0.5 (rounded) to 2 (pointed) (2). The stalk
             is one voxel wide.
  roll       the blade about the midrib, within +-90 deg, the same along the
             leaf: a young blade is not twisted (1)
  fold       both halves lifted towards the upper face, 0 flat, 90 deg closed (1)
Structural rules built in, none with a number: the midrib lies in the plane
through the shoot axis and the leaf (its median plane), and turns only towards
the axis (--recurve: also away from it, to 90 deg); the leaf's upper face looks
at the shoot (roll within +-90 deg), so its fold cups towards the centre; the
blade is not twisted. A young apex leaf is
symmetric about its median plane and curls in over the bud, never out. (The
version before, a free Bezier with free roll, fitted midribs turning 60 deg
median and up to 137, out of their median planes, rolls from -238 to +200
deg: spirals, blades cupped away from the centre -- the user's Blender look
of 2026-10-13. Kept as debug/apex_fit/apex_fit_v2.py.)

The bud: one more element at O, starting along the axis, closed (fold 70
deg), half as long as the shortest selected leaf -- the tiny leaf the user
sees at the centre of almost every apex. It is structure, like the stem: it is
there because the plant has one, sits at O, is fitted like a leaf, may claim a
SAM3 mask, but is never a false positive and never dropped (SAM3 seldom draws
it apart from the leaves around it; judged as a leaf, J dropped it at A, where
it is plainly there). Being the youngest pair it is no longer than the
shortest leaf of its apex (`bud` below; without it, never a false positive,
it grew into an 11 mm leaf on a leaf's mask). Written as a leaf with "bud":
true and not counted.

The renderer: each leaf is ALONG x ACROSS flat 3D Gaussians, one at the
centre of each cell of its blade, spread over the cell (covariance from the
grid's own steps, KAPPA of a step), drawn by gsplat (antialiased) into the
apex's square of every view at GRID px, depth-sorted, one colour channel per
leaf. Cell centres, not the grid's nodes: Gaussians on the edge itself drew
the leaf half a Gaussian too big all round, and the fit shrank it to match.
Cameras are pinhole on an undistorted grid; SAM3's masks and the plant mask
are resampled onto it through the SIMPLE_RADIAL distortion.

The rest of the plant, as context: every P5x leaf whose points lie mostly
outside the apex's region is drawn as its own instance (its P5x points, fixed)
-- so it can claim its SAM3 mask in the matching, and a leaf behind the apex
is no longer a target an apex leaf has to cover. It is never a false
positive (most of its mask is cut by the square's edge and dropped). The
stem and the unassigned points are one plain occluder. P5x leaves mostly
inside the region are the apex blob: dropped, the fitted leaves replace them.

The objective, with no leaf ids across views:
  masks      per view, leaves (apex and context) and zoomed SAM3 masks are
             matched one to one (Hungarian on IoU, recomputed every step, no
             gradient). Scored as instances, the panoptic-quality way: a pair
             above TP_IOU is a true positive, a visible apex leaf in no such
             pair a false positive, a mask in none a false negative; each pair
             earns its IoU, each FP and FN costs 1/2 (the F1 weights). Masks
             reaching the square's edge are leaves passing through
             (apex_zoom.py's rule): no target.
  on plant   every point of every leaf projects inside the plant mask: the
             distance outside it, in mm at the leaf, over the leaf's width (a
             leaf off by its own width is as wrong as one missing its mask).
  away       distance from the base never decreases along the midrib.
  bud        the bud's length beyond the shortest leaf's, over that length.
  J = sum over views of [IoU of the pairs - (FP + FN) / 2 - on-plant cost] / views - away - bud.

Leaf count, by the same J (section 3.8): the selection's leaves are fitted;
then, from a candidate pool, the candidate ranked best is added, everything
re-fitted, and it stays only if J went up -- the first that does not ends the
growing; then a leaf whose removal does not lower J goes. A leaf stays only if
the photos need it. The pool: each fitted leaf's partner -- the same leaf
turned half round the shoot (section 3.4: vogelmeere's leaves come in
opposite pairs at a node; a young pair often shows one leaf and hides the
other) -- and apex_select's candidates (--evidence, --candidates).

Evaluation (with --train even): the same scores on the odd frames, which the
fit never saw, for the starting leaves and the fitted ones.

Writes: skeleton.json (--init with its apex leaves replaced: one curve from
the shoot to the tip, blade_start, per leaf `blade` = attach, azimuth,
elevation, incurve, width, tapers, roll, fold, and `blade_mesh`, the fitted
surface; each apex's fitted shoot in the stem tree, `apices` = tip and axis),
fit.json (per apex: scores before/after, leaves added/dropped, the axis),
apex_<X>.jpg (photo | SAM3 masks | start | fitted, 8 views; context leaves grey).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from math import comb
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apex_select                                                      # noqa: E402
from apex_zoom import EDGE_PX, SAM_PX                                   # noqa: E402
from compare_runs import track_colour                                   # noqa: E402
from p5x_views import Run                                               # noqa: E402
from pose_estimator import stem_tree                                   # noqa: E402
from pose_estimator.fine_views import source_image                     # noqa: E402
from pose_estimator.leaf_skeleton import smooth_curve                  # noqa: E402
from skeleton_2d import resample                                        # noqa: E402

GRID = 128          # px of an apex square in the renderer (apex_select's grid)
ALONG, ACROSS = 40, 11
KAPPA = 0.6         # Gaussian spread / sample step: neighbours close into one surface
DENSE = 96          # Bezier samples before resampling by arc length
DEGREE = 2          # midrib Bezier: one arc, no inflection
MIN_MASK = 40       # SAM3 masks smaller than this at GRID are specks (apex_select); a leaf
                    # showing fewer pixels than this in a view is not seen there
TP_IOU = 0.5        # a leaf and a mask are one instance above this IoU: panoptic quality's
                    # rule (Kirillov et al. 2019), the one that makes one-to-one matches unique
ITERS, REFIT, ADD_REFIT = 600, 200, 100
LR = 0.03
START_INCURVE = np.radians(10)  # where a leaf whose start curve is straight or bent outwards starts:
                                # a start value inside the range the fit explores, not a bound
BUD_FOLD = np.radians(70)       # the bud starts closed; fitted like any fold
SAMPLE_TRACK = re.compile(r"sel_(apex_[A-Za-z0-9]+)_(\d+)_")


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------


class Views:
    """One apex in every frame: pinhole cameras on an undistorted GRID of its
    square, SAM3's apex-leaf masks and the plant mask resampled onto it."""

    def __init__(self, run: Run, zoom: Path, apex: str, frames_wanted, dev):
        self.frames, Ks, viewmats, masks, dts, self.maps, mm = [], [], [], [], [], [], []
        g = np.arange(GRID) + 0.5
        gx, gy = np.meshgrid(g, g)
        for path in sorted((zoom / "sam3").glob(f"apex{apex}_pass*.npz")):
            z = np.load(path)
            frames = [str(f) for f in z["frames"]]
            per = {}
            for key in z.files:
                if key.startswith("m_"):
                    per.setdefault(int(key.split("_")[1]), []).append(z[key])
            for i, frame in enumerate(frames):
                if frames_wanted is not None and frame not in frames_wanted:
                    continue
                x0, y0, side = (float(v) for v in z[f"crop_{i}"])
                up = float(z[f"up_{i}"])
                im = run.image_of[frame]
                cam = run.rec.cameras[im.camera_id]
                f, cx, cy, k = (float(v) for v in cam.params[:4])
                s = GRID / side
                Ks.append([[f * up * s, 0, (cx * up - x0) * s], [0, f * up * s, (cy * up - y0) * s], [0, 0, 1]])
                pose = im.cam_from_world()
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = pose.rotation.matrix(), np.asarray(pose.translation)
                viewmats.append(M)
                # undistorted grid -> distorted camera pixels (P1 frame, cam.width)
                xn = ((gx / s + x0) / up - cx) / f
                yn = ((gy / s + y0) / up - cy) / f
                fac = 1 + k * (xn * xn + yn * yn)
                u, v = f * xn * fac + cx, f * yn * fac + cy
                self.maps.append((u.astype(np.float32), v.astype(np.float32), up))
                mu = ((u * up - x0) * SAM_PX / side - 0.5).astype(np.float32)
                mv = ((v * up - y0) * SAM_PX / side - 0.5).astype(np.float32)
                ms = []
                for packed in per.get(i, []):
                    m = np.unpackbits(packed)[:SAM_PX * SAM_PX].reshape(SAM_PX, SAM_PX)
                    if (m[:EDGE_PX].any() or m[-EDGE_PX:].any() or m[:, :EDGE_PX].any()
                            or m[:, -EDGE_PX:].any()):
                        continue                       # a leaf passing through the square
                    t = cv2.remap(m.astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderValue=0)
                    if t.sum() >= MIN_MASK:
                        ms.append(t)
                masks.append(ms)
                plant = cv2.imread(str(run.workdir / "p2" / "masks" / "plant" / f"{frame}.png"), 0) > 0
                sx = plant.shape[1] / cam.width
                pm = cv2.remap(plant.astype(np.float32), (u * sx - 0.5).astype(np.float32),
                               (v * sx - 0.5).astype(np.float32), cv2.INTER_LINEAR, borderValue=0) > 0.5
                dts.append(cv2.distanceTransform((~pm).astype(np.uint8), cv2.DIST_L2, 5))
                mm.append(side / GRID / up / f)           # x depth / (units per mm) = mm at the leaf
                self.frames.append(frame)
        V = len(self.frames)
        self.Kmax = max(1, max(len(m) for m in masks))
        T = np.zeros((V, self.Kmax, GRID * GRID), np.float32)
        for v, ms in enumerate(masks):
            for j, m in enumerate(ms):
                T[v, j] = m.ravel()
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.T, self.Ks, self.viewmats = t(T), t(Ks), t(viewmats)
        self.T_area = self.T.sum(-1)
        self.dt = t(np.stack(dts))[:, None]                                  # (V, 1, G, G)
        self.px_mm = t(mm)
        self.n_masks = [len(m) for m in masks]

    def project(self, X: torch.Tensor):
        """(P, 3) world -> (V, P, 2) grid px, (V, P) depth."""
        c = torch.einsum("vij,pj->vpi", self.viewmats[:, :3, :3], X) + self.viewmats[:, None, :3, 3]
        z = c[..., 2].clamp(min=1e-6)
        uv = torch.einsum("vij,vpj->vpi", self.Ks[:, :2, :3], c) / z[..., None]
        return uv, z

    def off_plant_px(self, X: torch.Tensor):
        """(P, 3) -> (V, P) px outside the plant mask (0 inside), and depth."""
        uv, z = self.project(X)
        grid = (uv / GRID * 2 - 1)[:, :, None]                               # (V, P, 1, 2)
        d = F.grid_sample(self.dt, grid, mode="bilinear", padding_mode="border",
                          align_corners=False)[:, 0, :, 0]
        out = (F.relu(-uv) + F.relu(uv - GRID)).norm(dim=-1)                 # beyond the square
        return d + out, z


# --------------------------------------------------------------------------
# the apex and its leaves
# --------------------------------------------------------------------------


def bernstein(t):
    return torch.stack([comb(DEGREE, j) * t ** j * (1 - t) ** (DEGREE - j) for j in range(DEGREE + 1)], -1)


def logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1 - p)))


def unit_vec(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


class Apex(torch.nn.Module):
    """A pool of leaves of one apex in its shoot's frame (`active` says which
    are in the model); parameters in mm and radians, geometry in world units."""

    # outline exponents, from rounded (0.5: an ellipse) to sharply pointed (2). Chickweed
    # leaves are ovate, rounded to acute at both ends, never truncate; with 0.2 allowed,
    # a tenth of the leaves went square-ended and pulled their tips 6 px short of the
    # labelled ones (a square end cannot follow a rounded tip, so the midrib stops early)
    TAPER = (0.5, 2.0)
    # incurve: 0 (straight) .. 180 deg towards the axis; with --recurve also up to 90 deg away
    INCURVE = (0.0, np.pi)

    def __init__(self, frame, rows, unit, stalk_half, dev):
        super().__init__()
        self.unit, self.stalk_half = unit, stalk_half                      # world units per mm
        self.reach_mm = float(frame["reach_mm"])
        t = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
        p = lambda a: torch.nn.Parameter(t(a))
        self.O0, self.e1_0 = t(frame["node"]), t(frame["e1"])
        self.dO = p(np.zeros(3))                                            # mm
        self.a_raw = p(frame["axis"])
        n = len(rows)
        lo, hi = self.TAPER
        self.p_h = p([logit(np.clip(r["h_mm"] / self.reach_mm, 0.02, 0.98)) for r in rows])
        self.phi = p([r["phi"] for r in rows])
        self.p_psi = p([logit(r["psi"] / np.pi) for r in rows])
        k0, k1 = self.INCURVE
        self.p_kappa = p([logit((r["kappa"] - k0) / (k1 - k0)) for r in rows])
        self.p_legs = p([np.log(r["legs_mm"]) for r in rows])               # (n, 2)
        self.p_blade = p([logit(r["blade"]) for r in rows])
        self.p_width = p([0.0] * n)                                         # half-width 1/4 of the blade
        self.p_base = p([logit((0.7 - lo) / (hi - lo))] * n)
        self.p_tip = p([logit((0.7 - lo) / (hi - lo))] * n)
        self.p_roll = p([0.0] * n)                                          # level, upper face to the shoot
        self.p_fold = p([logit(r["fold"] / (0.5 * np.pi)) for r in rows])
        self.active = torch.tensor([bool(r["active"]) for r in rows], device=dev)
        self.bud = torch.tensor([bool(r.get("bud")) for r in rows], device=dev)

    def frame(self):
        """Tip O, axis a, and two directions e1, e2 across it (azimuth 0 and 90 deg)."""
        a = self.a_raw / self.a_raw.norm()
        e1 = self.e1_0 - (self.e1_0 @ a) * a
        e1 = e1 / e1.norm()
        return self.O0 + self.dO * self.unit, a, e1, torch.linalg.cross(a, e1)

    def shape(self, sel=slice(None)):
        lo, hi = self.TAPER
        attach = self.reach_mm * torch.sigmoid(self.p_h[sel])
        return {"attach_mm": torch.where(self.bud[sel], torch.zeros_like(attach), attach),   # the bud: at O
                "azimuth": self.phi[sel],
                "elevation": np.pi * torch.sigmoid(self.p_psi[sel]),
                "incurve": self.INCURVE[0] + (self.INCURVE[1] - self.INCURVE[0]) * torch.sigmoid(self.p_kappa[sel]),
                "legs_mm": torch.exp(self.p_legs[sel]),
                "blade": torch.sigmoid(self.p_blade[sel]),
                "half_share": 0.5 * torch.sigmoid(self.p_width[sel]),
                "a": lo + (hi - lo) * torch.sigmoid(self.p_base[sel]),
                "b": lo + (hi - lo) * torch.sigmoid(self.p_tip[sel]),
                "roll": 0.5 * np.pi * torch.tanh(self.p_roll[sel]),
                "fold": 0.5 * np.pi * torch.sigmoid(self.p_fold[sel])}

    def midrib(self, sel, m=ALONG):
        """(n, m, 3) by arc length, ends included, lengths, and each leaf's outward direction r."""
        O, a, e1, e2 = self.frame()
        sh = self.shape(sel)
        r = torch.cos(sh["azimuth"])[:, None] * e1 + torch.sin(sh["azimuth"])[:, None] * e2
        d = lambda th: torch.sin(th)[:, None] * r + torch.cos(th)[:, None] * a
        legs = sh["legs_mm"] * self.unit
        P0 = O - (sh["attach_mm"] * self.unit)[:, None] * a
        P1 = P0 + legs[:, :1] * d(sh["elevation"])
        P2 = P1 + legs[:, 1:] * d(sh["elevation"] - sh["incurve"])        # turned towards the axis
        P = torch.stack([P0, P1, P2], 1)
        t = torch.linspace(0, 1, DENSE, device=P.device)
        D = torch.einsum("tj,njc->ntc", bernstein(t), P)
        seg = (D[:, 1:] - D[:, :-1]).norm(dim=-1)
        cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
        L = cum[:, -1]
        target = torch.linspace(0, 1, m, device=P.device)[None] * L[:, None]
        idx = torch.searchsorted(cum.detach().contiguous(), target.detach().contiguous()).clamp(1, DENSE - 1)
        lo, hi = cum.gather(1, idx - 1), cum.gather(1, idx)
        w = ((target - lo) / (hi - lo).clamp(min=1e-12)).clamp(0, 1)[..., None]
        g = lambda k: D.gather(1, k[..., None].expand(-1, -1, 3))
        return g(idx - 1) * (1 - w) + g(idx) * w, L, r

    def surface(self, sel, cells=True):
        """Leaves `sel`: (n, ALONG, ACROSS, 3) blade points, (n, ALONG, 3) midrib, lengths, half-widths.

        cells: each point is the centre of its own ALONG x ACROSS cell of the
        blade, so the Gaussians drawn on them cover the blade up to its edge and
        no further. (Points on the edge itself drew the leaf ~1 render px too
        big all round -- half a Gaussian beyond the edge -- and the fit shrank
        it to match: tips came out 4 px short of the labelled ones.) Else the
        points are the grid's nodes, edge included: the surface itself."""
        if cells:
            Cn, L, r = self.midrib(sel, 2 * ALONG + 1)
            C = Cn[:, 1::2]
            s = (torch.arange(ALONG, device=Cn.device) + 0.5) / ALONG
            u = -1 + (2 * torch.arange(ACROSS, device=Cn.device) + 1) / ACROSS
        else:
            C, L, r = self.midrib(sel)
            s = torch.linspace(0, 1, ALONG, device=C.device)
            u = torch.linspace(-1, 1, ACROSS, device=C.device)
        a = self.frame()[1]
        across = torch.linalg.cross(a.expand_as(r), r)                     # the blade's width, roll 0
        T = torch.gradient(C, dim=1)[0]
        T = T / T.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        upper = torch.linalg.cross(T, across[:, None].expand_as(T))         # the upper face, to the axis
        sh = self.shape(sel)
        b, a_, e, fold = sh["blade"][:, None], sh["a"][:, None], sh["b"][:, None], sh["fold"]
        th = sh["roll"][:, None].expand(-1, len(s))
        W = torch.cos(th)[..., None] * across[:, None] + torch.sin(th)[..., None] * upper
        Nd = torch.linalg.cross(T, W)
        tb = ((s[None] - b) / (1 - b)).clamp(1e-4, 1 - 1e-4)
        peak = a_ / (a_ + e)
        outline = (tb ** a_ * (1 - tb) ** e) / (peak ** a_ * (1 - peak) ** e)
        blade = (s[None] > b).float() * outline
        half = self.stalk_half + sh["half_share"][:, None] * (1 - b) * L[:, None] * blade
        side = torch.cos(fold)[:, None, None, None] * torch.sign(u)[None, None, :, None] * W[:, :, None] \
            + torch.sin(fold)[:, None, None, None] * Nd[:, :, None]
        X = C[:, :, None] + half[..., None, None] * u.abs()[None, None, :, None] * side
        return X, C, L, half

    def covariances(self, X: torch.Tensor):
        """Each blade point spread over its grid steps."""
        a = torch.gradient(X, dim=1)[0]
        b = torch.gradient(X, dim=2)[0]
        eps = 1e-3 * self.unit
        cov = KAPPA ** 2 * (a[..., :, None] * a[..., None, :] + b[..., :, None] * b[..., None, :])
        return cov + eps ** 2 * torch.eye(3, device=X.device)


# --------------------------------------------------------------------------
# the rest of the plant
# --------------------------------------------------------------------------


def rows_in(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Which rows of `a` are rows of `b` (the regions are carved out of the same cloud)."""
    v = lambda x: np.ascontiguousarray(x, np.float64).view(np.dtype((np.void, 24))).ravel()
    return np.isin(v(a), v(b))


def context(run: Run, views: Views, in_region: np.ndarray, dev):
    """The plant around the apex as the renderer sees it: the points outside the apex's
    region in its squares, each P5x leaf mostly outside the region its own channel
    0..n-1, the rest (stem, unassigned) channel n, a plain occluder. P5x leaves mostly
    inside the region are the apex blob the fitted leaves replace: dropped."""
    lab = run.assignment
    ids = np.unique(lab[lab >= 0])
    share = {int(i): float(in_region[lab == i].mean()) for i in ids}
    outside = [i for i in ids if share[int(i)] <= 0.5]
    keep = ~in_region & ((lab < 0) | np.isin(lab, outside))
    X = torch.tensor(run.points[keep], dtype=torch.float32, device=dev)
    L = lab[keep]
    uv, _ = views.project(X)
    seen = ((uv >= 0) & (uv < GRID)).all(-1).any(0).cpu().numpy()
    X, L = X[torch.from_numpy(seen).to(dev)], L[seen]
    present = sorted({int(i) for i in L if i >= 0})
    chan = {i: j for j, i in enumerate(present)}
    channel = np.array([chan.get(int(i), len(present)) for i in L], np.int64)
    cov = (KAPPA * run.voxel) ** 2 * torch.eye(3, device=dev).expand(len(X), 3, 3).contiguous()
    return SimpleNamespace(X=X, cov=cov, channel=torch.tensor(channel, device=dev), n=len(present),
                           labels=present, dropped=sorted(int(i) for i in ids if share[int(i)] > 0.5))


# --------------------------------------------------------------------------
# render and score
# --------------------------------------------------------------------------


def rasterize(X, cov, channel, n_ch, views):
    """Points X (P, 3) with covariances, each drawn into one of n_ch channels."""
    import gsplat
    col = torch.zeros(len(X), n_ch, device=X.device)
    col[torch.arange(len(X), device=X.device), channel] = 1
    img, _alpha, _ = gsplat.rasterization(
        X, None, None, torch.full((len(X),), 0.99, device=X.device), col, views.viewmats, views.Ks,
        GRID, GRID, covars=cov, near_plane=1e-3, packed=False,
        rasterize_mode="antialiased")              # the low-pass dims, rather than grows, small footprints
    return img.reshape(len(views.frames), GRID * GRID, n_ch)


def render_with(X, cov, n, views, ctx):
    """Apex leaves' blade points X (n, ..., 3) and the context: (V, G*G, n) visible weight of each
    apex leaf, (V, G*G, ctx.n) of each context leaf."""
    m = X[0].numel() // 3 if n else 0
    ch_leaf = torch.arange(n, device=ctx.X.device).repeat_interleave(m)
    img = rasterize(torch.cat([X.reshape(-1, 3), ctx.X]), torch.cat([cov.reshape(-1, 3, 3), ctx.cov]),
                    torch.cat([ch_leaf, n + ctx.channel]), n + ctx.n + 1, views)
    return img[..., :n], img[..., n:n + ctx.n]


def render(model: Apex, views: Views, ctx, keep):
    """(V, G*G, k) visible weight of each kept leaf, (V, G*G, c) of each context leaf, and the
    kept leaves' geometry."""
    idx = torch.nonzero(keep).flatten()
    X, C, L, half = model.surface(idx)
    Pa, Pc = render_with(X, model.covariances(X), len(idx), views, ctx)
    return Pa, Pc, (X, C, L, half)


def match(Pa: torch.Tensor, Pc: torch.Tensor, views: Views):
    """Per view, leaves (apex, then context) and SAM3 masks matched one to one (Hungarian on
    IoU): IoU (V, n + c, K), the matched pairs (V, n + c, K), and which apex leaves are
    visible there (V, n) -- at least a SAM3 speck's worth of pixels, the rule the masks pass."""
    from scipy.optimize import linear_sum_assignment
    P = torch.cat([Pa, Pc], -1)
    inter = torch.einsum("vxn,vkx->vnk", P, views.T)
    area = P.sum(1)
    iou = inter / (area[..., None] + views.T_area[:, None] - inter).clamp(min=1e-6)
    I = iou.detach().cpu().numpy()
    pairs = np.zeros(I.shape, bool)
    for v, k in enumerate(views.n_masks):
        if k:
            r, c = linear_sum_assignment(-I[v, :, :k])
            ok = I[v, r, c] > 0
            pairs[v, r[ok], c[ok]] = True
    return iou, torch.from_numpy(pairs).to(P.device), Pa.sum(1) >= MIN_MASK


def counts(iou, pairs, visible, views, structure=None):
    """Per view: true positives (a pair above TP_IOU, apex or context leaf), of those by
    context leaves, false positives (visible apex leaves in none, the bud -- `structure` --
    excepted), false negatives (masks in none)."""
    n = visible.shape[1]
    tp = pairs & (iou.detach() > TP_IOU)
    n_tp = tp.sum((1, 2))
    judged = visible if structure is None else visible & ~structure[None]
    fp = (judged & ~tp[:, :n].any(-1)).sum(-1)
    fn = torch.tensor(views.n_masks, device=iou.device) - n_tp
    return n_tp, fp, fn, tp, tp[:, n:].sum((1, 2))


def objective(model, views, ctx, keep=None):
    keep = model.active if keep is None else keep
    Pa, Pc, (X, C, L, half) = render(model, views, ctx, keep)
    iou, pairs, visible = match(Pa, Pc, views)
    n_tp, fp, fn, _, tp_ctx = counts(iou, pairs, visible, views, model.bud[keep])
    n, V = X.shape[0], len(views.frames)
    d_px, z = views.off_plant_px(X.reshape(-1, 3))
    d_mm = d_px * z * views.px_mm[:, None] / model.unit
    width_mm = (2 * half.amax(1) / model.unit).detach()
    off = d_mm.view(V, n, -1).mean(-1) / width_mm[None]
    dist = (C - C[:, :1]).norm(dim=-1)
    away = (F.relu(dist[:, :-1] - dist[:, 1:]).sum(1) / L).sum()
    bud = model.bud[keep]
    over = torch.zeros((), device=L.device)
    if bud.any() and (~bud).any():
        shortest = L[~bud].min().detach()
        over = (F.relu(L[bud] - shortest) / shortest).sum()
    matched = (iou * pairs).sum()
    J = (matched - 0.5 * (fp.sum() + fn.sum()) - off.sum()) / V - away - over
    return J, {"iou_per_view": matched.item() / V, "tp": n_tp.sum().item() / V,
               "tp_context": tp_ctx.sum().item() / V, "fp": fp.sum().item() / V,
               "fn": fn.sum().item() / V, "off_plant_per_view": off.sum().item() / V, "away": away.item(),
               "bud_over": over.item()}


def scores(Pa, Pc, views, X, structure=None):
    """Label-free scores of drawn leaves on these views: the instance scores
    (F1 = 2TP / (2TP + FP + FN), SQ = mean IoU of the true positives, PQ = SQ x F1;
    TP of apex and context leaves alike, so a context leaf on its mask is no FN),
    the apex leaves' own true positives, the share of SAM3 apex-mask pixels some apex
    leaf covers, and leaf points on the plant."""
    iou, pairs, visible = match(Pa, Pc, views)
    n_tp, fp, fn, tp, tp_ctx = counts(iou, pairs, visible, views, structure)
    TP, FP, FN, TPc = int(n_tp.sum()), int(fp.sum()), int(fn.sum()), int(tp_ctx.sum())
    f1 = 2 * TP / max(1, 2 * TP + FP + FN)
    sq = float(iou[tp].mean()) if TP else 0.0
    covered = (Pa.sum(-1)[:, None] * views.T).sum(-1)
    d_px, _ = views.off_plant_px(X.reshape(-1, 3))
    return {"views": len(views.frames), "tp": TP, "tp_apex": TP - TPc, "tp_context": TPc, "fp": FP, "fn": FN,
            "f1": round(f1, 4), "sq": round(sq, 4), "pq": round(sq * f1, 4),
            "pixel_recall": round(float(covered.sum() / views.T.sum().clamp(min=1)), 4),
            "on_plant": round(float((d_px < 1).float().mean()), 4)}, (iou, pairs, visible, tp)


@torch.no_grad()
def evaluate(model, views, ctx, keep):
    Pa, Pc, (X, *_rest) = render(model, views, ctx, keep)
    out, _ = scores(Pa, Pc, views, X, model.bud[keep])
    out["leaves"] = int((keep & ~model.bud).sum())
    out["bud"] = bool((keep & model.bud).any())
    return out


def fit(model, views, ctx, iters, log, tag):
    opt = torch.optim.Adam([
        {"params": [model.dO], "lr": LR},                                                   # mm
        {"params": [model.p_legs], "lr": LR / 3},                                           # log mm
        {"params": [model.p_h, model.phi, model.p_psi, model.p_kappa, model.a_raw, model.p_roll], "lr": LR},
        {"params": [model.p_blade, model.p_width, model.p_base, model.p_tip, model.p_fold], "lr": 2 * LR}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=LR * 0.05)
    for it in range(iters):
        opt.zero_grad()
        J, parts = objective(model, views, ctx)
        (-J).backward()
        opt.step()
        sched.step()
        if log and (it % 200 == 0 or it == iters - 1):
            log(f"      {tag} {it:4d}  J {J.item():7.3f}  per view: matched IoU {parts['iou_per_view']:.2f}  "
                f"TP {parts['tp']:.1f} (context {parts['tp_context']:.1f}) FP {parts['fp']:.1f} "
                f"FN {parts['fn']:.1f}  off-plant {parts['off_plant_per_view']:.3f}  away {parts['away']:.3f}")


@torch.no_grad()
def best_change(model, views, ctx, options):
    """The option (index to toggle) that raises J most, and by how much."""
    J0, _ = objective(model, views, ctx)
    best, who = -np.inf, None
    for i in options:
        keep = model.active.clone()
        keep[i] = ~keep[i]
        if not keep.any():
            continue
        dJ = objective(model, views, ctx, keep)[0].item() - J0.item()
        if dJ > best:
            best, who = dJ, i
    return who, best, J0.item()


def grow_pairs(model, views, ctx, rows, log):
    """Each fitted leaf's partner proposal (pair_up), judged on its own, the best-ranked by J
    as it starts first: added, everything re-fitted, kept if J went up, else taken out again.
    (Ranked among all candidates, a partner was never reached: the first candidate tried was
    a geometry one, it did not raise J, and that ended the growing.)"""
    with torch.no_grad():
        J0 = objective(model, views, ctx)[0].item()
        gain = {}
        for i, r in enumerate(rows):
            if r.get("partner_of") is not None and model.active[r["partner_of"]] and not model.active[i]:
                keep = model.active.clone()
                keep[i] = True
                gain[i] = objective(model, views, ctx, keep)[0].item() - J0
    added = []
    for i in sorted(gain, key=lambda i: -gain[i]):
        with torch.no_grad():
            J0 = objective(model, views, ctx)[0].item()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.active[i] = True
        fit(model, views, ctx, ADD_REFIT, None, "pair")
        with torch.no_grad():
            J1 = objective(model, views, ctx)[0].item()
        if J1 <= J0:
            model.load_state_dict(before)
            model.active[i] = False
            log(f"      x partner of leaf {rows[i]['partner_of']}: J {J0:.3f} -> {J1:.3f} fitted: not kept")
        else:
            added.append(i)
            log(f"      + partner of leaf {rows[i]['partner_of']}: J {J0:.3f} -> {J1:.3f} fitted")
    return added


def grow(model, views, ctx, rows, log):
    """Add candidates while one raises J. The candidate ranked best by J as it starts is added
    and everything re-fitted; it stays if the fitted J is higher than before it, else it goes
    and the growing stops. A raw candidate rarely reaches a true positive by itself, so judging
    it unfitted turned away real leaves."""
    added, used = [], set()
    while True:
        options = [i for i, r in enumerate(rows) if r.get("group") is not None and r.get("partner_of") is None
                   and r["group"] not in used and not model.active[i]]
        who, dJ, J0 = best_change(model, views, ctx, options)
        if who is None:
            return added
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.active[who] = True
        used.add(rows[who]["group"])
        fit(model, views, ctx, ADD_REFIT, None, "add")
        with torch.no_grad():
            J1 = objective(model, views, ctx)[0].item()
        if J1 <= J0:
            model.load_state_dict(before)
            model.active[who] = False
            log(f"      x {rows[who]['source']:<20s} J {J0:.3f} -> {J1:.3f} fitted: not kept, growing stops")
            return added
        added.append(who)
        log(f"      + {rows[who]['source']:<20s} J {J0:.3f} -> {J1:.3f} fitted")


def prune(model, views, ctx, rows, log):
    """Drop leaves while dropping one does not lower J, the one that helps most first (the bud
    is structure: never)."""
    dropped = []
    while (model.active & ~model.bud).sum() > 1:
        who, dJ, J0 = best_change(model, views, ctx, torch.nonzero(model.active & ~model.bud).flatten().tolist())
        if who is None or dJ < 0:
            return dropped
        model.active[who] = False
        dropped.append(who)
        log(f"      - leaf {who} ({rows[who]['source']}): J {J0:.3f} -> {J0 + dJ:.3f} without it")
    return dropped


# --------------------------------------------------------------------------
# leaves from curves, in the shoot's frame
# --------------------------------------------------------------------------


def trunk_lines(run: Run, skeleton: dict):
    """The stem tree's axes (no petioles) in world."""
    return [run.to_world(a["points"]) for a in skeleton.get("axes") or []
            if a["kind"] != "petiole" and len(a["points"]) >= 2]


def apex_frame(node, tips, lines, reach, unit):
    """The apex's starting frame: tip O = node, and the axis. An apex is the end of a shoot, so
    its axis continues the shoot's last stretch: the direction, over its last `reach`, of the
    stem or branch of P5x's tree (`lines`, ended by end_axes where the cloud closes over the
    bud) whose end is nearest the node, if that end lies within the apex's reach of it. Else
    the axis the leaves radiate round: the direction their tips, seen from O, vary least along
    (the axis of the cone they lie on), turned away from the nearest stem. (The tree apex_select
    hands on is no guide here: it joins each node to the nearest trunk, sideways where the node
    is beside it -- 47-90 deg off on B and D. The cone alone was 61-82 deg off on A and D,
    whose leaves stand at 3-80 deg to the axis.)"""
    ends = [(float(np.linalg.norm(line[-1] - node)), line) for line in lines]
    d_end, line = min(ends, key=lambda e: e[0])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    back = np.array([np.interp(max(cum[-1] - reach, 0.0), cum, line[:, d]) for d in range(3)])
    shoot = unit_vec(line[-1] - back)
    U = np.stack([unit_vec(t - node) for t in tips])
    cone = np.linalg.eigh(np.cov(U.T))[1][:, 0] if len(U) >= 3 else shoot
    cone = cone if cone @ shoot >= 0 else -cone
    a = shoot if d_end <= reach else cone
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return {"node": node, "axis": a, "e1": unit_vec(np.cross(a, ref)), "reach_mm": reach / unit,
            "from": "shoot end" if d_end <= reach else "leaf cone", "shoot_end_mm": d_end / unit,
            "cone_vs_shoot_deg": float(np.degrees(np.arccos(np.clip(cone @ shoot, -1, 1))))}


def to_row(curve, frame, unit, blade, **extra):
    """A world curve, base first -> the shoot-frame numbers of a leaf starting like it.

    Its base joins the axis where the curve's base end projects onto it; its azimuth is its
    tip's; the rest is read in its median plane (the curve's part out of that plane is
    dropped -- the rule the model keeps): elevation from the first quarter's direction,
    incurve from the turn to the last quarter's, the legs so that the tip lands on the curve's."""
    O, a, e1 = frame["node"], frame["axis"], frame["e1"]
    e2 = np.cross(a, e1)
    X = resample(curve, 32)
    h = float(np.clip(-((X[0] - O) @ a) / unit, 0.0, frame["reach_mm"]))
    v = X[-1] - O
    v = v - (v @ a) * a
    phi = float(np.arctan2(v @ e2, v @ e1))
    r = np.cos(phi) * e1 + np.sin(phi) * e2
    xz = np.stack([(X - O) @ r, (X - O) @ a], 1) / unit                # mm, in the median plane
    xz[0] = (0.0, -h)
    ang = lambda d: float(np.arctan2(d[0], d[1]))                       # from the axis, + outwards
    q = len(xz) // 4
    th0, th1 = ang(xz[q] - xz[0]), ang(xz[-1] - xz[-1 - q])
    kap = th0 - th1
    lo = Apex.INCURVE[0] + START_INCURVE
    if kap < lo:
        kap = lo
        th0 = ang(xz[-1] - xz[0]) + kap / 2
    kap = min(kap, Apex.INCURVE[1] - np.radians(10))
    th0 = float(np.clip(th0, 0.02, np.pi - 0.02))
    D = np.array([[np.sin(th0), np.sin(th0 - kap)], [np.cos(th0), np.cos(th0 - kap)]])
    chord = xz[-1] - xz[0]
    try:
        legs = np.linalg.solve(D, chord)
    except np.linalg.LinAlgError:
        legs = np.array([-1.0, -1.0])
    if (legs <= 0.05 * np.linalg.norm(chord)).any():
        legs = np.full(2, np.linalg.norm(chord) / (2 * np.cos(kap / 2)))
    return dict(extra, h_mm=h, phi=phi, psi=th0, kappa=float(kap), legs_mm=legs.tolist(),
                blade=float(np.clip(blade, apex_select.STALK, 0.9)), fold=float(np.radians(4)))


def pool(run, skeleton, frame, name, unit, cands, sel_curves):
    """apex_select's leaves of this apex (active), the bud (active), each selected leaf's
    partner (inactive; its numbers are set from the fitted leaf by pair_up), then the
    candidate pool (inactive, one row per candidate: every roll starts level here)."""
    rows = []
    for curve, leaf in sel_curves:
        blade = (leaf.get("blade_start") or 0) / max(len(leaf["curve"]) - 1, 1)
        rows.append(to_row(curve, frame, unit, blade, active=True, source=leaf.get("source"),
                           track=leaf["track"]))
    for i in range(len(rows)):
        rows.append(dict(rows[i], phi=rows[i]["phi"] + np.pi, active=False, group=f"pair_{i}",
                         source="pair", partner_of=i, track=None))
    shortest = min(sum(r["legs_mm"]) for r in rows)
    rows.append(dict(h_mm=0.0, phi=0.0, psi=float(START_INCURVE), kappa=float(START_INCURVE),
                     legs_mm=[shortest / 4, shortest / 4], blade=0.05, fold=float(BUD_FOLD),
                     active=True, bud=True, source="bud", track=f"bud_{name}"))
    node = frame["node"]
    seen = set()
    for c in cands:
        if c["group"] in seen:
            continue
        seen.add(c["group"])
        P = resample(c["curve"], 24)
        if np.linalg.norm(P[-1] - node) < np.linalg.norm(P[0] - node):
            P = P[::-1]                                  # its base is the end nearer the node
        on_axis = node + ((P[0] - node) @ frame["axis"]) * frame["axis"]
        gap = np.linalg.norm(P[0] - on_axis)
        stalk = gap / max(gap + np.linalg.norm(np.diff(P, axis=0), axis=1).sum(), 1e-12)
        rows.append(to_row(P, frame, unit, stalk, active=False, group=c["group"], source=c["source"]))
    return rows


PER_LEAF = ("p_h", "phi", "p_psi", "p_kappa", "p_legs", "p_blade", "p_width", "p_base", "p_tip", "p_roll",
            "p_fold")


@torch.no_grad()
def pair_up(model, rows):
    """Each fitted leaf's partner proposal (section 3.4): the same leaf, turned half round the
    shoot -- the other leaf of its opposite pair."""
    for k, r in enumerate(rows):
        i = r.get("partner_of")
        if i is None:
            continue
        for name in PER_LEAF:
            getattr(model, name)[k] = getattr(model, name)[i]
        model.phi[k] = model.phi[i] + np.pi


# --------------------------------------------------------------------------
# pictures
# --------------------------------------------------------------------------


def montage(path: Path, run: Run, views: Views, P0, P1, Pc, mids, c0, c1, n=8, title=""):
    tiles, S = [], 320
    up_ = lambda a: cv2.resize(a.reshape(GRID, GRID, 3), (S, S), interpolation=cv2.INTER_NEAREST)

    def paint(Pm, colours, ctx=None):
        out = np.zeros((GRID * GRID, 3), np.float32)
        if ctx is not None:
            out += ctx.sum(-1, keepdims=True) * np.float32(90)
        for j, col in enumerate(colours):
            out += Pm[:, j:j + 1] * np.asarray(col, np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    for i in np.linspace(0, len(views.frames) - 1, n).round().astype(int):
        frame = views.frames[i]
        u, v, up = views.maps[i]
        photo = cv2.imread(str(source_image(run.workdir, frame)))
        mu = cv2.resize(u, (S, S), interpolation=cv2.INTER_LINEAR) * up - 0.5
        mv = cv2.resize(v, (S, S), interpolation=cv2.INTER_LINEAR) * up - 0.5
        img = cv2.remap(photo, mu.astype(np.float32), mv.astype(np.float32), cv2.INTER_AREA)
        lab = np.zeros((GRID * GRID, 3), np.uint8)
        T = views.T[i].cpu().numpy()
        for k in np.argsort(-T.sum(1)):
            if T[k].sum() > 0:
                lab[T[k] > 0.5] = track_colour(int(k) + 3)
        ctx = Pc[i].cpu().numpy()
        fit_img = cv2.addWeighted(img, 0.45, up_(paint(P1[i].cpu().numpy(), c1, ctx)), 0.55, 0)
        for uv, col in mids[i]:
            line = (uv * S / GRID).astype(np.int32)
            cv2.polylines(fit_img, [line], False, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.polylines(fit_img, [line], False, col, 1, cv2.LINE_AA)
            cv2.circle(fit_img, tuple(line[-1]), 3, col, -1)
        row = np.hstack([img, cv2.addWeighted(img, 0.4, up_(lab), 0.6, 0),
                         cv2.addWeighted(img, 0.45, up_(paint(P0[i].cpu().numpy(), c0, ctx)), 0.55, 0), fit_img])
        cv2.putText(row, f"{frame}: photo | SAM3 apex masks | start | fitted + midribs (grey: other leaves)",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(row)
    img = np.vstack(tiles)
    if title:
        cv2.rectangle(img, (0, img.shape[0] - 24), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, title, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True, help="apex_select.py output dir (skeleton.json)")
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--zoom", type=Path, required=True, help="apex_zoom.py --regions output dir")
    ap.add_argument("--evidence", type=Path, help="evidence_2d.json: geometry candidates (apex_select)")
    ap.add_argument("--candidates", type=Path, nargs="*", default=[],
                    help="skeleton dirs whose apex leaves are candidates too (apex_select)")
    ap.add_argument("--train", choices=("all", "even"), default="all",
                    help="frames to fit on; 'even' scores the odd ones as held out")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--recurve", action="store_true",
                    help="let a midrib also bend away from the shoot (to 90 deg), not only towards it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.recurve:
        Apex.INCURVE = (-0.5 * np.pi, np.pi)
    dev = "cuda"
    torch.manual_seed(0)
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    unit = 1.0 / mpu                                                        # world units per mm
    run = Run(args.workdir)
    regions = dict(np.load(args.regions))
    skeleton = json.loads((args.init / "skeleton.json").read_text())
    lines = trunk_lines(run, run.skeleton)                                   # P5x's own tree
    log = lambda s: print(s, flush=True)
    num = lambda f: int(f.split("_")[-1])
    frames = sorted(run.image_of)
    train = None if args.train == "all" else {f for f in frames if num(f) % 2 == 0}
    test = None if args.train == "all" else {f for f in frames if num(f) % 2 == 1}
    cands = {}
    if args.evidence or args.candidates:
        cands = apex_select.candidates(run, regions, apex_select.region_reach(regions), args.evidence,
                                       args.candidates, mpu)

    report, fitted = {}, {}
    for name in sorted(regions):
        apex = name.replace("apex_", "")
        pts = regions[name]
        centre = pts.mean(axis=0)
        reach = float(np.linalg.norm(pts - centre, axis=1).max())
        t0 = time.time()
        sel = []
        for leaf in skeleton["leaves"]:
            m = SAMPLE_TRACK.match(str(leaf.get("track", "")))
            if m and m.group(1) == name:
                sel.append((run.to_world(np.asarray(leaf["curve"], float)), leaf))
        if not sel:
            continue
        node = np.median(np.stack([c[0] for c, _ in sel]), axis=0)
        frame = apex_frame(node, [c[-1] for c, _ in sel], lines, reach, unit)
        rows = pool(run, skeleton, frame, name, unit, cands.get(name, []), sel)
        views = Views(run, args.zoom, apex, train, dev)
        ctx = context(run, views, rows_in(run.points, pts), dev)
        model = Apex(frame, rows, unit, 0.5 * run.voxel, dev)
        n_start = int((model.active & ~model.bud).sum())
        log(f"  {name}: {n_start} leaves from the selection + bud, {n_start} partners and "
            f"{len(rows) - 2 * n_start - 1} candidates in the pool, "
            f"{len(views.frames)} views, {sum(views.n_masks)} SAM3 apex masks; context: {ctx.n} P5x leaves "
            f"as instances, {len(ctx.dropped)} replaced (mostly in the region), {len(ctx.X)} points; "
            f"axis from the {frame['from']} ({frame['shoot_end_mm']:.1f} mm from the node; the leaf cone "
            f"{frame['cone_vs_shoot_deg']:.0f} deg off it)")
        start_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        start_keep = model.active.clone()
        before = evaluate(model, views, ctx, start_keep)
        fit(model, views, ctx, args.iters, log, "fit")
        pair_up(model, rows)
        added = grow_pairs(model, views, ctx, rows, log)
        added += grow(model, views, ctx, rows, log)
        dropped = prune(model, views, ctx, rows, log)
        fit(model, views, ctx, REFIT, log, "final")
        keep = model.active.clone()
        after = evaluate(model, views, ctx, keep)
        with torch.no_grad():
            O, a, *_ = model.frame()
        turned = float(np.degrees(np.arccos(np.clip(float(a.cpu().numpy() @ frame["axis"]), -1, 1))))
        rep = {"tip_moved_mm": round(float(model.dO.detach().norm()), 2), "leaves_start": n_start, "leaves_added": len(added),
               "added_from": [rows[i]["source"] for i in added], "leaves_dropped": len(dropped),
               "leaves_kept": after["leaves"], "bud_kept": after["bud"], "fit_views_start": before,
               "fit_views_fitted": after, "context_leaves": ctx.labels, "replaced_p5x_leaves": ctx.dropped,
               "axis_start": frame["from"], "cone_vs_shoot_deg": round(frame["cone_vs_shoot_deg"], 1), "axis_turned_deg": round(turned, 1),
               }
        say = lambda tag, b, a: log(
            f"    {tag}: PQ {b['pq']:.3f} -> {a['pq']:.3f} (F1 {b['f1']:.3f} -> {a['f1']:.3f}, "
            f"SQ {b['sq']:.3f} -> {a['sq']:.3f}; TP {b['tp']} -> {a['tp']} (apex {b['tp_apex']} -> "
            f"{a['tp_apex']}), FP {b['fp']} -> {a['fp']}, FN {b['fn']} -> {a['fn']}), mask pixels covered "
            f"{b['pixel_recall']:.3f} -> {a['pixel_recall']:.3f}, on plant {b['on_plant']:.3f} -> "
            f"{a['on_plant']:.3f}, leaves {b['leaves']} -> {a['leaves']}, bud {a['bud']}")
        say("fit views", before, after)
        log(f"    axis turned {turned:.0f} deg in the fit, tip moved {rep['tip_moved_mm']} mm")
        fitted_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if test is not None:
            tv = Views(run, args.zoom, apex, test, dev)
            tctx = context(run, tv, rows_in(run.points, pts), dev)
            model.load_state_dict(start_state)
            held_b = evaluate(model, tv, tctx, start_keep)
            model.load_state_dict(fitted_state)
            held_a = evaluate(model, tv, tctx, keep)
            rep["held_out_start"], rep["held_out_fitted"] = held_b, held_a
            say(f"HELD-OUT ({len(tv.frames)} odd frames)", held_b, held_a)
            del tv, tctx

        with torch.no_grad():
            model.load_state_dict(start_state)
            P0, _, _ = render(model, views, ctx, start_keep)
            model.load_state_dict(fitted_state)
            P1, Pc, (_, C, _, _) = render(model, views, ctx, keep)
            idx = torch.nonzero(keep).flatten()
            X, Cn, L, half = model.surface(idx, cells=False)              # the surface itself
            idx = idx.tolist()
            c1 = [track_colour(j) for j in range(len(idx))]
            c0 = [track_colour(j) for j in range(int(start_keep.sum()))]
            uv, _ = views.project(C.reshape(-1, 3))
            uv = uv.view(len(views.frames), len(idx), ALONG, 2).cpu().numpy()
            mids = [[(uv[v, j], c1[j]) for j in range(len(idx))] for v in range(len(views.frames))]
            montage(args.out / f"{name}.jpg", run, views, P0, P1, Pc, mids, c0, c1,
                    title=f"{name}: start {n_start} leaves + bud (selection), fitted {after['leaves']} "
                          f"(+{len(added)} -{rep['leaves_dropped']}), bud {'kept' if after['bud'] else 'dropped'}; "
                          f"colour = each leaf where it is visible, grey = other leaves")
            sh = {k: v[keep].cpu().numpy() for k, v in model.shape().items()}
            fitted[name] = {"rows": [rows[i] for i in idx], "X": X.cpu().numpy(), "C": Cn.cpu().numpy(),
                            "L": L.cpu().numpy(), "half": half.cpu().numpy(), "shape": sh, "colours": c1,
                            "O": O.cpu().numpy(), "axis": a.cpu().numpy()}
        rep["seconds"] = round(time.time() - t0, 1)
        report[name] = rep
        log(f"    {rep['seconds']:.0f}s")
        del views, ctx, model
        torch.cuda.empty_cache()

    # the result: --init with its apex leaves replaced by the fitted ones, and each apex's
    # fitted shoot in the stem tree: from the nearest stem or branch point to its lowest leaf,
    # then up its axis to its tip; every axis then ends at its last attachment (end_axes)
    to_plant = lambda X: (np.asarray(X) - run.origin) @ run.rotation.T
    out = [l for l in skeleton["leaves"] if not SAMPLE_TRACK.match(str(l.get("track", "")))]
    axes = [dict(x) for x in skeleton.get("axes") or []]
    apices = {}
    deg = lambda x: round(float(np.degrees(x)), 1)
    for name, f in fitted.items():
        sh = f["shape"]
        tip, axis = to_plant(f["O"]), f["axis"] @ run.rotation.T
        low = tip - float(sh["attach_mm"].max()) * unit * axis
        trunks = [i for i, x in enumerate(axes) if x["kind"] != "petiole" and len(x["points"]) >= 2]
        (_, near, d), i = min(((stem_tree._on_polyline(np.asarray(axes[i]["points"], float), low), i)
                               for i in trunks), key=lambda h: h[0][2])
        shoot = np.vstack([near, low, tip] if d > 0 else [low, tip])
        axes.append({"id": len(axes), "order": axes[i]["order"] + 1, "parent": i, "kind": "branch",
                     "leaf": None, "apex": name, "points": shoot.round(6).tolist()})
        apices[name] = {"tip": tip.round(6).tolist(), "axis": axis.round(6).tolist(),
                        "leaves": int(sum(not r.get("bud") for r in f["rows"])),
                        "bud": any(r.get("bud") for r in f["rows"])}
        for j, r in enumerate(f["rows"]):
            curve = to_plant(f["C"][j])
            b = float(sh["blade"][j])
            bi = int(round(b * (ALONG - 1)))
            mesh = to_plant(f["X"][j].reshape(-1, 3))
            a_, e_ = float(sh["a"][j]), float(sh["b"][j])
            track = f"fit_{name}_bud" if r.get("bud") else f"fit_{name}_{j}_{r.get('source')}"
            out.append({
                "id": None, "track": track, "apex": name, "bud": bool(r.get("bud")),
                "sam3_ids": [f"fit_{name}_bud" if r.get("bud") else f"fit_{name}_{j}"],
                "source": r.get("source"),
                "tip": curve[-1].tolist(), "base": curve[bi].tolist(), "curve": curve.round(6).tolist(),
                "blade_start": bi, "midrib": curve[bi:].round(6).tolist(),
                "petiole": curve[:bi + 1].round(6).tolist() if bi > 0 else [],
                "starts_on": "stem", "colour_bgr": list(f["colours"][j]),
                "midrib_mm": float(f["L"][j] * (1 - b) * mpu),
                "blade": {"length_mm": round(float(f["L"][j] * (1 - b) * mpu), 3),
                          "width_mm": round(float(2 * f["half"][j].max() * mpu), 3),
                          "widest_at": round(a_ / (a_ + e_), 3), "base_taper": round(a_, 3),
                          "tip_taper": round(e_, 3),
                          "attach_mm": round(float(sh["attach_mm"][j]), 2),
                          "azimuth_deg": deg(sh["azimuth"][j]), "elevation_deg": deg(sh["elevation"][j]),
                          "incurve_deg": deg(sh["incurve"][j]),
                          "roll_deg": deg(sh["roll"][j]),
                          "fold_deg": deg(sh["fold"][j]),
                          "frame": "angles about the apex's shoot axis (apices[apex].axis); roll 0 = blade "
                                   "level across its median plane, upper face to the shoot"},
                "blade_mesh": {"rows": ALONG, "cols": ACROSS, "points": mesh.round(6).tolist()},
            })
    starts = [l["curve"][0] for l in out if l.get("curve") and l.get("starts_on") == "stem"]
    axes, attach = stem_tree.end_axes(axes, starts)
    on_stem = [l for l in out if l.get("curve") and l.get("starts_on") == "stem"]
    for l, q in zip(on_stem, attach):
        if np.linalg.norm(q - np.asarray(l["curve"][0])) > 1e-9 and not l.get("apex"):
            whole = np.vstack([q, np.asarray(l["curve"], float)])
            blade_at = np.asarray(l["curve"], float)[int(l.get("blade_start") or 0)]
            curve = smooth_curve(whole, run.voxel)
            b = int(np.argmin(np.linalg.norm(curve - blade_at, axis=1)))
            l.update(curve=curve.tolist(), blade_start=b, base=curve[b].tolist(),
                     midrib=curve[b:].tolist(), petiole=curve[:b + 1].tolist() if b > 0 else [])
    for i, l in enumerate(out):
        l["id"] = i
    main_axis = next((x for x in axes if x["kind"] == "stem"), None)
    doc = dict(skeleton, leaves=out, axes=axes, apices=apices,
               stem=main_axis["points"] if main_axis else skeleton.get("stem", []),
               built={"by": "scripts/apex_fit.py", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "init": str(args.init), "train": args.train})
    (args.out / "skeleton.json").write_text(json.dumps(doc))
    report["settings"] = {"grid": GRID, "along": ALONG, "across": ACROSS, "kappa": KAPPA, "degree": DEGREE,
                          "iters": args.iters, "refit": REFIT, "add_refit": ADD_REFIT, "lr": LR,
                          "train": args.train, "init": str(args.init), "evidence": str(args.evidence),
                          "incurve_range_deg": [round(float(np.degrees(x))) for x in Apex.INCURVE],
                          "candidates": [str(c) for c in args.candidates]}
    (args.out / "fit.json").write_text(json.dumps(report, indent=1))
    n_leaves = sum(a["leaves"] for a in apices.values())
    log(f"  wrote {args.out / 'skeleton.json'} ({n_leaves} apex leaves, "
        f"{sum(a['bud'] for a in apices.values())} buds)")


if __name__ == "__main__":
    main()
