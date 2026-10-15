#!/usr/bin/env python
"""Each leaf's curvature, read off P4g: the leaf flattened, and how far it stands off its flat self.

    PYTHONPATH=src:scripts python scripts/leaf_curvature.py --workdir <workdir> [--cloud p5x_p4g] [--out DIR]

What a flat leaf asset (a scanned leaf's texture on a flat mesh, src/leaf_generator) lacks to
look like the real one is its curvature: the midrib's bend, the blade's cup across, and the waves of
its edge. This reads them per leaf of <workdir>/<cloud> (P5x on P4g), in mm (tag_scale.json).

The leaf's frame. Along the midrib (skeleton.json, resampled every STEP_MM) each stretch's points
give the across direction (their widest spread, the next being the normal, as leaf_margins.py
reads them). The frame is smoothed along the leaf over one half-width of the blade -- the leaf's
own scale (leaf_margins.py's trend window): a frame read per stretch turns with every wave and
hides it. A point's place: s along the midrib (arc length, plus its offset along the tangent),
a across it, h off it along the normal (+ = the upper face, towards plant up).

The mid-sheet. P4g draws a leaf as two skins 1-4 mm apart (docs/apex-plan.md 0.0d). In each cell
(STEP_MM along, CELL_MM across) the leaf's height is the middle of its two skins -- the mean of
its lowest and highest tenth -- and the skins' distance its apparent thickness.

Flattening. Each cross-section (one s) is unrolled along its own mid-sheet from the midrib out:
v, the arc length across, is where that point lies on the flat leaf (a cupped leaf is wider flat
than it looks). Along the leaf, s is already arc length. The flat leaf is the grid (s, v); its
outline the margins' v per s. (Each strip across is developable; a leaf's Gaussian curvature --
a ruffled edge longer than its midrib -- is what the height wave carries.)

The curvature, three scales:
  bend     the midrib's own curve: curvature in the leaf's plane and out of it, per s
  cup      per s, the cross-section's smooth part: h = c1 v + c2 v^2 on each side, its numbers
           smoothed along s over one half-width
  wave     what is left -- the wavy edge: its amplitude (robust standard deviation) against the
           distance from the midrib as a share of the half-width (rho = |v| / half-width at s),
           its wavelength along the margin (the residual's spectrum over rho > 1/2), and how the
           two halves move together (correlation of their margin traces)

Writes <out>/curvature.json (per leaf: outline, cup, wave; the plant's model from its leaves), per leaf
curvature_<id>.npz (s, v grids: h, cup, wave, mask) and curvature_<id>.json (its cup and wave on a
size-free grid, for blender_leaf_curvature.py), and curvature.jpg. <out> defaults to <workdir>/apex/curvature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import generic_filter, uniform_filter1d
from scipy.spatial import cKDTree

from leaf_margins import body
from pose_estimator.ply_io import read_ply_vertices
from skeleton_2d import resample

STEP_MM = 0.5        # along the midrib, and the flat grid's step
CELL_MM = 0.25       # across, in the cross-section before flattening (two point spacings, 0.12 mm)
SKIN_SHARE = 0.1     # a cell's two skins: its lowest and highest tenth of heights


def frame_along(P, midrib):
    """The leaf's frame about its own midline (see _frame), and how far P5x's midrib lies off it.

    P5x traces a wide leaf's midrib along one side of its blade (thistle_x, 2026-10-15: leaves 0
    and 1 had their whole cross-section 4-11 mm to one side of it, leaf 1's V-fold 4-6 mm off it),
    and a cup read about the wrong line leaves the fold in the wave. So the frame is read once
    about P5x's midrib, then again about the blade's midline: per stretch half-way between its
    two margins, at the middle of its two skins there, smoothed along the leaf over one half-width."""
    fr, s, a, h, half = _frame(P, midrib)
    M, T, W, N, n = fr["M"], fr["T"], fr["W"], fr["N"], len(fr["M"])
    i = np.clip((s / STEP_MM).astype(int), 0, n - 1)
    ac, hc = np.full(n, np.nan), np.full(n, np.nan)
    for r in range(n):
        sel = i == r
        if sel.sum() < 12:
            continue
        lo, hi = np.percentile(a[sel], [1, 99])
        ac[r] = 0.5 * (lo + hi)
        near = sel & (np.abs(a - ac[r]) <= CELL_MM)
        if near.sum() >= 2:
            hc[r] = 0.5 * sum(np.percentile(h[near], [100 * SKIN_SHARE, 100 * (1 - SKIN_SHARE)]))
    ok = np.isfinite(ac) & np.isfinite(hc)
    if ok.sum() < 4:
        return fr, s, a, h, half, None
    k = max(3, int(round(half / STEP_MM)) | 1)
    ac = uniform_filter1d(np.interp(np.arange(n), np.flatnonzero(ok), ac[ok]), k, mode="nearest")
    hc = uniform_filter1d(np.interp(np.arange(n), np.flatnonzero(ok), hc[ok]), k, mode="nearest")
    line = M + ac[:, None] * W + hc[:, None] * N
    fr2, s2, a2, h2, half2 = _frame(P, line)
    fr2["p5x_midrib_offset_mm"] = np.hypot(ac, hc)
    return fr2, s2, a2, h2, half2, float(np.median(np.hypot(ac, hc)))


def _frame(P, midrib):
    """Midrib samples M every STEP_MM, their arc length, and a frame T, W (across), N (normal)
    smoothed over one half-width; per point (s, a, h) and the leaf's half-width."""
    n = max(16, int(np.linalg.norm(np.diff(midrib, axis=0), axis=1).sum() / STEP_MM) + 1)
    M = resample(midrib, n)
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))]
    T = np.gradient(M, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    _, j = cKDTree(M).query(P)
    s = cum[j] + np.einsum("ij,ij->i", P - M[j], T[j])
    inside = (s > 0) & (s < cum[-1])
    P, s, j = P[inside], s[inside], j[inside]
    W = np.full((n, 3), np.nan)
    for i in range(n):
        wide = np.abs(j - i) <= 2
        if wide.sum() < 12:
            continue
        q = P[wide] - M[i]
        q = q - (q @ T[i])[:, None] * T[i]
        W[i] = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
    ok = np.flatnonzero(np.isfinite(W[:, 0]))
    for k in range(1, len(ok)):                                 # one side stays one side
        if W[ok[k]] @ W[ok[k - 1]] < 0:
            W[ok[k]] = -W[ok[k]]
    for d in range(3):
        W[:, d] = np.interp(np.arange(n), ok, W[ok, d])
    q = P - M[j]
    q -= np.einsum("ij,ij->i", q, T[j])[:, None] * T[j]
    a0 = np.einsum("ij,ij->i", q, W[j])
    half = max(float(np.percentile(np.abs(a0), 98)), 4 * STEP_MM)
    k = max(3, int(round(half / STEP_MM)) | 1)
    W = uniform_filter1d(W, k, axis=0, mode="nearest")          # over one half-width
    W -= np.einsum("ij,ij->i", W, T)[:, None] * T
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    N = np.cross(T, W)
    if np.nanmean(N[:, 2]) < 0:                                 # the upper face towards plant up
        W, N = -W, -N
    a = np.einsum("ij,ij->i", q, W[j])
    h = np.einsum("ij,ij->i", q, N[j])
    return dict(M=M, cum=cum, T=T, W=W, N=N), s, a, h, half


def mid_sheet(s, a, h, length):
    """Per stretch along and cell across: the mid-sheet height and apparent thickness (nan where
    the cell has too few points to show both skins)."""
    ns = int(np.ceil(length / STEP_MM))
    amax = float(np.abs(a).max())
    na = int(np.ceil(amax / CELL_MM))
    A = (np.arange(-na, na) + 0.5) * CELL_MM
    mid = np.full((ns, len(A)), np.nan)
    thick = np.full((ns, len(A)), np.nan)
    i = np.clip((s / STEP_MM).astype(int), 0, ns - 1)
    k = np.clip(np.floor(a / CELL_MM).astype(int) + na, 0, len(A) - 1)
    order = np.lexsort((h, k, i))
    i, k, hs = i[order], k[order], h[order]
    key = i * len(A) + k
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    for b, e in zip(starts, ends):
        m = e - b
        if m < 6:
            continue
        lo_n = max(1, int(round(SKIN_SHARE * m)))
        v = hs[b:e]
        lo, hi = v[:lo_n].mean(), v[-lo_n:].mean()
        mid[i[b], k[b]] = 0.5 * (lo + hi)
        thick[i[b], k[b]] = hi - lo
    # one-row features are not the leaf: a tooth or lobe spans several 0.5 mm rows, a patch of
    # another leaf's points (P5x leaks, docs/apex-plan.md 0.0d) cut by one row is one row thick
    # (thistle_x leaves 0-2: single rows reaching 20-30 mm beyond their neighbours)
    occ = np.isfinite(mid).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
    n, comp = cv2.connectedComponents(occ, connectivity=4)
    on_rib = comp[:, na - 1:na + 1][occ[:, na - 1:na + 1] > 0]
    if len(on_rib) and on_rib.max() > 0:
        occ = (comp == np.bincount(on_rib[on_rib > 0]).argmax()).astype(np.uint8)
    mid[occ == 0], thick[occ == 0] = np.nan, np.nan
    # a cell off its neighbours' median by more than half its own skin gap: a stray, not the sheet
    med = generic_filter(mid, np.nanmedian, size=3, mode="constant", cval=np.nan)
    bad = np.isfinite(mid) & np.isfinite(med) & (np.abs(mid - med) > 0.5 * np.nan_to_num(thick, nan=np.inf))
    mid[bad] = med[bad]
    return A, mid, thick


def flatten(A, mid):
    """Each row (one s) unrolled from the midrib out along its own mid-sheet: per row the flat v of
    every cell (nan outside the leaf), the margins' flat v, and the mid-sheet with its holes bridged."""
    ns, na = mid.shape
    c = na // 2                                              # A[c - 1] < 0 < A[c]
    V = np.full_like(mid, np.nan)
    edges = np.full((ns, 2), np.nan)                          # flat v of the left (-) and right (+) margin
    bridge = int(round(STEP_MM / CELL_MM))                   # holes up to one grid step are bridged
    for r in range(ns):
        row = mid[r]
        ok = np.flatnonzero(np.isfinite(row))
        if len(ok) < 4 or ok.min() >= c or ok.max() < c:
            continue
        # the blade connected to the midrib: a section leaving it at a sinus and meeting a lobe
        # further out ends at the sinus (bridging it unrolled the gap's diagonal: 30 mm spikes)
        start = ok[np.argmin(np.abs(A[ok]))]
        gaps = np.diff(ok)
        k0 = int(np.searchsorted(ok, start))
        lo_k, hi_k = k0, k0
        while lo_k > 0 and gaps[lo_k - 1] <= bridge + 1:
            lo_k -= 1
        while hi_k < len(ok) - 1 and gaps[hi_k] <= bridge + 1:
            hi_k += 1
        lo, hi = ok[lo_k], ok[hi_k]
        if lo >= c or hi < c - 1:
            continue
        ok = ok[lo_k:hi_k + 1]
        filled = np.interp(np.arange(lo, hi + 1), ok, row[ok])  # gaps inside the leaf: bridged
        mid[r, :lo] = np.nan
        mid[r, hi + 1:] = np.nan
        # unrolled along the section smoothed over five cells (1.25 mm, finer than any wave, which
        # runs along the leaf): a section whose cells zig-zag between two surfaces otherwise sums
        # every zig into its length (thistle_x: single rows 50 mm wide)
        k5 = min(5, len(filled) | 1 if len(filled) % 2 else len(filled) - 1)
        smooth = np.array([np.median(filled[max(0, q - k5 // 2):q + k5 // 2 + 1]) for q in range(len(filled))])
        seg = np.hypot(np.diff(A[lo:hi + 1]), np.diff(smooth))
        arc = np.r_[0, np.cumsum(seg)]
        arc -= np.interp(0.0, A[lo:hi + 1], arc)              # v = 0 at the midrib
        V[r, lo:hi + 1] = arc
        mid[r, lo:hi + 1] = filled
        edges[r] = arc[0], arc[-1]
    mid[~np.isfinite(V)] = np.nan
    # each margin the median of its row and the two either side: a feature under 1.5 mm along the
    # leaf is not the blade's (thistle_x leaves 0-1: two-row spikes where the leaf passes the heart)
    for side in range(2):
        e = edges[:, side].copy()
        for r in range(ns):
            win = e[max(0, r - 2):r + 3]
            win = win[np.isfinite(win)]
            if np.isfinite(e[r]) and len(win) >= 3:
                edges[r, side] = np.median(win)
    return V, edges, mid


def regrid(V, mid, edges, step=STEP_MM):
    """h on the flat leaf's regular grid (rows = s, columns = v), and its mask."""
    ns = len(V)
    vmax = float(np.nanmax(np.abs(edges))) if np.isfinite(edges).any() else 1.0
    nv = int(np.ceil(vmax / step))
    G = (np.arange(-nv, nv + 1)) * step
    H = np.full((ns, len(G)), np.nan)
    for r in range(ns):
        ok = np.isfinite(V[r])
        if ok.sum() < 4:
            continue
        inside = (G >= edges[r, 0]) & (G <= edges[r, 1])
        H[r, inside] = np.interp(G[inside], V[r, ok], mid[r, ok])
    return G, H, np.isfinite(H)


def cup(G, H, half):
    """The smooth part across: per row and side, h = c1 v + c2 v^2 (least squares, through the
    midrib's h), the numbers then smoothed along s over one half-width; returns the shape grid
    and the coefficients (ns, 2 sides, 2)."""
    ns = len(H)
    C = np.full((ns, 2, 2), np.nan)
    for r in range(ns):
        ok = np.isfinite(H[r])
        if ok.sum() < 6:
            continue
        h0 = np.interp(0.0, G[ok], H[r, ok])
        for side, sel in enumerate([(G < 0) & ok, (G > 0) & ok]):
            if sel.sum() < 3:
                continue
            v = np.abs(G[sel])
            X = np.stack([v, v * v], 1)
            C[r, side] = np.linalg.lstsq(X, H[r, sel] - h0, rcond=None)[0]
    k = max(3, int(round(half / STEP_MM)) | 1)
    Cs = C.copy()
    for side in range(2):
        for d in range(2):
            x = C[:, side, d]
            ok = np.isfinite(x)
            if ok.sum() >= 2:
                xf = np.interp(np.arange(ns), np.flatnonzero(ok), x[ok])
                Cs[:, side, d] = uniform_filter1d(xf, k, mode="nearest")
    h0 = np.array([np.interp(0.0, G[np.isfinite(H[r])], H[r, np.isfinite(H[r])]) if np.isfinite(H[r]).sum() >= 2
                   else np.nan for r in range(ns)])
    okh = np.isfinite(h0)
    h0 = uniform_filter1d(np.interp(np.arange(ns), np.flatnonzero(okh), h0[okh]), k, mode="nearest") if okh.any() \
        else np.zeros(ns)
    v = np.abs(G)[None]
    side = (G > 0).astype(int)[None].repeat(ns, 0)
    c1 = np.where(side == 1, Cs[:, 1, 0][:, None], Cs[:, 0, 0][:, None])
    c2 = np.where(side == 1, Cs[:, 1, 1][:, None], Cs[:, 0, 1][:, None])
    return h0[:, None] + c1 * v + c2 * v * v, Cs


def robust_sd(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def wave_model(G, R, edges, half):
    """The wave's amplitude against rho = |v| / (that row's half-width, per side), its wavelength
    along the margin, and the two margins' correlation."""
    ns = len(R)
    w = np.where(G[None] < 0, -edges[:, :1], edges[:, 1:])          # that side's half-width per row
    rho = np.abs(G)[None] / np.maximum(w, 1e-9)
    bands = np.linspace(0, 1, 6)
    env = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        sel = (rho >= lo) & (rho < hi) & np.isfinite(R)
        env.append(robust_sd(R[sel]) if sel.sum() > 20 else None)
    # the margin trace per side: the residual at rho in [1/2, 1], mean per row
    traces = []
    for side in (G < 0, G > 0):
        m = np.where(side[None] & (rho >= 0.5) & (rho <= 1.0) & np.isfinite(R), R, np.nan)
        cnt = np.isfinite(m).sum(1)
        traces.append(np.where(cnt > 0, np.nansum(np.nan_to_num(m), 1) / np.maximum(cnt, 1), np.nan))
    lam, spectra = [], []
    hw = float(np.nanmax(edges[:, 1] - edges[:, 0]) / 2)        # the flat leaf's half-width
    for t in traces:
        ok = np.isfinite(t)
        if ok.sum() < 16:
            lam.append(None)
            continue
        x = np.interp(np.arange(ns), np.flatnonzero(ok), t[ok])
        f = np.fft.rfftfreq(ns, STEP_MM)
        p = np.abs(np.fft.rfft(x - x.mean())) ** 2
        keep = (f > 1.0 / (ns * STEP_MM)) & (f < 1.0 / (2 * STEP_MM))
        lam.append(float(1.0 / f[keep][np.argmax(p[keep])]) if keep.any() else None)
        spectra.append((f[keep] * hw, p[keep] / max(p[keep].sum(), 1e-12)))     # cycles per half-width
    both = np.isfinite(traces[0]) & np.isfinite(traces[1])
    corr = float(np.corrcoef(traces[0][both], traces[1][both])[0, 1]) if both.sum() > 8 else None
    return {"rho_bands": bands.round(2).tolist(), "amplitude_mm": [None if e is None else round(e, 3) for e in env],
            "wavelength_mm": [None if l is None else round(l, 2) for l in lam],
            "left_right_correlation": None if corr is None else round(corr, 2),
            "margin_amplitude_mm": [round(robust_sd(t), 3) for t in traces]}, traces, spectra


def colour(Z, lim):
    z = np.clip((np.nan_to_num(Z) / max(lim, 1e-9) + 1) / 2, 0, 1)
    img = cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_COOL)
    img[~np.isfinite(Z)] = 255
    return img


HOW = ("per leaf of P5x on P4g: the leaf flattened (each cross-section unrolled along its mid-sheet, the "
       "middle of P4g's two skins), its height off the midrib's frame split into a smooth part (cup: per side "
       "c1 v + c2 v^2, smoothed along the leaf over one half-width) and the wave (the rest); the wave's robust "
       "standard deviation in five bands from the midrib (rho 0) to the margin (rho 1), its margin trace's "
       "wavelength and spectrum, the two margins' correlation (scripts/leaf_curvature.py)")
NU, NR = 96, 33          # the size-free grid of a leaf's maps: along (s / length), across (v / half-width)


def plant_model(per_leaf):
    """The plant's wave, from its leaves that show it whole (all five bands, a margin wavelength):
    medians of the envelope's shape (each band over the margin's), the margin's amplitude and
    wavelength over the leaf's half-width, the two sides' correlation, the cup's rise at the margin
    over the half-width, and the mean margin spectrum (power per cycle per half-width). Each leaf's
    amplitude and wavelength are also predicted from the others' (leave one out)."""
    use = []
    for row, spectra, *_ in per_leaf:
        w = row["wave"]
        if row["half_width_mm"] and None not in w["amplitude_mm"] and any(l is not None for l in w["wavelength_mm"]):
            use.append((row, spectra))
    if not use:
        return {"leaves_used": []}
    hw = np.array([r["half_width_mm"] for r, _ in use])
    amp = np.array([np.mean(r["wave"]["margin_amplitude_mm"]) for r, _ in use])
    env = np.array([np.array(r["wave"]["amplitude_mm"]) / r["wave"]["amplitude_mm"][-1] for r, _ in use])
    lam = [[l / r["half_width_mm"] for l in r["wave"]["wavelength_mm"] if l is not None] for r, _ in use]
    corr = [r["wave"]["left_right_correlation"] for r, _ in use if r["wave"]["left_right_correlation"] is not None]
    cupr = np.array([r["cup"]["rise_at_margin_mm_median"] / r["half_width_mm"] for r, _ in use])
    F = np.linspace(0, 4, 81)
    S = []
    for _, sp in use:
        for f, p in sp:
            if len(f) <= 2:
                continue
            # the reading's own noise is white: its floor, the median power over the shorter half of
            # the wavelengths read, is taken off (a model wave built on it is jagged, not wavy)
            floor = float(np.median(p[len(p) // 2:]))
            S.append(np.interp(F, f, np.clip(p - floor, 0, None), left=0, right=0))
    spec = np.mean(S, 0) if S else np.zeros_like(F)
    spec = spec / spec.sum() if spec.sum() > 0 else spec
    loo = []
    for k, (r, _) in enumerate(use):
        others = [i for i in range(len(use)) if i != k]
        if not others:
            break
        a_hat = float(np.median(amp[others] / hw[others]) * hw[k])
        l_hat = float(np.median([x for i in others for x in lam[i]]) * hw[k])
        loo.append({"leaf": r["leaf"], "amplitude_mm": round(float(amp[k]), 3), "predicted": round(a_hat, 3),
                    "wavelength_mm": [round(x * hw[k], 2) for x in lam[k]], "predicted_wavelength": round(l_hat, 2)})
    return {"leaves_used": [r["leaf"] for r, _ in use],
            "envelope_rho": [0.1, 0.3, 0.5, 0.7, 0.9],
            "envelope_share_of_margin": np.median(env, 0).round(3).tolist(),
            "margin_amplitude_per_half_width": round(float(np.median(amp / hw)), 4),
            "wavelength_per_half_width": round(float(np.median([x for l in lam for x in l])), 3),
            "left_right_correlation": round(float(np.median(corr)), 2) if corr else None,
            "cup_rise_per_half_width": round(float(np.median(cupr)), 4),
            "spectrum_cycles_per_half_width": F.round(3).tolist(), "spectrum_power": spec.round(5).tolist(),
            "leave_one_out": loo}


def normalised_maps(row, G, H, shape, Rw, mask, edges):
    """A leaf's curvature on the size-free grid: u = s / length (NU), r = v / that side's half-width
    at s (NR, -1 left .. 1 right): the wave and the cup (mm, cup with the midrib's own height
    removed), and the outline (half-width per side, mm). Applied to another leaf, heights scale
    with its half-width over this one's."""
    ns = len(H)
    U = np.linspace(0, 1, NU)
    R = np.linspace(-1, 1, NR)
    rows = np.clip((U * (ns - 1)).round().astype(int), 0, ns - 1)
    wave, cup_ = np.full((NU, NR), np.nan), np.full((NU, NR), np.nan)
    for i, r in enumerate(rows):
        # a row narrower than two grid cells a side cannot carry a cross-section (the base, the tip)
        if not np.isfinite(edges[r]).all() or min(-edges[r, 0], edges[r, 1]) < 2 * STEP_MM:
            continue
        v = np.where(R < 0, R * -edges[r, 0], R * edges[r, 1])
        ok = mask[r]
        if ok.sum() < 4:
            continue
        h0 = np.interp(0.0, G[ok], shape[r, ok])
        wave[i] = np.interp(v, G[ok], Rw[r, ok], left=np.nan, right=np.nan)
        cup_[i] = np.interp(v, G[ok], shape[r, ok] - h0, left=np.nan, right=np.nan)
    out = np.interp(U * (ns - 1), np.arange(ns), np.nan_to_num(edges[:, 1] - 0 * edges[:, 1]))
    return {"leaf": row["leaf"], "midrib_mm": row["midrib_mm"], "half_width_mm": row["half_width_mm"],
            "u": U.round(4).tolist(), "r": R.round(4).tolist(),
            "wave_mm": np.round(np.nan_to_num(wave), 4).tolist(), "cup_mm": np.round(np.nan_to_num(cup_), 4).tolist(),
            "inside": np.isfinite(wave).tolist(),
            "half_width_left_mm": np.round(np.interp(U * (ns - 1), np.arange(ns), np.nan_to_num(-edges[:, 0])), 3).tolist(),
            "half_width_right_mm": np.round(out, 3).tolist()}


def print_model(m):
    if not m.get("leaves_used"):
        print("  plant model: no leaf shows its wave whole")
        return
    print(f"  plant model from leaves {m['leaves_used']}: wave at the margin {m['margin_amplitude_per_half_width']} of "
          f"the half-width, wavelength {m['wavelength_per_half_width']} half-widths, envelope midrib->margin "
          f"{m['envelope_share_of_margin']}, L/R corr {m['left_right_correlation']}, cup rise "
          f"{m['cup_rise_per_half_width']} of the half-width")
    for r in m["leave_one_out"]:
        print(f"    leave one out, leaf {r['leaf']}: margin wave {r['amplitude_mm']} mm (predicted {r['predicted']}), "
              f"wavelength {r['wavelength_mm']} mm (predicted {r['predicted_wavelength']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--cloud", default="p5x_p4g")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--leaves", type=int, nargs="*")
    args = ap.parse_args()
    out = args.out or args.workdir / "apex" / "curvature"
    out.mkdir(parents=True, exist_ok=True)
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    folder = args.workdir / args.cloud
    f = read_ply_vertices(folder / "segmented.ply")
    X = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64) * mpu
    lab = np.asarray(f["label"])
    sk = json.loads((folder / "skeleton.json").read_text())
    rows, panels, per_leaf = [], [], []
    for leaf in sk.get("leaves", []):
        lid = int(leaf["id"])
        if args.leaves and lid not in args.leaves:
            continue
        rib = np.asarray(leaf.get("midrib") or [], float) * mpu
        P = X[lab == lid]
        if len(rib) < 4 or len(P) < 500:
            continue
        P = body(P)
        fr, s, a, h, half, rib_off = frame_along(P, rib)
        A, mid, thick = mid_sheet(s, a, h, fr["cum"][-1])
        V, edges, midf = flatten(A, mid.copy())
        G, H, mask = regrid(V, midf, edges)
        # on the flat grid: a cell off its neighbours' median by more than the leaf's own skin gap is
        # not the sheet (a sheet rising that much over one 0.5 mm step stands at > 70 deg; the waves
        # stand at < 30): another leaf's points where this one passes it (thistle_x 0-1: 6-9 mm)
        gap = float(np.nanmedian(thick))
        for _ in range(2):
            med = generic_filter(H, np.nanmedian, size=3, mode="constant", cval=np.nan)
            bad = mask & np.isfinite(med) & (np.abs(H - med) > gap)
            H[bad] = med[bad]
        shape, C = cup(G, H, half)
        shape = np.where(mask, shape, np.nan)
        Rw = np.where(mask, H - shape, np.nan)
        model, traces, spectra = wave_model(G, Rw, edges, half)
        # bend: the midrib's curvature in and out of the leaf's plane
        M, T, W, N = fr["M"], fr["T"], fr["W"], fr["N"]
        dT = np.gradient(T, STEP_MM, axis=0)
        k_in, k_out = np.einsum("ij,ij->i", dT, W), np.einsum("ij,ij->i", dT, N)
        width = edges[:, 1] - edges[:, 0]
        row = {"leaf": lid, "points": int(len(P)), "midrib_mm": round(float(fr["cum"][-1]), 2),
               "p5x_midrib_off_midline_mm": None if rib_off is None else round(rib_off, 2),
               "half_width_mm": round(float(np.nanmax(width) / 2), 2) if np.isfinite(width).any() else None,
               "flat_width_profile_mm": [round(float(x), 2) for x in np.interp(
                   np.linspace(0, len(width) - 1, 20), np.flatnonzero(np.isfinite(width)), width[np.isfinite(width)])]
               if np.isfinite(width).sum() > 2 else [],
               "thickness_mm_median": round(float(np.nanmedian(thick)), 2),
               "bend_per_mm": {"in_plane_median": round(float(np.median(k_in)), 4),
                               "out_of_plane_median": round(float(np.median(k_out)), 4),
                               "turn_out_of_plane_deg": round(float(np.degrees(np.arccos(np.clip(T[0] @ T[-1], -1, 1)))), 1)},
               "cup": {"c2_per_mm_left_right_median": [round(float(np.nanmedian(C[:, 0, 1])), 4),
                                                        round(float(np.nanmedian(C[:, 1, 1])), 4)],
                       "rise_at_margin_mm_median": round(float(np.nanmedian(
                           np.nanmean(np.stack([C[:, 0, 0] * -edges[:, 0] + C[:, 0, 1] * edges[:, 0] ** 2,
                                                C[:, 1, 0] * edges[:, 1] + C[:, 1, 1] * edges[:, 1] ** 2]), 0))), 2)},
               "wave": model}
        rows.append(row)
        per_leaf.append((row, spectra, G, H, shape, Rw, mask, edges))
        np.savez_compressed(out / f"curvature_{lid}.npz", s=np.arange(len(H)) * STEP_MM + STEP_MM / 2, v=G, h=H,
                            shape=shape, wave=Rw, mask=mask, edges=edges, cup=C,
                            midrib=M, T=T, W=W, N=N)
        lim = max(0.1, 3 * max(model["margin_amplitude_mm"]))      # the wave: 3 of its margin's deviations
        Hc = H - np.nanmedian(H)
        hl = max(0.25, 3 * robust_sd(Hc) + abs(float(np.nanmedian(Hc))))
        tiles = [np.rot90(colour(Hc, hl), 1), np.rot90(colour(shape - np.nanmedian(H), hl), 1),
                 np.rot90(colour(Rw, lim), 1)]
        px = max(2, int(round(180 / tiles[0].shape[0])))           # each tile ~180 px across the leaf
        tiles = [cv2.resize(t_, None, fx=px, fy=px, interpolation=cv2.INTER_NEAREST) for t_ in tiles]
        tiles = [cv2.copyMakeBorder(t_, 18, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t_ in tiles]
        for t_, name in zip(tiles, [f"height (+-{hl:.1f} mm)", "cup + bend", f"wave (+-{lim:.2f} mm)"]):
            cv2.putText(t_, name, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        img = np.concatenate(tiles, 1)
        img = cv2.copyMakeBorder(img, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, f"leaf {lid} ({row['midrib_mm']:.0f} mm, base left): wave amplitude midrib->margin "
                         f"{[a for a in model['amplitude_mm']]} mm, wavelength {model['wavelength_mm']} mm, "
                         f"L/R corr {model['left_right_correlation']}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        panels.append(img)
        print(f"  leaf {lid:2d}: midrib {row['midrib_mm']:5.1f} mm (P5x's {row['p5x_midrib_off_midline_mm']} mm off the midline), flat half-width {row['half_width_mm']} mm, "
              f"thickness {row['thickness_mm_median']} mm, cup rise {row['cup']['rise_at_margin_mm_median']} mm; "
              f"wave by rho {model['amplitude_mm']} mm, wavelength {model['wavelength_mm']} mm, "
              f"L/R corr {model['left_right_correlation']}", flush=True)
    model = plant_model(per_leaf)
    (out / "curvature.json").write_text(json.dumps({"cloud": str(folder), "mm_per_unit": mpu, "step_mm": STEP_MM,
                                                 "model": model, "leaves": rows, "how": HOW}, indent=1))
    for row, _, G, H, shape, Rw, mask, edges in per_leaf:
        (out / f"curvature_{row['leaf']}.json").write_text(json.dumps(normalised_maps(row, G, H, shape, Rw, mask, edges)))
    print_model(model)
    if panels:
        Wd = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 6, 0, Wd - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
        cv2.imwrite(str(out / "curvature.jpg"), np.concatenate(panels, 0))
    print(f"  wrote {out}/curvature.json, curvature.jpg, curvature_<leaf>.npz")



if __name__ == "__main__":
    main()
