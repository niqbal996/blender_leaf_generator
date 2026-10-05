#!/usr/bin/env python
"""Replay P5x's 2D-to-3D vote from the cached evidence, with every intermediate kept.

    python scripts/leafcount_replay.py --workdir <dataset>/plant --out runs/<specimen>_leafcount

Reads evidence.npz from `leafcount_funnel.py`. The merge and the cross-pass
combine are P5x's own functions, imported, not re-implemented -- so a variant
that changes a count here changes it for a reason that also holds in P5x.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pose_estimator.cli.leaf_instances import (ROOT, SKELETON, UNSEEN, _combine_passes,
                                               merge_across_passes)


class Evidence:
    def __init__(self, path: Path, workdir: Path):
        z = np.load(path, allow_pickle=True)
        self.points = z["points"]
        self.n = len(self.points)
        self.tracks = list(z["tracks"])
        self.views = list(z["views"])
        self.pix_view = z["pix_view"]
        self.pix_point = z["pix_point"]
        self.pix_w = z["pix_w"]
        self.claim_pix = z["claim_pix"]
        self.claim_label = z["claim_label"]
        sources = json.loads((workdir / "p1" / "sources.json").read_text())
        self.view_pass = np.array([sources[v] for v in self.views])
        self.track_pass = np.array([int(t.split("_")[0][4:]) for t in self.tracks])
        # frames per track (from the claims: a track claims >=1 rendered pixel in a view)
        self.track_frames = np.zeros(len(self.tracks), int)
        leaf = self.claim_label >= 0
        pairs = np.unique(np.stack([self.claim_label[leaf].astype(np.int64),
                                    self.pix_view[self.claim_pix[leaf]].astype(np.int64)], 1), axis=0)
        np.add.at(self.track_frames, pairs[:, 0], 1)


def p5x_order(ev: Evidence, workdir: Path, min_frames: int = 3):
    """Per pass, the track indices in P5x's painting order (usable_instances)."""
    root = workdir / "p2" / "masks" / "leaf_instances"
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    kept = [(d.name, len(list(d.glob("*.png")))) for d in dirs]
    kept = [(name, n) for name, n in kept if n >= min_frames]
    kept.sort(key=lambda pair: -pair[1])
    order = defaultdict(list)
    for name, _ in kept:
        order[int(name.split("_")[0][4:])].append(ev.tracks.index(name))
    return dict(order)


def pixel_labels(ev: Evidence, order_of_pass: dict, track_to_class=None):
    """One label per rendered pixel, painted exactly as build_label_map does.

    Leaves in `order`, later overwriting earlier; then stem, then root on top.
    Returns per-pixel class: >=0 a leaf class (position in the pass's order, or
    track_to_class[track] if given), -2 stem, -3 root, -1 nothing.
    """
    lab = np.full(len(ev.pix_point), -1, np.int32)
    for p, order in order_of_pass.items():
        rank = np.full(len(ev.tracks), -1, np.int64)
        rank[order] = np.arange(len(order))
        sel = ev.claim_label >= 0
        sel &= ev.track_pass[np.maximum(ev.claim_label, 0)] == p
        pix, tr = ev.claim_pix[sel], ev.claim_label[sel].astype(np.int64)
        r = rank[tr]
        keep = r >= 0
        pix, tr, r = pix[keep], tr[keep], r[keep]
        # paint in order: sort by rank, assignment keeps the last write per pixel
        o = np.argsort(r, kind="stable")
        cls = r if track_to_class is None else track_to_class[tr]
        lab[pix[o]] = cls[o]
    for code in (-2, -3):
        lab[ev.claim_pix[ev.claim_label == code]] = code
    return lab


def vote_pass(ev: Evidence, lab: np.ndarray, p: int, n_classes: int):
    """Weighted tally (points x (n_classes + 2)) from this pass's own views."""
    in_pass = ev.view_pass[ev.pix_view] == p
    use = in_pass & (lab != -1)
    cls = lab[use].astype(np.int64)
    cls = np.where(cls == -2, n_classes, np.where(cls == -3, n_classes + 1, cls))
    tally = np.zeros((ev.n, n_classes + 2), np.float64)
    np.add.at(tally, (ev.pix_point[use], cls), ev.pix_w[use])
    return tally


def argmax_labels(tally: np.ndarray, n_classes: int):
    total = tally.sum(1)
    win = tally.argmax(1)
    labels = np.full(len(tally), UNSEEN, np.int32)
    seen = total > 0
    leaf = seen & (win < n_classes)
    labels[leaf] = win[leaf]
    labels[seen & (win == n_classes)] = SKELETON
    labels[seen & (win == n_classes + 1)] = ROOT
    return labels


def replay_p5x(ev: Evidence, workdir: Path, min_frames=3, min_points=40, merge_overlap=0.30,
               verbose=True):
    order = p5x_order(ev, workdir, min_frames)
    lab = pixel_labels(ev, order)
    per_pass, counts, tallies = {}, {}, {}
    for p in sorted(order):
        k = len(order[p])
        tally = vote_pass(ev, lab, p, k)
        labels = argmax_labels(tally, k)
        prefix = f"pass{p}_"
        per_pass[prefix] = labels
        tallies[p] = tally
        for i in range(k):
            counts[(prefix, i)] = int((labels == i).sum())
        if verbose:
            won = sum(1 for i in range(k) if counts[(prefix, i)] > 0)
            won40 = sum(1 for i in range(k) if counts[(prefix, i)] >= min_points)
            print(f"    pass {p}: {k} ids (>= {min_frames} frames) -> {won} win any point, "
                  f"{won40} win >= {min_points}; {int((labels >= 0).sum())} leaf points")
    merged = merge_across_passes(per_pass, counts, merge_overlap)
    groups = defaultdict(list)
    for key, g in merged.items():
        groups[g].append(key)
    assignment = _combine_passes(per_pass, merged, ev.n)
    sizes = {g: int((assignment == g).sum()) for g in set(merged.values())}
    dropped = [g for g, s in sizes.items() if 0 < s < min_points]
    for g in dropped:
        assignment[assignment == g] = SKELETON
    surviving = sorted(g for g, s in sizes.items() if s >= min_points)
    if verbose:
        print(f"    merge: {len(counts)} per-pass ids, {sum(1 for c in counts.values() if c > 0)} "
              f"with points -> {len(groups)} groups; {len(surviving)} survive >= {min_points} "
              f"points ({len(dropped)} dropped)")
    return dict(order=order, per_pass=per_pass, counts=counts, merged=merged, groups=groups,
                assignment=assignment, surviving=surviving, tallies=tallies, lab=lab)


def cannot_links(ev: Evidence, min_shared_frames: int = 2) -> set:
    """Same-pass id pairs SAM3 showed as separate masks on the same frames.

    Measured on gaensefuss_1: of 2700 same-pass pairs ever on screen together,
    2689 have ~zero 2D overlap -- simultaneous SAM3 instances are disjoint
    objects, so two ids co-visible on several frames are two leaves in 2D.
    """
    leaf = ev.claim_label >= 0
    pairs = np.unique(np.stack([ev.claim_label[leaf].astype(np.int64),
                                ev.pix_view[ev.claim_pix[leaf]].astype(np.int64)], 1), axis=0)
    frames_of = defaultdict(set)
    for t, v in pairs:
        frames_of[int(t)].add(int(v))
    out = set()
    ids = sorted(frames_of)
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if ev.track_pass[a] != ev.track_pass[b]:
                continue
            if len(frames_of[a] & frames_of[b]) >= min_shared_frames:
                out.add((a, b))
    return out


def constrained_merge(per_pass, counts, keys_to_track, forbid: set, overlap_fraction: float):
    """P5x's cross-pass overlap test, but greedy and refusing any merge SAM3 contradicts.

    Union-find is transitive: a small id touching two leaves joins them, and
    on a crown every small id touches several. Here candidate links are taken
    strongest first, and a link is refused when it would put two ids that
    SAM3 showed simultaneously as separate masks into one leaf.
    """
    cands = []
    passes = sorted(per_pass)
    for a_index, first in enumerate(passes):
        for second in passes[a_index + 1:]:
            left, right = per_pass[first], per_pass[second]
            both = (left >= 0) & (right >= 0)
            if not both.any():
                continue
            pairs, overlap = np.unique(np.stack([left[both], right[both]], 1), axis=0,
                                       return_counts=True)
            for (a, b), n in zip(pairs, overlap):
                ka, kb = (first, int(a)), (second, int(b))
                smaller = min(counts.get(ka, 0), counts.get(kb, 0))
                if smaller and n >= overlap_fraction * smaller:
                    cands.append((n / smaller, n, ka, kb))
    cands.sort(key=lambda c: (-c[0], -c[1]))

    members = {k: {k} for k, c in counts.items() if c > 0}
    owner = {k: k for k in members}
    refused = 0
    for _score, _n, ka, kb in cands:
        ra, rb = owner[ka], owner[kb]
        if ra == rb:
            continue
        ta = {keys_to_track[k] for k in members[ra]}
        tb = {keys_to_track[k] for k in members[rb]}
        if any((min(x, y), max(x, y)) in forbid for x in ta for y in tb):
            refused += 1
            continue
        for k in members[rb]:
            owner[k] = ra
        members[ra] |= members.pop(rb)
    roots = {r: i for i, r in enumerate(sorted(members))}
    merged = {k: roots[owner[k]] for k in owner}
    return merged, refused


def pooled_assignment(ev, tallies, order, merged, n_points):
    """Sum each merged leaf's columns over every pass *before* the argmax.

    Fragments of one leaf stop splitting its weight (the measured cause of the
    stem residual winning at leaf borders), and a leaf only one pass labels
    keeps that pass's votes instead of being outvoted 2:1 by passes that
    simply did not have an id for it.
    """
    n_groups = max(merged.values()) + 1
    pooled = np.zeros((n_points, n_groups + 2))
    for p, order_p in order.items():
        k = len(order_p)
        t = tallies[p]
        prefix = f"pass{p}_"
        for i in range(k):
            g = merged.get((prefix, i))
            if g is not None:
                pooled[:, g] += t[:, i]
        pooled[:, n_groups] += t[:, k]
        pooled[:, n_groups + 1] += t[:, k + 1]
    return argmax_labels(pooled, n_groups), pooled


def summarise(name, assignment, min_points=40):
    sizes = {g: int((assignment == g).sum()) for g in np.unique(assignment) if g >= 0}
    keep = sorted(g for g, s in sizes.items() if s >= min_points)
    a = assignment.copy()
    for g, s in sizes.items():
        if s < min_points:
            a[a == g] = SKELETON
    pts = sorted((sizes[g] for g in keep), reverse=True)
    print(f"  {name:58s} leaves={len(keep):3d}  leaf pts={int((a >= 0).sum()):6d}  "
          f"skeleton={int((a == SKELETON).sum()):6d}   sizes {pts[:6]}..{pts[-4:]}")
    return a, keep


def shipped_co_visible(ev: Evidence, workdir: Path, order: dict, cache: Path) -> dict:
    """P5x's own co-visibility table, from its own `build_label_map` on the real masks.

    Every registered view of each pass, the pass's folders in P5x's order --
    so the cannot-link set is the one the pipeline would build, not this
    script's approximation of it. Cached: it is the slow part (every PNG).
    """
    from pose_estimator.cli.leaf_instances import build_label_map
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {p: {tuple(map(int, k.split(","))): n for k, n in t.items()} for p, t in raw.items()}
    root = workdir / "p2" / "masks"
    out = {}
    for p, order_p in sorted(order.items()):
        prefix = f"pass{p}_"
        dirs = [root / "leaf_instances" / ev.tracks[t] for t in order_p]
        table: dict = {}
        for view, vp in zip(ev.views, ev.view_pass):
            if vp == p:
                build_label_map(dirs, None, None, view, (1280, 1920), len(dirs), len(dirs) + 1,
                                co_visible=table)
        out[prefix] = table
        print(f"    {prefix}: {sum(1 for n in table.values() if n >= 2)} pairs co-visible on >= 2 frames")
    cache.write_text(json.dumps({p: {f"{a},{b}": n for (a, b), n in t.items()}
                                 for p, t in out.items()}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ev = Evidence(args.out / "evidence.npz", args.workdir)
    print("Run A0 -- P5x replayed exactly (current SAM3 ids, default knobs)")
    r = replay_p5x(ev, args.workdir)
    disk = np.load(args.workdir / "p5x" / "instances.npy")
    a = r["assignment"]
    print(f"    agreement with p5x/instances.npy on class kind (leaf/skel/root/unseen): "
          f"{np.mean(np.sign(np.minimum(a, 0)) == np.sign(np.minimum(disk, 0))):.4f}; "
          f"leaf count replay {len(r['surviving'])} vs disk {len(set(disk[disk >= 0]))}")

    keys_to_track = {(f"pass{p}_", i): t for p, o in r["order"].items() for i, t in enumerate(o)}
    print("\nVariants (same votes; only the association and the final argmax change)")
    summarise("A0  P5x: union-find merge, majority over passes", r["assignment"])
    pooled_a, _ = pooled_assignment(ev, r["tallies"], r["order"], r["merged"], ev.n)
    summarise("A1  union-find merge, pooled vote before argmax", pooled_a)
    for k in (1, 2, 3, 5):
        forbid = cannot_links(ev, k)
        merged, refused = constrained_merge(r["per_pass"], r["counts"], keys_to_track, forbid, 0.30)
        groups = len(set(merged.values()))
        maj = _combine_passes(r["per_pass"], merged, ev.n)
        summarise(f"B{k}m constrained (co-visible>={k}f, {refused} refused, {groups} grp), majority",
                  maj)
        pooled_b, _ = pooled_assignment(ev, r["tallies"], r["order"], merged, ev.n)
        summarise(f"B{k}p constrained (co-visible>={k}f), pooled vote", pooled_b)

    print("\nShipped code path: build_label_map co-visibility -> cannot_link_pairs -> "
          "merge_across_passes -> _combine_passes")
    from pose_estimator.cli.leaf_instances import cannot_link_pairs
    table = shipped_co_visible(ev, args.workdir, r["order"], args.out / "co_visible_shipped.json")
    for k in (0, 1, 2, 3, 5):
        forbid = cannot_link_pairs(table, k)
        stats = {}
        merged = merge_across_passes(r["per_pass"], r["counts"], 0.30, cannot_link=forbid,
                                     stats=stats)
        a, keep = summarise(f"S{k} --covisible-frames {k}: {len(forbid)} pairs, "
                            f"{stats['links_refused']} refused, {len(set(merged.values()))} grp",
                            _combine_passes(r["per_pass"], merged, ev.n))
        np.save(args.out / f"S{k}_assignment.npy", a)


if __name__ == "__main__":
    main()
