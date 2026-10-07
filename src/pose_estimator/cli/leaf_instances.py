"""P5x (alternative): leaf instances and skeleton, projected from P2's 2D ids.

    pose-leaf-instances --workdir runs/plant_9/

An alternative to P4c+P5, not a replacement. The existing path stays exactly
as it is: pose-classify, pose-fuse, pose-structure are untouched and still
write p4c/ and p5/. This writes p5x/ and nothing else reads it.

**How the two differ.**

The existing path reconstructs leaf identity in 3D. P4c labels each point
leaf/stem/root, then P5 locates the crown, finds candidate tips as maxima of
a geodesic depth field over the leaf tissue, groups them, and assigns
ownership outward from the crown -- drawing the leaves as geodesic lines from
crown to tip. Every one of those steps is inference standing in for
information the 2D segmentation never carried, because a SAM2 P2 only ever
produced one silhouette.

A SAM3 P2 carries that information. It tracks each leaf as its own object with
its own id, held across the rotation, and writes the ids to
p2/masks/leaf_instances/<id>/. So the correspondence that P5 rebuilds in 3D is
already solved in 2D, and this phase only has to move it onto the points:

    every view, for every point the render places under a leaf-id pixel,
    one vote for that leaf, weighted by how broad-side the point is to
    that camera.

Then the leaf a point belongs to is its argmax, and the skeleton is the
tissue that voted for the stem residual instead. No crown, no depth field, no
tip detection -- a tip becomes simply the far end of a leaf whose extent is
already known.

**The skeleton is voted, not left over.** Taking it as "every point that is
not a leaf" would quietly sweep two different things into one class: tissue
that really is stem, and tissue no camera ever saw. p2/masks/stem is a real
mask, so it is voted as its own class and an unseen point stays unlabelled
and is reported as such.

**What this cannot do.** It inherits P2's 2D mistakes without appeal. A leaf
the `leaf` prompt lost on a frame contributes no vote there, which the other
views absorb; but a leaf that fragments into two ids across the rotation
arrives here as two leaves unless another pass's id bridges them in 3D.
On gaensefuss_1 that fragmentation turned out to be moderate -- 43-45 ids
per pass for a plant the flat lay counts at 34 leaves -- and the dominant
error was the cross-pass merge chaining *distinct* leaves together; see
`merge_across_passes`.

Reads:
    p2/masks/leaf_instances/<id>/*.png   the 2D ids, from --backend sam3
    p2/masks/stem/*.png, masks/root/*.png
    p4b/surface.ply (or p4/hull_points.ply), p3/sparse/best

Writes into <workdir>/p5x:
    instances.npy        int32 per cloud point: >=0 leaf index, -1 unseen,
                         -2 skeleton, -3 root
    leaves.ply           leaf points only, one colour per leaf   <- Blender
    skeleton.ply         the stem-and-petiole points             <- Blender
    segmented.ply        everything, leaves coloured + skeleton grey
    instances.json       per-leaf point counts and acceptance checks
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from pose_estimator import cloud_source, plant_profiles
from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.semantic import (
    accumulate_votes,
    camera_from_colmap,
    cast_votes,
    colors_from_surfels,
    estimate_normals,
    finalise_votes,
    render_points,
    surface_scale,
    view_weights,
    visible_only,
)
from pose_estimator.leaf_skeleton import trace
from pose_estimator.cli.report import print_checks

# Sentinels in instances.npy, chosen so a leaf index stays a plain 0-based
# index and the three non-leaf outcomes are told apart rather than merged.
UNSEEN = -1
SKELETON = -2
ROOT = -3

SKELETON_RGB = (200, 200, 200)
ROOT_RGB = (240, 140, 40)

# finalise_votes returns int8 labels, so the vote can distinguish at most 127
# classes and two of those are the stem and root. Well clear of any real leaf
# count, but a fragmented run can produce more ids than leaves, so it is
# checked rather than assumed.
MAX_VOTED_CLASSES = 127

# Two ids of one pass that SAM3 shows on the same frame overlapping less than
# this share of the smaller are two objects, not one object twice. Measured on
# gaensefuss_1: of 2700 same-pass pairs ever on screen together, 2689 overlap
# by ~0 -- SAM3's simultaneous instances are disjoint -- so the bar only has to
# keep a genuine duplicate (an NMS miss) from being read as two leaves.
DUPLICATE_OVERLAP = 0.5


def leaf_palette(n: int) -> np.ndarray:
    """One saturated, well-separated RGB per leaf.

    Generated rather than a fixed list: a bushy specimen produces dozens of
    ids and a twelve-colour table would repeat, which in a 3D view reads as
    two leaves being the same leaf.
    """
    if n <= 0:
        return np.zeros((0, 3), np.uint8)
    # The golden-ratio hue step keeps successive ids far apart in hue, so
    # neighbouring leaves -- which get consecutive ids -- never look alike.
    hues = np.mod(np.arange(n) * 0.61803398875, 1.0)
    out = []
    for i, h in enumerate(hues):
        value = 0.95 if i % 2 == 0 else 0.72      # alternate brightness too
        r, g, b = colorsys.hsv_to_rgb(float(h), 0.85, value)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(out, np.uint8)


def usable_instances(leaf_root: Path, min_frames: int) -> List[Path]:
    """Instance directories worth projecting, longest-lived first.

    An id present on one or two frames is a fragment, not a leaf, and it costs
    a vote class and a colour while contributing a handful of points. Dropping
    it here is the one place a threshold enters this phase, and it is a
    statement about *tracking*, not about geometry.
    """
    if not leaf_root.is_dir():
        return []
    dirs = [d for d in sorted(leaf_root.iterdir()) if d.is_dir()]
    kept = [(d, len(list(d.glob("*.png")))) for d in dirs]
    kept = [(d, n) for d, n in kept if n >= min_frames]
    kept.sort(key=lambda pair: -pair[1])
    return [d for d, _ in kept]


def build_label_map(leaf_dirs: Sequence[Path], stem_dir: Optional[Path],
                    root_dir: Optional[Path], stem: str,
                    shape, stem_class: int, root_class: int,
                    co_visible: Optional[Dict[tuple, int]] = None) -> Optional[np.ndarray]:
    """One frame's 2D map in the form `cast_votes` already reads.

    Same contract as a P4c class map -- -1 where nothing is claimed, otherwise
    a class index -- so the fusion below is the one that has been validated
    against synthetic maps, not a second implementation of it.

    `co_visible`, if given, is incremented for every pair of leaf indices this
    frame shows as two separate masks -- the evidence `merge_across_passes`
    uses to refuse joining them. It is gathered here because this is where
    every mask is read anyway.
    """
    out = np.full(shape, -1, np.int16)
    found = False
    areas: Dict[int, int] = {}
    shared: Dict[tuple, int] = {}
    for index, folder in enumerate(leaf_dirs):
        path = folder / f"{stem}.png"
        if not path.exists():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        claim = raw > 127
        if co_visible is not None:
            under = out[claim]
            under = under[under >= 0]
            for other, n in zip(*np.unique(under, return_counts=True)):
                shared[(int(other), index)] = int(n)
            areas[index] = int(claim.sum())
        out[claim] = index
        found = True

    if co_visible is not None:
        present = sorted(i for i, a in areas.items() if a > 0)
        for a_index, a in enumerate(present):
            for b in present[a_index + 1:]:
                if shared.get((a, b), 0) < DUPLICATE_OVERLAP * min(areas[a], areas[b]):
                    co_visible[(a, b)] = co_visible.get((a, b), 0) + 1

    # The stem and root are painted after the leaves: where a leaf instance
    # and the stem residual disagree the residual is the narrower claim, and
    # it is the one built by subtracting the leaves in the first place.
    for folder, klass in ((stem_dir, stem_class), (root_dir, root_class)):
        if folder is None:
            continue
        path = folder / f"{stem}.png"
        if not path.exists():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        out[raw > 127] = klass
        found = True
    return out if found else None


def group_by_pass(leaf_dirs: Sequence[Path]) -> Dict[str, List[Path]]:
    """Instance folders split by the capture pass that produced them.

    Folders are named `pass<N>_<id>` because each pass is its own SAM3 session
    whose object ids restart from zero. A folder with no prefix comes from
    before that was true and is filed under "" -- which is correct only if the
    capture had one pass, and the caller checks that.
    """
    out: Dict[str, List[Path]] = {}
    for folder in leaf_dirs:
        name = folder.name
        prefix = name.split("_", 1)[0] + "_" if name.startswith("pass") and "_" in name else ""
        out.setdefault(prefix, []).append(folder)
    return out


def cannot_link_pairs(co_visible: Dict[str, Dict[tuple, int]],
                      min_frames: int) -> set:
    """{frozenset of two (pass, index) keys} that SAM3 says are two objects.

    `co_visible[prefix][(a, b)]` counts the frames on which that pass showed
    ids a and b as separate masks. One shared frame could be a tracker
    hiccup; `min_frames` of them is SAM3 stating, repeatedly, that these are
    different things. 0 turns the constraint off.
    """
    if min_frames <= 0:
        return set()
    return {frozenset(((prefix, a), (prefix, b)))
            for prefix, table in co_visible.items()
            for (a, b), n in table.items() if n >= min_frames}


def merge_across_passes(per_pass: Dict[str, np.ndarray], counts: Dict[str, int],
                        overlap_fraction: float, cannot_link: Optional[set] = None,
                        stats: Optional[dict] = None) -> Dict[tuple, int]:
    """Match each pass's leaf labels to the other passes', by 3D overlap.

    Every pass labels the *same* cloud, so the correspondence the 2D tracker
    cannot carry across a pass boundary -- the sessions are independent and
    their ids are unrelated -- is recoverable here: two ids from two passes
    that claim the same points are the same leaf. This is the step that keeps
    a three-pass capture from reporting every leaf three times.

    **Links are taken strongest first, and one SAM3 contradicts is refused.**
    This used to be a union-find over every link above `overlap_fraction`,
    and a union-find is transitive: a small id that touches two leaves joins
    them, and on a crown every small id touches several. Measured on
    gaensefuss_1 (34 leaves on the flat lay), each pass alone carried 33-37
    ids winning >= 40 points, and the union-find chained them into 17 -- one
    "leaf" of 26 ids held 10 from pass 0 alone, 8 of them on screen at once as
    separate masks. `cannot_link` (see `cannot_link_pairs`) is that statement
    from SAM3: a merge that would put two such ids in one leaf is skipped,
    and the weaker link that wanted it simply does not happen. Same votes,
    same threshold: 17 -> 33 leaves, with blade lengths that rank-match the
    flat lay at the camera-derived scale. With no constraints the result is
    exactly the union-find's: the same links, merged in a different order.

    `stats`, if given, receives the number of links refused.

    Returns {(pass, local index) -> merged leaf id}.
    """
    cannot_link = cannot_link or set()
    links = []
    passes = sorted(per_pass)
    for a_index, first in enumerate(passes):
        for second in passes[a_index + 1:]:
            left, right = per_pass[first], per_pass[second]
            both = (left >= 0) & (right >= 0)
            if not both.any():
                continue
            # One pass over the co-labelled points gives the whole contingency
            # table; looping leaf pairs would be O(leaves^2) reads of the cloud.
            pairs, overlap = np.unique(
                np.stack([left[both], right[both]], axis=1), axis=0, return_counts=True)
            for (a, b), n in zip(pairs, overlap):
                ka, kb = (first, int(a)), (second, int(b))
                smaller = min(counts.get(ka, 0), counts.get(kb, 0))
                if smaller and n >= overlap_fraction * smaller:
                    links.append((n / smaller, int(n), ka, kb))
    # Strongest share first, larger absolute overlap breaking ties, then the
    # keys themselves so the result never depends on dict order.
    links.sort(key=lambda link: (-link[0], -link[1], link[2], link[3]))

    # Every leaf that won any points is entered first, merged or not. Seeding
    # only from the pairs that matched drops the leaf that one elevation sees
    # and the others do not -- it gets no group, and `_combine_passes` then
    # reads it as unlabelled and deletes it.
    members = {key: {key} for key, size in counts.items() if size > 0}
    owner = {key: key for key in members}
    refused = 0
    for _share, _n, ka, kb in links:
        ra, rb = owner[ka], owner[kb]
        if ra == rb:
            continue
        if cannot_link and any(frozenset((x, y)) in cannot_link
                               for x in members[ra] for y in members[rb]):
            refused += 1
            continue
        for key in members[rb]:
            owner[key] = ra
        members[ra] |= members.pop(rb)

    if stats is not None:
        stats["links_refused"] = refused
    roots, merged = {}, {}
    for key in sorted(owner):
        root = owner[key]
        if root not in roots:
            roots[root] = len(roots)
        merged[key] = roots[root]
    return merged


# A piece of a leaf rather than a leaf: at least this share of its points lie
# on a larger leaf's surface. Measured on vogelmeere (2026-10-07), with the
# occlusion test's own surface tolerance: the four ids that are a strip of
# another leaf's blade score 0.87-1.00; the next, separate pieces, 0.71 and
# below. Placed in that gap.
FRAGMENT_CONTACT = 0.8


def fragment_contacts(points: np.ndarray, assignment: np.ndarray,
                      tolerance: float) -> Dict[int, tuple]:
    """{leaf: (larger leaf, share of its points within `tolerance` of that one's)}.

    Only leaves whose share reaches FRAGMENT_CONTACT, each with the one larger
    leaf it lies on most. Contact is necessary but not enough: on a dense shoot
    tip a real small leaf lies against a bigger one too, so `fragment_votes`
    asks SAM3 as well.
    """
    from scipy.spatial import cKDTree

    ids, sizes = np.unique(assignment[assignment >= 0], return_counts=True)
    size = dict(zip(ids.tolist(), sizes.tolist()))
    member = {i: points[assignment == i] for i in size}
    low = {i: p.min(axis=0) - tolerance for i, p in member.items()}
    high = {i: p.max(axis=0) + tolerance for i, p in member.items()}
    trees: Dict[int, object] = {}
    out: Dict[int, tuple] = {}
    for a in size:
        best = (None, 0.0)
        for b in size:
            if size[b] <= size[a] or (high[a] < low[b]).any() or (high[b] < low[a]).any():
                continue
            if b not in trees:
                trees[b] = cKDTree(member[b])
            d = trees[b].query(member[a], distance_upper_bound=tolerance)[0]
            share = float(np.isfinite(d).mean())
            if share > best[1]:
                best = (b, share)
        if best[0] is not None and best[1] >= FRAGMENT_CONTACT:
            out[a] = (best[0], best[1])
    return out


def fragment_votes(points: np.ndarray, assignment: np.ndarray, pairs, reconstruction,
                   by_pass: Dict[str, List[Path]], frames_of_pass: Dict[str, set],
                   spacing: float, tolerance: float, min_pixels: int = 5) -> Dict[tuple, list]:
    """{(a, b): [frames SAM3 shows them as one object, frames as two]}.

    Per photo where both are visible: the SAM3 mask covering most of each. The
    same mask is a vote for one object, different masks a vote for two.
    Neighbouring leaves get different masks whenever both are on screen, so
    this is what keeps a real small leaf lying on a big one apart -- vogelmeere
    5 + 15 touch, and SAM3 drew them as one mask in 57 photos but as two in 15.
    """
    votes = {pair: [0, 0] for pair in pairs}
    wanted = np.array(sorted({x for pair in pairs for x in pair}))
    if not len(wanted):
        return votes
    blank = np.zeros((len(points), 3), np.uint8)
    for image_id in sorted(reconstruction.reg_image_ids()):
        image = reconstruction.images[image_id]
        stem = Path(image.name).stem
        camera = camera_from_colmap(image, reconstruction.cameras[image.camera_id])
        _rgb, index = render_points(points, blank, camera)
        if spacing > 0:
            index = visible_only(points, camera, index, spacing, tolerance)
        if not np.isin(assignment[index[index >= 0]], wanted).any():
            continue
        under: Dict[int, Dict[str, int]] = {}
        for prefix, dirs in by_pass.items():
            if prefix in frames_of_pass and stem not in frames_of_pass[prefix]:
                continue
            for folder in dirs:
                path = folder / f"{stem}.png"
                raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
                if raw is None:
                    continue
                hit = index[raw > 127]
                hit = assignment[hit[hit >= 0]]
                values, counts = np.unique(hit[np.isin(hit, wanted)], return_counts=True)
                for v, c in zip(values.tolist(), counts.tolist()):
                    under.setdefault(v, {})[folder.name] = c
        dominant = {leaf: max(t, key=t.get) for leaf, t in under.items()
                    if max(t.values()) >= min_pixels}
        for a, b in pairs:
            if a in dominant and b in dominant:
                votes[(a, b)][0 if dominant[a] == dominant[b] else 1] += 1
    return votes


def absorb_fragments(assignment: np.ndarray, contacts: Dict[int, tuple],
                     votes: Dict[tuple, list], min_frames: int = 2) -> Dict[int, int]:
    """Fold each fragment into the leaf it is a piece of. Returns {fragment: leaf}.

    A fragment lies on a larger leaf (`fragment_contacts`) and SAM3 shows the
    two as one object in most photos that see both (`fragment_votes`).

    Where they come from: SAM3 hands a leaf to a new id partway through a pass.
    On vogelmeere, pass0_24 tracked leaf 6's blade for frames 0-28; from frame
    33 the blade was pass0_64 and pass0_24 had shrunk onto the petiole. The
    per-point vote kept most of the blade with 24, but a strip seen broadside
    only at the end went to 64 -- 190 points, a leaf of their own, a second tip.
    Ids of one pass are never merged with each other (`merge_across_passes`
    only links passes), so nothing else rejoins them.

    Only ever the smaller into the larger, and only when most of the smaller
    lies on the larger: two whole leaves cannot qualify, which is what keeps
    this from the chaining that once merged gaensefuss_1's 34 leaves into 17.
    """
    absorbed: Dict[int, int] = {}
    for a, (b, _share) in contacts.items():
        together, apart = votes.get((a, b), [0, 0])
        if together >= min_frames and together > apart:
            absorbed[a] = b

    def final(x: int) -> int:
        while x in absorbed:
            x = absorbed[x]
        return x

    absorbed = {a: final(a) for a in absorbed}
    for a, b in absorbed.items():
        assignment[assignment == a] = b
    return absorbed


def _absorb(points, assignment, reconstruction, by_pass, frames_of_pass,
            spacing: float, tolerance: float) -> Dict[int, int]:
    """Find fragments, ask SAM3, fold them in, and say what happened."""
    contacts = fragment_contacts(points, assignment, tolerance)
    if not contacts:
        print("  no leaf lies mostly on another's surface -- nothing to absorb")
        return {}
    pairs = [(a, b) for a, (b, _share) in contacts.items()]
    votes = fragment_votes(points, assignment, pairs, reconstruction, by_pass,
                           frames_of_pass, spacing, tolerance)
    absorbed = absorb_fragments(assignment, contacts, votes)
    print(f"  {len(contacts)} leaf/leaves lie mostly (>= {FRAGMENT_CONTACT:.0%}) on a larger "
          f"leaf's surface; SAM3 shows {len(absorbed)} of them as one object with it:")
    for a, (b, share) in sorted(contacts.items()):
        together, apart = votes[(a, b)]
        print(f"    {a} on {b}: {share:.0%} contact, one mask in {together} photo(s), two in "
              f"{apart} -> {'absorbed' if a in absorbed else 'kept'}")
    return absorbed


def frames_by_pass(workdir: Path, by_pass: Dict[str, List[Path]]) -> Dict[str, set]:
    """prefix -> the frame stems that pass captured, from p1/sources.json.

    "pass2_" is capture pass 2 there, which is how a view is matched to the
    session whose ids describe it.
    """
    sources_file = workdir / "p1" / "sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else {}
    out: Dict[str, set] = {}
    for prefix in by_pass:
        if not prefix:
            continue
        index = int(prefix[len("pass"):].rstrip("_"))
        out[prefix] = {stem for stem, p in sources.items() if p == index}
    return out


def _combine_passes(per_pass: Dict[str, np.ndarray], merged: Dict[tuple, int],
                    n_points: int) -> np.ndarray:
    """One label per point, from what each pass said about it.

    Majority across the passes, with ties broken toward the more specific
    claim -- leaf over root over skeleton. A pass that never saw a point
    abstains rather than voting it unlabelled, so a leaf visible from only one
    elevation still gets its points.
    """
    assignment = np.full(n_points, UNSEEN, np.int32)
    if not per_pass:
        return assignment

    # Re-express every pass's labels in the merged vocabulary first, so the
    # tally below is comparing leaf identities rather than pass-local indices.
    translated = []
    for prefix, labels in sorted(per_pass.items()):
        out = labels.copy()
        leaves = labels >= 0
        if leaves.any():
            lookup = np.full(int(labels.max()) + 1, UNSEEN, np.int32)
            for (p, index), group in merged.items():
                if p == prefix and index < len(lookup):
                    lookup[index] = group
            out[leaves] = lookup[labels[leaves]]
        translated.append(out)

    stacked = np.stack(translated, axis=0)                    # (passes, points)
    priority = {UNSEEN: 0, SKELETON: 1, ROOT: 2}              # leaf beats all
    for i in range(n_points):
        column = stacked[:, i]
        column = column[column != UNSEEN]
        if not len(column):
            continue
        values, counts = np.unique(column, return_counts=True)
        best = counts.max()
        tied = values[counts == best]
        # Among equally supported answers take the most specific one, and the
        # lowest leaf id when several leaves tie, so the result is stable.
        assignment[i] = max(tied, key=lambda v: (priority.get(int(v), 3), -int(v)))
    return assignment


def run(
    workdir: Path,
    min_frames: int = 3,
    min_points: int = 40,
    normal_weighting: bool = True,
    geometry_backend: str = cloud_source.BASELINE,
    cloud: Optional[Path] = None,
    source: str = "auto",
    merge_overlap: float = 0.30,
    covisible_frames: int = 2,
    occlusion_test: bool = True,
    keep_fragments: bool = False,
) -> dict:
    import pycolmap

    p2 = workdir / "p2"
    p5x = workdir / "p5x"
    p5x.mkdir(parents=True, exist_ok=True)

    leaf_dirs = usable_instances(p2 / "masks" / "leaf_instances", min_frames)
    if not leaf_dirs:
        raise SystemExit(
            f"no leaf instances in {p2 / 'masks' / 'leaf_instances'} with at least "
            f"{min_frames} frames.\nThey are written by the SAM3 P2 backend:\n"
            f"    pose-segment --workdir {workdir} --backend sam3 --reuse-frames")

    stem_dir = p2 / "masks" / "stem"
    stem_dir = stem_dir if any(stem_dir.glob("*.png")) else None
    root_dir = p2 / "masks" / "root"
    root_dir = root_dir if any(root_dir.glob("*.png")) else None

    by_pass = group_by_pass(leaf_dirs)
    sources_file = workdir / "p1" / "sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else {}
    n_capture_passes = len(set(sources.values())) if sources else 1
    if "" in by_pass and n_capture_passes > 1:
        raise SystemExit(
            f"{p2 / 'masks' / 'leaf_instances'} has un-prefixed instance folders but this "
            f"capture has {n_capture_passes} passes.\nEach pass is its own SAM3 session with "
            "its own ids starting at zero, so those folders each hold several unrelated\n"
            "leaves merged together and cannot be projected. Re-run P2:\n"
            f"    pose-segment --workdir {workdir} --backend sam3 --reuse-frames")
    frames_of_pass = frames_by_pass(workdir, by_pass)

    print(f"  {len(leaf_dirs)} leaf instance(s) with >= {min_frames} frames "
          f"across {len(by_pass)} capture pass(es)"
          + (", stem residual" if stem_dir else ", NO stem masks")
          + (", root tracked" if root_dir else ", no root"))
    widest = max(len(v) for v in by_pass.values())
    if widest + 2 > MAX_VOTED_CLASSES:
        raise SystemExit(
            f"{widest} leaf instances in one pass survive --min-frames {min_frames}, and the "
            f"vote can carry {MAX_VOTED_CLASSES - 2}.\nThat many ids on one plant is "
            "fragmentation rather than leaves -- raise --min-frames.")

    chosen = cloud_source.resolve(workdir, geometry_backend, cloud, source)
    cloud_path = chosen.path
    if not cloud_path.exists():
        raise SystemExit(
            f"{cloud_path} not found ({chosen.origin}) -- run pose-hull and ideally "
            "pose-surface first")
    print(f"  projecting onto {cloud_path} -- {chosen.origin}")

    fields = read_ply_vertices(cloud_path)
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)

    normals = None
    if normal_weighting:
        if all(k in fields for k in ("nx", "ny", "nz")):
            normals = np.stack([fields["nx"], fields["ny"], fields["nz"]],
                               axis=1).astype(np.float64)
        else:
            normals = estimate_normals(points).astype(np.float64)
            print("  no normals in the cloud; estimated by local PCA")

    surfel_file = workdir / "p4b" / "surfels.npz"
    if surfel_file.exists() and chosen.is_baseline:
        surfels = np.load(surfel_file)
        photo_rgb = colors_from_surfels(points, surfels["means"], surfels["colors"])
    else:
        photo_rgb = np.full((len(points), 3), 160, np.uint8)

    # Occlusion: a point votes only in views where it is the visible surface.
    # Both scales are measured on this cloud -- see `visible_only`.
    spacing = tolerance = 0.0
    hidden = [0, 0]                         # pixels dropped as hidden, pixels drawn
    if occlusion_test:
        spacing, thickness = surface_scale(points)
        tolerance = 2.0 * thickness
        print(f"  occlusion test on: point spacing {spacing:.5f}, sheet thickness "
              f"{thickness:.5f}; a point votes only within {tolerance:.5f} of the visible surface")

    sparse_model, _ = cloud_source.geometry(workdir, geometry_backend)
    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    image_ids = sorted(reconstruction.reg_image_ids())
    print(f"  {len(points)} points, {len(image_ids)} registered views")

    # One vote per capture pass, not one over all of them. A pass's ids mean
    # nothing to another pass, so pooling their votes would have two different
    # leaves arguing over the same class index.
    per_pass: Dict[str, np.ndarray] = {}
    leaf_counts: Dict[tuple, int] = {}
    co_visible: Dict[str, Dict[tuple, int]] = {}
    used = 0
    for prefix, dirs in sorted(by_pass.items()):
        stem_class, root_class = len(dirs), len(dirs) + 1
        tally = accumulate_votes(len(points), len(dirs) + 2)
        co_visible[prefix] = {}
        seen_here = 0
        # Only this pass's own frames. The stem and root masks exist for every
        # frame in the workdir, so a frame from another pass still produces a
        # label map -- one with no leaf ids on it, because this pass has none
        # there. Letting those vote means a leaf that only pass 1 can see
        # collects skeleton votes from passes 0 and 2, and the majority in
        # `_combine_passes` then calls a leaf the stem.
        mine = frames_of_pass.get(prefix)
        for image_id in image_ids:
            image = reconstruction.images[image_id]
            if mine is not None and Path(image.name).stem not in mine:
                continue
            camera = camera_from_colmap(image, reconstruction.cameras[image.camera_id])
            label_map = build_label_map(dirs, stem_dir, root_dir,
                                        Path(image.name).stem,
                                        (camera.height, camera.width),
                                        stem_class, root_class,
                                        co_visible=co_visible[prefix])
            if label_map is None:
                continue
            _rgb, index_map = render_points(points, photo_rgb, camera)
            if occlusion_test:
                drawn = int((index_map >= 0).sum())
                index_map = visible_only(points, camera, index_map, spacing, tolerance)
                hidden[0] += drawn - int((index_map >= 0).sum())
                hidden[1] += drawn
            weights = view_weights(points, normals, camera) if normals is not None else None
            cast_votes(tally, index_map, label_map, weights)
            seen_here += 1
        used += seen_here

        voted = finalise_votes(tally).labels.astype(np.int32)
        labels = np.full(len(points), UNSEEN, np.int32)
        is_leaf = (voted >= 0) & (voted < len(dirs))
        labels[is_leaf] = voted[is_leaf]
        labels[voted == stem_class] = SKELETON
        labels[voted == root_class] = ROOT
        per_pass[prefix] = labels
        for index in range(len(dirs)):
            leaf_counts[(prefix, index)] = int((labels == index).sum())
        print(f"    pass {prefix or '(single)'}: {len(dirs)} ids over {seen_here} views, "
              f"{int((labels >= 0).sum())} leaf points")

    if used == 0:
        raise SystemExit(
            "none of the registered views had a leaf-instance mask -- the frame names in "
            f"{sparse_model} do not match the mask filenames in {p2 / 'masks'}")

    if occlusion_test and hidden[1]:
        print(f"  occlusion test dropped {hidden[0] / hidden[1]:.1%} of rendered pixels: points "
              "hidden behind the visible surface, which would have voted for what is in front")

    total_views = sum(len(frames_of_pass.get(prefix, image_ids)) for prefix in by_pass)
    cannot_link = cannot_link_pairs(co_visible, covisible_frames)
    merge_stats: dict = {}
    merged = merge_across_passes(per_pass, leaf_counts, merge_overlap,
                                 cannot_link=cannot_link, stats=merge_stats)
    if len(by_pass) > 1:
        n_groups = len(set(merged.values())) if merged else 0
        print(f"  {len(leaf_counts)} per-pass ids merged into {n_groups} leaves by 3D overlap "
              f"(>= {merge_overlap:.0%} of the smaller)")
        if covisible_frames > 0:
            print(f"    {len(cannot_link)} same-pass pairs on screen together as separate masks "
                  f"on >= {covisible_frames} frames; {merge_stats['links_refused']} merge(s) "
                  "refused for joining such a pair")

    assignment = _combine_passes(per_pass, merged, len(points))

    # A leaf that won only a handful of points is not a leaf in 3D whatever it
    # was in 2D; its points are more usefully skeleton than a spurious organ.
    dropped = 0
    for index in sorted({int(v) for v in np.unique(assignment) if v >= 0}):
        mask = assignment == index
        if 0 < int(mask.sum()) < min_points:
            assignment[mask] = SKELETON
            dropped += 1

    # A strip of one leaf that SAM3 tracked under a second id is not a leaf.
    absorbed: Dict[int, int] = {}
    if not keep_fragments:
        if not occlusion_test:
            spacing, thickness = surface_scale(points)
            tolerance = 2.0 * thickness
        absorbed = _absorb(points, assignment, reconstruction, by_pass, frames_of_pass,
                           spacing if occlusion_test else 0.0, tolerance)

    # --- upright, and in the same frame P5 draws in ---
    #
    # The cloud is in the reconstruction's own frame, where "up" is wherever
    # COLMAP's first camera happened to look -- which is why an unrotated P5x
    # scene lies on its side in Blender. P5 already solves the upright plant
    # frame and records it, so it is read rather than solved again: two scenes
    # built from one capture must not disagree about which way the plant grew.
    frame = _plant_frame(workdir, geometry_backend)
    if frame is not None:
        origin, rotation = frame
        points = (points - origin) @ rotation.T
        print("  rotated into P5's upright plant frame (origin on the clamp line)")
    else:
        print("  WARNING: no p5/stem_graph.json, so the cloud is drawn in the raw")
        print("    reconstruction frame and will not stand upright. Run pose-structure,")
        print("    or accept an arbitrarily oriented scene.")

    # --- the skeleton, traced through tissue whose identity is already known ---
    voxel, voxel_origin = cloud_source.voxel_size(workdir, geometry_backend, points)
    architecture = _architecture(workdir, geometry_backend)
    skeleton = trace(points, assignment, voxel, architecture=architecture)
    skeleton["architecture"] = architecture
    if skeleton.get("crown_moved_to_largest_component"):
        moved = skeleton["crown_moved_to_largest_component"]
        print(f"  the crown fell on a {moved['from_component_size']}-point island, so it was "
              f"moved to the foot of the {moved['to_component_size']}-point plant body")
    if skeleton.get("graph_components", 1) > 1:
        print(f"  the cloud is {skeleton['graph_components']} disconnected pieces -- the clamp "
              "cuts the root off, and a thin petiole can be carved through")
    if skeleton["unreachable"]:
        print(f"  {len(skeleton['unreachable'])} leaf/leaves never reach the crown through "
              f"the cloud: {skeleton['unreachable']}")
        print("    their bridge to the plant was carved away, so they get no midrib.")
    _print_tree(skeleton, voxel)
    if skeleton["leaves"]:
        heights = [leaf["height"] for leaf in skeleton["leaves"]]
        print(f"  traced {len(skeleton['leaves'])} midribs from the crown; "
              f"tips span z {min(heights):.3f}..{max(heights):.3f} ({voxel_origin} voxel "
              f"{voxel:.5f})")

    surviving = sorted({int(v) for v in np.unique(assignment) if v >= 0})
    report = _write_outputs(p5x, points, assignment, surviving, leaf_dirs,
                            photo_rgb, used, total_views, dropped, min_points,
                            skeleton=skeleton,
                            merge={"overlap_fraction": merge_overlap,
                                   "covisible_frames": covisible_frames,
                                   "fragments_absorbed": {str(a): b for a, b in absorbed.items()},
                                   "per_pass_ids": len(leaf_counts),
                                   "groups": len(set(merged.values())) if merged else 0,
                                   "cannot_link_pairs": len(cannot_link),
                                   "links_refused": merge_stats.get("links_refused", 0)},
                            visibility={"occlusion_test": occlusion_test,
                                        "point_spacing": spacing, "depth_tolerance": tolerance,
                                        "hidden_pixels_dropped": (hidden[0] / hidden[1]
                                                                  if hidden[1] else 0.0)})

    print_checks("P5x", report)
    print(f"\n  {len(surviving)} leaf instance(s) in 3D, "
          f"{int((assignment == SKELETON).sum())} skeleton points, "
          f"{int((assignment == ROOT).sum())} root points, "
          f"{int((assignment == UNSEEN).sum())} unseen")
    print(f"  open in Blender: {p5x / 'segmented.ply'}  (leaves + skeleton)")
    print(f"                   {p5x / 'leaves.ply'} / {p5x / 'skeleton.ply'} separately")
    return report


def _architecture(workdir: Path, geometry_backend: str) -> Optional[str]:
    """P5's --architecture, read from its output like the plant frame is: it
    decides whether the plant has a stem the petioles branch off, or leaves
    that meet at a crown (rosette)."""
    p5 = (workdir / "p5" if geometry_backend in (None, "", cloud_source.BASELINE)
          else workdir / "p5" / "experiments" / geometry_backend)
    path = p5 / "instancing.json"
    if not path.exists():
        return plant_profiles.profile_for(workdir)[0].architecture
    return plant_profiles.normalise_architecture(json.loads(path.read_text()).get("architecture"))


def _write_curves(path: Path, polylines, colours) -> None:
    """Polylines as a PLY edge list -- real curves, not a cloud of samples.

    A midrib drawn as points looks like thin tissue that happened to be
    labelled; drawn as a line it reads as the measurement it is, and Blender
    can bevel it into a tube. `edge` elements are the standard way to say so
    and the repo's own reader ignores them, so the file stays loadable
    everywhere it was before.
    """
    usable = [(np.asarray(line, float), colour)
              for line, colour in zip(polylines, colours) if len(np.asarray(line)) >= 2]
    if not usable:
        return
    vertices, edges, rgb = [], [], []
    for line, colour in usable:
        start = len(vertices)
        vertices.extend(line)
        rgb.extend([colour] * len(line))
        edges.extend((start + i, start + i + 1) for i in range(len(line) - 1))

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for (x, y, z), colour in zip(vertices, rgb):
            f.write(f"{x} {y} {z} {int(colour[0])} {int(colour[1])} {int(colour[2])}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")


def _plant_frame(workdir: Path, geometry_backend: str):
    """(origin, rotation) from P5's stem graph, or None if it has not run.

    Read, not re-solved. `solve_plant_frame` needs the sparse model, the orbit
    and the holder masks, and re-deriving it here would give a second answer
    to a question the workdir has already answered -- two scenes of one
    capture disagreeing about which way is up is worse than one that is not
    upright at all.
    """
    p5 = (workdir / "p5" if geometry_backend in (None, "", cloud_source.BASELINE)
          else workdir / "p5" / "experiments" / geometry_backend)
    graph_path = p5 / "stem_graph.json"
    if not graph_path.exists():
        return None
    frame = json.loads(graph_path.read_text()).get("plant_frame")
    if not frame:
        return None
    return np.asarray(frame["origin"], float), np.asarray(frame["rotation"], float)


def _write_outputs(p5x: Path, points, assignment, surviving, leaf_dirs,
                   photo_rgb, views_used, views_total, dropped, min_points,
                   skeleton=None, merge=None, visibility=None) -> dict:
    palette = leaf_palette(max(len(leaf_dirs), 1))

    rgb = np.zeros((len(points), 3), np.uint8)
    rgb[:] = (70, 70, 70)                                  # unseen
    rgb[assignment == SKELETON] = SKELETON_RGB
    rgb[assignment == ROOT] = ROOT_RGB
    for index in surviving:
        rgb[assignment == index] = palette[index % len(palette)]

    def dump(path: Path, keep: np.ndarray, colours: np.ndarray,
             with_label: bool = False) -> int:
        if not keep.any():
            return 0
        fields = {
            "x": points[keep, 0].astype(np.float32),
            "y": points[keep, 1].astype(np.float32),
            "z": points[keep, 2].astype(np.float32),
            "red": colours[keep, 0], "green": colours[keep, 1], "blue": colours[keep, 2],
        }
        if with_label:
            # Carried in the file itself, not only in instances.npy, so a
            # reader that has the PLY has the segmentation. The Blender
            # viewer runs in Blender's bundled Python and splits the cloud
            # into one object per leaf from this column; matching a colour
            # back to a leaf would be guessing at what the palette did.
            fields["label"] = assignment[keep].astype(np.int32)
        write_ply_vertices(path, fields)
        return int(keep.sum())

    n_leaf_points = dump(p5x / "leaves.ply", assignment >= 0, rgb, with_label=True)
    n_skeleton = dump(p5x / "skeleton.ply", assignment == SKELETON, rgb)
    dump(p5x / "segmented.ply", assignment != UNSEEN, rgb, with_label=True)
    np.save(p5x / "instances.npy", assignment)

    # The skeleton as curves, one polyline per leaf, so Blender can draw the
    # midribs as curves rather than as another cloud of points.
    skeleton = skeleton or {"crown": None, "leaves": [], "unreachable": []}
    with open(p5x / "skeleton.json", "w") as f:
        json.dump(skeleton, f, indent=2)
    _write_curves(p5x / "midribs.ply", [leaf["midrib"] for leaf in skeleton["leaves"]],
                  [palette[leaf["id"] % len(palette)] for leaf in skeleton["leaves"]])
    _write_curves(p5x / "petioles.ply", [leaf["petiole"] for leaf in skeleton["leaves"]],
                  [SKELETON_RGB] * len(skeleton["leaves"]))

    per_leaf = {str(i): int((assignment == i).sum()) for i in surviving}
    unseen = int((assignment == UNSEEN).sum())
    checks = {
        "every_leaf_survived_to_3d": {
            "pass": len(surviving) > 0,
            "detail": f"{len(surviving)} of {len(leaf_dirs)} 2D instances won at least "
                      f"{min_points} points"
                      + (f"; {dropped} fell below that and became skeleton" if dropped else ""),
        },
        "skeleton_is_not_empty": {
            "pass": n_skeleton > 0,
            "detail": f"{n_skeleton} points voted stem. An empty skeleton means p2/masks/stem "
                      "was empty or never projected -- there is nothing to build a "
                      "centreline along.",
        },
        # A large unseen share is the honest signal that the 2D masks and the
        # cloud do not describe the same object -- a stale P2, or a solve whose
        # frames were renamed -- and it is invisible in the PLYs, which simply
        # look sparse.
        "most_points_were_labelled": {
            "pass": unseen < 0.25 * len(points),
            "detail": f"{unseen} of {len(points)} points ({unseen / max(len(points), 1):.1%}) "
                      "were never seen under any leaf, stem or root pixel",
        },
        "views_had_masks": {
            "pass": views_used >= 0.8 * views_total,
            "detail": f"{views_used}/{views_total} registered views had 2D masks to vote with",
        },
    }
    report = {
        "num_points": int(len(points)),
        "midribs_traced": len(skeleton["leaves"]),
        "leaves_not_reaching_the_crown": skeleton["unreachable"],
        "crown": skeleton["crown"],
        "num_leaves_2d": len(leaf_dirs),
        "num_leaves_3d": len(surviving),
        "merge": merge or {},
        "visibility": visibility or {},
        "points_per_leaf": per_leaf,
        "skeleton_points": n_skeleton,
        "root_points": int((assignment == ROOT).sum()),
        "unseen_points": unseen,
        "leaf_points": n_leaf_points,
        "views_fused": views_used,
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
    }
    with open(p5x / "instances.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def retrace(workdir: Path, geometry_backend: str = cloud_source.BASELINE,
            keep_fragments: bool = False) -> dict:
    """Redo everything after the votes, from the labels P5x already wrote.

    p5x/segmented.ply carries every point's leaf/stem label in P5's plant
    frame, which is all `trace` reads. So when only the tracer changed, this
    rewrites skeleton.json, midribs.ply and petioles.ply instead of projecting
    every mask again (11 min on vogelmeere).

    Fragments are folded in first (`absorb_fragments`, unless
    `keep_fragments`). That asks SAM3, so it renders every view once -- about
    2.5 min on vogelmeere locally -- and when it folds anything in it rewrites
    instances.npy, segmented.ply, leaves.ply and instances.json to match.
    """
    p5x = workdir / "p5x"
    if not keep_fragments:
        _retrace_absorb(workdir, geometry_backend, p5x)
    cloud = p5x / "segmented.ply"
    if not cloud.exists():
        raise SystemExit(f"{cloud} not found -- run pose-leaf-instances once without --retrace")
    fields = read_ply_vertices(cloud)
    if "label" not in fields:
        raise SystemExit(f"{cloud} has no label column; it predates --retrace. Re-run P5x fully.")
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
    assignment = np.asarray(fields["label"], np.int64)

    voxel, voxel_origin = cloud_source.voxel_size(workdir, geometry_backend, points)
    architecture = _architecture(workdir, geometry_backend)
    print(f"  {plant_profiles.describe(*plant_profiles.profile_for(workdir))}")
    print(f"  re-tracing {len(points)} points from {cloud.name}; architecture {architecture}, "
          f"voxel {voxel:.5f} ({voxel_origin})")
    skeleton = trace(points, assignment, voxel, architecture=architecture)
    skeleton["architecture"] = architecture

    report_path = p5x / "instances.json"
    n_2d = json.loads(report_path.read_text()).get("num_leaves_2d", 1) if report_path.exists() else 1
    palette = leaf_palette(max(int(n_2d), 1))
    with open(p5x / "skeleton.json", "w") as f:
        json.dump(skeleton, f, indent=2)
    _write_curves(p5x / "midribs.ply", [leaf["midrib"] for leaf in skeleton["leaves"]],
                  [palette[leaf["id"] % len(palette)] for leaf in skeleton["leaves"]])
    _write_curves(p5x / "petioles.ply", [leaf["petiole"] for leaf in skeleton["leaves"]],
                  [SKELETON_RGB] * len(skeleton["leaves"]))
    _print_tree(skeleton, voxel)
    print(f"  wrote {p5x / 'skeleton.json'}, midribs.ply, petioles.ply")
    return skeleton


def _retrace_absorb(workdir: Path, geometry_backend: str, p5x: Path) -> None:
    """`absorb_fragments` on a finished P5x, rewriting its label files if it folds anything."""
    import pycolmap

    report_path = p5x / "instances.json"
    if not (p5x / "instances.npy").exists() or not report_path.exists():
        print("  no p5x/instances.npy -- fragments not checked")
        return
    report = json.loads(report_path.read_text())
    chosen = cloud_source.resolve(workdir, geometry_backend, None, "auto")
    fields = read_ply_vertices(chosen.path)
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
    assignment = np.load(p5x / "instances.npy")
    if len(assignment) != len(points):
        print(f"  p5x/instances.npy has {len(assignment)} rows but {chosen.path.name} has "
              f"{len(points)} -- P5x is stale; fragments not checked")
        return
    visibility = report.get("visibility") or {}
    tolerance = visibility.get("depth_tolerance") or 2.0 * surface_scale(points)[1]
    spacing = visibility.get("point_spacing", 0.0) if visibility.get("occlusion_test") else 0.0
    by_pass = group_by_pass(usable_instances(workdir / "p2" / "masks" / "leaf_instances", 1))
    if not by_pass:
        print("  no p2/masks/leaf_instances to ask SAM3 with -- fragments not checked")
        return
    sparse_model, _ = cloud_source.geometry(workdir, geometry_backend)
    absorbed = _absorb(points, assignment, pycolmap.Reconstruction(str(sparse_model)), by_pass,
                       frames_by_pass(workdir, by_pass), spacing, tolerance)
    if not absorbed:
        return

    np.save(p5x / "instances.npy", assignment)
    palette = leaf_palette(max(int(report.get("num_leaves_2d", 1)), 1))
    for name in ("segmented.ply", "leaves.ply"):
        ply = read_ply_vertices(p5x / name)
        label = np.asarray(ply["label"]).copy()
        for a, b in absorbed.items():
            hit = label == a
            label[hit] = b
            for channel, value in zip(("red", "green", "blue"), palette[b % len(palette)]):
                ply[channel][hit] = value
        ply["label"] = label.astype(np.int32)
        write_ply_vertices(p5x / name, ply)
    per_leaf = {str(i): int((assignment == i).sum())
                for i in sorted({int(v) for v in np.unique(assignment) if v >= 0})}
    report["points_per_leaf"] = per_leaf
    report["num_leaves_3d"] = len(per_leaf)
    merge = report.setdefault("merge", {})
    merge["fragments_absorbed"] = {**merge.get("fragments_absorbed", {}),
                                   **{str(a): b for a, b in absorbed.items()}}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  rewrote instances.npy, segmented.ply, leaves.ply, instances.json: "
          f"{len(per_leaf)} leaves")


def _print_tree(skeleton: dict, voxel: float) -> None:
    """The stem tree and where each petiole came from, as numbers."""
    tree = skeleton.get("stem_tree")
    leaves = skeleton["leaves"]
    if tree:
        kinds = tree["axes"]
        print(f"  stem tree: {kinds.get('branch', 0)} branch(es), {kinds.get('petiole', 0)} "
              f"leaf stalk(s), {tree['forks']} forks, {tree['length']:.4f} long; stem radius "
              f"{tree['radius']:.5f} ({tree['radius'] / voxel:.1f} voxels), shells "
              f"{tree['shell']:.5f}")
    elif skeleton.get("architecture") != "rosette":
        print("  stem tree: none -- no stem tissue to build it from")
    lengths = np.array([leaf["petiole_length"] for leaf in leaves]) if leaves else np.zeros(0)
    stalks = sum(1 for leaf in leaves if leaf.get("petiole_from") == "stalk")
    print(f"  petioles: {stalks} from a stalk of their own, {len(leaves) - stalks} from the "
          f"path to the plant; {int((lengths == 0).sum())} of {len(leaves)} empty"
          + (f"; median length {np.median(lengths[lengths > 0]):.4f}" if (lengths > 0).any() else ""))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-frames", type=int, default=3,
                        help="ignore a 2D leaf id present on fewer frames than this. An id "
                             "on one or two frames is a tracking fragment, not a leaf")
    parser.add_argument("--min-points", type=int, default=40,
                        help="a leaf winning fewer 3D points than this becomes skeleton")
    parser.add_argument("--merge-overlap", type=float, default=0.30,
                        help="two ids from different capture passes are the same leaf when "
                             "they claim this fraction of the smaller one's points. Each pass "
                             "is an independent SAM3 session, so its ids mean nothing to the "
                             "others and the match has to be made in 3D")
    parser.add_argument("--covisible-frames", type=int, default=2,
                        help="refuse a merge that would join two ids of one pass that SAM3 "
                             "showed as separate masks on at least this many frames -- SAM3 "
                             "saying they are two objects. 0 turns it off, which is the old "
                             "transitive merge that chained a crown into one leaf")
    parser.add_argument("--no-occlusion-test", action="store_true",
                        help="let a point vote in views where it is hidden behind another "
                             "surface, as before. Only for comparison: on gaensefuss_1 a quarter "
                             "of rendered pixels were such hidden points")
    parser.add_argument("--no-normal-weighting", action="store_true",
                        help="count every view equally instead of weighting by how "
                             "broad-side the surface is to it. Only for comparison: an "
                             "edge-on camera cannot see a blade as a blade")
    parser.add_argument("--keep-fragments", action="store_true",
                        help="keep a leaf that lies on another leaf's surface and that SAM3 "
                             "shows as one object with it, instead of folding it in. Such a "
                             "piece is a strip of one leaf tracked under a second SAM3 id, "
                             "and as its own leaf it gets a second tip")
    parser.add_argument("--geometry-backend", default=cloud_source.BASELINE)
    parser.add_argument("--cloud", type=Path, help="explicit point cloud to label")
    parser.add_argument("--source", default="auto",
                        help="which cloud to prefer: auto, surface (P4b) or hull (P4a)")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    add_arguments(parser)
    parser.add_argument("--retrace", action="store_true",
                        help="re-trace the skeleton (stem tree, midribs, petioles) from "
                             "p5x/segmented.ply and stop -- seconds, for when only the "
                             "tracer changed")
    args = parser.parse_args(argv)
    if args.retrace:
        retrace(args.workdir, args.geometry_backend, keep_fragments=args.keep_fragments)
        return
    run(workdir=args.workdir, min_frames=args.min_frames, min_points=args.min_points,
        keep_fragments=args.keep_fragments,
        normal_weighting=not args.no_normal_weighting,
        occlusion_test=not args.no_occlusion_test,
        geometry_backend=args.geometry_backend, cloud=args.cloud, source=args.source,
        merge_overlap=args.merge_overlap, covisible_frames=args.covisible_frames)


if __name__ == "__main__":
    main()
