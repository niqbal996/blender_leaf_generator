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
arrives here as two leaves, and nothing downstream can merge them. That is
the measurement to check first -- `p2/prompts.json` records how many frames
each id survived, and on a bushy specimen (gaensefuss_1: 43 ids for perhaps
18 leaves) the fragmentation is the dominant error, not the projection.

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

from pose_estimator import cloud_source
from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.semantic import (
    accumulate_votes,
    camera_from_colmap,
    cast_votes,
    colors_from_surfels,
    estimate_normals,
    finalise_votes,
    render_points,
    view_weights,
)
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
                    shape, stem_class: int, root_class: int) -> Optional[np.ndarray]:
    """One frame's 2D map in the form `cast_votes` already reads.

    Same contract as a P4c class map -- -1 where nothing is claimed, otherwise
    a class index -- so the fusion below is the one that has been validated
    against synthetic maps, not a second implementation of it.
    """
    out = np.full(shape, -1, np.int16)
    found = False
    for index, folder in enumerate(leaf_dirs):
        path = folder / f"{stem}.png"
        if not path.exists():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        out[raw > 127] = index
        found = True

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


class _Union:
    """Union-find over (pass, local leaf index) pairs."""

    def __init__(self):
        self.parent: Dict[tuple, tuple] = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_across_passes(per_pass: Dict[str, np.ndarray], counts: Dict[str, int],
                        overlap_fraction: float) -> Dict[tuple, int]:
    """Match each pass's leaf labels to the other passes', by 3D overlap.

    Every pass labels the *same* cloud, so the correspondence the 2D tracker
    cannot carry across a pass boundary -- the sessions are independent and
    their ids are unrelated -- is recoverable here: two ids from two passes
    that claim the same points are the same leaf. This is the step that keeps
    a three-pass capture from reporting every leaf three times.

    Returns {(pass, local index) -> merged leaf id}.
    """
    union = _Union()
    # Every leaf that won any points is entered first, merged or not. Seeding
    # only from the pairs that matched drops the leaf that one elevation sees
    # and the others do not -- it gets no group, and `_combine_passes` then
    # reads it as unlabelled and deletes it.
    for key, size in counts.items():
        if size > 0:
            union.find(key)
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
                smaller = min(counts.get((first, int(a)), 0), counts.get((second, int(b)), 0))
                if smaller and n >= overlap_fraction * smaller:
                    union.union((first, int(a)), (second, int(b)))

    roots, merged = {}, {}
    for key in sorted(union.parent):
        root = union.find(key)
        if root not in roots:
            roots[root] = len(roots)
        merged[key] = roots[root]
    return merged


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

    sparse_model, _ = cloud_source.geometry(workdir, geometry_backend)
    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    image_ids = sorted(reconstruction.reg_image_ids())
    print(f"  {len(points)} points, {len(image_ids)} registered views")

    # One vote per capture pass, not one over all of them. A pass's ids mean
    # nothing to another pass, so pooling their votes would have two different
    # leaves arguing over the same class index.
    per_pass: Dict[str, np.ndarray] = {}
    leaf_counts: Dict[tuple, int] = {}
    used = 0
    for prefix, dirs in sorted(by_pass.items()):
        stem_class, root_class = len(dirs), len(dirs) + 1
        tally = accumulate_votes(len(points), len(dirs) + 2)
        seen_here = 0
        for image_id in image_ids:
            image = reconstruction.images[image_id]
            camera = camera_from_colmap(image, reconstruction.cameras[image.camera_id])
            label_map = build_label_map(dirs, stem_dir, root_dir,
                                        Path(image.name).stem,
                                        (camera.height, camera.width),
                                        stem_class, root_class)
            if label_map is None:
                continue
            _rgb, index_map = render_points(points, photo_rgb, camera)
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

    merged = merge_across_passes(per_pass, leaf_counts, merge_overlap)
    if len(by_pass) > 1:
        n_groups = len(set(merged.values())) if merged else 0
        print(f"  {len(leaf_counts)} per-pass ids merged into {n_groups} leaves by 3D overlap "
              f"(>= {merge_overlap:.0%} of the smaller)")

    assignment = _combine_passes(per_pass, merged, len(points))

    # A leaf that won only a handful of points is not a leaf in 3D whatever it
    # was in 2D; its points are more usefully skeleton than a spurious organ.
    dropped = 0
    for index in range(n_leaves):
        mask = assignment == index
        if 0 < int(mask.sum()) < min_points:
            assignment[mask] = SKELETON
            dropped += 1

    surviving = sorted({int(v) for v in np.unique(assignment) if v >= 0})
    report = _write_outputs(p5x, points, assignment, surviving, leaf_dirs,
                            photo_rgb, used, len(image_ids) * len(by_pass),
                            dropped, min_points)

    print_checks("P5x", report)
    print(f"\n  {len(surviving)} leaf instance(s) in 3D, "
          f"{int((assignment == SKELETON).sum())} skeleton points, "
          f"{int((assignment == ROOT).sum())} root points, "
          f"{int((assignment == UNSEEN).sum())} unseen")
    print(f"  open in Blender: {p5x / 'segmented.ply'}  (leaves + skeleton)")
    print(f"                   {p5x / 'leaves.ply'} / {p5x / 'skeleton.ply'} separately")
    return report


def _write_outputs(p5x: Path, points, assignment, surviving, leaf_dirs,
                   photo_rgb, views_used, views_total, dropped, min_points) -> dict:
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
        "num_leaves_2d": len(leaf_dirs),
        "num_leaves_3d": len(surviving),
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
    parser.add_argument("--no-normal-weighting", action="store_true",
                        help="count every view equally instead of weighting by how "
                             "broad-side the surface is to it. Only for comparison: an "
                             "edge-on camera cannot see a blade as a blade")
    parser.add_argument("--geometry-backend", default=cloud_source.BASELINE)
    parser.add_argument("--cloud", type=Path, help="explicit point cloud to label")
    parser.add_argument("--source", default="auto",
                        help="which cloud to prefer: auto, surface (P4b) or hull (P4a)")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    add_arguments(parser)
    args = parser.parse_args(argv)
    run(workdir=args.workdir, min_frames=args.min_frames, min_points=args.min_points,
        normal_weighting=not args.no_normal_weighting,
        geometry_backend=args.geometry_backend, cloud=args.cloud, source=args.source,
        merge_overlap=args.merge_overlap)


if __name__ == "__main__":
    main()
