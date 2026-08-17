"""Find leaf tips in the photographs and vote them onto the cloud.

    pose-tips --workdir runs/plant_9 --checkpoint checkpoints/sam2.1_hiera_large.pt

Reads p1/frames, p2/masks/plant, p4c/class_maps, p3/sparse/best and the P4b
surface. Writes into <workdir>/p4c:
    tips3d.json    voted tip positions, with the vote count behind each
    tips3d.ply     the same, for Blender     <- open this
    diag/tips_*.jpg  the frames with each accepted tip drawn on

Why this exists: two leaves lying on top of each other merge into one sheet in
the carved surface, so nothing done to that sheet can separate them. In a
photograph they are still two objects, and SAM2 tells them apart. This finds
each leaf's tip where the evidence is and carries it into 3D by voting, which
is the same trade P4c makes for organ class.

Runtime is dominated by SAM2, roughly a second or two per frame. `--stride 4`
is usually plenty: a tip only has to be seen from enough directions to
out-vote the views that disagree, not from all of them.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_estimator.classify2d import load_class_map, read_manifest
from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.semantic import camera_from_colmap, render_points
from pose_estimator.tip_votes import (
    TipCluster,
    TipVote,
    cluster_votes,
    mask_facing,
    leaf_masks_in_frame,
    point_at_pixel,
    tip_pixel,
    write_tips,
)


def run(
    workdir: Path,
    checkpoint: Path,
    stride: int = 4,
    min_votes: int = 3,
    min_hit_rate: float = 0.0,
    cluster_voxels: float = 6.0,
    points_per_side: int = 32,
    device: str = "cuda",
) -> dict:
    import pycolmap
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    from pose_estimator.segmentation import _resolve_model_cfg

    p4c = workdir / "p4c"
    (p4c / "diag").mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(p4c)
    order = manifest["class_order"]
    if "leaf" not in order or "stem" not in order:
        raise SystemExit(f"class maps have {order}; leaf and stem are both needed")
    leaf_class, stem_class = order.index("leaf"), order.index("stem")

    surface = workdir / "p4b" / "surface.ply"
    cloud_path = surface if surface.exists() else workdir / "p4" / "hull_points.ply"
    fields = read_ply_vertices(cloud_path)
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
    colors = np.full((len(points), 3), 160, np.uint8)
    normals = (np.stack([fields["nx"], fields["ny"], fields["nz"]], axis=1).astype(np.float64)
               if all(k in fields for k in ("nx", "ny", "nz")) else None)
    if normals is None:
        print("  no normals in the cloud -- every view will count equally, so wide "
              "leaves will contribute false tips down their sides")
    with open(workdir / "p4" / "hull.json") as f:
        voxel = json.load(f)["voxel_size"]

    # The cloud is read in the raw COLMAP frame, which is what the renders
    # need. P5 writes everything in the upright plant frame, so the tips are
    # converted on the way out -- otherwise they would not overlay the rest
    # of the scene in Blender.
    to_plant = None
    graph_path = workdir / "p5" / "stem_graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            frame = json.load(f).get("plant_frame")
        if frame:
            to_plant = (np.array(frame["origin"]), np.array(frame["rotation"]))

    reconstruction = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    image_ids = sorted(reconstruction.reg_image_ids())[::stride]
    print(f"  {len(points)} points, {len(image_ids)} views (stride {stride})")

    print(f"Loading SAM2 from {checkpoint} ...")
    model = build_sam2(_resolve_model_cfg(Path(checkpoint)), str(checkpoint), device=device)
    generator = SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side,
                                           pred_iou_thresh=0.7, stability_score_thresh=0.85,
                                           min_mask_region_area=60)

    tip_class = next((i for i, n in enumerate(order) if "tip" in n), None)
    if tip_class is not None:
        print(f"  using the '{order[tip_class]}' class to reject mask edges")

    frames_dir = workdir / "p1" / "frames"
    mask_dir = workdir / "p2" / "masks" / "plant"
    votes, diag_at = [], set(np.linspace(0, len(image_ids) - 1, 6).astype(int).tolist())
    # How often each point was rendered at all, so a vote can be read as a
    # share of the views that could have cast it.
    visibility = np.zeros(len(points), np.int32)
    gated = 0

    for n, image_id in enumerate(image_ids):
        image = reconstruction.images[image_id]
        stem_name = Path(image.name).stem
        bgr = cv2.imread(str(frames_dir / image.name))
        plant = cv2.imread(str(mask_dir / f"{stem_name}.png"), cv2.IMREAD_GRAYSCALE)
        class_map = load_class_map(p4c / "class_maps", stem_name)
        if bgr is None or plant is None or class_map is None:
            continue

        stem_ys, stem_xs = np.nonzero(class_map == stem_class)
        if len(stem_xs) == 0:
            continue
        stem_pixels = np.stack([stem_xs, stem_ys], axis=1)

        masks = leaf_masks_in_frame(generator, bgr, plant > 127, class_map, leaf_class)
        camera = camera_from_colmap(image, reconstruction.cameras[image.camera_id])
        _rgb, index_map = render_points(points, colors, camera)
        seen_here = np.unique(index_map[index_map >= 0])
        visibility[seen_here] += 1
        rotation = camera.world_to_camera[:3, :3]
        camera_centre = -rotation.T @ camera.world_to_camera[:3, 3]

        prior = (class_map == tip_class) if tip_class is not None else None
        found = []
        for mask in masks:
            hit = tip_pixel(mask, stem_pixels, prior)
            if hit is None:
                continue
            x, y, distance, used_prior = hit
            gated += int(used_prior)
            # The nearest rendered point, not strictly the one under the
            # pixel: a tip sits on the outer edge of the mask where the carve
            # has eroded the margin away, so an exact lookup silently drops
            # the longest leaves.
            index = point_at_pixel(index_map, x, y)
            if index < 0:
                continue
            facing = (mask_facing(index_map, mask, points, normals, camera_centre)
                      if normals is not None else 1.0)
            votes.append(TipVote(frame=stem_name, point_index=index, pixel=(x, y),
                                 mask_area=int(mask.sum()), distance_from_stem_px=distance,
                                 facing=facing))
            found.append((x, y))

        if n in diag_at:
            _write_diag(p4c / "diag", stem_name, bgr, plant > 127, masks, found)
        if (n + 1) % 8 == 0:
            print(f"    {n + 1}/{len(image_ids)} views, {len(votes)} votes so far")

    clusters = cluster_votes(votes, points, radius=voxel * cluster_voxels,
                             min_votes=min_votes, visibility=visibility,
                             min_hit_rate=min_hit_rate)
    settings = {"stride": stride, "min_votes": min_votes,
                "cluster_voxels": cluster_voxels, "views": len(image_ids),
                "raw_votes": len(votes), "min_hit_rate": min_hit_rate,
                "votes_gated_by_tip_class": gated}
    write_tips(p4c / "tips3d.json", clusters, settings)
    _write_ply(p4c / "tips3d.ply", clusters, to_plant)

    print(f"\n  {len(votes)} raw votes over {len(image_ids)} views "
          f"-> {len(clusters)} tip(s) with >= {min_votes} votes")
    print(f"  {gated} of {len(votes)} votes were placed inside the tip class")
    print(f"\n  {'rank':>4} {'weight':>7} {'votes':>6} {'facing':>7}  position")
    for i, c in enumerate(clusters):
        print(f"    {i:>2} {c.weight:>7.1f} {c.votes:>6} "
              f"{c.weight / max(c.votes, 1):>7.2f}  {np.round(c.position, 3).tolist()}")
    # Report where support falls off, so the split is visible rather than assumed.
    if len(clusters) > 2:
        w = np.array([c.weight or c.votes for c in clusters], float)
        ratio = w[:-1] / np.maximum(w[1:], 1e-9)
        cut = int(np.argmax(ratio))
        print(f"\n  support drops hardest after rank {cut} "
              f"({w[cut]:.1f} -> {w[cut + 1]:.1f}, {ratio[cut]:.1f}x): "
              f"{cut + 1} well-supported tip(s), {len(clusters) - cut - 1} weak")
    if len(clusters) < 2:
        print("  Few tips: try --stride 2, --min-votes 2, or --points-per-side 48.")
    print(f"\n  {p4c / 'tips3d.ply'}   open in Blender")
    print(f"  {p4c / 'diag'}          the masks and tips per frame")
    return {"votes": len(votes), "tips": len(clusters),
            "positions": [c.to_dict() for c in clusters]}


def _write_ply(path: Path, clusters, to_plant=None) -> None:
    if not clusters:
        write_ply_vertices(path, {k: np.zeros(0, np.float32) for k in ("x", "y", "z")})
        return
    xyz = np.array([c.position for c in clusters])
    if to_plant is not None:
        origin, rotation = to_plant
        xyz = (xyz - origin) @ rotation.T
    # Support, normalised, carried in the colour so a viewer can size or shade
    # by it. Nothing is dropped here: on plant_9 the well-supported tips are
    # all correct and the weak ones are noise, but the cut between them is a
    # property of the plant, not a constant to bake in.
    weight = np.array([c.weight or c.votes for c in clusters], float)
    strength = (weight / max(weight.max(), 1e-9) * 255).astype(np.uint8)
    write_ply_vertices(path, {
        "x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "red": (255 - strength), "green": strength, "blue": np.zeros_like(strength)})


def _write_diag(diag_dir: Path, stem: str, bgr, plant, masks, tips) -> None:
    ys, xs = np.nonzero(plant)
    pad = 60
    y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)
    panel = bgr[y0:y1, x0:x1].copy()

    palette = [(80, 80, 230), (120, 200, 80), (240, 140, 90), (70, 190, 240),
               (220, 100, 200), (210, 210, 90)]
    for i, mask in enumerate(masks):
        crop = mask[y0:y1, x0:x1]
        colour = palette[i % len(palette)]
        panel[crop] = (0.55 * panel[crop] + 0.45 * np.array(colour)).astype(np.uint8)
    for x, y in tips:
        cv2.drawMarker(panel, (x - x0, y - y0), (255, 255, 255), cv2.MARKER_CROSS, 26, 5)
        cv2.drawMarker(panel, (x - x0, y - y0), (0, 40, 0), cv2.MARKER_CROSS, 26, 2)

    label = f"{stem}: {len(masks)} leaf mask(s), {len(tips)} tip vote(s)"
    for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(panel, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thick)
    cv2.imwrite(str(diag_dir / f"tips_{stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--stride", type=int, default=4,
                        help="use every Nth registered view. A tip needs enough "
                             "directions to out-vote the views that disagree, not all of them")
    parser.add_argument("--min-votes", type=int, default=3,
                        help="votes a cluster needs to count as a real tip")
    parser.add_argument("--min-hit-rate", type=float, default=0.0,
                        help="votes as a share of the views that rendered that surface. "
                             "The number to raise when a leaf is only visible in a few "
                             "frames: it is seen rarely but voted nearly every time it "
                             "is, where a mask edge is visible constantly and voted "
                             "seldom")
    parser.add_argument("--cluster-voxels", type=float, default=6.0,
                        help="votes within this distance are the same tip")
    parser.add_argument("--points-per-side", type=int, default=32,
                        help="SAM2 sampling density; higher finds smaller blades, slower")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run(workdir=args.workdir, checkpoint=args.checkpoint, stride=args.stride,
        min_votes=args.min_votes, min_hit_rate=args.min_hit_rate,
        cluster_voxels=args.cluster_voxels,
        points_per_side=args.points_per_side, device=args.device)


if __name__ == "__main__":
    main()
