#!/usr/bin/env python
"""A workdir that holds only some capture passes, so P4a-P6 and P5x run on them alone.

    python scripts/make_pass_workdir.py --workdir <dataset>/plant --passes 0
    ./run_pipeline.sh --workdir <dataset>/plant_pass0 --skip-to p4a
    pose-leaf-instances --workdir <dataset>/plant_pass0

Why: on gaensefuss_1 the plant wilted between passes. Carving one pass alone
fills 79-85% of that pass's silhouettes; carving all three fills 59-63% of the
same silhouettes (`pose-hull --passes`). Every phase from P4a on assumes one
static plant, so it should see views from one time only.

Nothing downstream needs to know: P4a-P6 and P5x all walk the *registered
images of the P3 model*, so the model is filtered to the chosen passes and
everything follows. P3 itself is not re-solved -- the joint solve over all
passes is the better one (its poses do not care that the plant moved; the
plant is masked out of P3), so its poses are kept exactly.

What is written (nothing is written into --workdir):
    p1/frames/<frame>.jpg        links, chosen frames only
    p1/{sources,manifest}.json   filtered; intrinsics.json copied
    p2/masks/<class>/<frame>.png links, chosen frames only
    p2/masks/leaf_instances/     links to the chosen passes' id folders
    p2/alpha/<frame>.png         links; p2/*.json copied
    p3/sparse/best/              the P3 model with the other passes deregistered
    p3/poses.json                copied: the orbit axis is shared by all passes
    derived_from.json            where this came from
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def _link(target: Path, link: Path) -> None:
    """Relative, so <dataset>/plant and <dataset>/plant_passN still work after
    being copied together -- from the cluster to the local mirror, say."""
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def _deregister(rec, image_id: int) -> None:
    """Version-agnostic: pycolmap >= 3.12 groups images into frames and only
    deregisters a frame; older releases deregister the image itself."""
    if hasattr(rec, "deregister_frame"):
        rec.deregister_frame(rec.image(image_id).frame_id)
    else:
        rec.deregister_image(image_id)


def filtered_model(workdir: Path, keep: set):
    """The P3 model with every view outside `keep` deregistered, in memory."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    before = (rec.num_reg_images(), rec.num_points3D())
    for image_id in list(rec.reg_image_ids()):
        if Path(rec.images[image_id].name).stem not in keep:
            _deregister(rec, image_id)
    return rec, before


def _check_written(path: Path, keep: set) -> None:
    """Reload what was written: exactly the chosen views, and no 3D point
    still observed from a dropped one. Write/read semantics for unregistered
    images differ between pycolmap releases, so this is checked, not assumed."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(path))
    names = {Path(rec.images[i].name).stem for i in rec.reg_image_ids()}
    stray = names - keep
    reg = set(rec.reg_image_ids())
    dangling = sum(1 for p in rec.points3D.values()
                   if any(e.image_id not in reg for e in p.track.elements))
    if stray or dangling:
        raise SystemExit(f"{path}: the written model still has {len(stray)} views from other "
                         f"passes and {dangling} points observed from them -- this pycolmap "
                         f"({pycolmap.__version__}) does not drop deregistered views on write")


def build(workdir: Path, passes, out: Path, force: bool = False) -> dict:
    passes = sorted(set(int(p) for p in passes))
    sources = {k: int(v) for k, v in json.loads((workdir / "p1" / "sources.json").read_text()).items()}
    chosen = sorted(f for f, p in sources.items() if p in passes)
    if not chosen:
        raise SystemExit(f"no frames of passes {passes} in {workdir / 'p1' / 'sources.json'}")
    keep = set(chosen)

    # The model first and in memory, so a failure here leaves nothing on disk.
    rec, before = filtered_model(workdir, keep)

    if out.exists() and any(out.iterdir()):
        if (out / "derived_from.json").exists() and not force:
            raise SystemExit(f"{out} is a finished build -- pass --force to rebuild it "
                             "(its p4..p6 and p5x outputs are deleted with it)")
        if not (out / "derived_from.json").exists():
            print(f"  {out} is an unfinished build (no derived_from.json); replacing it")
        shutil.rmtree(out)

    # --- p1 ---
    p1, q1 = workdir / "p1", out / "p1"
    for frame in chosen:
        src = p1 / "frames" / f"{frame}.jpg"
        if src.exists():
            _link(src, q1 / "frames" / f"{frame}.jpg")
    (q1 / "sources.json").write_text(json.dumps({f: sources[f] for f in chosen}, indent=2))
    manifest = json.loads((p1 / "manifest.json").read_text())
    (q1 / "manifest.json").write_text(json.dumps([m for m in manifest if m["frame"] in keep],
                                                 indent=2))
    if (p1 / "intrinsics.json").exists():
        shutil.copy2(p1 / "intrinsics.json", q1 / "intrinsics.json")

    # --- p2 ---
    p2, q2 = workdir / "p2", out / "p2"
    for sub in sorted(d for d in (p2 / "masks").iterdir() if d.is_dir()):
        if sub.name == "leaf_instances":
            for folder in sorted(d for d in sub.iterdir() if d.is_dir()):
                prefix = folder.name.split("_", 1)[0]
                if prefix.startswith("pass") and int(prefix[4:]) in passes:
                    _link(folder, q2 / "masks" / "leaf_instances" / folder.name)
            continue
        for frame in chosen:
            src = sub / f"{frame}.png"
            if src.exists():
                _link(src, q2 / "masks" / sub.name / f"{frame}.png")
    for frame in chosen:
        src = p2 / "alpha" / f"{frame}.png"
        if src.exists():
            _link(src, q2 / "alpha" / f"{frame}.png")
    for meta in p2.glob("*.json"):
        shutil.copy2(meta, q2 / meta.name)

    # --- p3: same poses, other passes deregistered ---
    (out / "p3" / "sparse" / "best").mkdir(parents=True)
    rec.write(str(out / "p3" / "sparse" / "best"))
    _check_written(out / "p3" / "sparse" / "best", keep)
    shutil.copy2(workdir / "p3" / "poses.json", out / "p3" / "poses.json")

    record = {"source": str(workdir.resolve()), "passes": passes, "frames": len(chosen),
              "registered_images": rec.num_reg_images(), "points3D": rec.num_points3D(),
              "source_registered_images": before[0], "source_points3D": before[1],
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "derived_from.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True, help="the full run, e.g. <dataset>/plant")
    ap.add_argument("--passes", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path,
                    help="default: <dataset>/plant_pass<N>[_<M>...] beside --workdir")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args()
    out = args.out or args.workdir.parent / (
        f"{args.workdir.name}_pass" + "_".join(str(p) for p in sorted(set(args.passes))))
    record = build(args.workdir, args.passes, out, args.force)
    print(f"  {out}: passes {record['passes']}, {record['frames']} frames, "
          f"{record['registered_images']}/{record['source_registered_images']} views and "
          f"{record['points3D']}/{record['source_points3D']} sparse points kept")
    print(f"  next:  ./run_pipeline.sh --workdir {out} --skip-to p4a")
    print(f"         pose-leaf-instances --workdir {out}")


if __name__ == "__main__":
    main()
