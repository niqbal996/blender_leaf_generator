"""P5x's vote, kept per pass: which SAM3 ids make each 3D leaf, and how they overlapped.

    PYTHONPATH=src python scripts/p5x_passes.py <workdir> [--cloud p4g/surface_p5x.ply]

P5x votes every point once per capture pass (each pass is its own SAM3 session, ids
unrelated), then links the passes' ids by 3D overlap (>= 30% of the smaller, strongest
first, never two ids SAM3 showed apart) and takes the majority. Only the result is written;
this re-runs it unchanged (pose_estimator.cli.leaf_instances.run) and keeps what it threw
away, into <workdir>/debug/p5x_passes/<cloud stem>/:

  per_pass.npz   each pass's label per cloud point (pass-local id, -1 unseen, -2 stem, -3 root)
  passes.json    per id: its SAM3 folder, frames, points it won; the cross-pass overlaps
                 (shared points, share of the smaller) and which links merged; final leaf
                 -> its ids
  (and the run's own outputs: instances.npy, segmented.ply, ... -- the same as p5x_p4g/)
"""
import argparse
import json
from pathlib import Path

import numpy as np

import pose_estimator.cli.leaf_instances as li


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--cloud", default="p4g/surface_p5x.ply", help="relative to the workdir")
    args = ap.parse_args()
    wd = args.workdir.resolve()
    cloud = wd / args.cloud
    out = wd / "debug" / "p5x_passes" / cloud.stem
    out.mkdir(parents=True, exist_ok=True)

    seen = {}
    real_merge, real_group = li.merge_across_passes, li.group_by_pass

    def group(leaf_dirs):
        by_pass = real_group(leaf_dirs)
        seen["by_pass"] = {p: [d.name for d in dirs] for p, dirs in by_pass.items()}
        seen["frames"] = {p: [len(list(d.glob("*.png"))) for d in dirs] for p, dirs in by_pass.items()}
        return by_pass

    def merge(per_pass, counts, overlap_fraction, cannot_link=None, stats=None):
        merged = real_merge(per_pass, counts, overlap_fraction, cannot_link=cannot_link, stats=stats)
        seen.update(per_pass=per_pass, counts=counts, merged=merged, cannot_link=cannot_link or set())
        return merged

    li.group_by_pass, li.merge_across_passes = group, merge
    li.run(wd, cloud=cloud, out=str(out))      # an absolute `out` replaces workdir/out

    per_pass, counts, merged = seen["per_pass"], seen["counts"], seen["merged"]
    np.savez_compressed(out / "per_pass.npz", **{p or "single": v for p, v in per_pass.items()})
    final = np.load(out / "instances.npy")
    passes = sorted(per_pass)
    overlaps = []
    for i, a in enumerate(passes):
        for b in passes[i + 1:]:
            both = (per_pass[a] >= 0) & (per_pass[b] >= 0)
            pairs, n = np.unique(np.stack([per_pass[a][both], per_pass[b][both]], 1), axis=0,
                                 return_counts=True)
            for (x, y), k in zip(pairs.tolist(), n.tolist()):
                small = min(counts[(a, x)], counts[(b, y)])
                overlaps.append({"a": seen["by_pass"][a][x], "b": seen["by_pass"][b][y], "shared": k,
                                 "share_of_smaller": round(k / small, 3) if small else None,
                                 "same_leaf": merged.get((a, x)) == merged.get((b, y))})
    overlaps.sort(key=lambda o: -o["shared"])
    ids = {seen["by_pass"][p][i]: {"pass": p, "frames": seen["frames"][p][i], "points_won": c,
                                   "merged_group": merged.get((p, i))}
           for (p, i), c in counts.items()}
    # the merged group -> the final leaf label it ended as (most of its points)
    leaves = {}
    for (p, i), g in merged.items():
        mine = per_pass[p] == i
        lab, n = np.unique(final[mine], return_counts=True)
        best = int(lab[np.argmax(n)]) if len(lab) else None
        leaves.setdefault(str(best), []).append(seen["by_pass"][p][i])
    doc = {"cloud": str(cloud), "ids": ids, "final_leaf_to_ids": leaves,
           "cannot_link": sorted(sorted(seen["by_pass"][p][i] for p, i in pair)
                                 for pair in seen["cannot_link"]),
           "overlaps": overlaps}
    (out / "passes.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote {out / 'passes.json'}")


if __name__ == "__main__":
    main()
