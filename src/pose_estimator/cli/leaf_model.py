"""P6 CLI: per-leaf midrib, frame, curvature, width and insertion angle.

    pose-leaf --workdir runs/plant_9/

Reads <workdir>/p5 (leaf point subsets + stem graph). Writes into <workdir>/p6:
    leaves.json     per-leaf midrib samples, curvature, widths, angles
    midribs.ply     midrib polylines, coloured per leaf, for Blender
    p6.json         acceptance checks
    diag/           3D midrib plot + midribs reprojected on the frames
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from pose_estimator import cloud_source
from pose_estimator.leaf import fit_leaf, frame_continuity_degrees, resolve_leaf_base
from pose_estimator.ply_io import write_ply_vertices


def run(workdir: Path, num_samples: int = 20,
        geometry_backend: str = cloud_source.BASELINE) -> dict:
    import pycolmap

    p5_dir = cloud_source.phase_dirs(workdir, geometry_backend)[1]
    p6_dir = (workdir / "p6" if geometry_backend == cloud_source.BASELINE
              else workdir / "p6" / "experiments" / geometry_backend)
    (p6_dir / "diag").mkdir(parents=True, exist_ok=True)

    with open(p5_dir / "stem_graph.json") as f:
        graph = json.load(f)
    assignment = np.load(p5_dir / "leaf_points.npy")
    points = np.load(p5_dir / "leaf_points_xyz.npy")

    stem_xyz = np.array(graph["stem_path_xyz"]) if graph["stem_path_xyz"] else np.zeros((0, 3))
    stem_axis = np.array([0.0, 0.0, 1.0])  # plant frame is already Z-up
    if len(stem_xyz) > 1:
        direction = stem_xyz[-1] - stem_xyz[0]
        norm = np.linalg.norm(direction)
        stem_tangent = direction / norm if norm > 1e-9 else stem_axis
    else:
        stem_tangent = stem_axis

    print(f"  {len(graph['leaves'])} leaf instances from P5; stem path {len(stem_xyz)} nodes")

    models = []
    for leaf in graph["leaves"]:
        leaf_id = leaf["id"]
        subset = points[assignment == leaf_id]
        if len(subset) < 50:
            print(f"    leaf {leaf_id}: only {len(subset)} points -- skipped")
            continue
        axis = np.array(leaf["axis_xyz"])
        # The stem reference is the plant origin (clamp line) when the traced
        # stem collapses to a single node, which it does on this specimen.
        # A point on the stem axis at canopy height: leaf attachments are the
        # radially inner ends relative to this.
        stem_reference = stem_xyz[-1] if len(stem_xyz) > 1 else np.array([0.0, 0.0, float(subset[:, 2].mean())])
        base = resolve_leaf_base(axis, subset, stem_reference)

        model = fit_leaf(leaf_id, subset, base, stem_tangent, stem_axis, num_samples=num_samples)
        if model is None:
            print(f"    leaf {leaf_id}: midrib fit failed ({len(subset)} points)")
            continue
        models.append(model)
        print(
            f"    leaf {leaf_id}: {len(subset):>5} pts | length {model.arclength:.4f} | "
            f"insertion {model.insertion_angle_deg:6.1f} deg | azimuth {model.azimuth_deg:6.1f} deg | "
            f"mean kappa {model.curvature.mean():7.2f} | max half-width {model.half_width.max():.4f}"
        )

    report = _evaluate(models, graph)
    with open(p6_dir / "leaves.json", "w") as f:
        json.dump(
            {
                "units": "colmap",
                "units_note": "NOT metric -- no scale reference has been solved for this capture",
                "origin": graph["origin_definition"],
                "leaves": [m.to_dict() for m in models],
            },
            f,
            indent=2,
        )
    _write_midrib_ply(p6_dir / "midribs.ply", models)
    _write_diagnostics(p6_dir / "diag", models, points, assignment, stem_xyz, workdir)

    with open(p6_dir / "p6.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  P6 checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  artifacts + diagnostics in {p6_dir}")
    return report


def _evaluate(models, graph) -> dict:
    if not models:
        return {"checks": {"any_leaves_fitted": {"pass": False, "detail": "no leaf produced a midrib"}},
                "all_passed": False}

    continuity = [float(frame_continuity_degrees(m.normals).max()) for m in models]
    lengths = [m.arclength for m in models]
    insertions = [m.insertion_angle_deg for m in models]
    widths = [float(m.half_width.max()) for m in models]

    checks = {
        "all_leaves_fitted": {
            "pass": len(models) == len(graph["leaves"]),
            "detail": f"{len(models)} of {len(graph['leaves'])} P5 leaves produced a midrib",
        },
        "frames_are_continuous": {
            # The plan's criterion. A Frenet frame would flip ~180 deg at each
            # inflection; the rotation-minimising frame must not.
            "pass": max(continuity) <= 5.0,
            "detail": f"largest rotation-minimising-frame step {max(continuity):.2f} deg "
            f"between adjacent stations (limit 5)",
        },
        "lengths_are_plausible": {
            "pass": min(lengths) > 0 and max(lengths) / max(min(lengths), 1e-9) < 6.0,
            "detail": f"leaf arclengths {min(lengths):.4f}..{max(lengths):.4f} "
            f"(ratio {max(lengths) / max(min(lengths), 1e-9):.1f}x)",
        },
        "insertion_angles_in_range": {
            "pass": all(0.0 <= a <= 180.0 for a in insertions),
            "detail": f"insertion angles {min(insertions):.1f}..{max(insertions):.1f} deg",
        },
    }
    return {
        "num_leaves": len(models),
        "arclength": {"min": float(min(lengths)), "max": float(max(lengths)),
                      "mean": float(np.mean(lengths))},
        "insertion_angle_deg": {"min": float(min(insertions)), "max": float(max(insertions)),
                                "mean": float(np.mean(insertions))},
        "azimuth_deg": [float(m.azimuth_deg) for m in models],
        "max_half_width": {"min": float(min(widths)), "max": float(max(widths))},
        "max_frame_step_deg": float(max(continuity)),
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
    }


def _write_midrib_ply(path: Path, models) -> None:
    palette = np.array([[230, 80, 80], [80, 200, 120], [90, 140, 240], [240, 190, 70],
                        [200, 100, 220], [90, 210, 210], [240, 140, 90], [150, 220, 90]], np.uint8)
    xyz, rgb = [], []
    for model in models:
        dense = np.linspace(0, 1, 120)
        source = np.linspace(0, 1, len(model.midrib))
        curve = np.stack([np.interp(dense, source, model.midrib[:, i]) for i in range(3)], axis=1)
        xyz.append(curve)
        rgb.append(np.tile(palette[model.leaf_id % len(palette)], (len(curve), 1)))
    if not xyz:
        return
    xyz = np.vstack(xyz)
    rgb = np.vstack(rgb)
    write_ply_vertices(path, {
        "x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
    })


def _write_diagnostics(diag_dir: Path, models, points, assignment, stem_xyz, workdir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    palette = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(17, 5.2))

    for panel, (elev, azim, title) in enumerate(
        [(16, -60, "midribs, perspective"), (2, -90, "midribs, front"), (89, -90, "midribs, from above")]
    ):
        ax = fig.add_subplot(1, 4, panel + 1, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="#C9D2CD", s=0.6, alpha=0.25, linewidths=0)
        for model in models:
            colour = palette(model.leaf_id % 10)
            ax.plot(model.midrib[:, 0], model.midrib[:, 1], model.midrib[:, 2],
                    color=colour, lw=2.6)
            ax.scatter(*model.midrib[-1], color=colour, s=26)
        if len(stem_xyz) > 1:
            ax.plot(stem_xyz[:, 0], stem_xyz[:, 1], stem_xyz[:, 2], color="k", lw=3.0)
        ax.scatter([0], [0], [0], c="k", marker="x", s=70)
        span = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
        mid = (points.max(axis=0) + points.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - span, mid[0] + span); ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.view_init(elev=elev, azim=azim); ax.set_title(title, fontsize=10)
        for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            setter([])

    # Width and curvature profiles: the actual per-leaf measurements, plotted
    # against normalised arclength so leaves of different sizes overlay.
    ax = fig.add_subplot(1, 4, 4)
    for model in models:
        s = np.linspace(0, 1, len(model.half_width))
        ax.plot(s, model.half_width, color=palette(model.leaf_id % 10), lw=1.8,
                label=f"leaf {model.leaf_id}")
    ax.set_xlabel("normalised arclength (base to tip)")
    ax.set_ylabel("half-width")
    ax.set_title("lamina width profile", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"P6: {len(models)} leaf midribs with rotation-minimising frames", fontsize=12)
    fig.tight_layout()
    fig.savefig(diag_dir / "midribs_3d.png", dpi=110)
    plt.close(fig)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Stations along each midrib at which to report frame, curvature and width")
    parser.add_argument("--geometry-backend", default=cloud_source.BASELINE,
                        help="Fit this P3 branch's leaves, from p5/experiments/<backend>")
    args = parser.parse_args(argv)
    run(workdir=args.workdir, num_samples=args.num_samples,
        geometry_backend=args.geometry_backend)


if __name__ == "__main__":
    main()
