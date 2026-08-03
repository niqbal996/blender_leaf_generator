"""P2 quality control: does this segmentation deserve to be trusted?

The checks here are deliberately the ones that catch *silent* failures --
where SAM2 returns a confident, plausible-looking mask that is wrong. A mask
that is empty is obvious the moment you look at it; a mask that quietly
dropped every petiole, or that swapped onto the pliers halfway through the
rotation, is not, and it will propagate into the visual hull as missing
geometry that no later stage can recover.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .segmentation import excess_green


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def holder_contamination(bgr: np.ndarray, plant_mask: np.ndarray) -> float:
    """Fraction of the plant mask that is unmistakably holder plastic.

    This is the check that matters, and it runs in the safe direction. Asking
    "did SAM2 find all the green?" fails on this rig -- the pliers' yellow
    grip is green-dominant in RGB and yellow-green foliage overlaps it in
    hue, so a color prior cannot arbitrate between them (measured: 47% of
    true plant pixels fall in the same warm-hue window as the plastic).

    Asking "did SAM2 grab something that is definitely *not* plant?" is
    decidable, because `holder_color_mask` is deliberately conservative --
    under 1% of true plant, verified on DSC_0009. A rising number here means
    the mask is drifting onto the holder, which is the specific failure that
    would inject a fake basal branch into the skeleton.

    Only *sizeable* holder blobs count. Specular highlights on a glossy leaf
    are small, bright and warm-toned, so they trip the yellow-plastic rule
    exactly the way the pliers' grip does -- on DSC_0009 this alone produced
    a spurious 4.5% "contamination" on a frame whose mask was perfect. Real
    contamination is a contiguous piece of plier, never scattered speckle.
    """
    from .segmentation import _significant_components, holder_color_mask

    plant_mask = plant_mask > 0
    if not plant_mask.any():
        return 0.0
    holder = _significant_components(holder_color_mask(bgr), min_area_fraction=1e-3)
    return float((holder & plant_mask).sum() / plant_mask.sum())


def centroid_trajectory(mask_paths: Sequence[Path]) -> np.ndarray:
    """Plant mask centroid per frame -- Nx2, NaN where the mask is empty."""
    out = []
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not (mask > 0).any():
            out.append([np.nan, np.nan])
            continue
        ys, xs = np.nonzero(mask > 0)
        out.append([xs.mean(), ys.mean()])
    return np.array(out, dtype=float)


def run_qc(
    frames_dir: Union[str, Path],
    p2_dir: Union[str, Path],
    max_area_jump: float = 0.15,
    max_holder_contamination: float = 0.02,
    max_centroid_jump_fraction: float = 0.08,
) -> dict:
    """Score a finished P2 run. Returns a report dict (also written as qc.json)."""
    frames_dir, p2_dir = Path(frames_dir), Path(p2_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))

    plant_areas: List[int] = []
    holder_areas: List[int] = []
    contaminations: List[float] = []
    empty_frames: List[str] = []
    full_frames: List[str] = []
    overlap_fractions: List[float] = []

    for path in frame_paths:
        plant = cv2.imread(str(p2_dir / "masks" / "plant" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        holder = cv2.imread(str(p2_dir / "masks" / "holder" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if plant is None:
            continue
        holder = holder if holder is not None else np.zeros_like(plant)

        plant_b, holder_b = plant > 0, holder > 0
        area = int(plant_b.sum())
        plant_areas.append(area)
        holder_areas.append(int(holder_b.sum()))

        total_px = plant_b.size
        if area == 0:
            empty_frames.append(path.name)
        elif area > 0.9 * total_px:
            full_frames.append(path.name)

        # Plant and holder are meant to be mutually exclusive; overlap means
        # SAM2 is confusing the two objects, which is how a clamp ends up
        # inside the skeleton.
        if area:
            overlap_fractions.append(float((plant_b & holder_b).sum() / area))

        bgr = cv2.imread(str(path))
        contaminations.append(holder_contamination(bgr, plant_b))

    areas = np.array(plant_areas, dtype=float)
    median_area = float(np.median(areas)) if len(areas) else 0.0
    if len(areas) > 1 and median_area > 0:
        jumps = np.abs(np.diff(areas)) / median_area
        worst_jump = float(jumps.max())
        jump_frames = [frame_paths[i + 1].name for i in np.nonzero(jumps > max_area_jump)[0]]
    else:
        worst_jump, jump_frames = 0.0, []

    # The specimen orbits the turntable axis, so its mask centroid traces a
    # smooth closed curve. A discontinuity is tracking loss -- SAM2 jumping
    # onto a different object -- and this catches it without reference to any
    # color prior, which on this rig cannot tell foliage from yellow plastic.
    centroids = centroid_trajectory(
        [p2_dir / "masks" / "plant" / f"{p.stem}.png" for p in frame_paths]
    )
    frame_diag = float(np.hypot(*cv2.imread(str(frame_paths[0])).shape[:2])) if frame_paths else 1.0
    finite = np.isfinite(centroids).all(axis=1)
    if finite.sum() > 2:
        steps = np.linalg.norm(np.diff(centroids[finite], axis=0), axis=1)
        worst_centroid_jump = float(steps.max() / frame_diag)
    else:
        worst_centroid_jump = 0.0

    max_contamination = float(np.max(contaminations)) if contaminations else 0.0

    checks = {
        "no_empty_masks": {"pass": not empty_frames, "detail": f"{len(empty_frames)} empty frame(s)"},
        "no_full_frame_masks": {"pass": not full_frames, "detail": f"{len(full_frames)} saturated frame(s)"},
        "area_temporally_smooth": {
            "pass": worst_jump <= max_area_jump,
            "detail": f"worst frame-to-frame area jump {worst_jump:.1%} of median (limit {max_area_jump:.0%})",
        },
        "centroid_trajectory_smooth": {
            "pass": worst_centroid_jump <= max_centroid_jump_fraction,
            "detail": f"worst centroid jump {worst_centroid_jump:.1%} of frame diagonal "
            f"(limit {max_centroid_jump_fraction:.0%})",
        },
        "plant_mask_free_of_holder": {
            "pass": max_contamination <= max_holder_contamination,
            "detail": f"worst frame has {max_contamination:.2%} of its plant mask on holder plastic "
            f"(limit {max_holder_contamination:.0%})",
        },
        "plant_holder_disjoint": {
            "pass": bool(np.max(overlap_fractions) < 0.02) if overlap_fractions else True,
            "detail": f"max plant/holder overlap {np.max(overlap_fractions):.2%} of plant area"
            if overlap_fractions else "no holder mask",
        },
    }

    report = {
        "num_frames": len(frame_paths),
        "plant_area_px": {
            "median": median_area,
            "min": float(areas.min()) if len(areas) else 0.0,
            "max": float(areas.max()) if len(areas) else 0.0,
            "worst_frame_to_frame_jump_fraction": worst_jump,
            "frames_exceeding_jump_limit": jump_frames,
        },
        "holder_area_px_median": float(np.median(holder_areas)) if holder_areas else 0.0,
        "holder_contamination": {
            "max": max_contamination,
            "mean": float(np.mean(contaminations)) if contaminations else 0.0,
        },
        "worst_centroid_jump_fraction": worst_centroid_jump,
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
    }

    with open(p2_dir / "qc.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def write_overlays(
    frames_dir: Union[str, Path],
    p2_dir: Union[str, Path],
    n_samples: int = 6,
    crop_boxes: Optional[Sequence[Sequence[int]]] = None,
) -> List[Path]:
    """Dump mask-on-image overlays, evenly spread across the rotation.

    Evenly spread rather than random: the failure that matters most is SAM2
    losing the object partway through the turn, and a uniform sample over
    turntable angle is what makes that visible.
    """
    frames_dir, p2_dir = Path(frames_dir), Path(p2_dir)
    diag_dir = p2_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        return []
    picks = np.linspace(0, len(frame_paths) - 1, min(n_samples, len(frame_paths))).astype(int)

    written = []
    for i in picks:
        path = frame_paths[i]
        bgr = cv2.imread(str(path))
        plant = cv2.imread(str(p2_dir / "masks" / "plant" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        holder = cv2.imread(str(p2_dir / "masks" / "holder" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or plant is None:
            continue

        overlay = bgr.copy()
        overlay[plant > 0] = (0.45 * overlay[plant > 0] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        if holder is not None:
            overlay[holder > 0] = (0.45 * overlay[holder > 0] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)

        for mask, color in ((plant, (0, 255, 0)), (holder, (255, 0, 255))):
            if mask is None:
                continue
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        if crop_boxes is not None and i < len(crop_boxes):
            bx = crop_boxes[i]
            cv2.rectangle(overlay, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 200, 255), 2)

        label = f"{path.stem}  plant={int((plant > 0).sum())}px"
        cv2.putText(overlay, label, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(overlay, label, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        out = diag_dir / f"overlay_{path.stem}.jpg"
        cv2.imwrite(str(out), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(out)

    return written


def write_area_plot(p2_dir: Union[str, Path], report: dict) -> Optional[Path]:
    """Plot mask area against frame index -- a turntable sequence should trace
    a smooth, roughly periodic curve, so spikes and cliffs localise exactly
    which frames to go and look at.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p2_dir = Path(p2_dir)
    stats_path = p2_dir / "frame_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        per_frame = json.load(f)

    idx = [s["index"] for s in per_frame]
    plant = [s.get("plant_area_px", 0) for s in per_frame]
    holder = [s.get("holder_area_px", 0) for s in per_frame]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(idx, plant, label="plant", color="tab:green", lw=1.8)
    ax.plot(idx, holder, label="holder", color="tab:purple", lw=1.2, alpha=0.8)
    median = report["plant_area_px"]["median"]
    ax.axhline(median, color="tab:green", ls=":", lw=1, alpha=0.7, label="plant median")
    ax.set_xlabel("frame index (turntable angle)")
    ax.set_ylabel("mask area (px)")
    ax.set_title("P2 mask area across the rotation")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = p2_dir / "diag" / "mask_area.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
