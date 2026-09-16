"""One diagram holding the whole result.

Three panels, because the result answers three different questions and a
single panel can only answer one of them well:

*Where is each leaf, and did the measurement land on it?* -- the photograph
with every contour, midrib and keypoint drawn on it. This is the panel that
makes an error obvious: a midrib that has run into a lobe, or a tip and a
petiole origin the wrong way round, is visible at a glance and invisible in
any table.

*What shape are the leaves?* -- every midrib moved to a common origin and
turned to a common heading. Stacked like that, the spread of curvature and
length across one plant is one picture rather than thirty.

*How big are they?* -- petiole and blade length per leaf, ranked. In
millimetres when the markers in the frame gave a scale, in pixels when they
did not, and the axis says which.

Colour carries meaning and nothing else: **blade/midrib blue**, **petiole
magenta**, **tip yellow**, everywhere, in all three panels. The leaf outline
is deliberately *not* a fourth colour -- it is the object, not a
measurement, so it wears the neutral ink the labels do. Those three hues were
checked against the dark surface rather than chosen by eye (worst pair,
all-pairs: normal-vision dE 19.3, colour-deficient dE 13.2). Veins are the
one thing kept off the overlay: they would be a fourth simultaneous colour,
which no four-hue set on this surface survives, and forty polylines a leaf
would bury the panel that exists to be scanned. They are drawn in
`plot_leaf_detail` instead, where the midrib is the only other mark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .record import LeafRecord

# Dark-mode slots, stepped for a dark surface rather than flipped from light.
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_MUTED = "#c3c2b7"
GRID = "#393836"
BLADE = "#3987e5"
PETIOLE = "#d55181"
TIP = "#c98500"


def _pt(pixels: float, dpi: int) -> float:
    """Pixels as points, so the mark specs hold at whatever dpi is saved."""
    return pixels * 72.0 / dpi


def _canonical(record: LeafRecord) -> np.ndarray:
    """A leaf's midrib at the origin, heading up, in its own units.

    Translated to the petiole origin and rotated so the straight line to the
    tip points along +y. Rotating on the chord rather than on the first
    tangent is what keeps the panel readable: the tangent at the cut end is
    the noisiest part of the curve, and aligning on it makes every leaf's
    *far* end scatter instead.
    """
    path = record.midrib_xy - record.petiole_origin
    chord = path[-1]
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return path
    cosine, sine = chord[1] / length, chord[0] / length
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    turned = path @ rotation.T

    # The rotation puts the tip at +y in *image* axes, where y runs downward
    # and x runs right -- so plotting it unchanged draws every leaf hanging
    # down, and negating y instead turns the whole panel through 180 degrees
    # (self-consistent, and still upside down). The axis to flip is x: that
    # converts the left-handed image frame to the right-handed plot one, and
    # leaves each leaf pointing up with its two sides on the sides the
    # photograph shows them.
    return np.stack([-turned[:, 0], turned[:, 1]], axis=1)


def plot_overview(
    records: Sequence[LeafRecord],
    flat: np.ndarray,
    output: Path,
    title: str = "",
    dpi: int = 150,
    max_side: int = 2600,
    show: bool = False,
):
    """Draw the three-panel diagram and save it to `output`."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import cv2

    from .raw import tone_map

    # Cropped to the leaves before it is shrunk. A flat lay is laid out on
    # whatever backing was to hand, and on gaensefuss_31 the subject occupies
    # about half the frame -- showing all of it spends half the panel on felt
    # and halves the size of the thing the panel exists to let you check.
    view = _subject_box(records, flat.shape)
    cropped = flat[view[1]:view[3], view[0]:view[2]]

    scale = min(1.0, max_side / float(max(cropped.shape[:2])))
    shown = tone_map(cv2.resize(cropped, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
                     if scale < 1.0 else cropped)
    origin = np.array([view[0], view[1]], dtype=float)

    figure = plt.figure(figsize=(17, 10), dpi=dpi, facecolor=SURFACE)
    grid = figure.add_gridspec(2, 2, width_ratios=[2.05, 1.0], height_ratios=[1.0, 1.0],
                               wspace=0.14, hspace=0.22,
                               left=0.03, right=0.975, top=0.90, bottom=0.07)
    photo = figure.add_subplot(grid[:, 0])
    fan = figure.add_subplot(grid[0, 1])
    bars = figure.add_subplot(grid[1, 1])

    _draw_photo(photo, records, shown, scale, origin, dpi)
    _draw_fan(fan, records, dpi)
    _draw_bars(bars, records, dpi)

    units = records[0].units if records else "px"
    figure.suptitle(title or "Leaf poses", color=INK, fontsize=17,
                    x=0.03, y=0.965, ha="left", fontweight="medium")
    figure.text(0.03, 0.925,
                f"{len(records)} leaves  ·  lengths in {units}"
                + ("  ·  scale from the markers in frame" if units == "mm"
                   else "  ·  no scale measured, pass --marker-mm for millimetres"),
                color=INK_MUTED, fontsize=11, ha="left")

    # One legend for the whole figure: the three roles mean the same thing in
    # every panel, so repeating it per panel would be three copies of one key.
    handles = [
        Line2D([], [], color=BLADE, lw=_pt(3, dpi), label="midrib / blade"),
        Line2D([], [], color=PETIOLE, lw=_pt(3, dpi), marker="o", ls="none",
               markersize=_pt(9, dpi), markeredgecolor=SURFACE,
               markeredgewidth=_pt(2, dpi), label="petiole origin (to stem)"),
        Line2D([], [], color=TIP, marker="^", ls="none", markersize=_pt(10, dpi),
               markeredgecolor=SURFACE, markeredgewidth=_pt(2, dpi), label="leaf tip"),
        Line2D([], [], color=INK_MUTED, lw=_pt(1.5, dpi), label="leaf outline"),
    ]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.975),
                  ncol=4, frameon=False, labelcolor=INK_MUTED, fontsize=10.5,
                  handletextpad=0.6, columnspacing=1.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor=SURFACE)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output


def _subject_box(records: Sequence[LeafRecord], shape, pad_fraction: float = 0.03):
    """Bounding box of every leaf, padded, clipped to the frame."""
    height, width = shape[:2]
    if not records:
        return 0, 0, width, height
    boxes = np.array([r.instance.bbox for r in records])
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    pad = int(pad_fraction * max(x1 - x0, y1 - y0))
    return (max(0, int(x0) - pad), max(0, int(y0) - pad),
            min(width, int(x1) + pad), min(height, int(y1) + pad))


def _draw_photo(axes, records: Sequence[LeafRecord], shown, scale: float,
                origin: np.ndarray, dpi: int):
    axes.imshow(shown)
    axes.set_facecolor(SURFACE)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    for record in records:
        contour = (record.contour - origin) * scale
        axes.plot(contour[:, 0], contour[:, 1], color=INK_MUTED,
                  lw=_pt(1.5, dpi), alpha=0.85, solid_joinstyle="round")

        midrib = (record.midrib_xy - origin) * scale
        axes.plot(midrib[:, 0], midrib[:, 1], color=BLADE, lw=_pt(2.5, dpi),
                  solid_capstyle="round", solid_joinstyle="round")

        base = (record.petiole_origin - origin) * scale
        tip = (record.tip - origin) * scale
        axes.plot(base[0], base[1], "o", color=PETIOLE, markersize=_pt(9, dpi),
                  markeredgecolor=SURFACE, markeredgewidth=_pt(2, dpi), zorder=3)
        axes.plot(tip[0], tip[1], "^", color=TIP, markersize=_pt(10, dpi),
                  markeredgecolor=SURFACE, markeredgewidth=_pt(2, dpi), zorder=3)

        # The id goes on the far side of the cut end from the blade, so it
        # never lands on the leaf it names or on the next leaf along.
        away = base - tip
        away = away / max(float(np.linalg.norm(away)), 1e-9)
        label = base + away * (0.02 * max(shown.shape[:2]))
        axes.text(label[0], label[1], str(record.leaf_id), color=INK_MUTED,
                  fontsize=8.5, ha="center", va="center")


def _draw_fan(axes, records: Sequence[LeafRecord], dpi: int):
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
        axes.spines[side].set_linewidth(_pt(1, dpi))
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=_pt(1, dpi))
    axes.grid(True, color=GRID, lw=_pt(1, dpi), alpha=0.7)
    axes.set_axisbelow(True)

    units = records[0].units if records else "px"
    axes.set_title("Midribs on a common origin and heading", color=INK,
                   fontsize=11.5, loc="left", pad=8)
    axes.set_xlabel(f"across ({units})", color=INK_MUTED, fontsize=9.5)
    axes.set_ylabel(f"along ({units})", color=INK_MUTED, fontsize=9.5)

    for record in records:
        curve = _canonical(record)
        if record.mm_per_pixel:
            curve = curve * record.mm_per_pixel
        axes.plot(curve[:, 0], curve[:, 1], color=BLADE, lw=_pt(2, dpi),
                  alpha=0.55, solid_capstyle="round")
        axes.plot(curve[-1, 0], curve[-1, 1], "^", color=TIP,
                  markersize=_pt(8, dpi), markeredgecolor=SURFACE,
                  markeredgewidth=_pt(2, dpi))
    axes.plot(0, 0, "o", color=PETIOLE, markersize=_pt(10, dpi),
              markeredgecolor=SURFACE, markeredgewidth=_pt(2, dpi), zorder=4)
    axes.set_aspect("equal", adjustable="datalim")


def _draw_bars(axes, records: Sequence[LeafRecord], dpi: int):
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.spines["bottom"].set_linewidth(_pt(1, dpi))
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=_pt(1, dpi))
    axes.grid(True, axis="x", color=GRID, lw=_pt(1, dpi), alpha=0.7)
    axes.set_axisbelow(True)

    if not records:
        return

    units = records[0].units
    ranked = sorted(records, key=lambda r: -r.length("total"))
    positions = np.arange(len(ranked))
    petiole = np.array([r.length("petiole") for r in ranked])
    blade = np.array([r.length("blade") for r in ranked])

    # A 2px gap in the surface colour between the two segments rather than an
    # outline around either: at this bar height an outline is most of the mark.
    span = float((petiole + blade).max()) or 1.0
    gap = span * (2.0 / max(axes.bbox.width, 1.0))
    height = min(0.62, 24.0 / max(len(ranked), 1))

    axes.barh(positions, petiole, height=height, color=PETIOLE, label="petiole")
    axes.barh(positions, np.maximum(blade - gap, 0.0), left=petiole + gap,
              height=height, color=BLADE, label="blade")

    axes.set_yticks(positions)
    axes.set_yticklabels([str(r.leaf_id) for r in ranked], fontsize=8)
    axes.invert_yaxis()

    # The rounded data-end, drawn as a round cap on a zero-length line once
    # the y limits exist. The width has to be converted, not assumed: the bar
    # height is in data units (leaf index) and a line width is in points, and
    # multiplying the one by 72 to get the other gave a cap six times the
    # bar's own thickness.
    low, high = axes.get_ylim()
    points_per_unit = (axes.bbox.height * 72.0 / dpi) / max(abs(high - low), 1e-9)
    for y, total in zip(positions, petiole + blade):
        axes.plot([total, total], [y, y], color=BLADE,
                  lw=height * points_per_unit, solid_capstyle="round", zorder=2)
    axes.set_xlabel(f"length ({units})", color=INK_MUTED, fontsize=9.5)
    axes.set_ylabel("leaf", color=INK_MUTED, fontsize=9.5)
    axes.set_title("Petiole and blade length, ranked", color=INK,
                   fontsize=11.5, loc="left", pad=8)
    axes.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9.5,
                loc="lower right", ncol=2)

    # One direct label, on the longest leaf: the axis carries the other 29.
    longest = ranked[0]
    axes.text(float(petiole[0] + blade[0]), 0.0,
              f"  {longest.length('total'):.0f} {units}",
              color=INK, fontsize=9.5, va="center", ha="left")


def plot_leaf_detail(
    record: LeafRecord, crop: np.ndarray, output: Path, dpi: int = 150,
):
    """One leaf, large, with its veins -- the panel the overview leaves out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .raw import tone_map

    offset = np.array(record.instance.offset, dtype=float)
    figure, axes = plt.subplots(figsize=(6, 9), dpi=dpi, facecolor=SURFACE)
    axes.imshow(tone_map(crop))
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    for vein in record.veins:
        axes.plot(vein.path[:, 0], vein.path[:, 1], color=BLADE,
                  lw=_pt(1.5, dpi), alpha=0.6, solid_capstyle="round")

    midrib = record.midrib_xy - offset
    axes.plot(midrib[:, 0], midrib[:, 1], color=BLADE, lw=_pt(3, dpi),
              solid_capstyle="round")
    base = record.petiole_origin - offset
    tip = record.tip - offset
    axes.plot(base[0], base[1], "o", color=PETIOLE, markersize=_pt(11, dpi),
              markeredgecolor=SURFACE, markeredgewidth=_pt(2, dpi))
    axes.plot(tip[0], tip[1], "^", color=TIP, markersize=_pt(12, dpi),
              markeredgecolor=SURFACE, markeredgewidth=_pt(2, dpi))

    units = record.units
    axes.set_title(
        f"leaf {record.leaf_id}   blade {record.length('blade'):.1f} {units}"
        f"   petiole {record.length('petiole'):.1f} {units}"
        f"   {len(record.veins)} veins",
        color=INK, fontsize=11, loc="left", pad=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return output
