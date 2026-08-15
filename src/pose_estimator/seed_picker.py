"""Click organ seeds on a frame instead of reading coordinates off a grid.

The seeds P4c needs are a handful of points labelled "this is leaf", "this is
stem". Reading them off a printed coordinate grid works but is slow and easy
to get wrong, and the coordinates live in a *cropped* frame rather than the
original, which is the usual way people get them wrong.

Two things this enforces that hand-typed coordinates do not:

- **The crop matches.** Seeds are interpreted in the frame cropped to the
  plant mask with the same padding `dino.crop_to_plant` uses. The picker
  crops identically, so what you click is what the classifier reads.

- **Seeds land on the plant.** `classify_frame` zeroes everything outside the
  plant mask before extracting features, so a seed on the background samples
  a blacked-out patch and describes nothing. Those clicks are refused rather
  than silently stored.

Seeds may be taken from several frames. A leaf seen from one angle is one
example of "leaf"; seeded from three angles it is three, and because the
vectors are kept individually rather than averaged, that widens the class
instead of blurring it.

The session state is separated from the window so it can be tested without a
display -- see `SeedSession`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

SEED_FILE_VERSION = 1

# BGR, matching the organ colours the rest of the pipeline draws with.
CLASS_BGR = {
    "leaf": (40, 40, 220),
    "tiny leaf": (60, 90, 240),
    "stem": (200, 60, 160),
    "petiole": (200, 60, 160),
    "root": (40, 140, 240),
}
SPARE_BGR = [(110, 200, 90), (210, 190, 60), (70, 200, 220), (200, 200, 200)]


def class_color(name: str, index: int) -> Tuple[int, int, int]:
    return CLASS_BGR.get(name, SPARE_BGR[index % len(SPARE_BGR)])


@dataclass
class Seed:
    frame: str        # frame stem, e.g. "frame_0057"
    frame_index: int
    label: str
    x: int            # in the *cropped* frame's pixel space
    y: int


@dataclass
class SeedSession:
    """Seeds collected so far, and which class the next click gets.

    Deliberately window-free: every rule that could be wrong (coordinate
    mapping, refusing off-plant clicks, undo) lives here where it can be
    tested, and the window is only a way to drive it.
    """

    classes: List[str]
    seeds: List[Seed] = field(default_factory=list)
    current: int = 0

    @property
    def current_class(self) -> str:
        return self.classes[self.current]

    def set_class(self, index: int) -> bool:
        if 0 <= index < len(self.classes):
            self.current = index
            return True
        return False

    def add(self, frame: str, frame_index: int, x: int, y: int,
            plant_mask: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """Record a seed. Returns (accepted, message)."""
        if plant_mask is not None:
            height, width = plant_mask.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                return False, "outside the image"
            if not plant_mask[y, x]:
                return False, "not on the plant -- that patch is blacked out before features"
        self.seeds.append(Seed(frame=frame, frame_index=frame_index,
                               label=self.current_class, x=int(x), y=int(y)))
        return True, f"{self.current_class} at ({x}, {y})"

    def undo(self) -> Optional[Seed]:
        return self.seeds.pop() if self.seeds else None

    def clear(self) -> int:
        n = len(self.seeds)
        self.seeds.clear()
        return n

    def counts(self) -> Dict[str, int]:
        return {name: sum(1 for s in self.seeds if s.label == name) for name in self.classes}

    def missing_classes(self) -> List[str]:
        return [name for name, n in self.counts().items() if n == 0]

    def by_frame(self) -> Dict[str, List[Seed]]:
        grouped: Dict[str, List[Seed]] = {}
        for seed in self.seeds:
            grouped.setdefault(seed.frame, []).append(seed)
        return grouped

    def as_cli_string(self) -> str:
        """The equivalent --seeds argument, for a single-frame session."""
        return " ".join(f'"{s.label}:{s.x},{s.y}"' for s in self.seeds)


def save_seeds(path: Union[str, Path], session: SeedSession, pad: int = 60) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SEED_FILE_VERSION,
        "pad": pad,
        "classes": session.classes,
        "seeds": [
            {"frame": s.frame, "frame_index": s.frame_index,
             "label": s.label, "x": s.x, "y": s.y}
            for s in session.seeds
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_seeds(path: Union[str, Path]) -> Tuple[List[Seed], List[str], int]:
    """Returns (seeds, class order, pad). Class order follows first appearance."""
    with open(path) as f:
        payload = json.load(f)
    version = payload.get("version")
    if version != SEED_FILE_VERSION:
        raise SystemExit(f"{path} is seed-file version {version}, expected {SEED_FILE_VERSION}")
    seeds = [Seed(frame=s["frame"], frame_index=int(s.get("frame_index", -1)),
                  label=s["label"], x=int(s["x"]), y=int(s["y"]))
             for s in payload["seeds"]]
    if not seeds:
        raise SystemExit(f"{path} contains no seeds")
    # First-appearance order, not the stored class list: a class nobody clicked
    # must not become a column the classifier can never fill.
    class_order = list(dict.fromkeys(s.label for s in seeds))
    return seeds, class_order, int(payload.get("pad", 60))


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


def _crop(bgr: np.ndarray, plant: np.ndarray, pad: int):
    """Same crop as `dino.crop_to_plant`, but keeping the background visible.

    The classifier zeroes the background; here it is dimmed instead, because
    you need the surrounding context to tell which organ you are aiming at,
    and the mask is drawn on top so it stays obvious what counts as plant.
    """
    ys, xs = np.nonzero(plant)
    if len(xs) == 0:
        return None, None
    y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)
    return bgr[y0:y1, x0:x1].copy(), plant[y0:y1, x0:x1]


def _render(view: np.ndarray, plant: np.ndarray, session: SeedSession,
            frame: str, message: str, scale: float) -> np.ndarray:
    canvas = view.copy()
    # Dim what the classifier will blank out, so off-plant clicks look wrong
    # before you make them.
    canvas[~plant] = (canvas[~plant] * 0.35).astype(np.uint8)

    for seed in session.seeds:
        if seed.frame != frame:
            continue
        colour = class_color(seed.label, session.classes.index(seed.label)
                             if seed.label in session.classes else 0)
        cv2.drawMarker(canvas, (seed.x, seed.y), (255, 255, 255),
                       cv2.MARKER_CROSS, 15, 3)
        cv2.drawMarker(canvas, (seed.x, seed.y), colour, cv2.MARKER_CROSS, 15, 1)
        cv2.circle(canvas, (seed.x, seed.y), 9, colour, 2, cv2.LINE_AA)

    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    bar_h = 78
    bar = np.full((bar_h, canvas.shape[1], 3), 24, np.uint8)
    counts = session.counts()
    x = 12
    for i, name in enumerate(session.classes):
        colour = class_color(name, i)
        active = i == session.current
        label = f"[{i + 1}] {name} ({counts[name]})"
        if active:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(bar, (x - 5, 8), (x + tw + 5, 8 + th + 9), colour, -1)
        cv2.putText(bar, label, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if active else colour, 1, cv2.LINE_AA)
        x += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 26

    cv2.putText(bar, f"{frame}   click=add  1-9=class  u=undo  c=clear  n/p=frame  s=save  q=quit",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    if message:
        colour = (90, 90, 235) if message.startswith("!") else (150, 220, 150)
        cv2.putText(bar, message.lstrip("! "), (12, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def display_available() -> Tuple[bool, str]:
    """Can OpenCV open a window here? WSL without WSLg typically cannot."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        cv2.waitKey(1)
        return True, ""
    except cv2.error as exc:  # pragma: no cover - depends on the host
        return False, str(exc).strip().splitlines()[-1] if str(exc).strip() else "no GUI backend"


def pick(
    frames_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    classes: Sequence[str],
    start_index: int = 0,
    pad: int = 60,
    max_display: int = 1100,
    window: str = "pick organ seeds",
) -> Optional[SeedSession]:
    """Open the picker. Returns the session, or None if the user quit."""
    frames_dir, mask_dir = Path(frames_dir), Path(mask_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit(f"no frames in {frames_dir} -- run pose-segment first")

    session = SeedSession(classes=list(classes))
    index = max(0, min(start_index, len(frame_paths) - 1))
    state = {"message": "", "click": None}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    loaded: Dict[int, tuple] = {}
    try:
        while True:
            if index not in loaded:
                path = frame_paths[index]
                bgr = cv2.imread(str(path))
                mask = cv2.imread(str(mask_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
                if bgr is None or mask is None:
                    raise SystemExit(f"could not read {path.name} or its mask in {mask_dir}")
                view, plant = _crop(bgr, mask > 127, pad)
                if view is None:
                    raise SystemExit(f"{path.stem} has an empty plant mask")
                loaded[index] = (path.stem, view, plant)
            stem, view, plant = loaded[index]

            scale = min(1.0, max_display / max(view.shape[0], view.shape[1]))

            if state["click"] is not None:
                cx, cy = state["click"]
                state["click"] = None
                # Undo the display scaling before recording, so what is stored
                # is a coordinate in the crop the classifier will build.
                ok, msg = session.add(stem, index, int(round(cx / scale)),
                                      int(round(cy / scale)), plant)
                state["message"] = msg if ok else f"! {msg}"

            cv2.imshow(window, _render(view, plant, session, stem, state["message"], scale))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                return None
            if key in (ord("s"), 13, 10):
                return session
            if key == ord("u"):
                removed = session.undo()
                state["message"] = f"removed {removed.label}" if removed else "! nothing to undo"
            elif key == ord("c"):
                state["message"] = f"cleared {session.clear()} seed(s)"
            elif key in (ord("n"), 83):
                index = min(index + 1, len(frame_paths) - 1)
                state["message"] = ""
            elif key in (ord("p"), 81):
                index = max(index - 1, 0)
                state["message"] = ""
            elif ord("1") <= key <= ord("9"):
                if session.set_class(key - ord("1")):
                    state["message"] = f"class -> {session.current_class}"
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
