"""Click the SAM2 seed points for P2, instead of deriving them from colour.

`derive_prompts` picks the largest excess-green blob that is not obviously
holder plastic. That is a reasonable guess and it is wrong whenever the
holder out-competes the plant on area, because the pliers' amber grip is
green-dominant in RGB -- its blue channel is near zero, so `2G-R-B` reads as
foliage. The brightness rule meant to catch it (`val >= 215`) encodes the
lighting of one shoot: on thistle1's low side-on pass the same grip sits at
V=188, slips the rule, and wins on area at 62k px against the plant's 44k.
SAM2 then tracks the pliers faithfully for the whole pass, P4a intersects two
silhouette sets that describe different objects 1.3 units apart, and the
carve comes out empty.

No threshold fixes that in general -- which object is "the plant" is not a
colour question. One click per pass settles it.

Two things this enforces, mirroring the P4c seed picker:

- **One seed frame per pass.** SAM2's video propagation is seeded at frame 0
  of each pass and carries memory forward from there, so a point clicked on
  any other frame has nowhere to attach. The picker therefore shows each
  pass's first frame and no other -- there is nothing to navigate.

- **Full-frame coordinates.** The P4c picker works in the classifier's crop
  because that crop already exists. Here it does not: the tracking crop is
  solved *from* the prepass being corrected, so clicks are stored full-frame
  and converted by `Prompts.to_crop` once the box is known.

The session state is separated from the window so it can be tested without a
display -- see `PromptSession`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from pose_estimator.pose import rotating_region_mask
from pose_estimator.segmentation import Prompts, derive_prompts

PROMPT_FILE_VERSION = 1
# "root" is a tracking category, not a third output class: the root is part
# of the plant, but the jaws cut it into a separate blob, and a blob sharing
# the foliage object's SAM2 memory flickers out of the mask (51-77% of frames
# on thistle1, under the 86% the carve needs). Rooted in its own object id it
# is a single connected region, which SAM2 tracks well; P2 unions it back
# into masks/plant at write time. Old prompt files without root points load
# unchanged, so the file version stays at 1.
CLASSES = ("plant", "holder", "root")

# BGR, matching the colours p2/diag overlays already use for the two masks.
# Root is drawn orange -- it becomes part of the green plant mask, but at
# click time it must be legible as its own category.
CLASS_BGR = {"plant": (60, 220, 60), "holder": (220, 60, 220), "root": (40, 150, 255)}


@dataclass
class CapturePass:
    """One capture pass and the frame its prompts must be clicked on."""

    index: int
    frame: str  # frame stem, e.g. "frame_0096"
    path: Path


@dataclass
class PromptSession:
    """Points clicked so far, per pass, in full-frame coordinates.

    Deliberately window-free: the rules that could be wrong (coordinate
    mapping, which pass a click belongs to, undo, what counts as complete)
    live here where they can be tested, and the window only drives them.
    """

    passes: List[CapturePass]
    classes: List[str] = field(default_factory=lambda: list(CLASSES))
    points: Dict[int, Dict[str, List[Tuple[int, int]]]] = field(default_factory=dict)
    position: int = 0  # index into `passes`, not the pass number
    current: int = 0  # index into `classes`

    def __post_init__(self) -> None:
        for spec in self.passes:
            self.points.setdefault(spec.index, {name: [] for name in self.classes})

    @property
    def current_pass(self) -> CapturePass:
        return self.passes[self.position]

    @property
    def current_class(self) -> str:
        return self.classes[self.current]

    def set_class(self, index: int) -> bool:
        if 0 <= index < len(self.classes):
            self.current = index
            return True
        return False

    def step_pass(self, delta: int) -> bool:
        new = self.position + delta
        if 0 <= new < len(self.passes):
            self.position = new
            return True
        return False

    def add(self, x: int, y: int, subject: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """Record a click on the current pass. Returns (accepted, message).

        `subject` is P3's rotating-region mask: what sweeps past the lens, so
        the table, holder and plant, as against the backdrop that travels with
        the camera rig. A click outside it is on something P3 will refuse to
        match features on, which makes it a real mistake worth flagging --
        unlike a brightness gate, which on this rig's grey backdrop keeps 88%
        of the frame and would flag nothing.

        Still a warning rather than a refusal: the mask is derived by Otsu on
        temporal variance, not ground truth, so a wrong one must not be able
        to lock you out of seeding at all.
        """
        spec = self.current_pass
        if subject is not None:
            height, width = subject.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                return False, "! outside the image"
        self.points[spec.index][self.current_class].append((int(x), int(y)))
        if subject is not None and not subject[y, x]:
            return True, (f"? {self.current_class} at ({x}, {y}) -- that is the backdrop, which "
                          "does not rotate with the table; P3 ignores it")
        return True, f"{self.current_class} at ({x}, {y})"

    def undo(self) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Remove the most recent point of the current pass, any class."""
        by_class = self.points[self.current_pass.index]
        for name in reversed(self.classes):
            if by_class[name]:
                return name, by_class[name].pop()
        return None

    def clear(self) -> int:
        by_class = self.points[self.current_pass.index]
        n = sum(len(v) for v in by_class.values())
        for name in self.classes:
            by_class[name].clear()
        return n

    def counts(self, pass_index: Optional[int] = None) -> Dict[str, int]:
        index = self.current_pass.index if pass_index is None else pass_index
        return {name: len(self.points[index][name]) for name in self.classes}

    def passes_missing_plant(self) -> List[int]:
        """Passes with no plant point. These are what makes a save invalid.

        A holder point is optional -- P2 tracks the holder only to keep it out
        of the plant mask -- but a pass with no plant seed has nothing to
        propagate, so saving one would produce a file that fails halfway
        through the next run rather than now.
        """
        return [spec.index for spec in self.passes if not self.points[spec.index]["plant"]]

    def as_prompts(self, pass_index: int) -> Prompts:
        by_class = self.points[pass_index]
        return Prompts(plant=list(by_class["plant"]), holder=list(by_class["holder"]),
                       root=list(by_class.get("root", [])), space="full_frame")

    def to_payload(self) -> dict:
        return {
            "version": PROMPT_FILE_VERSION,
            "space": "full_frame",
            "passes": {
                str(spec.index): {
                    "frame": spec.frame,
                    "plant": [list(p) for p in self.points[spec.index]["plant"]],
                    "holder": [list(p) for p in self.points[spec.index]["holder"]],
                    "root": [list(p) for p in self.points[spec.index].get("root", [])],
                }
                for spec in self.passes
            },
        }


def capture_passes(workdir: Union[str, Path]) -> List[CapturePass]:
    """The passes in a workdir, each with the frame SAM2 will be seeded on.

    Falls back to a single pass when `p1/sources.json` is absent, so this
    works on a workdir made before multi-pass capture existed.
    """
    workdir = Path(workdir)
    frames_dir = workdir / "p1" / "frames"
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit(f"no frames in {frames_dir} -- run pose-segment on this workdir first")

    sources_file = workdir / "p1" / "sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else {}

    first: Dict[int, Path] = {}
    for path in frame_paths:
        first.setdefault(int(sources.get(path.stem, 0)), path)
    return [CapturePass(index=i, frame=first[i].stem, path=first[i]) for i in sorted(first)]


def frames_per_pass(workdir: Union[str, Path]) -> Dict[int, List[Path]]:
    """Every frame of every pass, grouped the way `pose-segment` groups them."""
    workdir = Path(workdir)
    sources_file = workdir / "p1" / "sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else {}

    grouped: Dict[int, List[Path]] = {}
    for path in sorted((workdir / "p1" / "frames").glob("frame_*.jpg")):
        grouped.setdefault(int(sources.get(path.stem, 0)), []).append(path)
    return grouped


def save_prompts(path: Union[str, Path], session: PromptSession) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(session.to_payload(), f, indent=2)
    return path


def load_prompts(path: Union[str, Path]) -> Dict[int, Prompts]:
    """Read a clicked prompt file into {pass index: Prompts}."""
    with open(path) as f:
        payload = json.load(f)
    version = payload.get("version")
    if version != PROMPT_FILE_VERSION:
        raise SystemExit(f"{path} is prompt-file version {version}, expected {PROMPT_FILE_VERSION}")
    space = payload.get("space", "full_frame")
    if space != "full_frame":
        raise SystemExit(f"{path} stores {space!r} coordinates, expected 'full_frame'")

    prompts = {}
    for key, entry in payload.get("passes", {}).items():
        plant = [tuple(p) for p in entry.get("plant", [])]
        if not plant:
            raise SystemExit(f"{path}: pass {key} has no plant point")
        prompts[int(key)] = Prompts(plant=plant,
                                    holder=[tuple(p) for p in entry.get("holder", [])],
                                    root=[tuple(p) for p in entry.get("root", [])],
                                    space="full_frame")
    if not prompts:
        raise SystemExit(f"{path} contains no passes")
    return prompts


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


def _render(
    view: np.ndarray,
    session: PromptSession,
    auto: Optional[Prompts],
    message: str,
    scale: float,
) -> np.ndarray:
    canvas = view.copy()
    spec = session.current_pass

    # What the colour heuristic would have chosen, drawn faintly. When it is
    # sitting on the holder this is the fastest possible explanation of why
    # the run needed clicking, and it costs one derive per pass.
    if auto is not None:
        for name, points in (("plant", auto.plant), ("holder", auto.holder)):
            for x, y in points:
                cv2.drawMarker(canvas, (x, y), (110, 110, 110), cv2.MARKER_TILTED_CROSS, 22, 2)

    for name in session.classes:
        colour = CLASS_BGR[name]
        for x, y in session.points[spec.index][name]:
            cv2.drawMarker(canvas, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 21, 4)
            cv2.drawMarker(canvas, (x, y), colour, cv2.MARKER_CROSS, 21, 2)
            cv2.circle(canvas, (x, y), 13, colour, 2, cv2.LINE_AA)

    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    bar_h = 78
    bar = np.full((bar_h, canvas.shape[1], 3), 24, np.uint8)
    counts = session.counts()
    x = 12
    for i, name in enumerate(session.classes):
        colour = CLASS_BGR[name]
        active = i == session.current
        label = f"[{i + 1}] {name} ({counts[name]})"
        if active:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(bar, (x - 5, 8), (x + tw + 5, 8 + th + 9), colour, -1)
        cv2.putText(bar, label, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if active else colour, 1, cv2.LINE_AA)
        x += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 26

    todo = session.passes_missing_plant()
    status = f"needs a plant point: pass {', '.join(map(str, todo))}" if todo else "ready to save"
    cv2.putText(bar, f"pass {session.position + 1}/{len(session.passes)} ({spec.frame})   "
                     f"click=add  1-3=class  u=undo  c=clear  n/p=pass  s=save  q=quit   [{status}]",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    if message:
        colour = {"!": (90, 90, 235), "?": (60, 190, 235)}.get(message[0], (150, 220, 150))
        cv2.putText(bar, message.lstrip("!? "), (12, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def pick(
    passes: Sequence[CapturePass],
    max_display: int = 1400,
    show_auto: bool = True,
    window: str = "pick SAM2 prompts",
    pass_frames: Optional[Dict[int, Sequence[Path]]] = None,
) -> Optional[PromptSession]:
    """Open the picker. Returns the session, or None if the user quit.

    `pass_frames` supplies each pass's frames so the rotating-region mask can
    be measured; without it, clicks are recorded without the backdrop check.
    """
    session = PromptSession(passes=list(passes))
    state = {"message": "", "click": None}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    loaded: Dict[int, tuple] = {}
    try:
        while True:
            spec = session.current_pass
            if spec.index not in loaded:
                bgr = cv2.imread(str(spec.path))
                if bgr is None:
                    raise SystemExit(f"could not read {spec.path}")
                auto = derive_prompts(bgr) if show_auto else None
                subject = None
                if pass_frames and spec.index in pass_frames:
                    print(f"  measuring what rotates in pass {spec.index}...", flush=True)
                    subject = rotating_region_mask(pass_frames[spec.index])[0] > 0
                loaded[spec.index] = (bgr, subject, auto)
            view, subject, auto = loaded[spec.index]

            scale = min(1.0, max_display / max(view.shape[0], view.shape[1]))

            if state["click"] is not None:
                cx, cy = state["click"]
                state["click"] = None
                # Undo the display scaling before recording, so what is stored
                # is a coordinate in the full frame.
                _, msg = session.add(int(round(cx / scale)), int(round(cy / scale)), subject)
                state["message"] = msg

            cv2.imshow(window, _render(view, session, auto, state["message"], scale))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                return None
            if key in (ord("s"), 13, 10):
                missing = session.passes_missing_plant()
                if missing:
                    state["message"] = (
                        f"! cannot save -- pass {', '.join(map(str, missing))} has no plant point")
                    continue
                return session
            if key == ord("u"):
                removed = session.undo()
                state["message"] = f"removed {removed[0]}" if removed else "! nothing to undo"
            elif key == ord("c"):
                state["message"] = f"cleared {session.clear()} point(s)"
            elif key in (ord("n"), 83):
                state["message"] = "" if session.step_pass(1) else "! last pass"
            elif key in (ord("p"), 81):
                state["message"] = "" if session.step_pass(-1) else "! first pass"
            elif ord("1") <= key <= ord("9"):
                if session.set_class(key - ord("1")):
                    state["message"] = f"class -> {session.current_class}"
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
