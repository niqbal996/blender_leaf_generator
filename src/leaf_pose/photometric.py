"""Surface normals from the 12 lights, and the crease they reveal.

Optional, and worth the extra pass. The flat image deliberately throws shape
away -- it is the average of 12 light directions precisely so that shading
cancels -- and what is left is colour. A midrib that happens to be the same
green as the lamina beside it is then nearly invisible in it, and the ridge
filter in `midrib.py` has little to respond to.

Photometric stereo puts the shape back. Twelve photographs of a static scene
under twelve known light directions over-determine the surface orientation at
every pixel: with Lambert's law, brightness is albedo times the cosine
between the normal and the light, so three lights already fix the normal and
the remaining nine make it robust. The crossed-polariser set is used, not the
parallel one, because Lambert's law describes diffuse reflection and crossed
polarisers are what remove everything that is not.

The midrib is then a crease in the normal field -- normals tip away from the
crest on both sides -- which is a far stronger and more local signal than
"slightly brighter than its surroundings". `ridge_from_normals` measures
exactly that, and `midrib.fit_midrib` takes it as evidence.

The normals are worth keeping in their own right: `leaf_generator` builds
Blender leaves from `..._NORMAL_GL_...` maps, and this is that map, measured
rather than inferred from a single photograph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# Observations discarded per pixel before the fit. Lambert's law describes
# neither a cast shadow nor a specular flare, and each biases the normal
# hard in a different direction; the cheapest sound answer with 12 lights is
# to rank the observations at each pixel and drop the extremes. Asymmetric
# because shadows are both more common and more damaging than the residual
# highlights that survive crossed polarisers.
DROP_DARKEST = 3
DROP_BRIGHTEST = 1


@dataclass
class Photometric:
    normals: np.ndarray  # (H, W, 3) unit vectors, +z toward the camera
    albedo: np.ndarray   # (H, W) float32
    ridge: np.ndarray    # (H, W) float32 0..1, high on a crest

    def normal_map_rgb(self) -> np.ndarray:
        """The normals as an OpenGL-convention 8-bit normal map.

        Green is flipped because image rows run downward and the GL
        convention has +G pointing up -- the difference between a leaf that
        lights correctly in Blender and one lit from the wrong side.
        """
        encoded = np.stack([self.normals[:, :, 0],
                            -self.normals[:, :, 1],
                            self.normals[:, :, 2]], axis=-1)
        return np.clip((encoded * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def crop_light_stack(
    frames: Sequence[Path],
    boxes: Sequence[Tuple[int, int, int, int]],
    progress: Optional[Callable[[str], None]] = None,
) -> List[np.ndarray]:
    """Decode each frame once and cut every ROI out of it.

    The loop is this way round on purpose. A 61 megapixel frame is 732 MB
    decoded, so the full 12-light stack is 8.8 GB and does not fit; the leaf
    ROIs together are a few per cent of the frame, so per-ROI stacks do. The
    cost is that the RAW files are read a second time, after detection has
    said where to cut.

    Returns one (N, h, w) luminance stack per box, in the frames' order.
    """
    from .raw import decode

    say = progress or (lambda _message: None)
    stacks = [[] for _ in boxes]
    for index, path in enumerate(frames, start=1):
        say(f"  light {index}/{len(frames)}  {path.name}")
        frame = decode(path)
        for slot, (x0, y0, x1, y1) in enumerate(boxes):
            stacks[slot].append(frame[y0:y1, x0:x1].mean(axis=2).astype(np.float32))
        del frame
    return [np.stack(s) for s in stacks]


def observation_weights(stack: np.ndarray) -> np.ndarray:
    """1 for observations to fit, 0 for the per-pixel extremes to discard."""
    lights = stack.shape[0]
    keep_low, keep_high = DROP_DARKEST, lights - DROP_BRIGHTEST
    if keep_high - keep_low < 3:
        return np.ones_like(stack, np.float32)

    rank = np.argsort(np.argsort(stack, axis=0), axis=0)
    return ((rank >= keep_low) & (rank < keep_high)).astype(np.float32)


def solve(
    stack: np.ndarray, directions: np.ndarray, mask: Optional[np.ndarray] = None,
) -> Photometric:
    """Least-squares normals from (N, H, W) intensities and (N, H, W, 3) lights.

    Solved per pixel rather than once for the whole crop, because the light
    directions vary across it -- see `rig.light_directions`. The normal
    equations are only 3x3, so the whole field is one batched solve.
    """
    weights = observation_weights(stack)
    weighted = directions * weights[..., None]

    normal_matrix = np.einsum("nhwi,nhwj->hwij", weighted, directions)
    right_hand = np.einsum("nhwi,nhw->hwi", weighted, stack)

    # Ridge regression, with a term far below any real signal: a pixel lit
    # from one direction only -- deep in a fold, or masked to almost nothing
    # -- leaves a singular 3x3, and a singular solve takes the whole crop
    # down rather than that pixel.
    normal_matrix += np.eye(3, dtype=np.float32) * 1e-6
    # The trailing axis is explicit: NumPy 2 reads a bare (H, W, 3) right-hand
    # side as a stack of (W, 3) matrices rather than as one vector per pixel.
    gradient = np.linalg.solve(normal_matrix, right_hand[..., None])[..., 0]

    albedo = np.linalg.norm(gradient, axis=-1)
    normals = gradient / np.maximum(albedo, 1e-9)[..., None]

    # The surface faces the camera; a fit that says otherwise has found the
    # mirror solution, which is a sign error rather than a measurement.
    flip = normals[:, :, 2] < 0
    normals[flip] *= -1.0

    if mask is not None:
        normals[~mask] = (0.0, 0.0, 1.0)
        albedo = albedo * mask

    return Photometric(normals=normals.astype(np.float32),
                       albedo=albedo.astype(np.float32),
                       ridge=ridge_from_normals(normals, mask))


def ridge_from_normals(normals: np.ndarray, mask: Optional[np.ndarray] = None,
                       blur: int = 5) -> np.ndarray:
    """Where the surface creases upward, normalised to 0..1.

    The divergence of the normals' in-plane part. Walk across a ridge and the
    normal tips away from the crest on each side -- leftward to the left of
    it, rightward to the right -- so the in-plane field points outward from
    the crest line, which is what a positive divergence is.

    The sign is the whole discriminating power of this: a groove, such as the
    seam where one leaf overlaps another, tips its normals *inward* and
    scores negative, so it cannot be mistaken for a midrib. Getting it
    backwards does not fail loudly -- it quietly returns the flanks instead
    of the crest, which is how it was caught: on a synthetic rib centred at
    column 45 the response peaked at 62.
    """
    field = normals[:, :, :2].astype(np.float32)
    if blur > 1:
        field = cv2.GaussianBlur(field, (0, 0), blur)

    dx = cv2.Sobel(field[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(field[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
    response = dx + dy

    inside = response[mask] if mask is not None and mask.any() else response.ravel()
    low, high = np.percentile(inside, [50.0, 99.0])
    if high - low < 1e-9:
        return np.zeros(response.shape, np.float32)
    scaled = np.clip((response - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return scaled * mask if mask is not None else scaled
