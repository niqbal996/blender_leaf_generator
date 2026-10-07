#!/usr/bin/env python
"""Would a stem mask from its own SAM3 session fix leaves leaking onto petioles?

    python scripts/stem_session_check.py --lab <out of sam3_p2_prompt_lab.py --save-masks> \
        --workdir <specimen>/plant --watch pass0_24 --out runs/<specimen>_stem_session

The lab run this reads (frames of ONE pass, "stem" in its own group):

    python scripts/sam3_p2_prompt_lab.py --images <wd>/p1/frames --p2 <wd>/p2 \
        --frames 22-36 --group plant "plant" "leaf" --group stem "stem" \
        --save-masks --out <wd>/p2_stem_session

P2 today runs "leaf", "stem" and "plant" in ONE session and writes
masks/stem = plant - leaf instances - holder - root. A session gives each
object to one phrase, so where SAM3's leaf track slides onto a petiole --
vogelmeere's pass0_24 sat on leaf 6's petiole from frame 33 -- the stem
phrase cannot also claim it, the residual loses it, and P5x votes that
petiole "leaf". The proposal under test:

    stem  = (plant - leaf) | stem_from_its_own_session
    leaf_k = leaf_k - stem                (each leaf instance ends at the blade)

Answers, per frame, with numbers:
  1. how much of the --watch track's mask the separate stem session claims
     (high on frames where the track sits on a petiole = the leak is fixed);
  2. per leaf instance, the share of its mask the trim removes (a blade
     losing a large share means the stem session strayed onto it);
  3. today's stem mask against the proposed one, in area.
Writes stem_session_check.jpg (today | proposed stem, --watch track outlined)
and stem_session_check.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def full_frame(crop_mask: np.ndarray, box, shape) -> np.ndarray:
    out = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    h, w = min(crop_mask.shape[0], shape[0] - y0), min(crop_mask.shape[1], shape[1] - x0)
    out[y0:y0 + h, x0:x0 + w] = crop_mask[:h, :w]
    return out


def read(path: Path, shape=None):
    if not path.exists():
        return None if shape is None else np.zeros(shape, bool)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--watch", default="pass0_24", help="leaf track to follow onto its petiole")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lab = json.loads((args.lab / "lab.json").read_text())
    names = [Path(n).stem for n in lab["frame_names"]]
    boxes = lab["crop"]["boxes"]
    shape = (lab["crop"]["frame_height"], lab["crop"]["frame_width"])
    p2 = args.workdir / "p2" / "masks"
    tracks = sorted(d for d in (p2 / "leaf_instances").iterdir() if d.is_dir())

    rows, tiles = [], []
    for name, box in zip(names, boxes):
        lab_mask = lambda phrase: full_frame(read(args.lab / "masks" / phrase / f"{name}.png",
                                                   (box[3] - box[1], box[2] - box[0])), box, shape)
        stem_own = lab_mask("stem")
        plant = lab_mask("plant") | lab_mask("leaf")
        leaf = lab_mask("leaf")
        today = read(p2 / "stem" / f"{name}.png", shape)
        holder = read(p2 / "holder" / f"{name}.png", shape)
        proposed = ((plant & ~leaf) | stem_own) & ~holder

        watch = read(p2 / "leaf_instances" / args.watch / f"{name}.png", shape)
        trims = {}
        for d in tracks:
            m = read(d / f"{name}.png")
            # In P2 a leaf instance and `plant - leaf` come from one session,
            # so they never overlap: only the separate stem session can trim a
            # leaf. (Measuring against `proposed` here would count leaves this
            # lab's own leaf prompt happened to miss.)
            if m is not None and m.sum() >= 200:
                trims[d.name] = float((m & stem_own & ~holder).sum() / m.sum())
        row = {"frame": name,
               "watch_px": int(watch.sum()),
               "watch_claimed_by_stem_session": float((watch & stem_own).sum() / max(watch.sum(), 1)),
               "stem_today_px": int(today.sum()), "stem_proposed_px": int(proposed.sum()),
               "stem_session_px": int(stem_own.sum()),
               "leaves_trimmed_over_30pct": sorted(k for k, v in trims.items() if v > 0.3),
               "median_trim": float(np.median(list(trims.values()))) if trims else 0.0}
        rows.append(row)

        photo = cv2.imread(str(args.workdir / "p1" / "frames" / f"{name}.jpg"))
        x0, y0, x1, y1 = box
        panels = []
        for title, mask in (("today: plant - leaf", today), ("proposed: (plant - leaf) | stem", proposed)):
            img = (photo * 0.45).astype(np.uint8)
            img[mask] = (0.4 * img[mask] + [0, 200, 255]).astype(np.uint8)
            if watch.any():
                cs, _ = cv2.findContours(watch.astype(np.uint8), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (255, 80, 255), 2)
            img = img[y0:y1, x0:x1]
            img = cv2.resize(img, (560, int(560 * img.shape[0] / img.shape[1])))
            cv2.putText(img, f"{name} {title}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(img)
        tiles.append(np.hstack(panels))

    print(f"{'frame':12s} {args.watch:>10s}px  claimed-by-stem  stem today  proposed  "
          f"stem-session  median-trim  leaves trimmed >30%")
    for r in rows:
        print(f"{r['frame']:12s} {r['watch_px']:10d}  {r['watch_claimed_by_stem_session']:14.0%}  "
              f"{r['stem_today_px']:10d}  {r['stem_proposed_px']:8d}  {r['stem_session_px']:12d}  "
              f"{r['median_trim']:10.1%}  {r['leaves_trimmed_over_30pct']}")
    (args.out / "stem_session_check.json").write_text(json.dumps(rows, indent=1))
    picks = sorted(range(len(rows)), key=lambda i: -rows[i]["watch_claimed_by_stem_session"])[:4]
    cv2.imwrite(str(args.out / "stem_session_check.jpg"), np.vstack([tiles[i] for i in sorted(picks)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {args.out / 'stem_session_check.jpg'} (the 4 frames where the stem session "
          f"claims most of {args.watch}), stem_session_check.json")


if __name__ == "__main__":
    main()
