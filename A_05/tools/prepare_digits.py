"""
Turn photos/scans of my handwritten digits into MNIST-style 28x28 grayscale PNGs.

MNIST convention: white digit on a black background, the digit fitted into a
20x20 box and centred by its centre of mass inside a 28x28 frame.

Usage
-----
    python tools/prepare_digits.py raw_digits/ images/

Every file in raw_digits/ must already be named n_m.<ext>  (e.g. 7_3.jpg).
One digit per photo; dark ink on light paper works best.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def to_mnist_style(path, threshold=0.25):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    # paper is usually brighter than ink -> invert so the digit becomes white
    if np.median(g) > 0.5:
        g = 1.0 - g

    # stretch contrast and remove background noise
    g = (g - g.min()) / max(g.max() - g.min(), 1e-6)
    g[g < threshold] = 0.0

    # crop to the bounding box of the ink
    rows, cols = np.nonzero(g)
    if rows.size == 0:
        raise ValueError(f"{path}: no ink found")
    g = g[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    # fit the longer side into 20 px, keep aspect ratio
    h, w = g.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = Image.fromarray((g * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS)

    # paste into 28x28 then shift so the centre of mass sits in the middle
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(digit, dtype=np.float32)
    ys, xs = np.indices(canvas.shape)
    mass = canvas.sum()
    dy = int(round(14 - (ys * canvas).sum() / mass))
    dx = int(round(14 - (xs * canvas).sum() / mass))
    canvas = np.roll(np.roll(canvas, dy, axis=0), dx, axis=1)

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="L")


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    for p in files:
        out = dst / f"{p.stem}.png"
        to_mnist_style(p).save(out)
        print(f"{p.name:>12} -> {out}")
    print(f"done: {len(files)} images")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
