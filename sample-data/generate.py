"""Generate sample data for the canvas examples.

Writes, into this folder:
- example-shapes.csv   : 784-feature (28x28) tabular form, 3 classes (for the ANN example)
- shapes-images.zip    : the SAME 3 shape classes as 28x28 grayscale PNGs in class
                         subfolders (for the image-ZIP dataset upload / CNN training)
- samples/*.png        : one loose image per class (for Observe "Upload an image")

Classes: 0=vertical bar, 1=horizontal bar, 2=diagonal. Each sample varies in
position, thickness, intensity and noise — plentiful and variable but learnable.

Run:  cd server && uv run python ../sample-data/generate.py
"""

import csv
import io
import os
import zipfile

import numpy as np
from PIL import Image

H = W = 28
PER_CLASS = 300
BASE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["vertical", "horizontal", "diagonal"]
rng = np.random.default_rng(7)


def sample(cls: int) -> np.ndarray:
    img = rng.normal(0.08, 0.05, (H, W))
    intensity = rng.uniform(0.6, 1.0)
    thick = rng.integers(2, 5)
    if cls == 0:
        c = rng.integers(3, W - 3 - thick)
        img[:, c : c + thick] += intensity
    elif cls == 1:
        r = rng.integers(3, H - 3 - thick)
        img[r : r + thick, :] += intensity
    else:
        off = rng.integers(-6, 7)
        for i in range(H):
            j = i + off
            for t in range(thick):
                if 0 <= j + t < W:
                    img[i, j + t] += intensity
    img += rng.normal(0, 0.04, (H, W))
    return np.clip(img, 0.0, 1.0).reshape(-1)


def _png_bytes(cls: int) -> bytes:
    a = (sample(cls).reshape(H, W) * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def write_csv() -> None:
    rows = [(sample(cls), cls) for cls in range(3) for _ in range(PER_CLASS)]
    rng.shuffle(rows)
    header = [f"p{i}" for i in range(H * W)] + ["label"]
    with open(os.path.join(BASE, "example-shapes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for px, lbl in rows:
            w.writerow([round(float(v), 3) for v in px] + [lbl])
    print(f"wrote {len(rows)} rows x {H*W} features -> example-shapes.csv")


def write_images(per: int = 150) -> None:
    zip_path = os.path.join(BASE, "shapes-images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls, name in enumerate(CLASS_NAMES):
            for i in range(per):
                zf.writestr(f"{name}/{name}_{i:03d}.png", _png_bytes(cls))
    samples_dir = os.path.join(BASE, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for cls, name in enumerate(CLASS_NAMES):
        with open(os.path.join(samples_dir, f"{name}.png"), "wb") as f:
            f.write(_png_bytes(cls))
    print(f"wrote {per * 3} images -> shapes-images.zip  + {len(CLASS_NAMES)} samples/*.png")


if __name__ == "__main__":
    write_csv()
    write_images()
