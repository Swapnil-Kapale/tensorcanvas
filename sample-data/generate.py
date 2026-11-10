"""Generate sample data for the canvas example network:
Input[1,28,28] -> Flatten -> Dense(128,relu) -> Dense(3) -> softmax.

Produces a CSV with 784 pixel features (row-major, values in [0,1]) + a final
`label` column with 3 classes:
  0 = vertical bar, 1 = horizontal bar, 2 = diagonal.
Each sample varies in position, thickness, intensity and noise, so the data is
plentiful and variable but still learnable.

Run:  cd server && uv run python ../sample-data/generate.py
"""

import csv
import numpy as np

H = W = 28
PER_CLASS = 300          # 900 samples total
OUT = __file__.rsplit("/", 1)[0] + "/example-shapes.csv"
rng = np.random.default_rng(7)


def sample(cls: int) -> np.ndarray:
    img = rng.normal(0.08, 0.05, (H, W))          # noisy dark background
    intensity = rng.uniform(0.6, 1.0)
    thick = rng.integers(2, 5)
    if cls == 0:                                    # vertical bar
        c = rng.integers(3, W - 3 - thick)
        img[:, c:c + thick] += intensity
    elif cls == 1:                                  # horizontal bar
        r = rng.integers(3, H - 3 - thick)
        img[r:r + thick, :] += intensity
    else:                                           # diagonal
        off = rng.integers(-6, 7)
        for i in range(H):
            j = i + off
            for t in range(thick):
                if 0 <= j + t < W:
                    img[i, j + t] += intensity
    img += rng.normal(0, 0.04, (H, W))              # per-pixel speckle
    return np.clip(img, 0.0, 1.0).reshape(-1)


def main() -> None:
    rows = []
    for cls in range(3):
        for _ in range(PER_CLASS):
            rows.append((sample(cls), cls))
    rng.shuffle(rows)                               # mix classes
    header = [f"p{i}" for i in range(H * W)] + ["label"]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for px, lbl in rows:
            w.writerow([round(float(v), 3) for v in px] + [lbl])
    print(f"wrote {len(rows)} samples x {H*W} features -> {OUT}")


if __name__ == "__main__":
    main()
