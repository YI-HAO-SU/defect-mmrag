"""Generate synthetic MVTec-AD-structured sample data for local development.

Creates a small dataset of procedurally generated defect images so the demo
pipeline can be exercised without downloading the real MVTec AD archive.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --out_dir ./data/mvtec --per_class 5
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Defect generators
# ---------------------------------------------------------------------------

_RNG = random.Random(42)


def _base_image(size: int = 224) -> Image.Image:
    """Uniform grey background — mimics a bare wafer / product surface."""
    gray = _RNG.randint(180, 210)
    return Image.new("RGB", (size, size), color=(gray, gray, gray))


def _add_noise(img: Image.Image, amount: int = 8) -> Image.Image:
    import struct
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            d = _RNG.randint(-amount, amount)
            px[x, y] = (
                max(0, min(255, r + d)),
                max(0, min(255, g + d)),
                max(0, min(255, b + d)),
            )
    return img


def make_good(size: int = 224) -> Image.Image:
    return _add_noise(_base_image(size))


def make_broken_large(size: int = 224) -> Image.Image:
    img = _add_noise(_base_image(size))
    draw = ImageDraw.Draw(img)
    x0 = _RNG.randint(20, 80)
    y0 = _RNG.randint(20, 80)
    w = _RNG.randint(60, 100)
    h = _RNG.randint(60, 100)
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(40, 40, 40))
    return img


def make_broken_small(size: int = 224) -> Image.Image:
    img = _add_noise(_base_image(size))
    draw = ImageDraw.Draw(img)
    for _ in range(_RNG.randint(2, 5)):
        x = _RNG.randint(10, size - 30)
        y = _RNG.randint(10, size - 30)
        r = _RNG.randint(5, 15)
        draw.ellipse([x, y, x + r, y + r], fill=(60, 60, 60))
    return img


def make_contamination(size: int = 224) -> Image.Image:
    img = _add_noise(_base_image(size))
    draw = ImageDraw.Draw(img)
    x = _RNG.randint(30, size - 60)
    y = _RNG.randint(30, size - 60)
    rx = _RNG.randint(20, 50)
    ry = _RNG.randint(15, 40)
    draw.ellipse([x, y, x + rx, y + ry], fill=(120, 80, 50))
    return img.filter(ImageFilter.GaussianBlur(1))


def make_crack(size: int = 224) -> Image.Image:
    img = _add_noise(_base_image(size))
    draw = ImageDraw.Draw(img)
    x0, y0 = _RNG.randint(0, size // 2), _RNG.randint(0, size // 2)
    x1, y1 = x0 + _RNG.randint(40, 120), y0 + _RNG.randint(20, 80)
    draw.line([(x0, y0), (x1, y1)], fill=(30, 30, 30), width=2)
    return img


def make_scratch(size: int = 224) -> Image.Image:
    img = _add_noise(_base_image(size))
    draw = ImageDraw.Draw(img)
    x0 = _RNG.randint(0, size // 3)
    x1 = x0 + _RNG.randint(80, 150)
    y = _RNG.randint(40, size - 40)
    draw.line([(x0, y), (x1, y + _RNG.randint(-10, 10))],
              fill=(50, 50, 50), width=1)
    return img


def make_hole(size: int = 224) -> Image.Image:
    img = _add_noise(_base_image(size))
    draw = ImageDraw.Draw(img)
    x = _RNG.randint(40, size - 60)
    y = _RNG.randint(40, size - 60)
    r = _RNG.randint(10, 25)
    draw.ellipse([x, y, x + r, y + r], fill=(15, 15, 15))
    return img


def make_cut(size: int = 224) -> Image.Image:
    img = _add_noise(_base_image(size))
    draw = ImageDraw.Draw(img)
    y = _RNG.randint(60, size - 60)
    draw.line([(0, y), (size, y + _RNG.randint(-5, 5))],
              fill=(25, 25, 25), width=3)
    return img


_GENERATORS: dict[str, callable] = {
    "good": make_good,
    "broken_large": make_broken_large,
    "broken_small": make_broken_small,
    "contamination": make_contamination,
    "crack": make_crack,
    "scratch": make_scratch,
    "hole": make_hole,
    "cut": make_cut,
}

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

CATEGORIES = ["bottle", "cable", "pill"]


def build_dataset(out_dir: Path, per_class: int = 5) -> None:
    for category in CATEGORIES:
        for defect, gen_fn in _GENERATORS.items():
            folder = out_dir / category / "test" / defect
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img = gen_fn()
                img.save(folder / f"{i:03d}.png")
        print(f"  [+] {category}: {len(_GENERATORS)} defect classes × {per_class} images")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=Path("./data/mvtec"))
    p.add_argument("--per_class", type=int, default=5,
                   help="Images to generate per defect class (default: 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Generating synthetic MVTec-style data → {args.out_dir.resolve()}")
    build_dataset(args.out_dir, args.per_class)
    total = len(CATEGORIES) * len(_GENERATORS) * args.per_class
    print(f"Done. {total} images across {len(CATEGORIES)} categories.")
    print()
    print("Next step:")
    print(f"  python scripts/ingest.py --data_dir {args.out_dir}")


if __name__ == "__main__":
    main()
