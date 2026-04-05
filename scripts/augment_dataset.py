"""
augment_dataset.py
==================
Dataset augmentation for LED Classifier (ESP32-CAM / TinyML)
Classes: red | green | blue | no_led

Produces ~400 images per class from 100 originals (4x).
IMPORTANT: no transformation alters chrominance (hue/saturation),
           to avoid corrupting labels based on LED color.

Expected input structure:
    dataset/raw/<class>/*.png

Produced output structure:
    dataset/augmented/<class>/*.png   ← originals + augmented
    dataset/augmented/dataset_stats.json

Usage:
    python augment_dataset.py
    python augment_dataset.py --input dataset/raw --output dataset/augmented --factor 4
"""

import os
import cv2
import numpy as np
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

# ── Default configuration ──────────────────────────────────────────────
CLASSES          = ["red", "green", "blue", "no_led"]
IMG_SIZE         = (96, 96)
DEFAULT_FACTOR   = 4          # total images = originals × factor
SEED             = 42

random.seed(SEED)
np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# TRANSFORMATIONS (all color-safe)
# ════════════════════════════════════════════════════════════════════════════

def aug_flip_horizontal(img: np.ndarray) -> np.ndarray:
    """Horizontal flip — simulates mirrored LED positions."""
    return cv2.flip(img, 1)


def aug_rotate(img: np.ndarray,
               max_angle: float = 15.0) -> np.ndarray:
    """Random rotation ±max_angle degrees around the center."""
    angle = random.uniform(-max_angle, max_angle)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # BORDER_REFLECT avoids artificial black borders
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def aug_brightness(img: np.ndarray,
                   low: float = 0.65,
                   high: float = 1.40) -> np.ndarray:
    """
    Modifies brightness by operating only on the V (HSV) channel.
    Preserves hue and saturation completely → LED color invariant.
    """
    factor = random.uniform(low, high)
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_contrast(img: np.ndarray,
                 low: float = 0.75,
                 high: float = 1.30) -> np.ndarray:
    """
    Contrast adjustment via alpha-blending with medium gray.
    Formula: out = alpha * img + (1 - alpha) * mean  → preserves hue.
    """
    alpha = random.uniform(low, high)
    mean  = np.mean(img)
    out   = np.clip(alpha * img.astype(np.float32) +
                    (1 - alpha) * mean, 0, 255)
    return out.astype(np.uint8)


def aug_gaussian_blur(img: np.ndarray,
                      max_ksize: int = 3) -> np.ndarray:
    """Light Gaussian blur — simulates optical blur / motion."""
    ksize = random.choice([k for k in range(1, max_ksize + 1, 2)])
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def aug_gaussian_noise(img: np.ndarray,
                       sigma_max: float = 12.0) -> np.ndarray:
    """Additive Gaussian noise — simulates OV2640 sensor noise."""
    sigma = random.uniform(2.0, sigma_max)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_zoom_crop(img: np.ndarray,
                  zoom_range: tuple = (1.05, 1.25)) -> np.ndarray:
    """
    Zoom in + center crop — simulates various distances from the camera.
    The image is enlarged and then re-cropped to IMG_SIZE.
    """
    h, w   = img.shape[:2]
    factor = random.uniform(*zoom_range)
    new_h  = int(h * factor)
    new_w  = int(w * factor)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Center crop
    y_start = (new_h - h) // 2
    x_start = (new_w - w) // 2
    return resized[y_start:y_start + h, x_start:x_start + w]


def aug_translate(img: np.ndarray,
                  max_shift_px: int = 6) -> np.ndarray:
    """
    Random translation ±max_shift_px pixels.
    Simulates the LED not perfectly centered in the frame.
    """
    dx = random.randint(-max_shift_px, max_shift_px)
    dy = random.randint(-max_shift_px, max_shift_px)
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REFLECT_101)


# ── Augmentation pipeline ──────────────────────────────────────────────────
# Each pipeline is a list of (function, probability_of_application).
# They are composed in random order for maximum variety.

AUGMENTATION_PIPELINE = [
    (aug_flip_horizontal, 0.50),
    (aug_rotate,          0.70),
    (aug_brightness,      0.80),
    (aug_contrast,        0.50),
    (aug_gaussian_blur,   0.40),
    (aug_gaussian_noise,  0.50),
    (aug_zoom_crop,       0.40),
    (aug_translate,       0.60),
]


def apply_random_augmentation(img: np.ndarray) -> np.ndarray:
    """
    Applies a random subset of the pipeline.
    Ensures that each augmented image is statistically unique.
    """
    aug_img = img.copy()
    # Shuffle the order to increase combinatorial variability
    pipeline = AUGMENTATION_PIPELINE.copy()
    random.shuffle(pipeline)

    for fn, prob in pipeline:
        if random.random() < prob:
            aug_img = fn(aug_img)

    return aug_img


# ════════════════════════════════════════════════════════════════════════════
# CORE: load, augment, save
# ════════════════════════════════════════════════════════════════════════════

def load_images(class_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Load all images from a class directory."""
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(class_dir.glob(ext)):
            img = cv2.imread(str(p))
            if img is None:
                print(f"  [WARN] Unable to load: {p.name}")
                continue
            # Resize if necessary (safety)
            if img.shape[:2] != IMG_SIZE:
                img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            images.append((p.stem, img))
    return images


def augment_class(class_name: str,
                  input_dir: Path,
                  output_dir: Path,
                  target_factor: int) -> dict:
    """
    Processes a single class:
    - copies originals to output_dir
    - generates (target_factor - 1) × N augmented images
    Returns statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = load_images(input_dir)

    if not images:
        print(f"  [SKIP] No images found in {input_dir}")
        return {"original": 0, "augmented": 0, "total": 0}

    n_original  = len(images)
    n_to_gen    = n_original * (target_factor - 1)
    aug_count   = 0

    # ── 1. Copy originals ──────────────────────────────────
    for stem, img in images:
        out_path = output_dir / f"{stem}_orig.png"
        cv2.imwrite(str(out_path), img)

    print(f"  Originals copied : {n_original}")

    # ── 2. Generate augmented ───────────────────────────────────────────────
    # Loops through originals uniformly to avoid favoring some
    for i in range(n_to_gen):
        stem, img = images[i % n_original]
        aug_img   = apply_random_augmentation(img)
        ts        = int(datetime.now().timestamp() * 1000) + i
        out_path  = output_dir / f"{stem}_aug_{i:04d}_{ts}.png"
        cv2.imwrite(str(out_path), aug_img)
        aug_count += 1

        if (i + 1) % 50 == 0:
            print(f"    ...generated {i + 1}/{n_to_gen}")

    total = n_original + aug_count
    print(f"  Generated augmented: {aug_count}")
    print(f"  Total class        : {total}")

    return {
        "original":  n_original,
        "augmented": aug_count,
        "total":     total
    }


# ════════════════════════════════════════════════════════════════════════════
# VISUAL VERIFICATION: generates a preview grid of augmentations
# ════════════════════════════════════════════════════════════════════════════

def generate_preview(input_dir: Path,
                     output_path: Path,
                     class_name: str,
                     n_rows: int = 4,
                     n_cols: int = 8) -> None:
    """
    Generates a grid image showing original + its augmentations.
    Useful for visually inspecting that colors are preserved.
    """
    images = load_images(input_dir)
    if not images:
        return

    # Take the first original and generate n_rows*n_cols-1 versions
    _, base_img = images[0]
    cells       = [base_img]  # first cell = original

    for _ in range(n_rows * n_cols - 1):
        cells.append(apply_random_augmentation(base_img))

    cell_size = 96
    padding   = 4
    grid_h    = n_rows * (cell_size + padding) + padding + 30
    grid_w    = n_cols * (cell_size + padding) + padding
    grid      = np.zeros((grid_h, grid_w, 3), dtype=np.uint8) + 40

    for idx, cell in enumerate(cells[:n_rows * n_cols]):
        row = idx // n_cols
        col = idx  % n_cols
        y   = padding + row * (cell_size + padding)
        x   = padding + col * (cell_size + padding)
        grid[y:y + cell_size, x:x + cell_size] = cell

    # Label "ORIG" on the first cell
    cv2.putText(grid, "ORIG", (padding + 2, padding + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

    # Title
    cv2.putText(grid, f"Preview augmentation: {class_name}",
                (padding, grid_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imwrite(str(output_path), grid)
    print(f"  Preview saved: {output_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dataset augmentation for LED Classifier (color-safe)")
    parser.add_argument("--input",   default="dataset/raw",
                        help="Root directory with original classes")
    parser.add_argument("--output",  default="dataset/augmented",
                        help="Output directory for the augmented dataset")
    parser.add_argument("--factor",  type=int, default=DEFAULT_FACTOR,
                        help="Total multiplier (default 4 = 400 images from 100)")
    parser.add_argument("--preview", action="store_true",
                        help="Generate grid preview for each class")
    args = parser.parse_args()

    input_root  = Path(args.input)
    output_root = Path(args.output)
    factor      = args.factor

    print("=" * 60)
    print(" LED Classifier — Dataset Augmentation")
    print("=" * 60)
    print(f" Input  : {input_root.resolve()}")
    print(f" Output : {output_root.resolve()}")
    print(f" Factor : {factor}x  (e.g. 100 orig → {100 * factor} total)")
    print("=" * 60)

    all_stats   = {}
    grand_total = 0

    for cls in CLASSES:
        in_dir  = input_root  / cls
        out_dir = output_root / cls

        if not in_dir.exists():
            print(f"\n[SKIP] {cls}: directory not found ({in_dir})")
            continue

        print(f"\n▶ Class: {cls.upper()}")
        stats = augment_class(cls, in_dir, out_dir, factor)
        all_stats[cls] = stats
        grand_total   += stats["total"]

        if args.preview:
            preview_path = output_root / f"_preview_{cls}.png"
            generate_preview(in_dir, preview_path, cls)

    # ── Save JSON statistics ────────────────────────────────────
    stats_payload = {
        "generated_at":  datetime.now().isoformat(),
        "augment_factor": factor,
        "img_size":       list(IMG_SIZE),
        "classes":        all_stats,
        "grand_total":    grand_total
    }
    stats_path = output_root / "dataset_stats.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats_payload, f, indent=2)

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    for cls, s in all_stats.items():
        bar = "█" * int(s["total"] / 20)
        print(f"  {cls:<10} orig={s['original']:>4}  "
              f"aug={s['augmented']:>4}  tot={s['total']:>4}  {bar}")
    print(f"\n  Total images in dataset: {grand_total}")
    print(f"  Statistics saved in     : {stats_path}")
    print("=" * 60)
    print("\n✅ Dataset ready for training on Google Colab!")
    print("   Next step: compress 'dataset/augmented/' and upload to Drive.")


if __name__ == "__main__":
    main()