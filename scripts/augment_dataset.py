"""
augment_dataset.py
==================
Augmentation del dataset per LED Classifier (ESP32-CAM / TinyML)
Classi: red | green | blue | no_led

Produce ~400 immagini per classe a partire da 100 originali (4x).
IMPORTANTE: nessuna trasformazione altera la crominanza (hue/saturation),
            per non corrompere le label basate sul colore del LED.

Struttura attesa in input:
    dataset/raw/<classe>/*.png

Struttura prodotta in output:
    dataset/augmented/<classe>/*.png   ← originali + augmentati
    dataset/augmented/dataset_stats.json

Uso:
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

# ── Configurazione di default ────────────────────────────────────────────────
CLASSES          = ["red", "green", "blue", "no_led"]
IMG_SIZE         = (96, 96)
DEFAULT_FACTOR   = 4          # immagini totali = originali × factor
SEED             = 42

random.seed(SEED)
np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# TRASFORMAZIONI (tutte color-safe)
# ════════════════════════════════════════════════════════════════════════════

def aug_flip_horizontal(img: np.ndarray) -> np.ndarray:
    """Flip orizzontale — simula posizioni specchiate del LED."""
    return cv2.flip(img, 1)


def aug_rotate(img: np.ndarray,
               max_angle: float = 15.0) -> np.ndarray:
    """Rotazione casuale ±max_angle gradi attorno al centro."""
    angle = random.uniform(-max_angle, max_angle)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # BORDER_REFLECT evita bordi neri artificiali
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def aug_brightness(img: np.ndarray,
                   low: float = 0.65,
                   high: float = 1.40) -> np.ndarray:
    """
    Modifica la luminosità operando solo sul canale V (HSV).
    Preserva completamente hue e saturation → colore LED invariato.
    """
    factor = random.uniform(low, high)
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_contrast(img: np.ndarray,
                 low: float = 0.75,
                 high: float = 1.30) -> np.ndarray:
    """
    Regolazione del contrasto tramite alpha-blending con grigio medio.
    Formula: out = alpha * img + (1 - alpha) * mean  → preserva hue.
    """
    alpha = random.uniform(low, high)
    mean  = np.mean(img)
    out   = np.clip(alpha * img.astype(np.float32) +
                    (1 - alpha) * mean, 0, 255)
    return out.astype(np.uint8)


def aug_gaussian_blur(img: np.ndarray,
                      max_ksize: int = 3) -> np.ndarray:
    """Sfocatura gaussiana leggera — simula sfocatura ottica / movimento."""
    ksize = random.choice([k for k in range(1, max_ksize + 1, 2)])
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def aug_gaussian_noise(img: np.ndarray,
                       sigma_max: float = 12.0) -> np.ndarray:
    """Rumore gaussiano additivo — simula il rumore del sensore OV2640."""
    sigma = random.uniform(2.0, sigma_max)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_zoom_crop(img: np.ndarray,
                  zoom_range: tuple = (1.05, 1.25)) -> np.ndarray:
    """
    Zoom in + crop centrale — simula diverse distanze dalla camera.
    L'immagine viene ingrandita e poi ricropata a IMG_SIZE.
    """
    h, w   = img.shape[:2]
    factor = random.uniform(*zoom_range)
    new_h  = int(h * factor)
    new_w  = int(w * factor)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Crop centrale
    y_start = (new_h - h) // 2
    x_start = (new_w - w) // 2
    return resized[y_start:y_start + h, x_start:x_start + w]


def aug_translate(img: np.ndarray,
                  max_shift_px: int = 6) -> np.ndarray:
    """
    Traslazione casuale ±max_shift_px pixel.
    Simula il LED non perfettamente centrato nel frame.
    """
    dx = random.randint(-max_shift_px, max_shift_px)
    dy = random.randint(-max_shift_px, max_shift_px)
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REFLECT_101)


# ── Pipeline di augmentation ─────────────────────────────────────────────────
# Ogni pipeline è una lista di (funzione, probabilità_di_applicazione).
# Vengono composte in sequenza casuale per massima varietà.

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
    Applica un sottoinsieme casuale della pipeline.
    Garantisce che ogni immagine augmentata sia statisticamente unica.
    """
    aug_img = img.copy()
    # Shuffle dell'ordine per aumentare la variabilità combinatoria
    pipeline = AUGMENTATION_PIPELINE.copy()
    random.shuffle(pipeline)

    for fn, prob in pipeline:
        if random.random() < prob:
            aug_img = fn(aug_img)

    return aug_img


# ════════════════════════════════════════════════════════════════════════════
# CORE: carica, aumenta, salva
# ════════════════════════════════════════════════════════════════════════════

def load_images(class_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Carica tutte le immagini da una directory di classe."""
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(class_dir.glob(ext)):
            img = cv2.imread(str(p))
            if img is None:
                print(f"  [WARN] Impossibile caricare: {p.name}")
                continue
            # Ridimensiona se necessario (sicurezza)
            if img.shape[:2] != IMG_SIZE:
                img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            images.append((p.stem, img))
    return images


def augment_class(class_name: str,
                  input_dir: Path,
                  output_dir: Path,
                  target_factor: int) -> dict:
    """
    Processa una singola classe:
    - copia gli originali nell'output_dir
    - genera (target_factor - 1) × N immagini augmentate
    Restituisce statistiche.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = load_images(input_dir)

    if not images:
        print(f"  [SKIP] Nessuna immagine trovata in {input_dir}")
        return {"original": 0, "augmented": 0, "total": 0}

    n_original  = len(images)
    n_to_gen    = n_original * (target_factor - 1)
    aug_count   = 0

    # ── 1. Copia originali ────────────────────────────────────────
    for stem, img in images:
        out_path = output_dir / f"{stem}_orig.png"
        cv2.imwrite(str(out_path), img)

    print(f"  Originali copiati : {n_original}")

    # ── 2. Genera augmentati ──────────────────────────────────────
    # Cicla sugli originali in modo uniforme per non favorirne alcuni
    for i in range(n_to_gen):
        stem, img = images[i % n_original]
        aug_img   = apply_random_augmentation(img)
        ts        = int(datetime.now().timestamp() * 1000) + i
        out_path  = output_dir / f"{stem}_aug_{i:04d}_{ts}.png"
        cv2.imwrite(str(out_path), aug_img)
        aug_count += 1

        if (i + 1) % 50 == 0:
            print(f"    ...generati {i + 1}/{n_to_gen}")

    total = n_original + aug_count
    print(f"  Augmentati generati: {aug_count}")
    print(f"  Totale classe      : {total}")

    return {
        "original":  n_original,
        "augmented": aug_count,
        "total":     total
    }


# ════════════════════════════════════════════════════════════════════════════
# VERIFICA VISIVA: genera una preview grid delle augmentation
# ════════════════════════════════════════════════════════════════════════════

def generate_preview(input_dir: Path,
                     output_path: Path,
                     class_name: str,
                     n_rows: int = 4,
                     n_cols: int = 8) -> None:
    """
    Genera un'immagine griglia che mostra originale + sue augmentazioni.
    Utile per ispezionare visivamente che i colori siano preservati.
    """
    images = load_images(input_dir)
    if not images:
        return

    # Prendi il primo originale e genera n_rows*n_cols-1 versioni
    _, base_img = images[0]
    cells       = [base_img]  # prima cella = originale

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

    # Label "ORIG" sulla prima cella
    cv2.putText(grid, "ORIG", (padding + 2, padding + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

    # Titolo
    cv2.putText(grid, f"Preview augmentation: {class_name}",
                (padding, grid_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imwrite(str(output_path), grid)
    print(f"  Preview salvata: {output_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Augmentation dataset LED Classifier (color-safe)")
    parser.add_argument("--input",   default="dataset/raw",
                        help="Directory radice con le classi originali")
    parser.add_argument("--output",  default="dataset/augmented",
                        help="Directory di output per il dataset aumentato")
    parser.add_argument("--factor",  type=int, default=DEFAULT_FACTOR,
                        help="Moltiplicatore totale (default 4 = 400 img da 100)")
    parser.add_argument("--preview", action="store_true",
                        help="Genera preview griglia per ogni classe")
    args = parser.parse_args()

    input_root  = Path(args.input)
    output_root = Path(args.output)
    factor      = args.factor

    print("=" * 60)
    print(" LED Classifier — Dataset Augmentation")
    print("=" * 60)
    print(f" Input  : {input_root.resolve()}")
    print(f" Output : {output_root.resolve()}")
    print(f" Factor : {factor}x  (es. 100 orig → {100 * factor} totali)")
    print("=" * 60)

    all_stats   = {}
    grand_total = 0

    for cls in CLASSES:
        in_dir  = input_root  / cls
        out_dir = output_root / cls

        if not in_dir.exists():
            print(f"\n[SKIP] {cls}: directory non trovata ({in_dir})")
            continue

        print(f"\n▶ Classe: {cls.upper()}")
        stats = augment_class(cls, in_dir, out_dir, factor)
        all_stats[cls] = stats
        grand_total   += stats["total"]

        if args.preview:
            preview_path = output_root / f"_preview_{cls}.png"
            generate_preview(in_dir, preview_path, cls)

    # ── Salva statistiche JSON ────────────────────────────────────
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

    # ── Riepilogo finale ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" RIEPILOGO")
    print("=" * 60)
    for cls, s in all_stats.items():
        bar = "█" * int(s["total"] / 20)
        print(f"  {cls:<10} orig={s['original']:>4}  "
              f"aug={s['augmented']:>4}  tot={s['total']:>4}  {bar}")
    print(f"\n  Immagini totali nel dataset: {grand_total}")
    print(f"  Statistiche salvate in     : {stats_path}")
    print("=" * 60)
    print("\n✅ Dataset pronto per il training su Google Colab!")
    print("   Prossimo step: comprimi 'dataset/augmented/' e caricalo su Drive.")


if __name__ == "__main__":
    main()