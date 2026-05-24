"""
scripts/augment_backgrounds.py

Genera variantes de imágenes de entrenamiento para mejorar detección
en fondos claros/blancos (principal punto ciego del modelo actual).

Dos estrategias por imagen:
  1. Bright   — gamma alta (simula iluminación de estudio / sobreexposición)
  2. Crop     — recorte por objeto sobre fondo blanco, con padding del 25%

Solo modifica el split de entrenamiento (val queda intacto para evaluación honesta).
Se puede ejecutar sobre un dataset_v2 ya existente sin reconstruirlo.

Uso:
    python scripts/augment_backgrounds.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT      = Path(__file__).parent.parent
DATASET   = ROOT / "dataset_v2"
IMG_EXTS  = {".jpg", ".jpeg", ".png"}
CROP_PAD  = 0.25   # padding alrededor del bbox al recortar
GAMMA_VAL = 2.8    # gamma para la variante bright (>1 → más claro)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _read_labels(lbl_path: Path) -> list:
    labels = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels


def _write_labels(lbl_path: Path, labels: list):
    lbl_path.write_text(
        "\n".join(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}"
                  for l in labels)
    )


# ── Augmentación bright ───────────────────────────────────────────────────────

def add_bright_variants(img_dir: Path, lbl_dir: Path) -> int:
    """
    Para cada imagen de entrenamiento crea una versión muy iluminada.
    Las labels son idénticas (no cambia la posición de los objetos).
    """
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
              and not p.stem.startswith("bright__")]
    added = 0
    for img_path in tqdm(images, desc="Bright variants", unit="img"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        bright = _apply_gamma(img, GAMMA_VAL)
        stem   = f"bright__{img_path.stem}"
        cv2.imwrite(str(img_dir / (stem + ".jpg")), bright,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Label idéntica, solo cambia el nombre de fichero
        import shutil
        shutil.copy2(lbl_path, lbl_dir / (stem + ".txt"))
        added += 1

    return added


# ── Augmentación crop+fondo blanco ───────────────────────────────────────────

def add_crop_variants(img_dir: Path, lbl_dir: Path) -> int:
    """
    Para cada objeto etiquetado: recorta la región (con padding) y la pega
    en el centro de un canvas blanco del mismo tamaño.
    Genera una imagen por objeto con su label recalculada.
    """
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
              and not p.stem.startswith(("bright__", "whitebg__"))]
    added = 0
    for img_path in tqdm(images, desc="Crop+white variants", unit="img"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        labels = _read_labels(lbl_path)
        if not labels:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for i, lbl in enumerate(labels):
            cls_id, xc, yc, bw, bh = lbl

            # Coordenadas del recorte con padding
            pad = CROP_PAD
            x1 = max(0, int((xc - bw / 2 - pad * bw) * w))
            y1 = max(0, int((yc - bh / 2 - pad * bh) * h))
            x2 = min(w, int((xc + bw / 2 + pad * bw) * w))
            y2 = min(h, int((yc + bh / 2 + pad * bh) * h))

            crop = img[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if ch < 20 or cw < 20:
                continue

            # Canvas blanco del mismo tamaño que el recorte
            canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)

            # Pegar el recorte; el fondo blanco queda en las esquinas
            # donde el recorte tiene bordes parciales
            canvas[:ch, :cw] = crop

            # Bbox relativo al canvas (solo este objeto)
            new_xc = (xc * w - x1) / cw
            new_yc = (yc * h - y1) / ch
            new_bw = bw * w / cw
            new_bh = bh * h / ch

            # Clamp por seguridad
            new_xc = float(np.clip(new_xc, 0.05, 0.95))
            new_yc = float(np.clip(new_yc, 0.05, 0.95))
            new_bw = float(np.clip(new_bw, 0.05, 0.95))
            new_bh = float(np.clip(new_bh, 0.05, 0.95))

            stem = f"whitebg__{img_path.stem}__obj{i}"
            cv2.imwrite(str(img_dir / (stem + ".jpg")), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            _write_labels(lbl_dir / (stem + ".txt"),
                          [[cls_id, new_xc, new_yc, new_bw, new_bh]])
            added += 1

    return added


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    img_dir = DATASET / "images" / "train"
    lbl_dir = DATASET / "labels" / "train"

    if not img_dir.exists():
        print(f"ERROR: No se encontró dataset_v2 en {DATASET}")
        print("Ejecuta primero: python scripts/train.py --phase all")
        return

    print(f"Dataset: {DATASET}")
    existing = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Imágenes de entrenamiento actuales: {existing}")

    print("\n--- Variantes bright (iluminación de estudio) ---")
    n_bright = add_bright_variants(img_dir, lbl_dir)

    print("\n--- Variantes crop + fondo blanco ---")
    n_crop = add_crop_variants(img_dir, lbl_dir)

    total = existing + n_bright + n_crop
    print(f"\nAugmentación completada:")
    print(f"  Variantes bright : {n_bright}")
    print(f"  Variantes crop   : {n_crop}")
    print(f"  Total imágenes   : {total}")


if __name__ == "__main__":
    main()
