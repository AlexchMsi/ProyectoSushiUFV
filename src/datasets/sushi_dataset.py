"""
Construcción del dataset YOLO para detección de sushi.
Combina etiquetas manuales con auto-etiquetado del modelo base.
"""

import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# ── Filtro de calidad de imagen ───────────────────────────────────────────────
# Usa la varianza del Laplacian como estimador de nitidez.
# Imágenes con varianza < BLUR_THRESHOLD se descartan del auto-etiquetado.
BLUR_THRESHOLD = 80.0


def _is_sharp(img_path: Path) -> bool:
    """Devuelve True si la imagen tiene nitidez suficiente (no está borrosa)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance >= BLUR_THRESHOLD

ROOT        = Path(__file__).parent.parent.parent
DATASET_SRC = ROOT / "Dataset" / "SushiType"
MANUAL_BASE = ROOT / "dataset_base"
OUT_DIR     = ROOT / "dataset_v2"

NAME_MAP = {
    "ball":         "ball",
    "onigiri":      "onigiri",
    "sashimi":      "sashimi",
    "sushi-gunkan": "gunkan",
    "sushi-maki":   "maki",
    "sushi-nigiri": "nigiri",
    "sushi-roll":   "roll",
    "wrap":         "wrap",
}
EXCLUDE  = {"sushi-other", "sushi-mix", "sushi-inari"}
CLASSES  = ["ball", "onigiri", "sashimi", "gunkan", "maki", "nigiri", "roll", "wrap"]
CLS2ID   = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_manual_samples() -> list:
    img_root = MANUAL_BASE / "images" / "train"
    lbl_root = MANUAL_BASE / "labels" / "train"
    samples  = []
    for cls_dir in img_root.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = NAME_MAP.get(cls_dir.name, cls_dir.name)
        for img in cls_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_root / cls_dir.name / img.with_suffix(".txt").name
            if lbl.exists():
                samples.append((img, lbl, cls_name, f"{cls_name}__{img.stem}"))
    return samples


def collect_unlabeled_images() -> list:
    samples = []
    for cls_dir in DATASET_SRC.iterdir():
        if not cls_dir.is_dir() or cls_dir.name in EXCLUDE:
            continue
        cls_name = NAME_MAP.get(cls_dir.name)
        if cls_name is None:
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                samples.append((img, cls_name))
    return samples


def run_auto_label(model: YOLO, samples: list, tmp_lbl_dir: Path,
                   conf_thresh: float = 0.65) -> list:
    tmp_lbl_dir.mkdir(parents=True, exist_ok=True)
    accepted, skipped = [], 0

    for img_path, cls_name in tqdm(samples, desc="Auto-etiquetando", unit="img"):
        # Descartar imágenes borrosas antes de inferir
        if not _is_sharp(img_path):
            skipped += 1
            continue
        results = model(str(img_path), conf=conf_thresh, verbose=False)[0]
        if len(results.boxes) == 0:
            skipped += 1
            continue

        unique_stem = f"{cls_name}__{img_path.stem}"
        lbl_path    = tmp_lbl_dir / (unique_stem + ".txt")
        h, w        = results.orig_shape
        lines = []
        for box in results.boxes:
            cid = int(box.cls[0])
            xc  = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2 / w)
            yc  = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2 / h)
            bw  = float((box.xyxy[0][2] - box.xyxy[0][0]) / w)
            bh  = float((box.xyxy[0][3] - box.xyxy[0][1]) / h)
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        lbl_path.write_text("\n".join(lines))
        accepted.append((img_path, lbl_path, cls_name, unique_stem))

    print(f"Auto-etiquetado: {len(accepted)} aceptadas, {skipped} omitidas")
    return accepted


def write_split(samples: list, out_dir: Path, val_split: float = 0.20, seed: int = 42):
    random.seed(seed)
    random.shuffle(samples)
    n_val   = int(len(samples) * val_split)
    val_set = samples[:n_val]
    trn_set = samples[n_val:]

    for split_name, split in [("train", trn_set), ("val", val_set)]:
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path, _, unique_stem in tqdm(split, desc=f"Copiando {split_name}"):
            shutil.copy2(img_path, img_out / (unique_stem + img_path.suffix))
            shutil.copy2(lbl_path, lbl_out / (unique_stem + ".txt"))

    print(f"Split: {len(trn_set)} train / {len(val_set)} val")
    return len(trn_set), len(val_set)


def write_yaml(out_dir: Path) -> Path:
    cfg = {
        "path":  str(out_dir),
        "train": "images/train",
        "val":   "images/val",
        "names": {i: c for i, c in enumerate(CLASSES)},
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def build_dataset(model_path: Path, conf_thresh: float = 0.65,
                  val_split: float = 0.20) -> Path:
    """Construye dataset_v2 combinando etiquetas manuales + auto-etiquetado."""
    print("=" * 60)
    print(f"Cargando modelo base: {model_path}")
    model = YOLO(str(model_path))

    print("\nRecopilando muestras manuales...")
    manual = collect_manual_samples()
    print(f"  Etiquetas manuales: {len(manual)} imágenes")

    print("\nRecopilando imágenes sin etiquetar...")
    unlabeled = collect_unlabeled_images()
    manual_stems = {p.stem for p, _, _, _ in manual}
    unlabeled = [(p, c) for p, c in unlabeled if p.stem not in manual_stems]
    print(f"  A auto-etiquetar: {len(unlabeled)} imágenes")

    print(f"\nAuto-etiquetando (conf >= {conf_thresh})...")
    tmp_lbl = OUT_DIR.parent / "_tmp_autolabels"
    auto = run_auto_label(model, unlabeled, tmp_lbl, conf_thresh)

    all_samples = manual + auto
    print(f"\nTotal: {len(all_samples)} ({len(manual)} manuales + {len(auto)} automáticas)")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    write_split(all_samples, OUT_DIR, val_split)
    yaml_path = write_yaml(OUT_DIR)
    shutil.rmtree(tmp_lbl, ignore_errors=True)

    print("=" * 60)
    print(f"Dataset listo: {yaml_path}")
    return yaml_path
