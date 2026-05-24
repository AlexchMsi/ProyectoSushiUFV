"""
scripts/train.py - Pipeline completo de entrenamiento.

Uso:
    python scripts/train.py --phase all    # fase 1 + autolabel + fase 2
    python scripts/train.py --phase 1      # solo entrenamiento base
    python scripts/train.py --phase 2      # solo fine-tuning (requiere fase 1 previa)
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(ROOT))

from src.datasets.sushi_dataset import build_dataset
from src.training.trainer import train_base, retrain
from src.utils.logger import get_logger

OUTPUTS_DIR  = ROOT / "outputs"
DATA_YAML    = ROOT / "dataset_v2" / "data.yaml"
BASE_DATA    = ROOT / "dataset_base" / "data.yaml"
# Busca el modelo base: outputs/sushi_base_v1 → outputs/sushi_base_v12 (nombre
# auto-incrementado por YOLO si la carpeta ya existía) → runs/ (entrenamiento antiguo)
BASE_MODEL   = next(
    (p for p in [
        OUTPUTS_DIR / "sushi_base_v1"  / "weights" / "best.pt",
        OUTPUTS_DIR / "sushi_base_v12" / "weights" / "best.pt",
        ROOT / "runs" / "sushi_base_v1" / "weights" / "best.pt",
    ] if p.exists()),
    OUTPUTS_DIR / "sushi_base_v1" / "weights" / "best.pt",
)
FINAL_MODEL  = ROOT / "models" / "best.pt"


def phase1():
    logger = get_logger("train.phase1")
    if not BASE_DATA.exists():
        logger.error(f"dataset_base no encontrado en {BASE_DATA}")
        return None
    logger.info("=== FASE 1: Entrenamiento base ===")
    best = train_base(BASE_DATA, OUTPUTS_DIR)
    return best


def phase2():
    logger = get_logger("train.phase2")
    if not BASE_MODEL.exists():
        logger.error(f"Modelo base no encontrado. Ejecuta primero la fase 1.")
        return None
    if not DATA_YAML.exists():
        logger.info("dataset_v2 no encontrado — construyendo con auto-etiquetado...")
        build_dataset(BASE_MODEL, conf_thresh=0.65)
    logger.info("=== FASE 2: Fine-tuning ===")
    best = retrain(DATA_YAML, BASE_MODEL, OUTPUTS_DIR)
    # Copiar al directorio de modelos para Docker
    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, FINAL_MODEL)
    logger.info(f"Modelo final copiado a: {FINAL_MODEL}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento Sushi YOLO")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        phase1()

    if args.phase in ("2", "all"):
        phase2()


if __name__ == "__main__":
    main()
