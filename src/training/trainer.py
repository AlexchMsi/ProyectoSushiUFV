"""
Pipeline de entrenamiento YOLO para detección de sushi.
Fase 1: entrenamiento base sobre etiquetas manuales.
Fase 2: fine-tuning sobre dataset completo (dataset_v2).
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent.parent


def train_base(data_yaml: Path, output_dir: Path, **overrides) -> Path:
    """
    Fase 1: entrena YOLO11s desde cero sobre las 440 etiquetas manuales.

    Returns:
        Ruta al mejor checkpoint.
    """
    params = dict(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        amp=True,
        cache=True,
        workers=0,
        patience=20,
        degrees=10.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        project=str(output_dir),
        name="sushi_base_v1",
    )
    params.update(overrides)

    model = YOLO("yolo11m.pt")
    model.train(**params)

    best = output_dir / "sushi_base_v1" / "weights" / "best.pt"
    print(f"\nFase 1 completada. Mejor modelo: {best}")
    return best


def retrain(data_yaml: Path, base_model: Path, output_dir: Path, **overrides) -> Path:
    """
    Fase 2: fine-tuning desde checkpoint base sobre el dataset completo.

    Returns:
        Ruta al mejor checkpoint.
    """
    # Nombre único basado en el tamaño del modelo base para evitar colisiones
    # con experimentos anteriores (yolo11s → sushi_v2, yolo11m → sushi_v2m)
    base_size = base_model.stat().st_size if base_model.exists() else 0
    exp_name = "sushi_v2m" if base_size > 30_000_000 else "sushi_v2"

    params = dict(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        amp=True,
        cache=False,
        workers=0,
        patience=25,
        cos_lr=True,
        degrees=10.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        project=str(output_dir),
        name=exp_name,
    )
    params.update(overrides)

    model = YOLO(str(base_model))
    model.train(**params)

    best = output_dir / exp_name / "weights" / "best.pt"
    print(f"\nFase 2 completada. Mejor modelo: {best}")
    return best
