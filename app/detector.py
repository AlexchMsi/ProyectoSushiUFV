import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

_APP_DIR = Path(__file__).parent
_RUNS    = _APP_DIR.parent / "runs"

# Prioridad: MODEL_PATH env (Docker) → models/best.pt → runs/sushi_v2 → runs/sushi_base_v1 → pretrained
_ENV_MODEL     = Path(os.environ["MODEL_PATH"]) if "MODEL_PATH" in os.environ else None
CUSTOM_MODEL   = _APP_DIR / "models" / "best.pt"
RUNS_V2        = _RUNS / "sushi_v2"     / "weights" / "best.pt"
RUNS_V1        = _RUNS / "sushi_base_v1"/ "weights" / "best.pt"
FALLBACK_MODEL = "yolo11n.pt"

_model: YOLO | None = None


def load_model() -> None:
    global _model
    candidates = [p for p in [_ENV_MODEL, CUSTOM_MODEL, RUNS_V2, RUNS_V1] if p and p.exists()]
    if candidates:
        chosen = candidates[0]
        print(f"[detector] Cargando modelo: {chosen}")
        _model = YOLO(str(chosen))
    else:
        print(f"[detector] Sin modelo personalizado, usando preentrenado: {FALLBACK_MODEL}")
        _model = YOLO(FALLBACK_MODEL)


def get_model() -> YOLO:
    if _model is None:
        load_model()
    return _model


def predict(frame_bytes: bytes, conf_threshold: float = 0.25) -> dict:
    """
    Run YOLO inference on a JPEG frame.

    Args:
        frame_bytes: Raw JPEG bytes from the client canvas.
        conf_threshold: Minimum confidence to include a detection.

    Returns:
        {
            "detections": [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}],
            "counts": {"class_name": int, ...},
            "total": int
        }
    """
    model = get_model()

    # Decode JPEG bytes to numpy BGR array
    np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detections": [], "counts": {}, "total": 0}

    # CLAHE en canal V (HSV): normaliza brillo para mejorar detección
    # con iluminación irregular (técnica de preprocesado clásico aplicada)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    results = model(frame, conf=conf_threshold, iou=0.3, verbose=False)[0]

    detections = []
    counts: dict[str, int] = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        detections.append({
            "class": cls_name,
            "confidence": round(confidence, 3),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
        })
        counts[cls_name] = counts.get(cls_name, 0) + 1

    return {
        "detections": detections,
        "counts": counts,
        "total": len(detections),
    }
