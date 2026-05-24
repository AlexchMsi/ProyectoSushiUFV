"""
Utilidades de visualización: dibujo de bboxes sobre imágenes OpenCV.
"""

import cv2
import numpy as np

# Paleta de colores BGR por clase (determinista)
_PALETTE: dict[str, tuple[int, int, int]] = {}


def _color_for_class(cls: str) -> tuple[int, int, int]:
    if cls not in _PALETTE:
        h = abs(hash(cls)) % 180
        hsv = np.array([[[h, 220, 200]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        _PALETTE[cls] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return _PALETTE[cls]


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Dibuja bboxes y etiquetas sobre un frame BGR.

    Args:
        frame:      imagen BGR (numpy array)
        detections: lista de {class, confidence, bbox:[x1,y1,x2,y2]}

    Returns:
        Copia del frame con anotaciones.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = _color_for_class(det["class"])
        label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 6, y1 + th + 6), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
