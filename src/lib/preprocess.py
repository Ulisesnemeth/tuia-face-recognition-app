"""Pipeline compartido de detección, alineación y normalización facial.

Este módulo es la **única** fuente de verdad del preprocesamiento. Lo usan:
  - la notebook 1 (EDA y curado del dataset),
  - la notebook 2 (entrenamiento y evaluación),
  - el backend (src/lib/services/face_service.py).

Convenciones (no cambiar sin cambiar todo lo demás):
  - Las imágenes que entran y salen están en **BGR uint8** (convención OpenCV).
  - El modelo de reconocimiento recibe **RGB**, escala (x - 127.5) / 127.5,
    en formato NCHW con dimensión de batch.
  - El embedding final se normaliza L2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Plantilla canónica de 5 puntos usada por ArcFace para 112x112.
# Orden: ojo izquierdo, ojo derecho, nariz, comisura izquierda, comisura derecha.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    """Un rostro detectado en la imagen original."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) en píxeles
    score: float
    kps: np.ndarray | None = None  # (5, 2) en coordenadas absolutas


def clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, height: int, width: int
) -> tuple[int, int, int, int]:
    """Recorta una caja a los límites de la imagen, garantizando área > 0."""
    x1i = int(max(0, min(round(x1), width - 1)))
    y1i = int(max(0, min(round(y1), height - 1)))
    x2i = int(max(0, min(round(x2), width)))
    y2i = int(max(0, min(round(y2), height)))
    if x2i <= x1i:
        x2i = min(x1i + 1, width)
    if y2i <= y1i:
        y2i = min(y1i + 1, height)
    return x1i, y1i, x2i, y2i


def iou(a: np.ndarray | tuple, b: np.ndarray | tuple) -> float:
    """Intersection over Union entre dos cajas (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


class FaceDetector:
    """Detector SCRFD de InsightFace cargado desde un archivo .onnx.

    Se instancia **una sola vez** y se reutiliza: cargar un modelo por
    request es el error de performance más común.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        score_threshold: float = 0.5,
        min_face_size: int = 20,
        ctx_id: int = -1,  # -1 = CPU
    ) -> None:
        # Import local: insightface es una dependencia pesada y este módulo
        # también se importa desde contextos que solo necesitan las constantes.
        from insightface.model_zoo import get_model

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Detector model not found: {model_path}")

        self.input_size = (int(input_size), int(input_size))
        self.score_threshold = float(score_threshold)
        self.min_face_size = int(min_face_size)
        self.model = get_model(str(model_path))
        self.model.prepare(
            ctx_id=ctx_id,
            input_size=self.input_size,
            det_thresh=self.score_threshold,
        )

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Devuelve los rostros aceptados, con caja recortada y landmarks."""
        height, width = image_bgr.shape[:2]
        bboxes, kpss = self.model.detect(
            image_bgr, input_size=self.input_size, max_num=0
        )
        if bboxes is None or len(bboxes) == 0:
            return []

        results: list[DetectedFace] = []
        for i, raw in enumerate(bboxes):
            score = float(raw[4]) if len(raw) > 4 else 1.0
            if score < self.score_threshold:
                continue
            box = clip_xyxy(raw[0], raw[1], raw[2], raw[3], height, width)
            if min(box[2] - box[0], box[3] - box[1]) < self.min_face_size:
                continue
            kps = None
            if kpss is not None and len(kpss) > i:
                kps = np.asarray(kpss[i], dtype=np.float32)
            results.append(DetectedFace(bbox=box, score=score, kps=kps))
        return results


def align_face(
    image_bgr: np.ndarray, kps: np.ndarray, image_size: int = 112
) -> np.ndarray:
    """Alinea un rostro con una transformación de similitud hacia la plantilla.

    Usa los 5 landmarks (ojos, nariz, comisuras) para que la cara quede
    centrada, escalada y sin rotación. Devuelve BGR uint8 de image_size x image_size.
    """
    from insightface.utils import face_align

    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape[0] < 5:
        raise ValueError(f"Se esperaban 5 landmarks, llegaron {kps.shape[0]}")
    # insightface 0.7.x: norm_crop(img, landmark, image_size, mode)
    return face_align.norm_crop(
        image_bgr, kps[:5], image_size=int(image_size)
    )


def crop_without_alignment(
    image_bgr: np.ndarray, box: tuple[int, int, int, int], image_size: int = 112
) -> np.ndarray:
    """Recorte de respaldo cuando no hay landmarks disponibles.

    Solo para casos borde: sin alineación el embedding pierde calidad.
    """
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for box {box}")
    return cv2.resize(crop, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)


def to_tensor(
    face_bgr: np.ndarray, image_size: int = 112, mean: float = 127.5, std: float = 127.5
) -> np.ndarray:
    """BGR uint8 -> RGB float32 NCHW normalizado, listo para la red.

    Este es exactamente el preprocesamiento de ArcFace en InsightFace
    ((x - 127.5) / 127.5 sobre RGB). Si se cambia el modelo, se cambia acá
    y en ningún otro lado.
    """
    if face_bgr.shape[0] != image_size or face_bgr.shape[1] != image_size:
        face_bgr = cv2.resize(
            face_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normaliza un vector para que la similitud coseno dependa solo del ángulo."""
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))
