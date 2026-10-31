from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
from lib.schemas import EmbeddingRecord, FaceDetection, PredictResult, AlignedFace
from lib.storage.base import EmbeddingStoreProtocol
from lib.preprocess import (
    FaceDetector,
    align_face as align_face_with_kps,
    crop_without_alignment,
    iou as box_iou,
    l2_normalize,
    to_tensor,
)
import os 
import logging

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        face_size: int,
        model_path: Path,
        output_path: Path = Path("output"),
        detector_path: Path | None = None,
        det_score_threshold: float = 0.5,
        det_input_size: int = 640,
        min_face_size: int = 20,
        align_margin: float = 0.5,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        self.model: any = self._load_model(model_path)
        self.output_path = output_path
        self.det_score_threshold = det_score_threshold
        self.det_input_size = det_input_size
        self.min_face_size = min_face_size
        self.align_margin = align_margin
        self.detector: any = self._load_detector(detector_path)

        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    @staticmethod
    def _kps_to_keypoints_dict(kps: np.ndarray | None) -> dict[str, list[int]]:
        if kps is None or len(kps) == 0:
            return {}
        return {
            f"k{i}": [int(round(float(kps[i, 0]))), int(round(float(kps[i, 1])))]
            for i in range(len(kps))
        }


    def _load_model(self, model_path: Path) -> any:
        mp = Path(model_path)
        if not mp.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        suf = mp.suffix.lower()
        if suf == ".pth":
            return torch.load(mp, map_location="cpu", weights_only=False)
        if suf == ".onnx":
            return onnxruntime.InferenceSession(str(mp))
        raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

    def _load_detector(self, detector_path: Path | None) -> any:
        """
        Load the face detector once (called from __init__).

        For the InsightFace SCRFD detector we do not use a raw onnxruntime
        session: `insightface.model_zoo.get_model` builds the model with the
        anchors/strides logic that turns raw outputs into boxes, scores and the
        5 landmarks. That is why this override exists and why
        DETECTOR_MODEL_NAME must point to det_10g.onnx (buffalo_l pack).
        """
        if detector_path is None:
            logger.warning(
                "DETECTOR_MODEL_NAME is not set: no detector will be available."
            )
            return None
        return FaceDetector(
            model_path=detector_path,
            input_size=self.det_input_size,
            score_threshold=self.det_score_threshold,
            min_face_size=self.min_face_size,
        )

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image
    
    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """Intersection over union between two boxes (x1, y1, x2, y2)."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return float(inter / union) if union > 0 else 0.0
    
    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Each box is (x1, y1, x2, y2) in pixels (InsightFace convention).
        Return a list of tuples with the coordinates of the faces detected in the image.
        """
        if self.detector is None:
            raise RuntimeError(
                "Detector not configured: set DETECTOR_MODEL_NAME to det_10g.onnx"
            )
        # FaceDetector already applies score threshold, minimum size and clipping.
        return [face.bbox for face in self.detector.detect(image)]

    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> AlignedFace:
        """
        Crop using box (x1, y1, x2, y2) and run FaceAnalysis on the crop.
        Return an AlignedFace object.
        """
        if self.detector is None:
            raise RuntimeError(
                "Detector not configured: set DETECTOR_MODEL_NAME to det_10g.onnx"
            )

        height, width = image.shape[:2]
        x1, y1, x2, y2 = box

        # Re-detect to get landmarks and match the requested box by IoU.
        faces = self.detector.detect(image)
        target = max(faces, key=lambda f: box_iou(f.bbox, box), default=None)
        kps = target.kps if target is not None else None
        if target is None or kps is None or box_iou(target.bbox, box) < 0.3:
            # Fallback: look for landmarks on a padded crop (small or partially
            # occluded faces are sometimes missed in the full image).
            margin = int(max(x2 - x1, y2 - y1) * self.align_margin)
            cx1, cy1, cx2, cy2 = self._clip_xyxy(
                x1 - margin, y1 - margin, x2 + margin, y2 + margin, height, width
            )
            crop = image[cy1:cy2, cx1:cx2]
            crop_faces = self.detector.detect(crop) if crop.size > 0 else []
            if crop_faces:
                best = max(crop_faces, key=lambda f: f.score)
                kps = best.kps + np.array([cx1, cy1], dtype=np.float32)

        if kps is not None:
            aligned = align_face_with_kps(image, kps, self.face_size)
            # Keypoints must be relative to the top-left corner of the box:
            # the frontend adds (x1, y1) back when drawing them.
            relative_kps = (
                np.asarray(kps, dtype=np.float32)
                - np.array([x1, y1], dtype=np.float32)
            )
        else:
            logger.warning("No landmarks for box %s: falling back to plain crop", box)
            aligned = crop_without_alignment(image, box, self.face_size)
            relative_kps = None

        return AlignedFace(
            bbox=list(box),
            keypoints=relative_kps,
            image=aligned,
            embedding=None,
        )

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extract embedding from face.
        Return a list of floats representing the embedding of the face.
        """
        # If the pipeline already computed it (e.g. a custom model), reuse it.
        if getattr(face, "embedding", None) is not None:
            embedding = np.asarray(face.embedding, dtype=np.float32)
        elif isinstance(self.model, onnxruntime.InferenceSession):
            tensor = to_tensor(face.image, self.face_size)
            input_name = self.model.get_inputs()[0].name
            embedding = self.model.run(None, {input_name: tensor})[0]
        else:  # torch nn.Module
            tensor = torch.from_numpy(to_tensor(face.image, self.face_size))
            self.model.eval()
            with torch.no_grad():
                output = self.model(tensor)
            embedding = output.detach().cpu().numpy()

        embedding = l2_normalize(np.asarray(embedding, dtype=np.float32).ravel())
        return embedding.astype(float).tolist()
        
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    def identify(self, query_embedding: list[float]) -> tuple[str, float]:
        records = self.store.all()
        if not records:
            return "unknown", 0.0

        best_label = "unknown"
        best_score = -1.0
        for record in records:
            score = self.similarity(query_embedding, record.embedding)
            if score > best_score:
                best_score = score
                best_label = record.etiqueta

        if best_score < self.similarity_threshold:
            return "unknown", max(best_score, 0.0)
        return best_label, best_score

    def register_identity(
        self, identity: str, image_path: str, metadata: dict[str, object]
    ) -> EmbeddingRecord:
        image = self._load_image(image_path)
        faces = self.detect_faces(image)

        if len(faces) != 1:
            raise ValueError("Exactly one face must be detected for identity registration.")
        
        logger.info(f"Face detected: {faces[0]}")

        box = faces[0]
        aligned = self.align_face(image, box)
        embedding = self.extract_embedding_from_face(aligned)

        img_id = str(uuid4())
        img_output_path = self.output_path / f"img_{img_id}.jpg"
        
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(img_output_path),
            etiqueta=identity,
            metadata=metadata,
        )
        self.store.append(record)

        cv2.imwrite(str(img_output_path), aligned.image)
        logger.info(f"Identity registered: {identity} with image: {image_path}")
        return record

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        faces = self.detect_faces(image)
        detections: list[FaceDetection] = []
        for (x1, y1, x2, y2) in faces:
            aligned = self.align_face(image, (x1, y1, x2, y2))
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            kps = getattr(aligned, "keypoints", None)
            kps_arr = np.asarray(kps) if kps is not None else None
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=self._kps_to_keypoints_dict(kps_arr),
                    label=label,
                    score=round(float(score), 4),
                )
            )

        detected_people = sorted({item.label for item in detections if item.label != "unknown"})
        result_payload = PredictResult(
            source_path=source_path,
            detections=detections,
            detected_people=detected_people,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(result_payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
