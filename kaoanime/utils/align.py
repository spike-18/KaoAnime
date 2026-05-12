"""Face alignment: MediaPipe FaceLandmarker (Tasks API) + OpenCV similarity transform."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# 5-point canonical landmarks at 128×128 (ArcFace standard scaled from 112px).
# Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
_CANONICAL_128 = np.array(
    [
        [43.7,  59.1],
        [84.0,  58.9],
        [64.0,  81.6],
        [47.5, 105.6],
        [80.8, 105.4],
    ],
    dtype=np.float32,
)

# MediaPipe Face Mesh landmark indices for each key point group.
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
_NOSE_IDX      = [4]
_L_MOUTH_IDX   = [61]
_R_MOUTH_IDX   = [291]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "kaoanime" / "face_landmarker.task"


def _ensure_model() -> str:
    if not _MODEL_CACHE.exists():
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
    return str(_MODEL_CACHE)


def _src_pts(landmarks: list, h: int, w: int) -> np.ndarray:
    def mean_xy(idxs: list[int]) -> list[float]:
        return [
            float(np.mean([landmarks[i].x * w for i in idxs])),
            float(np.mean([landmarks[i].y * h for i in idxs])),
        ]

    return np.array(
        [
            mean_xy(_LEFT_EYE_IDX),
            mean_xy(_RIGHT_EYE_IDX),
            mean_xy(_NOSE_IDX),
            mean_xy(_L_MOUTH_IDX),
            mean_xy(_R_MOUTH_IDX),
        ],
        dtype=np.float32,
    )


def _canonical(size: int) -> np.ndarray:
    return _CANONICAL_128 * (size / 128.0)


class AlignFaceProcessor:
    """Holds one MediaPipe FaceLandmarker instance to avoid reloading per image.

    Create one instance per process (or per inference session). Not thread-safe —
    use one per thread or per process.
    """

    def __init__(self) -> None:
        model_path = _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def align(self, image: np.ndarray, size: int = 128) -> np.ndarray | None:
        """Detect the largest face and warp it to canonical ArcFace alignment.

        Args:
            image: RGB uint8 array, any resolution.
            size:  Output square side in pixels.

        Returns:
            Aligned RGB image of shape ``(size, size, 3)`` uint8,
            or ``None`` if no face was detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        h, w = image.shape[:2]
        pts_src = _src_pts(result.face_landmarks[0], h, w)
        pts_dst = _canonical(size)

        M, _ = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.LMEDS)
        if M is None:
            return None

        return cv2.warpAffine(
            image, M, (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "AlignFaceProcessor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def align_face(image: np.ndarray, size: int = 128) -> np.ndarray | None:
    """Single-image convenience wrapper around :class:`AlignFaceProcessor`.

    Creates a fresh processor per call — fine for inference on individual images.
    For batch processing prefer :class:`AlignFaceProcessor` to load the model
    once per worker.

    Args:
        image: RGB uint8 array, any resolution.
        size:  Output square side in pixels.

    Returns:
        Aligned RGB image ``(size, size, 3)`` uint8, or ``None``.
    """
    with AlignFaceProcessor() as proc:
        return proc.align(image, size)
