"""Detector interface, a classical colour-blob detector and an optional YOLO adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Axis-aligned detection in pixel coordinates (x1, y1, x2, y2)."""

    box: tuple[float, float, float, float]
    score: float
    label: str = "object"

    @property
    def xyxy(self) -> np.ndarray:
        return np.asarray(self.box, dtype=np.float64)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground contact point used for flat-ground range estimation."""
        return 0.5 * (self.box[0] + self.box[2]), self.box[3]


class Detector(Protocol):
    """Anything that turns a BGR image into a list of detections."""

    def detect(self, image: np.ndarray) -> list[Detection]: ...


@dataclass
class HueClass:
    """A named object class defined by an OpenCV hue interval (0..179)."""

    label: str
    hue_lo: int
    hue_hi: int


DEFAULT_HUE_CLASSES = (
    HueClass("red", 170, 10),
    HueClass("yellow", 22, 36),
    HueClass("blue", 105, 128),
    HueClass("magenta", 140, 162),
)


class ColorBlobDetector:
    """Detects saturated colour blobs with HSV thresholding and connected components.

    This is a deliberately simple classical detector for the synthetic scenes. The
    confidence combines how well the blob fills its bounding box (occluded or merged
    blobs fill it poorly) with a size term (small, distant blobs are less certain), so
    the tracker receives a realistic mix of high and low confidence detections.
    """

    def __init__(
        self,
        classes: tuple[HueClass, ...] = DEFAULT_HUE_CLASSES,
        min_saturation: int = 150,
        min_value: int = 80,
        min_area: int = 30,
        full_confidence_area: int = 400,
    ) -> None:
        self.classes = classes
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_area = min_area
        self.full_confidence_area = full_confidence_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _class_mask(self, hsv: np.ndarray, cls: HueClass) -> np.ndarray:
        lo_sv, hi_sv = (self.min_saturation, self.min_value), (255, 255)
        if cls.hue_lo <= cls.hue_hi:
            return cv2.inRange(hsv, (cls.hue_lo, *lo_sv), (cls.hue_hi, *hi_sv))
        # Hue interval that wraps around 180 (red).
        return cv2.inRange(hsv, (cls.hue_lo, *lo_sv), (179, *hi_sv)) | \
            cv2.inRange(hsv, (0, *lo_sv), (cls.hue_hi, *hi_sv))

    def detect(self, image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []
        for cls in self.classes:
            mask = self._class_mask(hsv, cls)
            x0, y0, w0, h0 = cv2.boundingRect(mask)
            if w0 == 0:
                continue
            roi = cv2.morphologyEx(mask[y0:y0 + h0, x0:x0 + w0], cv2.MORPH_OPEN, self._kernel)
            n, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < self.min_area:
                    continue
                fill = area / float(w * h)
                size_term = min(1.0, area / float(self.full_confidence_area))
                score = float(np.clip(fill * (0.4 + 0.6 * size_term), 0.0, 1.0))
                # Pixel centres sit at integer coordinates, so the blob edges are half a
                # pixel outside the first and last covered pixel.
                x, y = x + x0, y + y0
                box = (x - 0.5, y - 0.5, x + w - 0.5, y + h - 0.5)
                detections.append(Detection(tuple(float(v) for v in box), score, cls.label))
        return detections


class YoloAdapter:
    """OPTIONAL adapter for an Ultralytics YOLO model.

    Not used by the tests or the demo and not exercised in this repository's CI: it
    needs the optional ``ultralytics`` package (which pulls in torch) and model weights.
    It only maps the library output onto the ``Detection`` type.
    """

    def __init__(self, weights: str = "yolov8n.pt", min_score: float = 0.05) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "YoloAdapter needs the optional dependency: pip install ultralytics"
            ) from exc
        self._model = YOLO(weights)
        self._min_score = min_score

    def detect(self, image: np.ndarray) -> list[Detection]:  # pragma: no cover - optional
        result = self._model.predict(image, conf=self._min_score, verbose=False)[0]
        names = result.names
        out: list[Detection] = []
        for box, score, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(),
            strict=True,
        ):
            out.append(Detection(tuple(float(v) for v in box), float(score), names[int(cls)]))
        return out
