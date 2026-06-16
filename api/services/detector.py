"""
Core detection pipeline — reusable across HTTP endpoints.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def decode_image(data: bytes | str) -> np.ndarray:
    """Decode raw bytes (or base64 string) into an OpenCV BGR image."""
    if isinstance(data, str):
        data = base64.b64decode(data)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data")
    return img


def encode_image(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR image to a base64 string."""
    success, buf = cv2.imencode(fmt, img)
    if not success:
        raise RuntimeError("Could not encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def run_detection(image: np.ndarray, tasks: list[str]) -> dict[str, Any]:
    """
    Run the requested detection tasks on a single BGR image.

    Returns a dict with:
      - ``detections``: list of per-task result dicts
      - ``processed_image``: base64 annotated BGR image (or None if no tasks)
      - ``processing_time_ms``: wall-clock time
    """
    processed = image.copy()
    all_detections: list[dict[str, Any]] = []
    start = time.perf_counter()

    # Lazily load models (will fall back gracefully if missing)
    from api.models import (
        get_vehicle_model,
        get_license_plate_model,
        get_fire_smoke_model,
        get_accident_model,
        get_face_cascade,
    )

    if "Vehicle Detection" in tasks:
        model, ok = get_vehicle_model()
        if ok:
            results = model(processed)
            processed = results[0].plot()
            boxes = _extract_boxes(results[0])
            all_detections.append({"task": "Vehicle Detection", "count": len(boxes), "boxes": boxes})
        else:
            all_detections.append({"task": "Vehicle Detection", "error": "Model not available"})

    if "License Plate Detection" in tasks:
        model, ok = get_license_plate_model()
        if ok:
            results = model(processed)
            processed = results[0].plot()
            boxes = _extract_boxes(results[0])
            all_detections.append({"task": "License Plate Detection", "count": len(boxes), "boxes": boxes})
        else:
            all_detections.append({"task": "License Plate Detection", "error": "Model not available"})

    if "Fire/Smoke Detection" in tasks:
        model, ok = get_fire_smoke_model()
        if ok:
            results = model(processed)
            processed = results[0].plot()
            boxes = _extract_boxes(results[0])
            all_detections.append({"task": "Fire/Smoke Detection", "count": len(boxes), "boxes": boxes})
        else:
            all_detections.append({"task": "Fire/Smoke Detection", "error": "Model not available"})

    if "Accident Detection" in tasks:
        model, ok = get_accident_model()
        if ok:
            results = model(processed)
            processed = results[0].plot()
            boxes = _extract_boxes(results[0])
            all_detections.append({"task": "Accident Detection", "count": len(boxes), "boxes": boxes})
        else:
            all_detections.append({"task": "Accident Detection", "error": "Model not available"})

    if "Face Detection" in tasks:
        cascade, ok = get_face_cascade()
        if ok and cascade:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5)
            boxes = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
            for (x, y, w, h) in faces:
                cv2.rectangle(processed, (x, y), (x + w, y + h), (0, 255, 0), 2)
            all_detections.append({"task": "Face Detection", "count": len(boxes), "boxes": boxes})
        else:
            all_detections.append({"task": "Face Detection", "error": "Face cascade not available"})

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "detections": all_detections,
        "processed_image": encode_image(processed),
        "processing_time_ms": round(elapsed, 2),
    }


def _extract_boxes(yolo_result) -> list[dict[str, Any]]:
    """Extract bounding-box info from a single YOLO result."""
    boxes = []
    if yolo_result.boxes is None:
        return boxes
    xyxy = yolo_result.boxes.xyxy.cpu().numpy()
    conf = yolo_result.boxes.conf.cpu().numpy() if yolo_result.boxes.conf is not None else None
    cls_ids = yolo_result.boxes.cls.cpu().numpy() if yolo_result.boxes.cls is not None else None
    names = yolo_result.names

    for i, box in enumerate(xyxy):
        entry: dict[str, Any] = {
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
        }
        if conf is not None:
            entry["confidence"] = round(float(conf[i]), 4)
        if cls_ids is not None:
            cls_id = int(cls_ids[i])
            entry["class_id"] = cls_id
            entry["label"] = names.get(cls_id, str(cls_id))
        boxes.append(entry)
    return boxes
