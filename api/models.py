"""
Lazy-load ML models with graceful fallback when files / packages are missing.
"""

from __future__ import annotations

import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#  Model registry: name -> lazy-loader
# ------------------------------------------------------------------

_models: dict[str, object] = {}
_model_status: dict[str, bool] = {}


def _try_load(key: str, loader, fallback=None) -> tuple[object, bool]:
    if key in _models:
        return _models[key], _model_status[key]
    try:
        instance = loader()
        _models[key] = instance
        _model_status[key] = True
        logger.info("Model '%s' loaded successfully.", key)
        return instance, True
    except Exception as exc:
        logger.warning("Model '%s' failed to load: %s", key, exc)
        _models[key] = fallback
        _model_status[key] = False
        return fallback, False


def get_vehicle_model():
    from ultralytics import YOLO
    from api.config import settings

    return _try_load(
        "vehicle",
        lambda: YOLO(settings.VEHICLE_DETECTION_MODEL),
    )


def get_license_plate_model():
    from ultralytics import YOLO
    from api.config import settings

    return _try_load(
        "license_plate",
        lambda: YOLO(settings.LICENSE_PLATE_MODEL),
    )


def get_fire_smoke_model():
    from ultralytics import YOLO
    from api.config import settings

    return _try_load(
        "fire_smoke",
        lambda: YOLO(settings.FIRE_SMOKE_MODEL),
    )


def get_accident_model():
    from ultralytics import YOLO
    from api.config import settings

    return _try_load(
        "accident",
        lambda: YOLO(settings.ACCIDENT_MODEL),
    )


def get_face_cascade():
    return _try_load(
        "face_cascade",
        lambda: cv2.CascadeClassifier(
            cv2.data.haarcascades + settings.FACE_CASCADE_PATH
        ),
    )


def get_gunshot_model():
    from tensorflow.keras.models import load_model
    from api.config import settings

    return _try_load(
        "gunshot",
        lambda: load_model(settings.GUNSHOT_MODEL),
    )


# ------------------------------------------------------------------
#  Health summary
# ------------------------------------------------------------------

def all_model_status() -> dict[str, bool]:
    """Trigger lazy-load for every model and return status dict."""
    get_vehicle_model()
    get_license_plate_model()
    get_fire_smoke_model()
    get_accident_model()
    get_face_cascade()
    get_gunshot_model()
    return dict(_model_status)
