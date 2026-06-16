from __future__ import annotations

import io
import logging
import tempfile
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.config import settings
from api.schemas import (
    BatchDetectionResponse,
    DetectionRequest,
    DetectionResponse,
    ErrorResponse,
    GunshotRequest,
    GunshotResponse,
)
from api.services.detector import decode_image, encode_image, run_detection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detect", tags=["Detection"])


SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SUPPORTED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


@router.post(
    "/",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Run detection tasks on an uploaded image",
)
async def detect_image(
    file: UploadFile = File(..., description="Image file (jpg, png, etc.)"),
    tasks: str = Form(
        default='["Vehicle Detection"]',
        description="JSON array of detection task names",
    ),
):
    """
    Upload an image and run one or more detection models on it.

    **Supported tasks** (pass as JSON array):
    - ``Vehicle Detection``
    - ``License Plate Detection``
    - ``Fire/Smoke Detection``
    - ``Accident Detection``
    - ``Face Detection``
    """
    import json

    # --- validate tasks ---
    try:
        task_list: list[str] = json.loads(tasks)
    except json.JSONDecodeError:
        raise HTTPException(400, detail="'tasks' must be a valid JSON array of strings")

    if not isinstance(task_list, list) or not all(isinstance(t, str) for t in task_list):
        raise HTTPException(400, detail="'tasks' must be a JSON array of strings")

    invalid = set(task_list) - set(settings.AVAILABLE_TASKS)
    if invalid:
        raise HTTPException(
            422,
            detail=f"Unknown task(s): {invalid}. Supported: {settings.AVAILABLE_TASKS}",
        )

    # --- read & decode image ---
    ext = "." + (file.filename or "image.jpg").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_IMAGE_EXT:
        raise HTTPException(400, detail=f"Unsupported image format: {ext}")

    body = await file.read()
    image = decode_image(body)

    # --- run ---
    result = run_detection(image, task_list)

    return DetectionResponse(
        tasks_run=task_list,
        detections=result["detections"],
        processed_image=result["processed_image"],
        processing_time_ms=result["processing_time_ms"],
    )


@router.post(
    "/batch",
    response_model=BatchDetectionResponse,
    summary="Run detection tasks on all frames of a video",
)
async def detect_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv)"),
    tasks: str = Form(
        default='["Vehicle Detection"]',
        description="JSON array of detection task names",
    ),
    max_frames: int = Form(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of frames to process",
    ),
):
    """
    Upload a video and run detection on each frame.

    Returns results for up to ``max_frames`` evenly-spaced frames.
    """
    import json

    try:
        task_list: list[str] = json.loads(tasks)
    except json.JSONDecodeError:
        raise HTTPException(400, detail="'tasks' must be a valid JSON array")

    ext = "." + (file.filename or "video.mp4").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_VIDEO_EXT:
        raise HTTPException(400, detail=f"Unsupported video format: {ext}")

    body = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise HTTPException(400, detail="Could not read video or video has no frames")

    step = max(1, total // max_frames)
    frame_responses: list[DetectionResponse] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            result = run_detection(frame, task_list)
            frame_responses.append(
                DetectionResponse(
                    tasks_run=task_list,
                    detections=result["detections"],
                    processed_image=result["processed_image"],
                    processing_time_ms=result["processing_time_ms"],
                )
            )
        idx += 1

    cap.release()

    return BatchDetectionResponse(frames=frame_responses, total_frames=total)


@router.post(
    "/gunshot",
    response_model=GunshotResponse,
    summary="Detect gunshot from an audio file",
)
async def detect_gunshot_from_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, etc.)"),
):
    """
    Upload an audio file and run gunshot detection.

    Internally converts the audio to a mel-spectrogram and classifies it.
    """
    body = await file.read()

    # --- write to temp file so librosa can read it ---
    ext = "." + (file.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(tmp_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        from api.models import get_gunshot_model

        gunshot_model, ok = get_gunshot_model()
        if not ok or gunshot_model is None:
            return GunshotResponse(result="Model not available", confidence=None)

        input_data = S_db[np.newaxis, ..., np.newaxis]
        pred = gunshot_model.predict(input_data)
        confidence = float(pred[0][0])
        result = "Gunshot Detected" if confidence > 0.5 else "No Gunshot"
        return GunshotResponse(result=result, confidence=confidence)
    finally:
        import os

        os.unlink(tmp_path)
