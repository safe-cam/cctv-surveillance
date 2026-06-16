from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Detection ───────────────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    tasks: list[str] = Field(
        default=...,
        description="List of detection tasks to run",
        examples=[["Vehicle Detection", "Face Detection"]],
    )

class DetectionResponse(BaseModel):
    tasks_run: list[str]
    detections: list[dict[str, Any]] = Field(
        default=...,
        description="Per-task detection results (bounding boxes, labels, scores)",
    )
    processed_image: str | None = Field(
        None,
        description="Base64-encoded annotated image (JPEG)",
    )
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

class BatchDetectionResponse(BaseModel):
    frames: list[DetectionResponse]
    total_frames: int

class GunshotRequest(BaseModel):
    audio_bytes: str = Field(..., description="Base64-encoded WAV audio data")
    filename: str = Field(default="audio.wav", description="Original filename hint")

class GunshotResponse(BaseModel):
    result: str = Field(
        ...,
        description="Detection result",
        examples=["Gunshot Detected", "No Gunshot", "Model not available"],
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0)

# ── Auth ────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, examples=["johndoe"])
    password: str = Field(..., min_length=1)

class LoginResponse(BaseModel):
    success: bool
    customer_name: str | None = None
    role: str | None = None
    plan: str | None = None
    email: str | None = None
    message: str | None = None

class RegisterRequest(BaseModel):
    customer_name: str = Field(..., min_length=1, examples=["johndoe"])
    password: str = Field(..., min_length=4)
    email: str = Field(..., examples=["john@example.com"])
    role: str = Field(default="User", pattern="^(User|Admin)$")
    plan: str = Field(default="Basic", pattern="^(Basic|Standard|Premium)$")

class RegisterResponse(BaseModel):
    success: bool
    message: str | None = None

# ── Health ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    models: dict[str, bool]
    database: bool
    version: str = "0.1.0"

# ── Error ───────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
