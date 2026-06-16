import os
from pathlib import Path


class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # ------------------------------------------------------------------
    #  Model paths (relative to PROJECT_ROOT)
    # ------------------------------------------------------------------
    VEHICLE_DETECTION_MODEL: str = "yolov8m.pt"

    LICENSE_PLATE_MODEL: str = str(
        PROJECT_ROOT / "web" / "models" / "license_plate_detection" / "weights" / "best.pt"
    )

    FIRE_SMOKE_MODEL: str = str(
        PROJECT_ROOT / "web" / "models" / "fire_detection" / "weights" / "best.pt"
    )

    ACCIDENT_MODEL: str = str(
        PROJECT_ROOT / "web" / "models" / "accident_dataset" / "best.pt"
    )

    GUNSHOT_MODEL: str = str(
        PROJECT_ROOT / "ML models" / "gun shot" / "gunshot_model"
    )

    FACE_CASCADE_PATH: str = "haarcascade_frontalface_default.xml"

    # ------------------------------------------------------------------
    #  MongoDB
    # ------------------------------------------------------------------
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB: str = os.getenv("MONGO_DB", "surveillance_db")

    # ------------------------------------------------------------------
    #  CORS
    # ------------------------------------------------------------------
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "*"
    ).split(",")

    # ------------------------------------------------------------------
    #  Server
    # ------------------------------------------------------------------
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # ------------------------------------------------------------------
    #  Detection task names
    # ------------------------------------------------------------------
    AVAILABLE_TASKS: list[str] = [
        "Vehicle Detection",
        "License Plate Detection",
        "Fire/Smoke Detection",
        "Accident Detection",
        "Face Detection",
        "Gunshot Detection",
    ]


settings = Settings()
