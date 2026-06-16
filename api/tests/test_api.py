"""
Unit tests for the FastAPI backend.
All external dependencies (YOLO, MongoDB, etc.) are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# =============================================================================
#  Helpers
# =============================================================================

def _fake_jpeg_bytes():
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buf, "JPEG")
    return buf.getvalue()


def _detection_result():
    return {
        "detections": [
            {"task": "Vehicle Detection", "count": 1, "boxes": []}
        ],
        "processed_image": None,
        "processing_time_ms": 1.23,
    }


# =============================================================================
#  Root
# =============================================================================

class TestRoot:
    def test_root_endpoint(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "version" in data


# =============================================================================
#  Health
# =============================================================================

class TestHealth:
    def test_health_returns_status(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "models" in data
        assert "database" in data


# =============================================================================
#  Auth
# =============================================================================

class TestLogin:
    LOGIN_URL = "/api/auth/login"

    @patch("api.routers.auth.pymongo.MongoClient")
    def test_successful_login(self, mock_mongo):
        fake_user = {
            "customer_name": "alice",
            "password": "pass",
            "role": "User",
            "plan": "Basic",
            "email": "alice@example.com",
        }
        mock_col = MagicMock()
        mock_col.find_one.return_value = fake_user
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_col
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_mongo.return_value = mock_client

        resp = client.post(self.LOGIN_URL, json={"username": "alice", "password": "pass"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["customer_name"] == "alice"

    @patch("api.routers.auth.pymongo.MongoClient")
    def test_wrong_password(self, mock_mongo):
        mock_col = MagicMock()
        mock_col.find_one.return_value = {"customer_name": "alice", "password": "pass"}
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_col
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_mongo.return_value = mock_client

        resp = client.post(self.LOGIN_URL, json={"username": "alice", "password": "wrong"})
        assert resp.status_code == 401

    @patch("api.routers.auth.pymongo.MongoClient")
    def test_user_not_found(self, mock_mongo):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_col
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_mongo.return_value = mock_client

        resp = client.post(self.LOGIN_URL, json={"username": "nobody", "password": "x"})
        assert resp.status_code == 401


class TestRegister:
    REGISTER_URL = "/api/auth/register"

    @patch("api.routers.auth.pymongo.MongoClient")
    def test_successful_registration(self, mock_mongo):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_col
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_mongo.return_value = mock_client

        resp = client.post(
            self.REGISTER_URL,
            json={
                "customer_name": "newuser",
                "password": "secret123",
                "email": "new@example.com",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @patch("api.routers.auth.pymongo.MongoClient")
    def test_duplicate_username(self, mock_mongo):
        mock_col = MagicMock()
        mock_col.find_one.return_value = {"customer_name": "existing"}
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_col
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_mongo.return_value = mock_client

        resp = client.post(
            self.REGISTER_URL,
            json={
                "customer_name": "existing",
                "password": "secret123",
                "email": "unique@example.com",
            },
        )
        assert resp.status_code == 409


# =============================================================================
#  Detection
# =============================================================================

class TestDetectImage:
    DETECT_URL = "/api/detect/"

    def test_unknown_task_returns_422(self):
        resp = client.post(
            self.DETECT_URL,
            files={"file": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")},
            data={"tasks": json.dumps(["Unknown Task"])},
        )
        assert resp.status_code == 422

    def test_unsupported_format_returns_400(self):
        resp = client.post(
            self.DETECT_URL,
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"tasks": json.dumps(["Vehicle Detection"])},
        )
        assert resp.status_code == 400

    @patch("api.routers.detection.run_detection", return_value=_detection_result())
    def test_valid_request_returns_200(self, mock_run):
        resp = client.post(
            self.DETECT_URL,
            files={"file": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")},
            data={"tasks": json.dumps(["Vehicle Detection"])},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks_run"] == ["Vehicle Detection"]
        mock_run.assert_called_once()

    @patch("api.routers.detection.run_detection", return_value=_detection_result())
    def test_empty_tasks_returns_200(self, mock_run):
        resp = client.post(
            self.DETECT_URL,
            files={"file": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")},
            data={"tasks": json.dumps([])},
        )
        assert resp.status_code == 200


class TestDetectVideo:
    BATCH_URL = "/api/detect/batch"

    @patch("api.routers.detection.run_detection", return_value=_detection_result())
    @patch("cv2.VideoCapture")
    def test_video_processing(self, mock_vc, mock_run):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30
        mock_cap.read.side_effect = [
            (True, np.zeros((10, 10, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_vc.return_value = mock_cap

        resp = client.post(
            self.BATCH_URL,
            files={"file": ("test.mp4", b"fake video bytes", "video/mp4")},
            data={"tasks": json.dumps(["Vehicle Detection"]), "max_frames": "10"},
        )
        assert resp.status_code == 200


class TestDetectGunshot:
    GUNSHOT_URL = "/api/detect/gunshot"

    def _do_request(self):
        return client.post(
            self.GUNSHOT_URL,
            files={"file": ("test.wav", b"fake audio", "audio/wav")},
        )

    def test_model_unavailable(self, monkeypatch):
        monkeypatch.setattr("api.models.get_gunshot_model", lambda: (None, False))
        monkeypatch.setattr("librosa.load", lambda path, sr: (MagicMock(), 22050))
        monkeypatch.setattr("librosa.feature.melspectrogram", lambda y=None, sr=None: MagicMock())
        monkeypatch.setattr("librosa.power_to_db", lambda S, ref=None: MagicMock())
        resp = self._do_request()
        assert resp.status_code == 200
        assert resp.json()["result"] == "Model not available"

    def test_gunshot_detected(self, monkeypatch):
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.85]]
        monkeypatch.setattr("api.models.get_gunshot_model", lambda: (mock_model, True))
        monkeypatch.setattr("librosa.load", lambda path, sr: (MagicMock(), 22050))
        monkeypatch.setattr("librosa.feature.melspectrogram", lambda y=None, sr=None: MagicMock())
        monkeypatch.setattr("librosa.power_to_db", lambda S, ref=None: MagicMock())

        resp = self._do_request()
        assert resp.status_code == 200
        assert resp.json()["result"] == "Gunshot Detected"

    def test_no_gunshot(self, monkeypatch):
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.15]]
        monkeypatch.setattr("api.models.get_gunshot_model", lambda: (mock_model, True))
        monkeypatch.setattr("librosa.load", lambda path, sr: (MagicMock(), 22050))
        monkeypatch.setattr("librosa.feature.melspectrogram", lambda y=None, sr=None: MagicMock())
        monkeypatch.setattr("librosa.power_to_db", lambda S, ref=None: MagicMock())

        resp = self._do_request()
        assert resp.status_code == 200
        assert resp.json()["result"] == "No Gunshot"
