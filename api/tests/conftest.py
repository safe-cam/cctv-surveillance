"""
Pytest configuration for API tests.
Mocks external dependencies so tests run without real ML/audio packages.
"""

import sys
from unittest.mock import MagicMock

import pytest


# ------------------------------------------------------------------
#  Module-level mocks for packages that may not be installed
# ------------------------------------------------------------------
def _ensure_mock(modname):
    if modname not in sys.modules:
        m = MagicMock(name=modname)
        m.__name__ = modname
        sys.modules[modname] = m
    return sys.modules[modname]


# librosa
librosa = _ensure_mock("librosa")
librosa.load = MagicMock()
librosa.feature = MagicMock()
librosa.feature.melspectrogram = MagicMock()
librosa.power_to_db = MagicMock()

# ------------------------------------------------------------------
#  Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model_loaders(monkeypatch):
    """Mock all model loaders in api.models before every test."""

    fake_model = MagicMock()
    fake_cascade = MagicMock()

    def _loader(result, ok):
        def _fn():
            return result, ok
        return _fn

    monkeypatch.setattr("api.models.get_vehicle_model", _loader(fake_model, False))
    monkeypatch.setattr("api.models.get_license_plate_model", _loader(fake_model, False))
    monkeypatch.setattr("api.models.get_fire_smoke_model", _loader(fake_model, False))
    monkeypatch.setattr("api.models.get_accident_model", _loader(fake_model, False))
    monkeypatch.setattr("api.models.get_face_cascade", _loader(fake_cascade, False))
    monkeypatch.setattr("api.models.get_gunshot_model", _loader(fake_model, False))

    yield
