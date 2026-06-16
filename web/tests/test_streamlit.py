import re
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# =============================================================================
#  Helper: import & reference the streamlit module once, then patch per test
# =============================================================================
import web.streamlit as st_mod


# =============================================================================
#  1. Indian Plate Regex
# =============================================================================
class TestIndianPlateRegex:
    def setup_method(self):
        self.regex = st_mod.IND_PLATE_REGEX

    def test_valid_plates(self):
        valid = [
            "MH01AB1234",
            "DL2C1234",
            "KA05MG3456",
            "UP14AB1234",
            "GJ01CD5678",
            "TN07BC2345",
            "AP03XY1234",
        ]
        for plate in valid:
            assert self.regex.match(plate), f"{plate!r} should be valid"

    def test_invalid_plates(self):
        invalid = [
            "",
            "ABC123",
            "MH1@AB1234",
            "12ABCD34",
            "MH01AB12345",
            "MH 01 AB 1234",
            "mh01ab1234",
            "MH01-AB-1234",
            "ABCDEFGHIJ",
            "A",
            "1234567890",
        ]
        for plate in invalid:
            assert not self.regex.match(plate), f"{plate!r} should be invalid"


# =============================================================================
#  2. send_email
# =============================================================================
class TestSendEmail:
    def test_sends_email_successfully(self):
        sender = MagicMock()
        sender.__enter__.return_value = sender
        with patch("web.streamlit.smtplib.SMTP", return_value=sender):
            st_mod.send_email("Subject", "Body", "to@example.com")

        sender.starttls.assert_called_once()
        sender.login.assert_called_once()
        sender.send_message.assert_called_once()
        msg = sender.send_message.call_args[0][0]
        assert msg["Subject"] == "Subject"
        assert msg["To"] == "to@example.com"

    def test_default_alert_address(self):
        sender = MagicMock()
        sender.__enter__.return_value = sender
        with patch("web.streamlit.smtplib.SMTP", return_value=sender):
            st_mod.send_email("S", "B")

        msg = sender.send_message.call_args[0][0]
        assert msg["To"] == st_mod.ALERT_EMAIL


# =============================================================================
#  3. login
# =============================================================================
class TestLogin:
    def test_successful_login(self):
        user_doc = {
            "customer_name": "alice",
            "password": "secret",
            "role": "User",
            "plan": "Basic",
        }
        mock_col = MagicMock()
        mock_col.find_one.return_value = user_doc

        with patch("web.streamlit.users_collection", mock_col):
            result = st_mod.login("alice", "secret")

        assert result == user_doc
        mock_col.find_one.assert_called_once_with({"customer_name": "alice"})

    def test_wrong_password(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = {
            "customer_name": "alice",
            "password": "secret",
        }

        with patch("web.streamlit.users_collection", mock_col):
            result = st_mod.login("alice", "wrongpass")

        assert result is None

    def test_user_not_found(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None

        with patch("web.streamlit.users_collection", mock_col):
            result = st_mod.login("unknown", "pass")

        assert result is None

    def test_user_doc_without_password_field(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = {"customer_name": "bob"}

        with patch("web.streamlit.users_collection", mock_col):
            result = st_mod.login("bob", "pass")

        assert result is None

    def test_mongo_connection_error(self):
        mock_col = MagicMock()
        mock_col.find_one.side_effect = Exception("Connection refused")

        with patch("web.streamlit.users_collection", mock_col):
            with pytest.raises(Exception, match="Connection refused"):
                st_mod.login("alice", "secret")


# =============================================================================
#  4. register_user
# =============================================================================
class TestRegisterUser:
    def test_successful_registration(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None

        with patch("web.streamlit.users_collection", mock_col):
            ok = st_mod.register_user("newuser", "mypass", "new@example.com")

        assert ok is True
        assert mock_col.insert_one.called
        inserted = mock_col.insert_one.call_args[0][0]
        assert inserted["customer_name"] == "newuser"
        assert inserted["password"] == "mypass"
        assert inserted["email"] == "new@example.com"
        assert inserted["role"] == "User"
        assert inserted["plan"] == "Basic"

    def test_custom_role_and_plan(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None

        with patch("web.streamlit.users_collection", mock_col):
            ok = st_mod.register_user("admin", "x", "a@b.com", role="Admin", plan="Premium")

        assert ok is True
        inserted = mock_col.insert_one.call_args[0][0]
        assert inserted["role"] == "Admin"
        assert inserted["plan"] == "Premium"

    def test_duplicate_username(self):
        mock_col = MagicMock()
        mock_col.find_one.side_effect = [
            {"customer_name": "dup"},  # exists
        ]

        with patch("web.streamlit.users_collection", mock_col):
            ok = st_mod.register_user("dup", "p", "unique@email.com")

        assert ok is False
        mock_col.insert_one.assert_not_called()

    def test_duplicate_email(self):
        mock_col = MagicMock()
        # find_one with $or query returns a result when email matches
        def fake_find_one(query):
            if "$or" in query:
                return {"email": "exists@example.com"}
            return None

        mock_col.find_one.side_effect = fake_find_one

        with patch("web.streamlit.users_collection", mock_col):
            ok = st_mod.register_user("unique", "p", "exists@example.com")

        assert ok is False
        mock_col.insert_one.assert_not_called()

    def test_mongo_insert_failure(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None
        mock_col.insert_one.side_effect = Exception("Write failed")

        with patch("web.streamlit.users_collection", mock_col):
            with pytest.raises(Exception, match="Write failed"):
                st_mod.register_user("u", "p", "e@e.com")


# =============================================================================
#  5. predict_gunshot
# =============================================================================
class TestPredictGunshot:
    def test_gunshot_detected(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.85]]

        with patch("web.streamlit.gunshot_model", mock_model):
            result = st_mod.predict_gunshot(np.zeros((128, 128)))

        assert result == "Gunshot Detected"
        mock_model.predict.assert_called_once()

    def test_no_gunshot(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.15]]

        with patch("web.streamlit.gunshot_model", mock_model):
            result = st_mod.predict_gunshot(np.zeros((128, 128)))

        assert result == "No Gunshot"

    def test_model_not_available(self):
        with patch("web.streamlit.gunshot_model", None):
            result = st_mod.predict_gunshot(np.zeros((128, 128)))

        assert result == "Model not available"


# =============================================================================
#  6. match_wanted_face
# =============================================================================
class TestMatchWantedFace:
    def test_match_found(self):
        fake_bytes = b"fake_image_bytes"
        mock_col = MagicMock()
        mock_col.find.return_value = [{"image": fake_bytes}]

        mock_np_arr = MagicMock()
        with (
            patch("web.streamlit.wanted_face_collection", mock_col),
            patch("numpy.frombuffer", return_value=mock_np_arr) as mock_frombuffer,
            patch("cv2.imdecode", return_value=MagicMock()) as mock_imdecode,
            patch("cv2.cvtColor", return_value=MagicMock()) as mock_cvt,
            patch("web.streamlit.DeepFace.verify", return_value={"verified": True}),
        ):
            result = st_mod.match_wanted_face(MagicMock())

        assert result is True

    def test_no_match(self):
        fake_bytes = b"fake_image_bytes"
        mock_col = MagicMock()
        mock_col.find.return_value = [{"image": fake_bytes}]

        with (
            patch("web.streamlit.wanted_face_collection", mock_col),
            patch("numpy.frombuffer"),
            patch("cv2.imdecode"),
            patch("cv2.cvtColor"),
            patch("web.streamlit.DeepFace.verify", return_value={"verified": False}),
        ):
            result = st_mod.match_wanted_face(MagicMock())

        assert result is False

    def test_no_documents_in_collection(self):
        mock_col = MagicMock()
        mock_col.find.return_value = []

        with patch("web.streamlit.wanted_face_collection", mock_col):
            result = st_mod.match_wanted_face(MagicMock())

        assert result is False
        mock_col.find.assert_called_once_with({})

    def test_document_missing_image_field(self):
        mock_col = MagicMock()
        mock_col.find.return_value = [{"no_image": True}]

        with patch("web.streamlit.wanted_face_collection", mock_col):
            result = st_mod.match_wanted_face(MagicMock())

        assert result is False

    def test_deepface_verification_raises(self):
        fake_bytes = b"fake_bytes"
        mock_col = MagicMock()
        mock_col.find.return_value = [{"image": fake_bytes}]

        with (
            patch("web.streamlit.wanted_face_collection", mock_col),
            patch("numpy.frombuffer"),
            patch("cv2.imdecode"),
            patch("cv2.cvtColor"),
            patch("web.streamlit.DeepFace.verify", side_effect=ValueError("no face")),
        ):
            result = st_mod.match_wanted_face(MagicMock())

        assert result is False

    def test_multiple_documents_second_matches(self):
        fake_bytes = b"fake"
        mock_col = MagicMock()
        mock_col.find.return_value = [
            {"image": fake_bytes, "_id": 1},
            {"image": fake_bytes, "_id": 2},
        ]

        call_count = [0]

        def fake_verify(*a, **kw):
            call_count[0] += 1
            return {"verified": call_count[0] == 2}

        with (
            patch("web.streamlit.wanted_face_collection", mock_col),
            patch("numpy.frombuffer"),
            patch("cv2.imdecode"),
            patch("cv2.cvtColor"),
            patch("web.streamlit.DeepFace.verify", side_effect=fake_verify),
        ):
            result = st_mod.match_wanted_face(MagicMock())

        assert result is True
        assert call_count[0] == 2


# =============================================================================
#  7. extract_audio
# =============================================================================
class TestExtractAudio:
    def test_extracts_audio_from_video(self):
        mock_clip = MagicMock()
        with patch("web.streamlit.VideoFileClip", return_value=mock_clip):
            st_mod.extract_audio("input.mp4", "output.wav")

        mock_clip.audio.write_audiofile.assert_called_once_with(
            "output.wav", verbose=False, logger=None
        )


# =============================================================================
#  8. audio_to_mel_spectrogram
# =============================================================================
class TestAudioToMelSpectrogram:
    def test_returns_spectrogram(self):
        mock_y = MagicMock()
        mock_s = MagicMock()
        mock_s_db = MagicMock()

        with (
            patch("web.streamlit.librosa.load", return_value=(mock_y, 22050)),
            patch("web.streamlit.librosa.feature.melspectrogram", return_value=mock_s),
            patch("web.streamlit.librosa.power_to_db", return_value=mock_s_db),
        ):
            result = st_mod.audio_to_mel_spectrogram("audio.wav")

        assert result == mock_s_db


# =============================================================================
#  9. logout
# =============================================================================
class TestLogout:
    def test_clears_session_and_reruns(self):
        class FakeSession(dict):
            def __setattr__(self, k, v):
                self[k] = v
            def __getattr__(self, k):
                try:
                    return self[k]
                except KeyError:
                    raise AttributeError(k)

        fake_session = FakeSession({"logged_in": True, "user": {"name": "x"}})

        with (
            patch("web.streamlit.st.session_state", fake_session),
            patch("web.streamlit.st.experimental_rerun") as mock_rerun,
        ):
            st_mod.logout()

        assert fake_session["logged_in"] is False
        assert "user" not in fake_session
        mock_rerun.assert_called_once()


# =============================================================================
#  10. process_frame
# =============================================================================
class TestProcessFrame:
    def test_no_tasks_returns_copy(self):
        frame = MagicMock()
        frame.copy.return_value = MagicMock()

        with patch.multiple(
            "web.streamlit",
            vehicle_detection_model=MagicMock(),
            license_plate_detection_model=MagicMock(),
            fire_and_smoke_detection_model=MagicMock(),
            accident_detection_model=MagicMock(),
            face_cascade=MagicMock(),
        ):
            result = st_mod.process_frame(frame, [])

        frame.copy.assert_called_once()
        assert result is not frame

    def test_vehicle_detection(self):
        frame = MagicMock()
        frame.copy.return_value = MagicMock(name="copy")
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.plot.return_value = MagicMock(name="plotted")
        mock_model.return_value = [mock_result]

        with patch("web.streamlit.vehicle_detection_model", mock_model):
            result = st_mod.process_frame(frame, ["Vehicle Detection"])

        mock_model.assert_called_once()
        assert result is not frame

    def test_vehicle_and_fire_detection(self):
        frame = MagicMock()
        frame.copy.return_value = MagicMock(name="copy")

        mock_veh = MagicMock()
        mock_veh_result = MagicMock()
        mock_veh_result.plot.return_value = MagicMock(name="veh_plot")
        mock_veh.return_value = [mock_veh_result]

        mock_fire = MagicMock()
        mock_fire_result = MagicMock()
        mock_fire_result.plot.return_value = MagicMock(name="fire_plot")
        mock_fire.return_value = [mock_fire_result]

        with (
            patch("web.streamlit.vehicle_detection_model", mock_veh),
            patch("web.streamlit.fire_and_smoke_detection_model", mock_fire),
            patch("web.streamlit.license_plate_detection_model", MagicMock()),
            patch("web.streamlit.accident_detection_model", MagicMock()),
            patch("web.streamlit.face_cascade", MagicMock()),
        ):
            result = st_mod.process_frame(
                frame, ["Vehicle Detection", "Fire/Smoke Detection"]
            )

        mock_veh.assert_called_once()
        mock_fire.assert_called_once()

    def test_face_detection_no_faces(self):
        frame = MagicMock()
        frame.copy.return_value = MagicMock(name="copy")
        gray = MagicMock()

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = []

        with (
            patch("web.streamlit.face_cascade", mock_cascade),
            patch("cv2.cvtColor", return_value=gray),
            patch("web.streamlit.st.session_state", {"wanted_detected": False}),
        ):
            result = st_mod.process_frame(frame, ["Face Detection"])

        mock_cascade.detectMultiScale.assert_called_once_with(gray, 1.1, 5)

    def test_face_detection_with_faces(self):
        frame = MagicMock()
        frame.copy.return_value = MagicMock(name="copy")
        gray = MagicMock()

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(10, 20, 50, 60)]

        with (
            patch("web.streamlit.face_cascade", mock_cascade),
            patch("web.streamlit.face_cascade", mock_cascade),
            patch("cv2.cvtColor", return_value=gray),
            patch("web.streamlit.DeepFace.analyze", return_value={"dominant_gender": "Man"}),
            patch("web.streamlit.match_wanted_face", return_value=False),
            patch("web.streamlit.st.session_state", {"wanted_detected": False, "alerted_faces": 0}),
        ):
            result = st_mod.process_frame(frame, ["Face Detection"])

        assert result is not None

    def test_face_detection_wanted_match(self):
        frame = MagicMock()
        frame.copy.return_value = MagicMock(name="copy")
        gray = MagicMock()

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(10, 20, 50, 60)]

        with (
            patch("web.streamlit.face_cascade", mock_cascade),
            patch("cv2.cvtColor", return_value=gray),
            patch("web.streamlit.DeepFace.analyze", return_value={"dominant_gender": "Woman"}),
            patch("web.streamlit.match_wanted_face", return_value=True),
            patch("web.streamlit.st.session_state", {"wanted_detected": False, "alerted_faces": 0}),
            patch("web.streamlit.send_email") as mock_email,
        ):
            result = st_mod.process_frame(frame, ["Face Detection"])

        mock_email.assert_called_once()

    def test_invalid_plate_ignored(self):
        frame = MagicMock()
        frame.copy.return_value = MagicMock(name="copy")

        mock_lp_model = MagicMock()
        result = MagicMock()
        result.plot.return_value = MagicMock(name="lp_plot")
        mock_box = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 10, 10]])
        result.boxes = MagicMock()
        result.boxes.xyxy = mock_box.xyxy
        mock_lp_model.return_value = [result]

        fake_plate_roi = MagicMock()
        with (
            patch("web.streamlit.license_plate_detection_model", mock_lp_model),
            patch("cv2.imdecode"),
            patch("web.streamlit.pytesseract.image_to_string", return_value="INVALID"),
            patch("web.streamlit.st.session_state", {}),
            patch("web.streamlit.user", {"customer_name": "test"}),
            patch("web.streamlit.email", "test@test.com"),
        ):
            st_mod.process_frame(frame, ["License Plate Detection"])


# =============================================================================
#  11. process_batch_frames
# =============================================================================
class TestProcessBatchFrames:
    def test_no_tasks_returns_copy(self):
        frames = [MagicMock()]

        with patch.multiple(
            "web.streamlit",
            vehicle_detection_model=MagicMock(),
            license_plate_detection_model=MagicMock(),
            fire_and_smoke_detection_model=MagicMock(),
            accident_detection_model=MagicMock(),
            face_cascade=MagicMock(),
        ):
            result = st_mod.process_batch_frames(frames, [])

        assert len(result) == 1

    def test_vehicle_detection_batch(self):
        frames = [MagicMock(), MagicMock()]
        mock_model = MagicMock()

        def fake_call(input_frames):
            return [MagicMock() for _ in input_frames]

        mock_model.side_effect = fake_call

        with patch("web.streamlit.vehicle_detection_model", mock_model):
            result = st_mod.process_batch_frames(frames, ["Vehicle Detection"])

        assert len(result) == 2

    def test_face_detection_batch(self):
        frames = [MagicMock()]
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(0, 0, 10, 10)]

        with (
            patch("web.streamlit.face_cascade", mock_cascade),
            patch("cv2.cvtColor"),
            patch("web.streamlit.DeepFace.analyze", return_value={"dominant_gender": "Man"}),
            patch("web.streamlit.match_wanted_face", return_value=False),
            patch("web.streamlit.st.session_state", {"wanted_detected": False}),
        ):
            result = st_mod.process_batch_frames(frames, ["Face Detection"])

        assert len(result) == 1

    def test_all_detection_types_batch(self):
        frames = [MagicMock()]

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = []

        mock_model = MagicMock()
        res = MagicMock()
        res.plot.return_value = MagicMock()
        mock_model.return_value = [res]

        with (
            patch.multiple(
                "web.streamlit",
                vehicle_detection_model=mock_model,
                license_plate_detection_model=mock_model,
                fire_and_smoke_detection_model=mock_model,
                accident_detection_model=mock_model,
            ),
            patch("web.streamlit.face_cascade", mock_cascade),
            patch("cv2.cvtColor"),
            patch("web.streamlit.st.session_state", {"wanted_detected": False}),
        ):
            result = st_mod.process_batch_frames(
                frames,
                [
                    "Vehicle Detection",
                    "License Plate Detection",
                    "Fire/Smoke Detection",
                    "Accident Detection",
                    "Face Detection",
                ],
            )

        assert len(result) == 1
