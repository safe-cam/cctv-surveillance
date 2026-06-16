import sys
from unittest.mock import MagicMock
from collections import UserDict

import pytest

# ============================================================
# Mock all external modules at sys.modules level BEFORE any
# test imports, so that importing web.streamlit doesn't
# require any real ML/DB packages.
# ============================================================

_MOCKED_MODULES: dict[str, MagicMock] = {}


def _m(modname: str, attrs: dict | None = None) -> MagicMock:
    """Create or retrieve a mock module and inject it into sys.modules."""
    if modname not in _MOCKED_MODULES:
        m = MagicMock(name=modname)
        m.__name__ = modname
        _MOCKED_MODULES[modname] = m
        sys.modules[modname] = m
    m = _MOCKED_MODULES[modname]
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    return m


# --- ultralytics tree ---
_m('ultralytics', {'YOLO': MagicMock()})
_m('ultralytics.nn')
_m('ultralytics.nn.tasks', {'DetectionModel': MagicMock()})
_m('ultralytics.nn.modules')
_m('ultralytics.nn.modules.container', {'Sequential': MagicMock()})
_m('ultralytics.nn.modules.conv', {'Conv': MagicMock()})
_m('ultralytics.nn.modules.block', {'C2f': MagicMock()})
_m('ultralytics.nn.modules.head', {'Detect': MagicMock()})

# --- torch (only if absent) ---
if 'torch' not in sys.modules:
    _m('torch')
    _m('torch.serialization', {'add_safe_globals': MagicMock()})
    _m('torch.nn')
    _m('torch.nn.modules')
    _m('torch.nn.modules.container')

# --- tensorflow / keras ---
_m('tensorflow')
tf_mod = _m('tensorflow.keras')
tf_conf = MagicMock()
tf_conf.set_visible_devices = MagicMock()
tf_mod.config = tf_conf
_m('tensorflow.keras.models', {'load_model': MagicMock()})

# --- deepface ---
_m('deepface')
_m('deepface.DeepFace', {'verify': MagicMock(), 'analyze': MagicMock()})

# --- moviepy ---
_m('moviepy')
_m('moviepy.editor', {'VideoFileClip': MagicMock()})

# --- librosa ---
librosa = _m('librosa', {'load': MagicMock()})
librosa.feature = MagicMock()
librosa.feature.melspectrogram = MagicMock()
librosa.power_to_db = MagicMock()

# --- pymongo ---
_m('pymongo', {'MongoClient': MagicMock()})

# --- smtplib (stdlib, always available) ---
# Not mocked; tests patch web.streamlit.smtplib.SMTP directly.

# --- pytesseract ---
_m('pytesseract', {'image_to_string': MagicMock()})

# --- cv2 ---
cv2 = _m('cv2')
cv2.imread = MagicMock()
cv2.imdecode = MagicMock()
cv2.imwrite = MagicMock()
cv2.cvtColor = MagicMock()
cv2.rectangle = MagicMock()
cv2.putText = MagicMock()
cv2.CascadeClassifier = MagicMock(return_value=MagicMock())
cv2.data = MagicMock()
cv2.data.haarcascades = ''
cv2.VideoCapture = MagicMock()
cv2.VideoWriter = MagicMock()
cv2.VideoWriter_fourcc = MagicMock(return_value='mp4v')
cv2.COLOR_BGR2GRAY = 6
cv2.COLOR_BGR2RGB = 4
cv2.IMREAD_COLOR = 1
cv2.FONT_HERSHEY_SIMPLEX = 0

# --- streamlit (mocked to prevent st.stop() from crashing) ---
class _MockSessionState(UserDict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self[name] = value

def _noop(*a, **kw): return None

mock_st = MagicMock(name='streamlit')
mock_st.session_state = _MockSessionState()
mock_st.session_state.logged_in = True
mock_st.session_state.user = {
    "customer_name": "test_user",
    "password": "test_pass",
    "email": "test@example.com",
    "role": "User",
    "plan": "Basic",
    "created_at": None,
}
mock_st.session_state.email = "test@example.com"
for name in [
    'title', 'error', 'warning', 'success', 'info', 'stop', 'image',
    'video', 'write', 'markdown', 'subheader', 'caption', 'spinner',
    'experimental_rerun',
]:
    setattr(mock_st, name, _noop)
mock_st.button = MagicMock(return_value=False)
mock_st.text_input = MagicMock(return_value='')
mock_st.selectbox = MagicMock(return_value='')
mock_st.multiselect = MagicMock(return_value=[])
mock_st.radio = MagicMock(return_value='Webcam')
mock_st.tabs = MagicMock(return_value=(MagicMock(), MagicMock()))
mock_st.columns = MagicMock(return_value=(MagicMock(), MagicMock()))
mock_st.form = MagicMock()
mock_st.form_submit_button = MagicMock()
mock_st.file_uploader = MagicMock(return_value=None)
mock_st.camera_input = MagicMock(return_value=None)
mock_st.date_input = MagicMock()
mock_st.empty = MagicMock()
mock_st.sidebar = MagicMock()
mock_st.sidebar.radio = MagicMock(return_value='Detection')
mock_st.sidebar.multiselect = MagicMock(return_value=[])
mock_st.expander = MagicMock()

if 'streamlit' not in sys.modules:
    sys.modules['streamlit'] = mock_st

# After import, subsequent accesses to streamlit return the same mock.
# Tests can reassign attributes at will.


# ============================================================
#  Shared fixtures
# ============================================================

@pytest.fixture
def mock_users_col():
    return MagicMock(name='users_collection')

@pytest.fixture
def mock_detections_col():
    return MagicMock(name='detections_collection')

@pytest.fixture
def mock_cameras_col():
    return MagicMock(name='cameras_collection')

@pytest.fixture
def mock_subscriptions_col():
    return MagicMock(name='subscriptions_collection')

@pytest.fixture
def mock_wanted_lp_col():
    return MagicMock(name='wanted_lp_collection')

@pytest.fixture
def mock_wanted_face_col():
    return MagicMock(name='wanted_face_collection')
