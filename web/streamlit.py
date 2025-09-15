

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.head import Detect
from ultralytics import YOLO
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import librosa
import pymongo
from tensorflow.keras.models import load_model
import smtplib
from email.message import EmailMessage
import pytesseract
import re


# Uncomment to force CPU for DeepFace if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import os
# Force CPU usage before importing any GPU-related libraries
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# ============================================================
# ALLOW SAFE GLOBALS FOR YOLO CHECKPOINT LOADING (Trusted Source Only)
# ============================================================
add_safe_globals([DetectionModel, Sequential, Conv, C2f, Detect])

# ============================================================
# MongoDB Setup
# ============================================================
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["surveillance_db"]
users_collection = db["users"]
detections_collection = db["detections"]
cameras_collection = db["cameras"]
subscriptions_collection = db["subscriptions"]

# Collections to store wanted data:
wanted_lp_collection = db["wanted_license_plates"]
wanted_face_collection = db["wanted_faces"]

# Connecting the SMTP
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "mohitkumharmm@gmail.com"
SMTP_PASS = "rcpv rmmh zyep xdnd"
ALERT_EMAIL = "mohitmolela@gmail.com"

def send_email(subject: str, body: str, to_address=ALERT_EMAIL):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_address
    msg.set_content(body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)

# Regression of Indian Licence Plate
IND_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$")

# ============================================================
# Utility Functions for Authentication
# ============================================================
def login(username, password):
    user = users_collection.find_one({"customer_name": username})
    if user and user.get("password") == password:
        return user
    return None

def register_user(customer_name, password, email, role="User", plan="Basic"):
    if users_collection.find_one({
        "$or": [
            {"customer_name": customer_name},
            {"email": email}
        ]
    }):
        return False
    users_collection.insert_one({
        "customer_name": customer_name,
        "password": password,
        "email": email,
        "role":   role,
        "plan":   plan,
        "created_at": datetime.now()
    })
    return True

# ============================================================
# Utility Functions for Gunshot Detection
# ============================================================
def extract_audio(video_path, audio_output_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_output_path, verbose=False, logger=None)

def audio_to_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def predict_gunshot(spectrogram):
    if gunshot_model is None:
        return "Model not available"
    input_data = spectrogram[np.newaxis, ..., np.newaxis]
    prediction = gunshot_model.predict(input_data)
    return "Gunshot Detected" if prediction[0][0] > 0.5 else "No Gunshot"

# ============================================================
# Helper Function for Matching a Face Against the Wanted Faces
# ============================================================
def match_wanted_face(face_image_rgb):
    """
    Check if face_image_rgb matches any face stored in the 'wanted_faces' collection.
    The wanted faces are stored as raw image bytes in MongoDB.
    face_image_rgb is an RGB image array.
    """
    docs = wanted_face_collection.find({})
    for doc in docs:
        wanted_bytes = doc.get("image")
        if not wanted_bytes:
            continue
        np_arr = np.frombuffer(wanted_bytes, np.uint8)
        wanted_face_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Convert stored face to RGB for consistency
        wanted_face_rgb = cv2.cvtColor(wanted_face_bgr, cv2.COLOR_BGR2RGB)
        try:
            # Remove threshold parameter since it's unsupported in your version.
            result = DeepFace.verify(
                face_image_rgb,
                wanted_face_rgb,
                enforce_detection=False,
                detector_backend='opencv'
            )
            if result["verified"]:
                print("DEBUG: Wanted face match found.")
                return True
        except Exception as e:
            print("DEBUG: Error in face verification:", e)
            pass
    return False

# ============================================================
# Load AI Models for Detection
# ============================================================
try:
    vehicle_detection_model = YOLO("yolov8m.pt")  # Adjust path if needed
except Exception as e:
    st.error(f"Error loading vehicle detection model: {e}")
    st.stop()

try:
    license_plate_detection_model = YOLO(r"F:\camera\web\models\license_plate_detection\weights\best.pt")
except Exception as e:
    st.error(f"Error loading license plate detection model: {e}")
    st.stop()

try:
    fire_and_smoke_detection_model = YOLO(r"F:\camera\web\models\fire_detection\weights\best.pt")
except Exception as e:
    st.error(f"Error loading fire/smoke detection model: {e}")
    st.stop()

# ------------------------------------------------------------
# Accident Detection Model Loading
# ------------------------------------------------------------
try:
    accident_detection_model = YOLO(r"F:\camera\accident detection\runs\detect\train2\weights\best.pt")
except Exception as e:
    st.error("Error loading accident detection model: "
             "Your model checkpoint may be incompatible with ultralytics v8.3+. "
             "Re-export your model or downgrade ultralytics.\n"
             f"Details: {e}")
    st.stop()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    gunshot_model = load_model(r"F:\camera\gun shot\gunshot_model_epoch_05_acc_0.90_valacc_0.96.h5")
except Exception as e:
    st.warning("Gunshot model not found. Gunshot detection will be disabled.")
    gunshot_model = None

# ============================================================
# Processing Functions for Media
# ============================================================
def process_frame(frame, selected_tasks):
    processed = frame.copy()

    if "Vehicle Detection" in selected_tasks:
        results = vehicle_detection_model(processed)
        processed = results[0].plot()

    if "License Plate Detection" in selected_tasks:
        results = license_plate_detection_model(processed)
        processed = results[0].plot()
        
        # Crop out each detected box and OCR:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = box.astype(int)
            plate_roi = frame[y1:y2, x1:x2]
            plate_text = pytesseract.image_to_string(
                plate_roi, config="--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ).strip().replace(" ", "").upper()
            if IND_PLATE_REGEX.match(plate_text):
                # valid plate
                st.write("Detected plate:", plate_text)
                # dedupe
                if st.session_state.get("alerted_plates") is None:
                    st.session_state["alerted_plates"] = set()
                if plate_text not in st.session_state["alerted_plates"]:
                    st.session_state["alerted_plates"].add(plate_text)
                    # check wanted
                    if wanted_lp_collection.find_one({"customer_name":user["customer_name"], "plate": plate_text}):
                        st.warning(f"Alert: Wanted plate {plate_text} detected!")
                        # send email
                        send_email(
                            subject="Wanted Plate Detected",
                            body=f"User {user['customer_name']} had wanted plate {plate_text} detected at {datetime.now()}.",
                            to_address=email,
                        )
            else:
                st.info(f"Ignored invalid plate: {plate_text}")

        if "Fire/Smoke Detection" in selected_tasks:
            results = fire_and_smoke_detection_model(processed)
            processed = results[0].plot()

    if "Accident Detection" in selected_tasks:
        results = accident_detection_model(processed)
        processed = results[0].plot()

    if "Face Detection" in selected_tasks:
        st.session_state["wanted_detected"] = False
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            face_region_bgr = processed[y:y+h, x:x+w]
            # Convert face region to RGB for DeepFace analysis
            face_region_rgb = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2RGB)

            try:
                # Use only gender detection (remove age)
                analysis = DeepFace.analyze(
                    face_region_rgb,
                    actions=['gender'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                gender = analysis.get('dominant_gender', 'N/A')
            except Exception as e:
                print("DEBUG: DeepFace analyze error:", e)
                gender = 'N/A'

            # Check if this face matches any wanted face
            if match_wanted_face(face_region_rgb):
                st.session_state["wanted_detected"] = True
                color = (0, 0, 255)  # Red for wanted faces
                if st.session_state.get("alerted_faces") is None:
                    st.session_state["alerted_faces"] = 0
                # only alert & email once per session
                if st.session_state["alerted_faces"] == 0:
                    st.session_state["alerted_faces"] = 1
                    st.warning("Alert: Wanted person detected!")
                    send_email(
                        subject="Wanted Person Detected",
                        body=f"User {user['customer_name']} had a wanted face detected at {datetime.now()}.",
                        to_address=email,
                    )
            else:
                color = (0, 255, 0)  # Green for normal faces

            cv2.rectangle(processed, (x, y), (x+w, y+h), color, 2)
            label = f"{gender}"
            cv2.putText(processed, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return processed

def process_batch_frames(frames, selected_tasks):
    proc_frames = frames.copy()

    if "Vehicle Detection" in selected_tasks:
        vehicle_results = vehicle_detection_model(proc_frames)
        proc_frames = [res.plot() for res in vehicle_results]

    if "License Plate Detection" in selected_tasks:
        lp_results = license_plate_detection_model(proc_frames)
        proc_frames = [res.plot() for res in lp_results]

    if "Fire/Smoke Detection" in selected_tasks:
        fire_results = fire_and_smoke_detection_model(proc_frames)
        proc_frames = [res.plot() for res in fire_results]

    if "Accident Detection" in selected_tasks:
        accident_results = accident_detection_model(proc_frames)
        proc_frames = [res.plot() for res in accident_results]

    final_frames = []
    if "Face Detection" in selected_tasks:
        st.session_state["wanted_detected"] = False
        for frame in proc_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                face_region_bgr = frame[y:y+h, x:x+w]
                face_region_rgb = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2RGB)
                try:
                    analysis = DeepFace.analyze(
                        face_region_rgb,
                        actions=['gender'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    gender = analysis.get('dominant_gender', 'N/A')
                except Exception as e:
                    print("DEBUG: DeepFace analyze error in batch:", e)
                    gender = 'N/A'

                if match_wanted_face(face_region_rgb):
                    st.session_state["wanted_detected"] = True
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{gender}"
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            final_frames.append(frame)
    else:
        final_frames = proc_frames

    return final_frames

# ============================================================
# Logout Functionality
# ============================================================
def logout():
    st.session_state.logged_in = False
    st.session_state.pop("user", None)
    st.experimental_rerun()

# ============================================================
# Application State: Login Screen
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Welcome — please log in or register")
    login_tab, register_tab = st.tabs(["🔑 Login", "📝 Register"])

    # ─── LOGIN TAB ─────────────────────────────────────────────
    with login_tab:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            user = login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user      = user
                st.session_state.email     = user.get("email", "")
                st.success(f"Logged in as {username} ({user.get('role')})")
            else:
                st.error("❌ Invalid username or password.")

    # ─── REGISTER TAB ─────────────────────────────────────────
    with register_tab:
        new_user  = st.text_input("Choose a Username", key="reg_user")
        new_email = st.text_input("Your Email Address", key="reg_email")
        new_pass  = st.text_input("Choose a Password", type="password", key="reg_pass")

        if st.button("Register", key="register_btn"):
            # Basic email format check
            import re
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', new_email):
                st.error("❌ Please enter a valid email address.")
            elif register_user(new_user, new_pass, new_email, role="User", plan="Basic"):
                st.success("✅ Registered successfully! Please switch to the Login tab.")
            else:
                st.error("❌ Username or email already exists.")

    st.stop()

user = st.session_state.user
email = st.session_state.email
role = user.get("role", "User")

# ============================================================
# Dashboard for Customer (User)
# ============================================================
if role == "User":
    dashboard_mode = st.sidebar.radio(
        "Dashboard",
        ["Detection", "Manage Cameras", "Search Database", "Upload Wanted"],
        key="user_dashboard_mode"
    )
    plan = user.get("plan", "Basic")
    allowed_cameras = {"Basic": 5, "Standard": 10, "Premium": 20}.get(plan, 5)
    
    if dashboard_mode == "Detection":
        st.title("Customer Dashboard - AI Surveillance")
        main_mode = st.radio("Select Input Type:", ["Webcam", "Upload Media"], key="user_main_mode")
        output_dir = "processed_media"
        os.makedirs(output_dir, exist_ok=True)

        detection_options = st.sidebar.multiselect(
            "Select detection tasks to perform:",
            options=["Vehicle Detection", "License Plate Detection", "Fire/Smoke Detection", "Accident Detection", "Face Detection", "Gunshot Detection"],
            default=["Face Detection"],
            key="detection_options"
        )
        
        if main_mode == "Webcam":
            st.subheader("Real-time Webcam Analysis")
            webcam_feed = st.camera_input("Capture a photo from your webcam")
            if webcam_feed:
                img_bytes = webcam_feed.getvalue()
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                processed_frame = process_frame(frame, detection_options)
                st.image(processed_frame, channels="BGR", use_column_width=True)
                image_path = os.path.join(output_dir, f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(image_path, processed_frame)
                detections_collection.insert_one({
                    "customer_name": user["customer_name"],
                    "detection_tasks": detection_options,
                    "timestamp": datetime.now(),
                    "file": image_path
                })
                if st.session_state.get("wanted_detected", False):
                    st.warning("Alert: Wanted person detected!")
            st.button("Back to Dashboard", key="back_webcam", on_click=lambda: st.experimental_rerun())

        elif main_mode == "Upload Media":
            st.subheader("Media File Analysis")
            uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "mp4", "avi"], key="upload_file")
            if uploaded_file:
                file_ext = uploaded_file.name.split(".")[-1].lower()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                temp_file.write(uploaded_file.read())
                if file_ext in ["jpg", "png", "jpeg"]:
                    img = cv2.imread(temp_file.name)
                    processed_img = process_frame(img, detection_options)
                    st.image(processed_img, channels="BGR", caption="Processed Image")
                    image_path = os.path.join(output_dir, f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(image_path, processed_img)
                    detections_collection.insert_one({
                        "customer_name": user["customer_name"],
                        "detection_tasks": detection_options,
                        "timestamp": datetime.now(),
                        "file": image_path
                    })
                    if st.session_state.get("wanted_detected", False):
                        st.warning("Alert: Wanted person detected!")
                elif file_ext in ["mp4", "avi"]:
                    st.info("Processing video... This may take a while.")
                    cap = cv2.VideoCapture(temp_file.name)
                    frame_width = int(cap.get(3))
                    frame_height = int(cap.get(4))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    output_video_path = os.path.join(output_dir, f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        processed_frame = process_frame(frame, detection_options)
                        out.write(processed_frame)
                    cap.release()
                    out.release()
                    st.video(output_video_path)
                    record = {
                        "customer_name": user["customer_name"],
                        "detection_tasks": detection_options,
                        "timestamp": datetime.now(),
                        "file": output_video_path
                    }
                    if "Gunshot Detection" in detection_options and gunshot_model:
                        audio_path = os.path.join(output_dir, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                        extract_audio(temp_file.name, audio_path)
                        spec = audio_to_mel_spectrogram(audio_path)
                        gunshot_res = predict_gunshot(spec)
                        record["gunshot_result"] = gunshot_res
                        st.success(f"Gunshot Detection: {gunshot_res}")
                    detections_collection.insert_one(record)
                else:
                    st.error("Unsupported file type.")
            st.button("Back to Dashboard", key="back_upload", on_click=lambda: st.experimental_rerun())

    elif dashboard_mode == "Manage Cameras":
        st.title("Manage Your Cameras")
        cam_docs = list(cameras_collection.find({"customer_name": user["customer_name"]}))
        st.write(f"You are allowed {allowed_cameras} cameras as per your {plan} plan.")
        st.subheader("Your Cameras")
        
        if "user_view_camera" in st.session_state and st.session_state["user_view_camera"]:
            st.info(f"Showing live feed for camera: {st.session_state['user_view_camera']}")
            st.write("This is where a WebRTC stream would be displayed (placeholder).")
            if st.button("Back to Cameras", key="user_cam_back"):
                st.session_state["user_view_camera"] = None
                st.experimental_rerun()
            st.stop()
        else:
            if cam_docs:
                for i, doc in enumerate(cam_docs):
                    camera_name = doc["camera_url"]
                    st.markdown(f"- **{camera_name}**")
                    col1, col2 = st.columns(2)
                    if col1.button("View", key=f"user_view_btn_{i}"):
                        st.session_state["user_view_camera"] = camera_name
                        st.experimental_rerun()
                    if col2.button("Delete", key=f"user_del_btn_{i}"):
                        cameras_collection.delete_one({"_id": doc["_id"]})
                        st.experimental_rerun()
            else:
                st.info("No cameras registered yet.")
            if len(cam_docs) < allowed_cameras:
                new_cam_url = st.text_input("Add a new camera URL", key="new_cam_url_user")
                if st.button("Add Camera", key="add_camera_user"):
                    if new_cam_url:
                        cameras_collection.insert_one({
                            "customer_name": user["customer_name"],
                            "camera_url": new_cam_url,
                            "added_at": datetime.now()
                        })
                        st.success("Camera added!")
                        st.experimental_rerun()
                    else:
                        st.error("Please enter a valid URL.")
            else:
                st.error("You have reached your camera limit.")

    elif dashboard_mode == "Search Database":
        st.title("Search Detection History")
        with st.form("search_form"):
            search_date = st.date_input("Select Date", key="search_date")
            submitted = st.form_submit_button("Search", key="search_btn")
        if submitted:
            query = {"customer_name": user["customer_name"]}
            if search_date:
                query["timestamp"] = {
                    "$gte": datetime.combine(search_date, datetime.min.time()),
                    "$lte": datetime.combine(search_date, datetime.max.time())
                }
            results = list(detections_collection.find(query).sort("timestamp", -1).limit(20))
            if results:
                for doc in results:
                    with st.expander(f"{doc['timestamp']}"):
                        st.write(f"Detection Tasks: {', '.join(doc.get('detection_tasks', []))}")
                        st.write(f"File: {doc.get('file', 'N/A')}")
                        if doc.get("gunshot_result"):
                            st.write(f"Gunshot Detection: {doc.get('gunshot_result')}")
            else:
                st.warning("No records found.")
        st.button("Back to Dashboard", key="back_search", on_click=lambda: st.experimental_rerun())

    elif dashboard_mode == "Upload Wanted":
        st.title("Upload Wanted Images")
        st.write("Upload a wanted face *and/or* enter a wanted license‑plate number:")
        if plan != "Premium":
            st.info("**Premium Feature:** Upgrade to Premium to access this.")
            st.text_input("Enter Wanted Plate Number", value="", disabled=True)
            st.file_uploader("Upload Wanted Face", type=["jpg","png"], disabled=True)
        else:
            plate_number = st.text_input("Enter Wanted Plate Number", key="wanted_lp_number")
            face_file = st.file_uploader("Upload Wanted Face", type=["jpg","png"], key="wanted_face_uploader")
            if plate_number:
                # Avoid duplicates
                if not wanted_lp_collection.find_one({"customer_name":user["customer_name"], "plate": plate_number}):
                    wanted_lp_collection.insert_one({
                        "customer_name": user["customer_name"],
                        "timestamp": datetime.now(),
                        "plate": plate_number
                    })
                    st.success(f"Wanted plate `{plate_number}` saved.")
                else:
                    st.warning("That plate is already in your wanted list.")
            if face_file:
                face_bytes = face_file.read()
                # same duplicate check for face images if you want
                wanted_face_collection.insert_one({
                    "customer_name": user["customer_name"],
                    "timestamp": datetime.now(),
                    "image": face_bytes
                })
                st.success("Wanted face uploaded successfully!")
        st.button("Back to Dashboard", key="back_wanted", on_click=lambda: st.experimental_rerun())


elif role == "Admin":
    st.title("Admin Dashboard")
    if "admin_view_camera" in st.session_state and st.session_state["admin_view_camera"]:
        st.info(f"Showing live feed for camera: {st.session_state['admin_view_camera']}")
        st.write("This is where a WebRTC stream would be displayed.")
        if st.button("Back to Admin Dashboard", key="admin_back_from_view"):
            st.session_state["admin_view_camera"] = None
            st.experimental_rerun()
        st.stop()
    else:
        st.subheader("Customer Management")
        customers = list(users_collection.find({"role": "User"}))
        customer_names = [cust["customer_name"] for cust in customers]
        selected_customer = st.selectbox("Select Customer", options=customer_names, key="admin_select_customer")
        st.write("Manage subscription and camera details for", selected_customer)
        cust_doc = users_collection.find_one({"customer_name": selected_customer})
        current_plan = cust_doc.get("plan", "Not Set")
        st.write(f"Current Plan: {current_plan}")
        new_plan = st.selectbox("Update Plan", options=["Basic", "Standard", "Premium"], key="admin_new_plan")
        if st.button("Update Customer Plan", key="admin_update_plan"):
            users_collection.update_one({"customer_name": selected_customer}, {"$set": {"plan": new_plan}})
            subscriptions_collection.update_one(
                {"customer_name": selected_customer},
                {"$set": {"plan": new_plan, "updated": datetime.now()}},
                upsert=True
            )
            st.success(f"Updated {selected_customer}'s plan to {new_plan}.")
        st.subheader("Customer Camera Management")
        cust_cameras = list(cameras_collection.find({"customer_name": selected_customer}))
        st.write(f"{selected_customer} has {len(cust_cameras)} cameras registered.")
        if cust_cameras:
            for i, cam in enumerate(cust_cameras):
                camera_name = cam["camera_url"]
                st.markdown(f"- **{camera_name}**")
                colA, colB = st.columns(2)
                if colB.button("Delete", key=f"delcam_{i}"):
                    cameras_collection.delete_one({"_id": cam["_id"]})
                    st.experimental_rerun()
        else:
            st.info("No cameras registered for this customer.")
        new_cam = st.text_input("Add a new camera for this customer", key="admin_new_cam")
        if st.button("Add Camera for Customer", key="admin_add_cam_btn"):
            if new_cam:
                cameras_collection.insert_one({
                    "customer_name": selected_customer,
                    "camera_url": new_cam,
                    "added_at": datetime.now()
                })
                st.success("Camera added!")
                st.experimental_rerun()
            else:
                st.error("Please enter a valid camera URL.")
        if st.button("Back to Admin Dashboard", key="admin_back"):
            st.info("Already on admin dashboard.")
