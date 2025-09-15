from flask import Flask, render_template, request, redirect, url_for, session, send_file
import cv2
import numpy as np
import os
from datetime import datetime
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics import YOLO
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import librosa
import pymongo
from tensorflow.keras.models import load_model
import tempfile

# Add safe globals for YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.head import Detect

add_safe_globals([DetectionModel, Sequential, Conv, C2f, Detect])

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MongoDB Setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["surveillance_db"]
detections_collection = db["detections"]
subscriptions_collection = db["subscriptions"]

# Load Models
try:
    vehicle_detection_model = YOLO("yolov8m.pt")
except Exception as e:
    print(f"Error loading vehicle detection model: {str(e)}")
    exit()

try:
    license_plate_detection_model = YOLO(r"F:\camera\web\models\license_plate_detection\weights\best.pt")
except Exception as e:
    print(f"Error loading license plate detection model: {str(e)}")

try:
    fire_and_smoke_detection_model = YOLO(r"F:\camera\web\models\fire_detection\weights\best.pt")
except Exception as e:
    print(f"Error loading fire/smoke detection model: {str(e)}")

try:
    accident_detection_model = YOLO(r"F:\camera\accident detection\delete\best.pt")
except Exception as e:
    print(f"Error loading accident detection model: {str(e)}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    gunshot_model = load_model(r"F:\camera\gun shot\gunshot_model_epoch_05_acc_0.90_valacc_0.96.h5")
except Exception as e:
    print("Gunshot model not found. Gunshot detection will be disabled.")
    gunshot_model = None

# Utility Functions (keep same as original)
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

def process_frame(frame, selected_tasks):
    processed = frame.copy()
    if "Vehicle Detection" in selected_tasks:
        results = vehicle_detection_model(processed)
        processed = results[0].plot()
    if "License Plate Detection" in selected_tasks:
        results = license_plate_detection_model(processed)
        processed = results[0].plot()
    if "Fire/Smoke Detection" in selected_tasks:
        results = fire_and_smoke_detection_model(processed)
        processed = results[0].plot()
    if "Accident Detection" in selected_tasks:
        results = accident_detection_model(processed)
        processed = results[0].plot()
    if "Face Detection" in selected_tasks:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(processed, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return processed

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['detection_options'] = request.form.getlist('detection_tasks')
        session['customer_name'] = request.form.get('customer_name', 'default')
        
        if 'admin_mode' in request.form:
            if request.form.get('admin_password') == 'admin123':
                new_plan = request.form.get('new_plan')
                subscriptions_collection.update_one(
                    {"customer_name": session['customer_name']},
                    {"$set": {"plan": new_plan}},
                    upsert=True
                )
        return redirect(url_for('process'))
    
    return render_template('index.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                
                # Image processing
                if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(file_path)
                    processed_img = process_frame(img, session.get('detection_options', []))
                    output_path = os.path.join(temp_dir, 'processed.jpg')
                    cv2.imwrite(output_path, processed_img)
                    return send_file(output_path, mimetype='image/jpeg')
                
                # Video processing
                elif file.filename.lower().endswith(('.mp4', '.avi')):
                    cap = cv2.VideoCapture(file_path)
                    frame_width = int(cap.get(3))
                    frame_height = int(cap.get(4))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    output_path = os.path.join(temp_dir, 'processed.mp4')
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                                        (frame_width, frame_height))
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        processed_frame = process_frame(frame, session.get('detection_options', []))
                        out.write(processed_frame)
                    
                    cap.release()
                    out.release()
                    
                    # Gunshot detection
                    if "Gunshot Detection" in session.get('detection_options', []) and gunshot_model:
                        audio_path = os.path.join(temp_dir, "temp_audio.wav")
                        extract_audio(file_path, audio_path)
                        spectrogram = audio_to_mel_spectrogram(audio_path)
                        result = predict_gunshot(spectrogram)
                        # Store result in session
                        session['gunshot_result'] = result
                    
                    return send_file(output_path, mimetype='video/mp4')
        
        # Webcam processing
        if 'webcam_image' in request.files:
            file = request.files['webcam_image']
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, 'webcam.jpg')
            file.save(file_path)
            
            img = cv2.imread(file_path)
            processed_img = process_frame(img, session.get('detection_options', []))
            output_path = os.path.join(temp_dir, 'processed.jpg')
            cv2.imwrite(output_path, processed_img)
            
            # Store in MongoDB
            detections_collection.insert_one({
                "customer_name": session.get('customer_name'),
                "timestamp": datetime.now(),
                "detection_tasks": session.get('detection_options'),
                "file": output_path
            })
            
            return send_file(output_path, mimetype='image/jpeg')
    
    return render_template('process.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_customer = request.form.get('search_customer')
        search_date = request.form.get('search_date')
        
        query = {}
        if search_customer:
            query["customer_name"] = search_customer
        if search_date:
            query_date = datetime.strptime(search_date, '%Y-%m-%d')
            query["timestamp"] = {
                "$gte": datetime.combine(query_date, datetime.min.time()),
                "$lte": datetime.combine(query_date, datetime.max.time())
            }
        
        results = list(detections_collection.find(query).sort("timestamp", -1).limit(20))
        return render_template('search_results.html', results=results)
    
    return render_template('search.html')

if __name__ == '__main__':
    app.run()