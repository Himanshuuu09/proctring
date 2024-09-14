import cv2
import face_recognition
import numpy as np
from fer import FER
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import mediapipe as mp
from flask_cors import CORS
from threading import Thread

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=["http://127.0.0.1:5000"])
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000"]}})

emotion_detector = FER()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

reference_frame = None
reference_encoding = None
frame_counter = 0
background_frame_interval = 20
face_detection_interval = 5
emotion_detection_interval = 10
background_reference = None

def get_face_encoding(frame, detection, results_dict):
    h, w, _ = frame.shape
    box = detection.location_data.relative_bounding_box
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    width = int(box.width * w)
    height = int(box.height * h)
    face_roi = frame[y:y + height, x:x + width]
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    face_encodings = face_recognition.face_encodings(face_rgb)
    
    if len(face_encodings) == 0:
        results_dict['No_faces'] = 1
        emit('message', {'text': "No face encoding found in the video frame."})
        return None  # Return None if no encodings are found
    
    return face_encodings[0]

def compare_faces(reference_encoding, video_frame, detection, results_dict):
    h, w, _ = video_frame.shape
    box = detection.location_data.relative_bounding_box
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    width = int(box.width * w)
    height = int(box.height * h)
    face_roi = video_frame[y:y + height, x:x + width]
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    face_encodings = face_recognition.face_encodings(face_rgb)
    
    if len(face_encodings) == 0:
        results_dict['No_faces'] = 1
        emit('message', {'text': "No face encoding found in the video frame."})
        return None
    
    match = face_recognition.compare_faces([reference_encoding], face_encodings[0], tolerance=0.6)
    return match[0]

def detect_emotion(video_frame, detection):
    try:
        h, w, _ = video_frame.shape
        box = detection.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        width = int(box.width * w)
        height = int(box.height * h)
        face_roi = video_frame[y:y + height, x:x + width]
        emotion_data = emotion_detector.detect_emotions(face_roi)
        if not emotion_data:
            emotion = "neutral"
            return emotion
        top_emotion = emotion_detector.top_emotion(face_roi)
        if top_emotion:
            emotion, score = top_emotion
            return emotion
    except Exception as e:
        return f"Error during emotion detection: {e}"

def check_background_change(video_frame, reference_frame, threshold=30):
    diff = cv2.absdiff(video_frame, reference_frame)
    mean_diff = np.mean(diff)
    emit('message', {'text': f"stable background"})
    return mean_diff > threshold

@socketio.on('video_data')
def handle_video_frame(data):
    global reference_frame, reference_encoding, frame_counter, background_reference

    np_data = np.frombuffer(base64.b64decode(data), np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        emit('message', {'text': 'Invalid or empty frame received.'})
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    frame_counter += 1
    results_dict = {
        'No_faces': 0,
        'multifaces': 0,
        'background_change': 0,
        'emotion': "neutral",
        'face_mismatch': 0
    }

    results = None  # Initialize results variable

    if frame_counter % face_detection_interval == 0:
        results = face_detection.process(rgb_frame)

        if results is None or results.detections is None:
            emit('message', {'text': 'Error processing frame or no faces detected.'})
            results_dict['No_faces'] = 1
            results_dict['emotion'] = None
            print(results_dict)
            return

        if reference_frame is None or frame_counter % 50 == 0:
            if len(results.detections) > 1:
                emit('message', {'text': "Multiple faces detected in the reference frame."})
                results_dict['multifaces'] = 1

            detection = results.detections[0]
            reference_encoding = get_face_encoding(frame, detection, results_dict)
            if reference_encoding is None:  # Check if reference_encoding was set
                return
            reference_frame = frame
            background_reference = frame
            emit('message', {'text': "Reference frame set."})
        if len(results.detections) == 1:
            face_match = compare_faces(reference_encoding, frame, results.detections[0], results_dict)
            if face_match is None:
                results_dict['No_faces']=1  
            elif not face_match:
                emit('message', {'text': "Face mismatch detected. Possible cheating!"})
                results_dict['face_mismatch'] = 1
    if background_reference is not None and frame_counter % background_frame_interval == 0:
        if check_background_change(frame, background_reference):
            emit('message', {'text': "Significant background change detected. Possible cheating!"})
            results_dict['background_change'] = 1
    if frame_counter % emotion_detection_interval == 0:
        if results is not None and results.detections is not None and len(results.detections) == 1:
            emotion = detect_emotion(frame, results.detections[0])
            results_dict['emotion'] = emotion

    if results is not None and results.detections is not None and len(results.detections) == 1:
        detection = results.detections[0]
        box = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        width = int(box.width * w)
        height = int(box.height * h)
        face_data = {"x": x, "y": y, "width": width, "height": height}
        emit('result', {"face": face_data})
        print(results_dict)
        return {"result": results_dict, "success": True}, 200
    else:
        emit('result', {"face": None})
        return {"result": "no data found", "success": False}, 400

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)