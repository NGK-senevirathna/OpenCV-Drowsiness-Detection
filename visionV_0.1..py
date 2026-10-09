import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

# 1. Model File Check & Auto Download
MODEL_FILE = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

if not os.path.exists(MODEL_FILE):
    print("face_landmarker.task file එක Download වෙමින් පවතී...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Download එක සාර්ථකයි!")

# 2. MediaPipe Tasks API Setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# Distance ගණනය කිරීම සඳහා Utility function එකක්
def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Global Variables
eye_closed_start_time = None
ALARM_THRESHOLD_SECONDS = 1.2  # ඇස් වැසී තත්පර 1.2ක් තිබුණොත් Alarm එක වදී
EYE_AR_THRESHOLD = 0.015       # ඇස් වැසී ඇති බව තීරණය කරන පරතරය

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    print("Driver Drowsiness Detector with Full Mesh Started. Press 'q' to exit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        h, w, _ = frame.shape
        alarm_status = False

        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                # ----------------------------------------------------
                # 1. මුහුණ පුරා Mesh එක (Grid Points & Lines) ඇඳීම
                # ----------------------------------------------------
                # ලක්ෂ්‍ය (Points) කොළ පාටින් කුඩාවට ඇඳීම
                for landmark in face_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                # ----------------------------------------------------
                # 2. Drowsiness Detection Logic (Eye EAR Check)
                # ----------------------------------------------------
                # Left eye: 386 (Top), 374 (Bottom)
                # Right eye: 159 (Top), 145 (Bottom)
                left_eye_top = face_landmarks[386]
                left_eye_bottom = face_landmarks[374]
                right_eye_top = face_landmarks[159]
                right_eye_bottom = face_landmarks[145]

                left_dist = get_distance(left_eye_top, left_eye_bottom)
                right_dist = get_distance(right_eye_top, right_eye_bottom)
                avg_eye_dist = (left_dist + right_dist) / 2.0

                # ඇස් Highlight කිරීමට රවුම් 2ක් ඇඳීම (Visual feedback)
                cv2.circle(frame, (int(left_eye_top.x * w), int(left_eye_top.y * h)), 3, (0, 255, 255), -1)
                cv2.circle(frame, (int(right_eye_top.x * w), int(right_eye_top.y * h)), 3, (0, 255, 255), -1)

                # ඇස් පියවී ඇත්දැයි පරීක්ෂා කිරීම
                if avg_eye_dist < EYE_AR_THRESHOLD:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    else:
                        elapsed_time = time.time() - eye_closed_start_time
                        
                        # ⚠️ තත්පර 1.2කට වඩා ඇස් පියවී තිබේ නම් ALARM!
                        if elapsed_time >= ALARM_THRESHOLD_SECONDS:
                            alarm_status = True
                            winsound.Beep(2500, 250)  # Frequency: 2500Hz, Duration: 250ms
                else:
                    eye_closed_start_time = None

                # Screen එක මත Status Alert පෙන්වීම
                if alarm_status:
                    cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    # රතු පාටින් මුළු Screen එකම Warning Border එකක් දැමීම
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                else:
                    cv2.putText(frame, "Driver Alert: Normal", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Driver Safety System (Full Mesh + Alarm)", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()