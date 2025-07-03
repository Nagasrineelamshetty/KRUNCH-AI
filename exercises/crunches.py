import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from threading import Thread
from time import time

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking rate

def speak_feedback(text):
    """Speak feedback in a separate thread to avoid blocking"""
    def speak():
        engine.say(text)
        engine.runAndWait()
    Thread(target=speak).start()

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Initialize Variables
counter = 0
stage = None
started = False
prev_angle = 0
last_voice_time = 0
VOICE_COOLDOWN = 2  # Seconds between voice feedback

# Thresholds
CRUNCH_DOWN_ANGLE = 160  # Angle when lying flat
CRUNCH_UP_ANGLE = 130    # Angle when crunched up
SMOOTHING_FACTOR = 0.5   # For angle smoothing

# Video capture
cap = cv2.VideoCapture(0)

# Initial voice guidance
speak_feedback("Starting crunch counter. Position yourself with camera on your side")

with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_feedback = None
        current_time = time()
        
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get key points for core crunch movement
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                
                # Calculate crunch angle
                current_angle = calculate_angle(shoulder, hip, knee)
                smoothed_angle = (current_angle * SMOOTHING_FACTOR) + (prev_angle * (1 - SMOOTHING_FACTOR))
                angle = smoothed_angle
                prev_angle = angle
                
                # Check core engagement
                if stage == 'up' and angle > CRUNCH_UP_ANGLE + 20:
                    form_feedback = "Engage your core more"
                
                # Rep counting logic
                if not started and angle > CRUNCH_DOWN_ANGLE:
                    started = True
                    stage = 'down'
                    if current_time - last_voice_time > VOICE_COOLDOWN:
                        speak_feedback("Start position")
                        last_voice_time = current_time
                
                if started:
                    if stage == 'down' and angle < CRUNCH_UP_ANGLE:
                        stage = 'up'
                        counter += 1
                        if current_time - last_voice_time > VOICE_COOLDOWN:
                            speak_feedback(f"Rep {counter}")
                            last_voice_time = current_time
                    elif stage == 'up' and angle > CRUNCH_DOWN_ANGLE:
                        stage = 'down'
                
                # Voice form feedback
                if form_feedback and current_time - last_voice_time > VOICE_COOLDOWN:
                    speak_feedback(form_feedback)
                    last_voice_time = current_time
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2))
                
        except Exception as e:
            print(f"Error: {e}")
            pass
            
        # Display metrics
        cv2.putText(image, f'Reps: {counter}', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Stage: {stage}', 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display form feedback
        if form_feedback:
            cv2.putText(image, form_feedback, 
                        (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(image, 'Position camera to see full body side view', 
                    (10, image.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show image
        cv2.imshow('Crunch Counter with Voice Feedback', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            speak_feedback(f"Workout complete. Total reps: {counter}")
            break

cap.release()
cv2.destroyAllWindows()