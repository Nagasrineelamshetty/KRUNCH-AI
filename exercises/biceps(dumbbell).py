import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

def speak(text):
    """Function to speak a given text."""
    engine.say(text)
    engine.runAndWait()

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Initialize Variables
counter = 0
stage = None
started = False
prev_angle = 0
form_issue = ""

# Thresholds
CURL_DOWN_ANGLE = 150
CURL_UP_ANGLE = 70
SMOOTHING_FACTOR = 0.5  # Angle smoothing

# Video capture
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                # Get coordinates for right arm
                shoulder = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
                          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y]
                elbow = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x,
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y]
                wrist = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x,
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y]
                
                # Calculate angle with smoothing
                current_angle = calculate_angle(shoulder, elbow, wrist)
                smoothed_angle = (current_angle * SMOOTHING_FACTOR) + (prev_angle * (1 - SMOOTHING_FACTOR))
                angle = smoothed_angle
                prev_angle = angle
                
                # Form correction logic
                form_issue = ""
                if elbow[1] < shoulder[1] - 0.1:  # If elbow is too high
                    form_issue = "Lower your elbow."
                elif wrist[1] > elbow[1] + 0.1:  # If wrist is too low
                    form_issue = "Keep your wrist aligned."

                # Improved rep counting logic
                if not started and angle > CURL_DOWN_ANGLE:
                    started = True
                    stage = 'down'

                if started:
                    if stage == 'down' and angle < CURL_UP_ANGLE:
                        stage = 'up'
                    elif stage == 'up' and angle > CURL_DOWN_ANGLE:
                        stage = 'down'
                        counter += 1
                        speak(f"Rep {counter}")  # Speak rep count

                # Voice feedback for form correction
                if form_issue:
                    speak(form_issue)

                # Visual feedback
                color = (255, 0, 0) if stage == 'down' else (0, 255, 0)
                
                # Draw arm angle
                cv2.putText(image, f'Angle: {int(angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2))

        except Exception as e:
            print(f"Error: {e}")

        # Display counter, stage, and form correction
        cv2.putText(image, f'Reps: {counter}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Stage: {stage}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, form_issue, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Add instructions
        cv2.putText(image, 'Stand sideways to camera', (10, image.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, 'Press q to quit', (10, image.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show image
        cv2.imshow('Bicep Curl Counter with Voice & Form Correction', image)

        # Break loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()