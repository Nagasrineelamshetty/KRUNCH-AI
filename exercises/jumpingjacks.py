import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
from collections import deque

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

counter = 0
stage = None
last_voice_time = 0
voice_cooldown = 5
speak_milestone = 10

form_check_frequency = 15
frame_counter = 0
previous_corrections = []
speak_form_correction_counter = 0
form_correction_speak_threshold = 5

# Smoothing for distances
buffer_size = 5
hand_dist_buffer = deque(maxlen=buffer_size)
leg_dist_buffer = deque(maxlen=buffer_size)

def speak_feedback(text):
    global last_voice_time
    current_time = time.time()
    if current_time - last_voice_time >= voice_cooldown:
        engine.say(text)
        engine.runAndWait()
        last_voice_time = current_time

def get_smoothed(buffer):
    if not buffer:
        return 0
    return sum(buffer) / len(buffer)

def check_jack_form(landmarks, hands_dist, legs_dist, stage):
    global frame_counter, previous_corrections, speak_form_correction_counter

    frame_counter += 1
    if frame_counter % form_check_frequency != 0:
        return previous_corrections

    corrections = []
    try:
        # Coords: left, right wrists; left, right ankles; left, right shoulders
        lw = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        rw = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        la = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        ra = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        rs = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        # Hands above head?
        head_y = min(ls[1], rs[1])
        hands_above_head = lw[1] < head_y and rw[1] < head_y

        # Hands together above head?
        hands_together = np.linalg.norm(np.array(lw) - np.array(rw)) < 0.12

        # Legs wide enough? (threshold is shoulder width * 1.5)
        shoulder_width = np.linalg.norm(np.array(ls) - np.array(rs))
        legs_apart = np.linalg.norm(np.array(la) - np.array(ra)) > (shoulder_width * 1.4)

        if stage == "open":
            if not hands_above_head:
                corrections.append("Raise arms above head")
            if not hands_together:
                corrections.append("Bring hands together above head")
            if not legs_apart:
                corrections.append("Spread legs wider")

        if corrections and corrections == previous_corrections:
            speak_form_correction_counter += 1
            if speak_form_correction_counter >= form_correction_speak_threshold:
                speak_feedback(corrections[0])
                speak_form_correction_counter = 0
        else:
            speak_form_correction_counter = 0

        previous_corrections = corrections
    except Exception:
        pass

    return corrections

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1) as pose:
    speak_feedback("Jumping jack counter ready")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Get keypoint coords
                lw = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                rw = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                la = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                ra = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                rs = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                # Smoothed distances
                hands_dist = np.linalg.norm(np.array(lw) - np.array(rw))
                legs_dist = np.linalg.norm(np.array(la) - np.array(ra))
                hand_dist_buffer.append(hands_dist)
                leg_dist_buffer.append(legs_dist)
                smoothed_hands = get_smoothed(hand_dist_buffer)
                smoothed_legs = get_smoothed(leg_dist_buffer)

                # Rep logic thresholds
                # "Open" when hands far apart & high above head, legs apart
                # "Closed" when hands low and close, legs together
                hand_high_y = (lw[1] + rw[1]) / 2
                shoulders_y = (ls[1] + rs[1]) / 2

                shoulder_width = np.linalg.norm(np.array(ls) - np.array(rs))
                legs_apart = smoothed_legs > (shoulder_width * 1.4)
                hands_apart = smoothed_hands > (shoulder_width * 1.2)
                hands_together = smoothed_hands < (shoulder_width * 0.7)

                hands_above = hand_high_y < shoulders_y - 0.08  # hands clearly above shoulders

                if hands_above and legs_apart and hands_apart and stage != "open":
                    stage = "open"
                if (not hands_above or not legs_apart or hands_together) and stage == "open":
                    stage = "closed"
                    counter += 1
                    if counter % speak_milestone == 0:
                        speak_feedback(f"{counter}")

                # Form check
                if stage is not None:
                    corrections = check_jack_form(landmarks, smoothed_hands, smoothed_legs, stage)
                    for i, correction in enumerate(corrections):
                        cv2.putText(image, correction, (10, 110 + 30*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            except Exception:
                pass

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style())

        # Status area
        status_area = image.copy()
        cv2.rectangle(status_area, (0, 0), (300, 100), (0, 0, 0), -1)
        image = cv2.addWeighted(status_area, 0.3, image, 0.7, 0)

        cv2.putText(image, f'Jacks: {counter}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        stage_color = (0, 255, 0) if stage == "closed" else (0, 165, 255) if stage == "open" else (255, 255, 255)
        cv2.putText(image, f'Stage: {stage if stage else "ready"}', (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, stage_color, 2, cv2.LINE_AA)

        if counter > 0:
            progress_width = min(counter * 20, image.shape[1] - 20)
            cv2.rectangle(image, (10, image.shape[0] - 30), (10 + progress_width, image.shape[0] - 20), (0, 255, 0), -1)

        cv2.imshow('Jumping Jack Counter', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if counter > 0:
        speak_feedback(f"Complete. {counter} jumping jacks.")

    cap.release()
    cv2.destroyAllWindows()