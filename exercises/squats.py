import cv2
import mediapipe as mp
import numpy as np
import math 
import pyttsx3
import time
from collections import deque

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize variables for squat counting
counter = 0
stage = None
knee_angle_threshold = 120  # Threshold for squat detection (in degrees)
last_voice_time = 0
voice_cooldown = 5  # Increased cooldown between voice prompts

# Form correction parameters
min_squat_angle = 70  # Minimum angle for proper depth
back_angle_threshold = 160  # Threshold for back straightness
knee_forward_threshold = 0.1  # Threshold for knees going too far forward

# Smoothing parameters
angle_buffer_size = 5
angle_buffer = deque(maxlen=angle_buffer_size)
form_check_frequency = 15  # Reduced form check frequency
frame_counter = 0
previous_corrections = []

# Voice feedback settings - minimal speaking
speak_milestone = 10  # Only announce every 10 reps
speak_form_correction_counter = 0
form_correction_speak_threshold = 5  # Only speak form correction if seen 5 consecutive times

def calculate_angle(a, b, c):
    """Calculate the angle between three points (in degrees)"""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (joint)
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def speak_feedback(text):
    """Provide voice feedback with cooldown"""
    global last_voice_time
    current_time = time.time()
    if current_time - last_voice_time >= voice_cooldown:
        engine.say(text)
        engine.runAndWait()
        last_voice_time = current_time

def get_smoothed_angle():
    """Return the smoothed angle from the buffer"""
    if not angle_buffer:
        return 0
    return sum(angle_buffer) / len(angle_buffer)

def check_form(landmarks, knee_angle):
    """Check squat form and return corrections"""
    global frame_counter, previous_corrections, speak_form_correction_counter
    
    # Only check form periodically to avoid jitter in feedback
    frame_counter += 1
    if frame_counter % form_check_frequency != 0:
        return previous_corrections
    
    corrections = []
    
    try:
        # Get coordinates for form analysis
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Check squat depth
        if stage == "down" and knee_angle > min_squat_angle:
            corrections.append("Squat deeper")
        
        # Check back angle (should be straight)
        back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        if back_angle < back_angle_threshold:
            corrections.append("Keep back straight")
        
        # Check if knees go too far forward over toes
        if left_knee[0] > left_ankle[0] + knee_forward_threshold:
            corrections.append("Knees too far forward")
        
        # Only voice feedback for persistent form issues
        if corrections and corrections == previous_corrections:
            speak_form_correction_counter += 1
            if speak_form_correction_counter >= form_correction_speak_threshold:
                speak_feedback(corrections[0])
                speak_form_correction_counter = 0
        else:
            speak_form_correction_counter = 0
            
        previous_corrections = corrections
    except:
        pass
        
    return corrections

# Set up video capture with proper resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Setup MediaPipe Pose instance with higher confidence thresholds
with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1) as pose:
    
    # Minimal initial greeting
    speak_feedback("Squat counter ready")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for better performance
        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        
        # Recolor frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Get coordinates for the right leg
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Calculate knee angle
                current_angle = calculate_angle(hip, knee, ankle)
                
                # Add to smoothing buffer
                angle_buffer.append(current_angle)
                
                # Get smoothed angle
                smoothed_angle = get_smoothed_angle()
                
                # Visualize angle
                angle_coords = tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int))
                cv2.putText(image, f'Angle: {smoothed_angle:.1f}', 
                           angle_coords,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Squat counter logic with minimal voice feedback
                if smoothed_angle < knee_angle_threshold and stage != "down":
                    stage = "down"
                    
                if smoothed_angle > knee_angle_threshold and stage == "down":
                    stage = "up"
                    counter += 1
                    # Only announce milestones
                    if counter % speak_milestone == 0:
                        speak_feedback(f"{counter}")
                
                # Ongoing form check
                if stage is not None:
                    corrections = check_form(landmarks, smoothed_angle)
                    
                    # Display form corrections on screen
                    for i, correction in enumerate(corrections):
                        cv2.putText(image, correction, (10, 110 + 30*i), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    
            except:
                pass
            
            # Draw pose landmarks with improved visibility
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style())
            
        # Status area with semi-transparent background
        status_area = image.copy()
        cv2.rectangle(status_area, (0, 0), (300, 100), (0, 0, 0), -1)
        image = cv2.addWeighted(status_area, 0.3, image, 0.7, 0)
        
        # Display counter and status
        cv2.putText(image, f'Squats: {counter}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display current stage with color coding
        stage_color = (0, 255, 0) if stage == "up" else (0, 165, 255) if stage == "down" else (255, 255, 255)
        cv2.putText(image, f'Stage: {stage if stage else "ready"}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, stage_color, 2, cv2.LINE_AA)
        
        # Add progress bar
        if counter > 0:
            progress_width = min(counter * 20, image.shape[1] - 20)
            cv2.rectangle(image, (10, image.shape[0] - 30), (10 + progress_width, image.shape[0] - 20), (0, 255, 0), -1)
        
        # Display the frame
        cv2.imshow('Squat Counter', image)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Minimal closing feedback
    if counter > 0:
        speak_feedback(f"Complete. {counter} squats.")
            
    cap.release()
    cv2.destroyAllWindows()