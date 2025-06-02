import cv2
import mediapipe as mp
import numpy as np
import torch

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load video or webcam feed
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' for file input

# Define a function to analyze batter's stance
def analyze_stance(landmarks, image):
    if landmarks:
        # Extract key points (adjust based on needed joints)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate angles for stance analysis
        shoulder_angle = np.arctan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x) * 180 / np.pi
        hip_angle = np.arctan2(left_hip.y - right_hip.y, left_hip.x - right_hip.x) * 180 / np.pi

        # Print angles for debugging
        cv2.putText(image, f"Shoulder Angle: {shoulder_angle:.2f}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"Hip Angle: {hip_angle:.2f}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Stance validation logic (modify as per your analysis needs)
        if abs(shoulder_angle) < 10 and abs(hip_angle) < 10:
            cv2.putText(image, "Good Stance", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
        else:
            cv2.putText(image, "Adjust Your Stance!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

# OpenCV video loop
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            analyze_stance(result.pose_landmarks.landmark, image)

        cv2.imshow('Batter Stance Analysis', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()