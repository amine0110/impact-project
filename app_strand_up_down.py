import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.title("Real-Time Pose Detection and Fall Detection")

def process_frame(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame

# Detect if person is lying down
def is_lying_down(landmarks, threshold=0.1):
    if landmarks:
        shoulder = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y,
               landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        ankle = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                 landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        print(abs(max(shoulder) - min(ankle)))
        return abs(max(shoulder) - min(ankle)) < threshold
    return False

def run_pose_detection(cap):
    is_person_standing = False
    was_person_standing = False

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame = process_frame(frame, results)

        if is_lying_down(results.pose_landmarks):
            is_person_standing = False
        else:
            is_person_standing = True

        if was_person_standing and not is_person_standing:
            status_placeholder.markdown("**Alert: A fall has been detected!**")
            was_person_standing = False
        elif is_person_standing:
            was_person_standing = True

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels='RGB')

        time.sleep(0.01)

    cap.release()

# Buttons for demo and real-time
if st.button('Start Demo'):
    # Path to the demo video file
    demo_video_path = './test_videos/left_right.mp4'
    video_cap = cv2.VideoCapture(demo_video_path)
    run_pose_detection(video_cap)

if st.button('Start Real-Time'):
    webcam_cap = cv2.VideoCapture(0)
    run_pose_detection(webcam_cap)
