import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / 'test_videos'
VIDEO_FILE = VIDEOS_DIR / 'left_right.mp4'

MP_POSE = mp.solutions.pose
MP_DRAWER = mp.solutions.drawing_utils

def path(path: Path) -> str:
    return str(VIDEO_FILE.absolute())

class PoseEstimator:
    def __init__(self):
        # Initialize MediaPipe Pose model
        self.pose = MP_POSE.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, frame):
        pose = self.pose.process(frame)
        if pose.pose_landmarks:
            frame = self.draw_pose(frame, pose)
        return frame, pose
    
    def draw_pose(self, frame, pose):
        MP_DRAWER.draw_landmarks(frame, pose.pose_landmarks, MP_POSE.POSE_CONNECTIONS)
        return frame

class PositionDetector:
    def __init__(self):
        pass

    # Detect if person is lying down
    def lying_down(self, landmarks, threshold = 0.1):
        if landmarks:
            shoulder = [landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y,
                        landmarks.landmark[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks.landmark[MP_POSE.PoseLandmark.LEFT_HIP.value].y,
                landmarks.landmark[MP_POSE.PoseLandmark.RIGHT_HIP.value].y]
            ankle = [landmarks.landmark[MP_POSE.PoseLandmark.LEFT_ANKLE.value].y,
                    landmarks.landmark[MP_POSE.PoseLandmark.RIGHT_ANKLE.value].y]
            return abs(max(shoulder) - min(ankle)) < threshold


def run_pose_detection(cap):
    person_is_standing = False
    person_was_standing = False

    estimator = PoseEstimator()
    position = PositionDetector()

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    frame_count = 0
    
    speed = None
    previous_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, pose = estimator.process_frame(frame)

        person_is_standing = not position.lying_down(pose.pose_landmarks)
        
        frame_count += 1
        # each second
        if frame_count % 30 == 0:
            if previous_landmarks:
                speed = pose.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y - previous_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y
                print(f"Speed : {speed}")
            previous_landmarks = pose.pose_landmarks
        

        if person_is_standing:
            person_was_standing = person_is_standing
        elif person_was_standing and not person_is_standing and speed > 0.2:
            status_placeholder.markdown("**Alert: A fall has been detected!**")
            person_was_standing = not person_is_standing

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels = 'RGB')

        time.sleep(0.01)

    cap.release()


if __name__ == '__main__':
    st.title("Real-Time Pose Detection and Fall Detection")

    # Buttons for demo and real-time
    if st.button('Start Demo'):
        # Path to the demo video file
        print('Using file', path(VIDEO_FILE))
        video_cap = cv2.VideoCapture(path(VIDEO_FILE))
        run_pose_detection(video_cap)

    if st.button('Start Real-Time'):
        webcam_cap = cv2.VideoCapture(0)
        run_pose_detection(webcam_cap)
