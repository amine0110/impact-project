import cv2
import mediapipe as mp
import time
from pathlib import Path
import streamlit as st
import simpleaudio as sa

# MediaPipe components
MP_POSE = mp.solutions.pose
MP_DRAWER = mp.solutions.drawing_utils

class PoseEstimator:
    def __init__(self):
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

    def lying_down(self, landmarks, threshold=0.1):
        if landmarks:
            shoulder = [landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y,
                        landmarks.landmark[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks.landmark[MP_POSE.PoseLandmark.LEFT_HIP.value].y,
                   landmarks.landmark[MP_POSE.PoseLandmark.RIGHT_HIP.value].y]
            ankle = [landmarks.landmark[MP_POSE.PoseLandmark.LEFT_ANKLE.value].y,
                     landmarks.landmark[MP_POSE.PoseLandmark.RIGHT_ANKLE.value].y]
            return abs(max(shoulder) - min(ankle)) < threshold
        return False

def play_alert_sound():
    ALERT_SOUND_PATH = "./assets/alert.wav"
    wave_obj = sa.WaveObject.from_wave_file(ALERT_SOUND_PATH)
    wave_obj.play()

def run_pose_detection(video_source):
    frame_placeholder, status_placeholder = st.empty(), st.empty()
    speed = None
    estimator = PoseEstimator()
    position_detector = PositionDetector()

    cap = cv2.VideoCapture(video_source)

    frame_count = 0
    person_was_standing = False
    previous_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, pose = estimator.process_frame(frame)

        person_is_standing = not position_detector.lying_down(pose.pose_landmarks)

        frame_count += 1
        if frame_count % 30 == 0 and previous_landmarks:
            speed = pose.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y - \
                    previous_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y
            previous_landmarks = pose.pose_landmarks

        if person_is_standing:
            person_was_standing = person_is_standing
        elif person_was_standing and not person_is_standing:
            status_placeholder.error("⚠️ Alert: A fall has been detected!")
            play_alert_sound()
            person_was_standing = not person_is_standing

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels='RGB')
        time.sleep(0.01)

    cap.release()
