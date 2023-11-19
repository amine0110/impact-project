import streamlit as st
from yolo_tiny_main import yolo_detect
from mediapipe_main import run_pose_detection
from streamlit_lottie import st_lottie
import requests

class CHUAssistant:
    def __init__(self):
        self.video_path = './test_videos/left_right.mp4'
        self.webcam_str = '0'
        self.webcam_int = 0
        self.initialize_ui()

    def process_video_tiny(self):
        yolo_detect(self.video_path)
    
    def process_video_mediapipe(self):
        run_pose_detection(self.video_path)

    def access_webcam_tiny(self):
        yolo_detect(self.webcam_str)
    
    def access_webcam_mediapipe(self):
        run_pose_detection(self.webcam_int)

    def yolo_tiny_integration(self):
        st.subheader("YOLO-Tiny Integration")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Launch YOLO-Tiny Demo Video"):
                self.process_video_tiny()
        with col2:
            if st.button("Launch YOLO-Tiny Webcam"):
                self.access_webcam_tiny()

    def chu_assist(self):
        st.subheader("CHU-Assist")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Launch CHU-Assist Demo Video"):
                self.process_video_mediapipe()
        with col2:
            if st.button("Launch CHU-Assist Webcam"):
                self.access_webcam_mediapipe()

    def initialize_ui(self):
        st.title("AI Assistant for your patients")

        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio("Choose the App Mode", 
                                    ["Home", "YOLO-Tiny Integration", "CHU-Assist"])

        if app_mode == "Home":
            st.write("Welcome to the platform")
        elif app_mode == "YOLO-Tiny Integration":
            self.yolo_tiny_integration()
        elif app_mode == "CHU-Assist":
            self.chu_assist()

if __name__ == "__main__":
    app = CHUAssistant()
