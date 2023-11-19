import streamlit as st
from utils.yolo_tiny_main import yolo_detect
from utils.mediapipe_main import run_pose_detection
from streamlit_lottie import st_lottie
import requests
from codecarbon import EmissionsTracker

class CHUAssistant:
    def __init__(self):
        self.video_path = './test_videos/left_right.mp4'
        self.webcam_str = '0'
        self.webcam_int = 0
        self.lottie_animation = self.load_lottieurl("https://lottie.host/2846e1e3-bd70-424d-9a7d-47eacb42b3d4/iyJTZj8HpU.json")
        self.initialize_ui()
    
    def load_lottieurl(self, url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    def process_video_tiny(self):
        tracker = EmissionsTracker(project_name="yolo_test", measure_power_secs=10)
        tracker.start()
        yolo_detect(self.video_path)
        emission = tracker.stop()
        print('Yolo Emission: ', emission)
    
    def process_video_mediapipe(self):
        tracker = EmissionsTracker(project_name="mediapipe_test", measure_power_secs=10)
        tracker.start()
        run_pose_detection(self.video_path)
        emission = tracker.stop()
        print('MediaPipe Emission: ', emission)

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
            st_lottie(self.lottie_animation, key="animation",height=500, width=500)
        elif app_mode == "YOLO-Tiny Integration":
            self.yolo_tiny_integration()
        elif app_mode == "CHU-Assist":
            self.chu_assist()

if __name__ == "__main__":
    app = CHUAssistant()
