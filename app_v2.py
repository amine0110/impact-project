import streamlit as st

class CHUAssistant:
    def __init__(self):
        self.video_path = './test_videos/left_right.mp4'
        self.initialize_ui()

    def process_video(self, video_path):
        st.video(video_path)

    def access_webcam(self):
        st.write("Webcam access not implemented yet.")

    def yolo_tiny_integration(self):
        st.subheader("YOLO-Tiny Integration")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Launch YOLO-Tiny Demo Video"):
                self.process_video(self.video_path)
        with col2:
            if st.button("Launch YOLO-Tiny Webcam"):
                self.access_webcam()

    def chu_assist(self):
        st.subheader("CHU-Assist")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Launch CHU-Assist Demo Video"):
                self.process_video(self.video_path)
        with col2:
            if st.button("Launch CHU-Assist Webcam"):
                self.access_webcam()

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
