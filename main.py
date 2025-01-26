import cv2
import streamlit as st
import time
import mediapipe as mp
import numpy as np

# Define a function to handle camera access
def check_camera_access():
    cap = None
    for i in range(3):  # Try indices 0, 1, and 2 to find the active camera
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap, i
    return None, -1

# FRONTEND
scanner = False

# Set Streamlit page configuration for wide layout
st.set_page_config(
    page_title="PABLO",
    page_icon="🖥️",
    layout="wide",
)

# CSS Styling for the app
st.markdown(
    """
    <style>
        h1 { font-size: 24px; color: #333; text-align: center; animation: fade-in 2s ease-in-out; }
        p { font-size: 16px; color: #555; }
        .fade-in { animation: fade-in 1.5s ease-in-out; }
        @keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
        .penguin-container { position: relative; top: -50px; left: 20px; }
        .button-container { position: relative; top: -50px; left: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Lottie animation
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error: {e}")
        return None

# Frontend navigation logic
col1, col2, _ = st.columns([9, 1, 1])  # Allocate more space for the empty column on the right

with col1:
    if st.button("Home"):
        st.switch_page("main.py")  # Replace with the appropriate file name for the home page

with col2:
    if st.button("About"):
        st.switch_page("pages/page_about.py")  # Replace with the appropriate file name for the about page

# Check camera availability before starting Streamlit's webcam stream
st.header("Webcam Access Test")
cap, camera_index = check_camera_access()

if cap is None:
    st.error(f"Unable to access any webcam. Please check your camera connections or browser permissions.")
else:
    st.success(f"Camera found at index {camera_index}.")
    # Proceed with webcam access, video processing logic here

# Class to track body and face using Mediapipe
class BodyTracker:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.mp_drawing = mp.solutions.drawing_utils
        self.keypoints = {}

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        return frame

    def release(self):
        self.holistic.close()
        self.face_mesh.close()

# If camera is available, proceed with video streaming and analysis
if cap and cap.isOpened():
    frame_placeholder = st.empty()

    stime = time.time()
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)  # Simulate progress
        progress_bar.progress(i + 1)
    
    progress_bar = None
    tracker = BodyTracker()

    running = True
    stop_tracking = st.button("Stop Tracking")

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        frame = tracker.process_frame(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        if stop_tracking:
            st.header("Please wait a moment, PABLO is analyzing your performance!")
            tottime = time.time() - stime
            running = False
            tracker.release()
            cap.release()

    if stop_tracking:
        st.write("Feedback for your performance will go here.")
else:
    st.warning("Camera is not accessible. Please check your browser permissions or try a different camera.")
