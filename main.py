import time
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import json
from streamlit_lottie import st_lottie
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import ai

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

# Function to load Lottie animation
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error: {e}")
        return None

# Load Lottie animation
lottie_penguin = load_lottiefile("lottiefiles/penguin.json")

# Add navigation buttons
col1, col2, _ = st.columns([9, 1, 1])  # Allocate more space for the empty column on the right

with col1:
    if st.button("Home"):
        st.switch_page("main.py")  # Replace with the appropriate file name for the home page

with col2:
    if st.button("About"):
        st.switch_page("pages/page_about.py")  # Replace with the appropriate file name for the about page

# Set up session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

def navigate_to(page_name):
    st.session_state.current_page = page_name

if st.session_state.current_page == "home":
    st.markdown('<h1 class="fade-in">Say Hello to PABLO - Your Presentation Assistant for Body Language and Observation</h1>', unsafe_allow_html=True)
    st.write("Click below to get started!")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("Start Presenting"):
            navigate_to("uploader")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if lottie_penguin:
            st.markdown('<div class="penguin-container">', unsafe_allow_html=True)
            st_lottie(lottie_penguin, speed=1, loop=True, quality="low", height=400, width=500, key="penguin_home")
            st.markdown('</div>', unsafe_allow_html=True)
elif st.session_state.current_page == "uploader":
    scanner = True
    st.markdown('<h1 class="fade-in">Start Your Video Presentation</h1>', unsafe_allow_html=True)
    st.write("start the presentation video below to get started.")
    if st.button("⬅️ Back to Home"):
        navigate_to("home")
elif st.session_state.current_page == "about":
    st.markdown('<h1 class="fade-in">About PABLO</h1>', unsafe_allow_html=True)
    st.write("This app provides actionable feedback on presentation skills.")
    if st.button("⬅️ Back to Home"):
        navigate_to("home")

head_score, body_score, hand_score = 0, 0, 0
timer_active, get_pos = False, True
last_hand_score_time, hand_score_cooldown = 0, 4

class BodyTracker(VideoTransformerBase):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.mp_drawing = mp.solutions.drawing_utils
        self.keypoints = {}
        self.previous_pitch, self.previous_yaw, self.previous_roll = 0, 0, 0

    def transform(self, frame):
        global hand_score, last_hand_score_time
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        if not (results.left_hand_landmarks or results.right_hand_landmarks):
            current_time = time.time()
            if current_time - last_hand_score_time >= hand_score_cooldown:
                hand_score += 1
                last_hand_score_time = current_time

        if results.face_landmarks:
            self.keypoints['nose'] = self._get_coordinates(results, 1)
            self.keypoints['left_eye'] = self._get_coordinates(results, 33)
            self.keypoints['right_eye'] = self._get_coordinates(results, 133)
            left_hip = self._get_coordinates(results, 23)
            right_hip = self._get_coordinates(results, 24)
            self.hip_midpoint = self._calculate_midpoint(left_hip, right_hip)
        else:
            self.keypoints = {'nose': (0, 0, 0), 'left_eye': (0, 0, 0), 'right_eye': (0, 0, 0)}
            cv2.putText(frame, "GET BACK IN FRAME", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 15)
        return frame

    def _get_coordinates(self, results, index):
        h, w, d = frame.shape
        landmark = results.face_landmarks.landmark[index]
        return int(landmark.x * w), int(landmark.y * h), landmark.z * d

    def _calculate_midpoint(self, left_hip, right_hip):
        return tuple((l + r) / 2 for l, r in zip(left_hip, right_hip))

    def calculate_face_rotation(self):
        if 'nose' in self.keypoints:
            nose = np.array(self.keypoints['nose'])
            left_eye, right_eye = np.array(self.keypoints['left_eye']), np.array(self.keypoints['right_eye'])
            yaw = np.degrees(np.arctan2(right_eye[0] - left_eye[0], right_eye[2] - left_eye[2]))
            roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
            pitch = np.degrees(np.arctan2(nose[2], np.linalg.norm(nose[:2] - left_eye[:2])))
            return {'yaw': yaw, 'roll': roll, 'pitch': pitch}
        return {}

    def count_head_turn(self, roll, yaw, pitch):
        global head_score
        threshold = 2
        if any(abs(curr - prev) > threshold for curr, prev in zip([pitch, roll, yaw], [self.previous_pitch, self.previous_roll, self.previous_yaw])):
            head_score += 1
            self.previous_pitch, self.previous_roll, self.previous_yaw = pitch, roll, yaw

    def release(self):
        self.holistic.close()
        self.face_mesh.close()

frame_placeholder = st.empty()

if scanner:
    tracker = BodyTracker()

    running, timer_active, get_pos, start_time = True, False, True, None

    if start_time is None:
        start_time = time.time()
        
    stop_tracking = st.button("Stop Tracking")

    # WebRTC video streamer
    webrtc_streamer(key="example", video_transformer_factory=lambda: tracker)

    while running:
        if stop_tracking:
            st.header("Please wait a moment, PABLO is analyzing your amazing performance!")
            tottime = time.time() - stime
            running = False
            feedback = ai.get_feedback({
                "head_score": head_score / tottime,
                "hand_score": hand_score / tottime,
                "body_score": body_score / tottime,
                "total_time": tottime
            })
            lottie_penguin = load_lottiefile("lottiefiles/penguin.json")
            if lottie_penguin:
                st.markdown('<div class="penguin-container">', unsafe_allow_html=True)
                st_lottie(lottie_penguin, speed=1, loop=False, quality="low", height=200, width=1000, key="penguin_home")
                st.markdown('</div>', unsafe_allow_html=True)

    tracker.release()

    st.write(feedback)
