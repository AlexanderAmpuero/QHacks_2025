import time
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import json
from streamlit_lottie import st_lottie

# FRONTEND
scanner = False
# Set Streamlit page configuration for wide layout
st.set_page_config(
    page_title="Presentation Feedback Assistant",
    page_icon="🖥️",
    layout="wide",  # Enables wide layout
)
# CSS Styling for the app
st.markdown(
    """
    <style>
        h1 {
            font-size: 24px;
            color: #333;
            text-align: center;
            animation: fade-in 2s ease-in-out;
        }
        p {
            font-size: 16px;
            color: #555;
        }
        .fade-in {
            animation: fade-in 1.5s ease-in-out;
        }
        @keyframes fade-in {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        /* Move the penguin animation */
        .penguin-container {
            position: relative;
            top: -50px;  /* Moves the penguin up */
            left: 20px;  /* Moves the penguin to the right */
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Function to load Lottie animation
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: File '{filepath}' not found!")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: File '{filepath}' is not a valid Lottie JSON file!")
        return None
# Load Lottie animation for the penguin
lottie_penguin = load_lottiefile("lottiefiles/penguin.json")

# Set up session state to manage page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"  # Default to home page

# Helper function to switch pages
def navigate_to(page_name):
    st.session_state.current_page = page_name

# Main navigation logic
if st.session_state.current_page == "home":
    # Home Page
    st.markdown('<h1 class="fade-in">Welcome to Presentation Feedback Assistant</h1>', unsafe_allow_html=True)
    st.write("Choose one of the options below to get started!")

    # Grouping content into columns for better layout
    col1, col2 = st.columns([2, 3])  # Adjusted column widths for better balance

    with col1:
        # Instruction or option buttons in the main content area
        st.markdown('<div class="button-container">', unsafe_allow_html=True)  # Open div container for buttons
        if st.button("Upload Your Video"):
            navigate_to("uploader")
        if st.button("Learn More About This App"):
            navigate_to("about")
        st.markdown('</div>', unsafe_allow_html=True)  # Close div container for buttons

    with col2:
        # Display the penguin animation beside the buttons
        if lottie_penguin:
            st.markdown('<div class="penguin-container">', unsafe_allow_html=True)  # Open div container for penguin animation
            st_lottie(
                lottie_penguin,
                speed=1,
                reverse=False,
                loop=True,
                quality="low",
                height=400,  # Keep the same size
                width=500,  # Keep the same size
                key="penguin_home",
            )
            st.markdown('</div>', unsafe_allow_html=True)  # Close div container for penguin animation

elif st.session_state.current_page == "uploader":
    scanner = True
    # Video Uploader Page
    st.markdown('<h1 class="fade-in">Upload Your Presentation Video</h1>', unsafe_allow_html=True)
    st.write("Upload your presentation video below to get started. Ensure your video has good lighting and clear audio for better analysis.")

    # Back to Home button
    if st.button("⬅️ Back to Home"):
        navigate_to("home")

elif st.session_state.current_page == "about":
    # About Page
    st.markdown('<h1 class="fade-in">About Presentation Feedback Assistant</h1>', unsafe_allow_html=True)
    st.write("""
        This app is designed to provide actionable feedback on your presentation skills.
        By analyzing your movements, gestures, and overall delivery, you'll receive
        suggestions to improve your confidence and engagement with your audience.
    """)

    # Back to Home button
    if st.button("⬅️ Back to Home"):
        navigate_to("home")

# Scores used for Gen AI feedback
head_score = 0  # number of times the head turns
body_score = 0  # number of times the person stands still
hand_score = 0  # number of times the person uses their hands

# Booleans used to set a timer for standing too long
timer_active = False
get_pos = True

# Timer for hand detection
last_hand_score_time = 0
hand_score_cooldown = 4


class BodyTracker:
    def __init__(self):
        # Initialize Mediapipe Holistic solution and FaceMesh

        # Uses holistic with facemesh to get more accurate rotations on the face while having the rest of the body
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic()

        self.mp_face_mesh = mp.solutions.face_mesh  # FaceMesh for extracting face landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh()

        self.mp_drawing = mp.solutions.drawing_utils
        self.keypoints = {}

        # Sets the previous rotations of the face to 0
        self.previous_pitch = 0
        self.previous_yaw = 0
        self.previous_roll = 0

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.holistic.process(rgb_frame)

        global hand_score, last_hand_score_time
        if not (results.left_hand_landmarks or results.right_hand_landmarks):
            current_time = time.time()
            if current_time - last_hand_score_time >= hand_score_cooldown:
                hand_score += 1
                last_hand_score_time = current_time

        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.face_landmarks)
            self.keypoints['nose'] = self._get_coordinates(results, 1)
            self.keypoints['left_eye'] = self._get_coordinates(results, 33)
            self.keypoints['right_eye'] = self._get_coordinates(results, 133)
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            left_hip = self._get_coordinates(results, 23)
            right_hip = self._get_coordinates(results, 24)
            self.hip_midpoint = self._calculate_midpoint(left_hip, right_hip)

        else:
            fail_safe = (0, 0, 0)
            left_hip = 0
            right_hip = fail_safe
            self.keypoints['nose'] = fail_safe
            self.keypoints['right_eye'] = fail_safe
            self.keypoints['left_eye'] = fail_safe
            cv2.putText(frame, "GET BACK IN FRAME", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 15)

        return frame

    def _get_coordinates(self, results, index):
        h, w, d = frame.shape
        landmark = results.face_landmarks.landmark[index]
        return int(landmark.x * w), int(landmark.y * h), landmark.z * d

    def _calculate_midpoint(self, left_hip, right_hip):
        midpoint_x = (left_hip[0] + right_hip[0]) / 2
        midpoint_y = (left_hip[1] + right_hip[1]) / 2
        midpoint_z = (left_hip[2] + right_hip[2]) / 2
        return (midpoint_x, midpoint_y, midpoint_z)

    def calculate_face_rotation(self):
        if 'nose' in self.keypoints and 'left_eye' in self.keypoints and 'right_eye' in self.keypoints:
            nose = np.array(self.keypoints['nose'])
            left_eye = np.array(self.keypoints['left_eye'])
            right_eye = np.array(self.keypoints['right_eye'])

            yaw = np.degrees(np.arctan2(right_eye[0] - left_eye[0], right_eye[2] - left_eye[2]))
            roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
            pitch = np.degrees(np.arctan2(nose[2], np.linalg.norm(nose[:2] - left_eye[:2])))

            return {'yaw': yaw, 'roll': roll, 'pitch': pitch}
        return {}

    def release(self):
        self.holistic.close()
        self.face_mesh.close()

    def count_head_turn(self, roll, yaw, pitch):
        global head_score
        threshold = 2  # Adjust this value to capture larger head turns (in degrees)
        if abs(self.previous_pitch - pitch) > threshold or \
           abs(self.previous_roll - roll) > threshold or \
           abs(self.previous_yaw - yaw) > threshold:
            head_score += 1
            self.previous_pitch = pitch
            self.previous_roll = roll
            self.previous_yaw = yaw\
            

frame_placeholder = st.empty()

if scanner:
    cap = cv2.VideoCapture(0)  # Open webcam
    tracker = BodyTracker()
    running = True
    get_pos = True
    timer_active = False
    start_time = None
    stop_tracking = st.button("Stop Tracking")

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        # Process the frame
        frame = tracker.process_frame(frame)
        face_metrics = tracker.calculate_face_rotation()

        if face_metrics:
            cv2.putText(
                frame, f"Yaw: {face_metrics['yaw']:.2f} degrees",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Roll: {face_metrics['roll']:.2f} degrees",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Pitch: {face_metrics['pitch']:.2f} degrees",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            # Handle case when no face is detected
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display frame using Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Tracking head turns
        if face_metrics:
            tracker.count_head_turn(face_metrics['roll'], face_metrics['yaw'], face_metrics['pitch'])

        # Position tracking logic
        if get_pos:
            curr_pos = tracker.hip_midpoint[0]
            get_pos = False
            timer_active = True
            start_time = time.time()

        # Timer logic for standing still
        if timer_active:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 10:
                if curr_pos - 200 <= tracker.hip_midpoint[0] <= curr_pos + 200:
                    body_score += 1
                get_pos = True
                timer_active = False

        # Stop if "Stop Tracking" button is pressed
        if stop_tracking:
            running = False
            navigate_to("home")

    tracker.release()
    cap.release()
    st.write(f"Head Score: {head_score}")
    st.write(f"Body Score: {body_score}")
    st.write(f"Hand Score: {hand_score}")
    st.warning("Tracking stopped.")
