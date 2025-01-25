import streamlit as st
import cv2
import numpy as np
import time

# Title and description
st.title("Presentation Feedback Assistant")
st.write("Upload a video of your presentation, and receive feedback on your movements. Get visual insights and suggestions to improve your presentation!")

# File uploader
video_file = st.file_uploader("Upload Your Presentation Video", type=["mp4", "mov", "avi"])

# Progress bar for video analysis
progress_bar = st.progress(0)
status_text = st.empty()

# Function to simulate video analysis (with animation)
def analyze_video(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
    
    feedback = []
    if not cap.isOpened():
        return "Error: Couldn't open the video."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Simulating movement detection (simplified motion analysis)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        non_zero_pixels = np.count_nonzero(thresholded)

        if non_zero_pixels > 1000:  # Arbitrary threshold for movement
            feedback.append("Movement detected!")
        else:
            feedback.append("No significant movement detected.")
        
        # Update status text with each frame's progress
        status_text.text(f"Analyzing... {frame_count}/{total_frames} frames processed.")

    cap.release()
    return '\n'.join(feedback)

# Display video and analyze when button is pressed
if video_file is not None:
    # Display uploaded video in the app
    st.video(video_file)
    
    # Show "Analyze" button with visual animation
    analyze_button = st.button("Start Analysis", key="analyze_button")
    
    if analyze_button:
        with st.spinner("Analyzing video... Please wait."):
            time.sleep(1)  # Simulate delay
            feedback = analyze_video(video_file)  # Analyze video content
            # Reset progress bar and status text once analysis is done
            progress_bar.empty()
            status_text.empty()

            # Show feedback
            st.subheader("Presentation Feedback")
            st.write(feedback)
            
            # Add additional suggestions
            st.subheader("Suggestions for Improvement")
            st.write("Consider incorporating more hand gestures and making more eye contact.")
            st.write("Avoid staying static in one position for too long during your presentation.")
        
            # Animation to highlight feedback (could be color-coded or animated text)
            st.markdown("""
            <style>
            .feedback-text {
                animation: bounce 1s infinite alternate;
                color: #d93f3a; /* Highlighted color for feedback */
            }
            @keyframes bounce {
                0% { transform: translateY(0); }
                100% { transform: translateY(-10px); }
            }
            </style>
            <div class="feedback-text">Ensure to maintain an active stance and vary your movements during your speech!</div>
            """, unsafe_allow_html=True)

else:
    st.info("Please upload a video to start the analysis.")

