import streamlit as st
import json
from streamlit_lottie import st_lottie

st.markdown("""
<style>
   h1 {
      font-size: 16px;
      text-align: center;
   }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the penguin and sleeping penguin animations
lottie_penguin = load_lottiefile("lottiefiles/penguin.json")
lottie_sleepingpenguin = load_lottiefile("lottiefiles/sleepingpenguin.json")

# Set up session state to manage page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"  # Default to home page

# Function to switch pages
def switch_page(page_name):
    st.session_state.current_page = page_name

# Custom CSS for the app
st.markdown(
    """
    <style>
    body {
        background-color: #EAE0E0;  /* Light background color */
        font-family: monospace;  /* Monospace font */
    }

    .fade-in {
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .stButton>button {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True
)

# Page 1: Home Page
if st.session_state.current_page == "home":
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="fade-in">Welcome to Presentation Feedback Assistant</h1>', unsafe_allow_html=True)
        st.write("Choose one of the options below to get started!")

    with col2:
        st_lottie(
            lottie_penguin,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=150,
            width=150,
            key="penguin_home",
        )

    # Navigation buttons
    if st.button("Upload Your Video"):
        switch_page("uploader")
    
    if st.button("Learn More About This App"):
        switch_page("about")

# Page 2: Video Uploader Page
elif st.session_state.current_page == "uploader":
    st.markdown('<h1 class="fade-in">Upload Your Presentation Video</h1>', unsafe_allow_html=True)
    st.write("Upload your presentation video below to get started. Ensure your video has good lighting and clear audio for better analysis.")

    # File uploader widget
    video_file = st.file_uploader("Upload Your Presentation Video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        st.video(video_file)
        st.write("Thank you for uploading! Click 'Analyze' to proceed.")

        # Analyze button
        if st.button("Analyze Video"):
            st.markdown('<p class="fade-in">Analyzing your video... Please wait.</p>', unsafe_allow_html=True)
            st.write("Analysis complete! Suggestions will be displayed here.")

    # Back to Home button
    if st.button("⬅️ Back to Home"):
        switch_page("home")

# Page 3: About Page
elif st.session_state.current_page == "about":
    st.markdown('<h1 class="fade-in">About Presentation Feedback Assistant</h1>', unsafe_allow_html=True)
    st.write("""
        This app is designed to provide actionable feedback on your presentation skills.
        By analyzing your movements, gestures, and overall delivery, you'll receive
        suggestions to improve your confidence and engagement with your audience.
    """)

    # Display the sleeping penguin animation
    st_lottie(
        lottie_sleepingpenguin,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height=150,
        width=150,
        key="penguin_sleeping"
    )

    # Back to Home button
    if st.button("⬅️ Back to Home"):
        switch_page("home")
