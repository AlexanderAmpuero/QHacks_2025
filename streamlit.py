import streamlit as st
import json
from streamlit_lottie import st_lottie

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
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<h1 class="fade-in">Welcome to Presentation Feedback Assistant</h1>',
            unsafe_allow_html=True,
        )
        st.write("Choose one of the options below to get started!")

    with col2:
        if lottie_penguin:
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
        navigate_to("uploader")
    if st.button("Learn More About This App"):
        navigate_to("about")

elif st.session_state.current_page == "uploader":
    # Video Uploader Page
    st.markdown(
        '<h1 class="fade-in">Upload Your Presentation Video</h1>',
        unsafe_allow_html=True,
    )
    st.write(
        "Upload your presentation video below to get started. Ensure your video has good lighting and clear audio for better analysis."
    )

    # File uploader widget
    video_file = st.file_uploader("Upload Your Presentation Video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        st.video(video_file)
        st.write("Thank you for uploading! Click 'Analyze' to proceed.")

        # Analyze button
        if st.button("Analyze Video"):
            st.markdown('<p class="fade-in">Analyzing your video... Please wait.</p>', unsafe_allow_html=True)
            st.success("Analysis complete!")
            st.write("Suggestions: Improve eye contact and reduce filler words.")

    # Back to Home button
    if st.button("⬅️ Back to Home"):
        navigate_to("home")

elif st.session_state.current_page == "about":
    # About Page
    st.markdown(
        '<h1 class="fade-in">About Presentation Feedback Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.write(
        """
        This app is designed to provide actionable feedback on your presentation skills.
        By analyzing your movements, gestures, and overall delivery, you'll receive
        suggestions to improve your confidence and engagement with your audience.
        """
    )

    # Back to Home button
    if st.button("⬅️ Back to Home"):
        navigate_to("home")
