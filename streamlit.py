import streamlit as st
import json
from streamlit_lottie import st_lottie

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
        /* Move buttons */
        .button-container {
            position: relative;
            top: -50px;  /* Move buttons upwards */
            left: 30px;  /* Move buttons to the right */
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
    # Video Uploader Page
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
            st.success("Analysis complete!")
            st.write("Suggestions: Improve eye contact and reduce filler words.")
            
            # Celebrate with balloons!
            st.balloons()  # This will show the celebratory balloons

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
