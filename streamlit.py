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
        .penguin-container {
            position: relative;
            top: -50px;
            left: 20px;
        }
        .button-container {
            position: relative;
            top: -50px;
            left: 30px;
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

# Add navigation buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Home"):
        st.switch_page("streamlit.py")  # Replace with the appropriate file name for the home page

with col2:
    if st.button("Uploader"):
        st.switch_page("pages/page_uploader.py")  # Replace with the appropriate file name for the uploader page

with col3:
    if st.button("About"):
        st.switch_page("pages/page_about.py")  # Replace with the appropriate file name for the about page

# Home Page Content
st.markdown('<h1 class="fade-in">Welcome to Presentation Feedback Assistant</h1>', unsafe_allow_html=True)
st.write("Choose one of the options above to navigate the app!")

# Display the penguin animation on the home page
if lottie_penguin:
    st.markdown('<div class="penguin-container">', unsafe_allow_html=True)
    st_lottie(
        lottie_penguin,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height=400,
        width=500,
        key="penguin_home",
    )
    st.markdown('</div>', unsafe_allow_html=True)
