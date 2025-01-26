import streamlit as st

# Set page configuration for the uploader page
st.set_page_config(
    page_title="Video Uploader",
    page_icon="📹",
    layout="wide",
)

if st.button("Go Back Home"):
    st.switch_page("streamlit.py")  # Replace with the correct file name for the home page

# Content for the Uploader Page
st.markdown('<h1 style="text-align:center; animation: fade-in 2s;">Record Your Presentation</h1>', unsafe_allow_html=True)
st.write("Click 'Start' to begin recording and 'Stop' when you are done.")

# Center the buttons using Streamlit columns
col1, col2, col3 = st.columns([2, 1, 2])  # Create columns for centering the buttons

with col2:
    # Display Start and Stop buttons
    if st.button("Start", key="start_button"):
        st.markdown('<p style="text-align:center; animation: fade-in 1.5s;">🎥 Recording started...</p>', unsafe_allow_html=True)

    if st.button("Stop", key="stop_button"):
        st.markdown('<p style="text-align:center; animation: fade-in 1.5s;">⏹️ Recording stopped.</p>', unsafe_allow_html=True)

# CSS for fade-in animation
st.markdown(
    """
    <style>
        @keyframes fade-in {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        h1, p {
            animation: fade-in 1.5s ease-in-out;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
