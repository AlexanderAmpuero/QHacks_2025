import streamlit as st

# Set page configuration for the about page
st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide",
)

if st.button("Back to Home"):
    st.switch_page("main.py")  

# Content for the About Page
st.markdown(
    '<h1 style="text-align: center; animation: fade-in 2s;">About Presentation Feedback Assistant</h1>',
    unsafe_allow_html=True,
)
st.write(
    """
    Welcome to the **Presentation Feedback Assistant**! This app is designed to provide actionable feedback on your presentation skills.
    By analyzing your movements, gestures, and overall delivery, you'll receive suggestions to improve your confidence and engagement
    with your audience.
    """
)

# FAQ Section with expandable questions
st.markdown('<h2 style="animation: fade-in 1.5s;">FAQs</h2>', unsafe_allow_html=True)

faqs = [
    (
        "How does this app provide feedback on my presentation?",
        "The app analyzes your video or live recording using AI to assess key aspects of your presentation, such as body language, speech clarity, pacing, and eye contact.",
    ),
    (
        "Can I upload any video format?",
        "Currently, the app supports video formats such as MP4, MOV, and AVI.",
    ),
    (
        "Do I need a webcam to use this app?",
        "No, you do not need a webcam to use the app. You can upload an existing video for analysis. If you want to record your presentation live, you can use the Start/Stop buttons without requiring a webcam, though having a camera is recommended.",
    ),
    (
        "Is the feedback truly personalized?",
        "Yes, the feedback is personalized based on your presentation’s content. The AI analyzes your specific delivery and suggests ways to improve.",
    ),
    (
        "Is my video data kept private?",
        "Absolutely! Your videos are processed securely, and we do not share your data with third parties.",
    ),
]

for question, answer in faqs:
    with st.expander(question):
        st.write(answer)

# CSS for animations
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
        h1, h2 {
            animation: fade-in 1.5s ease-in-out;
        }
    </style>
    """,
    unsafe_allow_html=True,
)