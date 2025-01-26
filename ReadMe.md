# 🐧 PABLO: Your Presentation Assistant for Body Language Observation

PABLO is an AI-powered assistant designed to provide real-time feedback on your presentation skills by analyzing your body language, including head movement, hand gestures, and posture. This tool leverages advanced computer vision and generative AI techniques to help you improve your presentation style, making your delivery more effective and engaging. 

---

## 🎯 Key Features
- **Real-Time Body Language Tracking**: Tracks head movement, hand gestures, and posture using Mediapipe and OpenCV.
- **Session Feedback**: Provides actionable feedback on your presentation performance based on tracked metrics.
- **User-Friendly Interface**: A streamlined, interactive interface built with Streamlit for ease of use.
- **Lottie Animations**: Engaging animations for a visually appealing user experience.
- **Customizable Navigation**: Seamless transitions between pages like Home, About, and the Presentation Interface.

---

## 🚀 How It Works
1. **Start Presenting**: Begin by uploading or recording a video of your presentation.
2. **Real-Time Analysis**: PABLO captures your movements using your webcam and processes them using advanced AI models.
3. **Performance Metrics**: Tracks:
   - **Head Movement**: Evaluates your head pitch, yaw, and roll.
   - **Hand Gestures**: Monitors hand activity and penalizes inactivity.
   - **Posture Stability**: Analyzes body movement to maintain consistent posture.
4. **Feedback Generation**: At the end of your session, receive a detailed analysis of your performance, including scores for head movement, hand gestures, and posture.

---

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Animation**: Lottie Animations
- **AI & Vision Libraries**:
  - OpenCV
  - Mediapipe
  - NumPy
- **Backend Support**: Generative AI for feedback analysis
- **Other Tools**: JSON for data handling

---

## 🖥️ Local Setup
1. **Clone the Repository**:
```bashgit clone https://github.com/your-username/pablo-presentation-assistant.git
cd pablo-presentation-assistant
```
2. **Install Dependencies**: Ensure you have Python 3.9 or above installed, then install required libraries:
``` bash
pip install -r requirements.txt
````
3. Run the Application:
``` bash
streamlit run main.py
```
   
