import mediapipe as mp
import cv2
import numpy as np

class BodyTracker:
    def __init__(self):
        # Initialize Mediapipe pose solution
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.keypoints = {}

    def process_frame(self, frame):
        # Convert the frame to RGB as Mediapipe works with RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect pose
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Extract key body points
            self.keypoints = {
                'nose': self._get_coordinates(results, self.mp_pose.PoseLandmark.NOSE),
                'left_shoulder': self._get_coordinates(results, self.mp_pose.PoseLandmark.LEFT_SHOULDER),
                'right_shoulder': self._get_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
                'left_hip': self._get_coordinates(results, self.mp_pose.PoseLandmark.LEFT_HIP),
                'right_hip': self._get_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_HIP),
                'left_knee': self._get_coordinates(results, self.mp_pose.PoseLandmark.LEFT_KNEE),
                'right_knee': self._get_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_KNEE),
                'left_ankle': self._get_coordinates(results, self.mp_pose.PoseLandmark.LEFT_ANKLE),
                'right_ankle': self._get_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            }

            # Draw landmarks on the frame for visualization
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        return frame

    def _get_coordinates(self, results, landmark):
        # Extract normalized coordinates for a specific landmark
        h, w, _ = frame.shape
        landmark = results.pose_landmarks.landmark[landmark]
        return int(landmark.x * w), int(landmark.y * h)

    def calculate_metrics(self):
        # Example: Calculate neck tilt angle
        if 'nose' in self.keypoints and 'left_shoulder' in self.keypoints and 'right_shoulder' in self.keypoints:
            nose = self.keypoints['nose']
            left_shoulder = self.keypoints['left_shoulder']
            right_shoulder = self.keypoints['right_shoulder']

            # Find shoulder midpoint
            shoulder_midpoint = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )

            # Calculate angle between nose and shoulder midpoint
            angle = np.degrees(np.arctan2(nose[1] - shoulder_midpoint[1], nose[0] - shoulder_midpoint[0]))
            return {
                'neck_tilt_angle': angle
            }

        return {}

    def release(self):
        # Clean up resources
        self.pose.close()

if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    tracker = BodyTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame = tracker.process_frame(frame)

        # Display keypoints and metrics
        metrics = tracker.calculate_metrics()
        if metrics:
            cv2.putText(
                frame, f"Neck Tilt: {metrics['neck_tilt_angle']:.2f} degrees",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Show the frame
        cv2.imshow('Body Tracker', frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
