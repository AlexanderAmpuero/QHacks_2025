import mediapipe as mp
import cv2
import numpy as np
import time

# Global variables
score = 0
timer_active = False
start_time = 0
get_pos = True

def calc_distance(p1: tuple, p2: tuple) -> int:
    """Takes two coordinates in (x, y, z) format and calculates the distance between them."""
    distance = np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )
    return distance

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
                'right_hip': self._get_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_HIP)
            }

            # Draw landmarks on the frame for visualization
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        return frame

    def _get_coordinates(self, results, landmark):
        # Extract normalized coordinates for a specific landmark
        h, w, d = frame.shape
        landmark = results.pose_landmarks.landmark[landmark]
        return int(landmark.x * w), int(landmark.y * h), int(landmark.z * d)

    def calculate_metrics(self):
        # Example: Calculate neck tilt angle
        if 'nose' and 'left_shoulder' and 'right_shoulder' and 'left_hip' and 'right_hip' in self.keypoints:
            nose = self.keypoints['nose']
            left_shoulder = self.keypoints['left_shoulder']
            right_shoulder = self.keypoints['right_shoulder']
            left_hip = self.keypoints['left_hip']
            right_hip = self.keypoints['right_hip']

            # Find shoulder midpoint
            shoulder_midpoint = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2,
                (left_shoulder[2] + right_shoulder[2]) // 2
            )
            shoulder_width = calc_distance(left_shoulder, right_shoulder)
            if shoulder_width == 0:  # Avoid division by zero
                return {}
            
            # Find hip midpoint
            hips_midpoint = (
                (left_hip[0] + right_hip[0]) // 2
            )

            # Calculate angles and distances
            angle_roll = np.degrees(np.arctan2(nose[1] - left_shoulder[1], nose[0] - right_shoulder[0]))
            angle_yaw = np.degrees(np.arctan2(nose[1] - shoulder_midpoint[1], nose[0] - shoulder_midpoint[0]))

            n_ls_distance = calc_distance(nose, left_shoulder)
            n_rs_distance = calc_distance(nose, right_shoulder)
            abs_deviation = abs(n_ls_distance - n_rs_distance)
            abs_shoulder_rotation = abs(shoulder_width)

            return {
                'neck_tilt_angle': angle_roll,
                'neck_nod_angle': angle_yaw,
                'neck_rotation_abs': abs_deviation,
                'shoulder_rotation': abs_shoulder_rotation,
                'hip_midpoint': hips_midpoint
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
            cv2.putText(
                frame, f"Neck Yaw: {metrics['neck_nod_angle']:.2f} degrees",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Rotation Deviation: {metrics['neck_rotation_abs']:.2f} mm",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Shoulder Rotation: {metrics['shoulder_rotation']:.2f} mm",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Centre of mass location: {metrics['hip_midpoint']:.2f} mm",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            if get_pos:
                curr_pos = metrics['hip_midpoint']
                get_pos = False
                timer_active = True
                start_time = time.time()  # Start the timer
            
            if timer_active:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 10:  # Timer ends after 3 seconds
                    if curr_pos - 200 <= metrics['hip_midpoint'] <= curr_pos + 200:
                        score += 1
                        print(f"Score: {score}")
                    else:
                        print("Position moved, resetting.")
                    get_pos = True
                    timer_active = False

        # Show the frame
        cv2.imshow('Body Tracker', frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
