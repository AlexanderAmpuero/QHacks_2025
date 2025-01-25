import time
import cv2
import mediapipe as mp
import numpy as np
import os

# Scores used for Gen AI feedback
head_score = 0 # number of times the head turns
body_score = 0 # number of times the person stands still
arm_score = 0 # number of times the person uses their hands

# booleans used to set a timer for standing too long
timer_active = False
get_pos = True

class BodyTracker:
    def __init__(self):
        # Initialize Mediapipe Holistic solution and FaceMesh

        # uses holistic with facemesh to get more accurate roations on the face while having rest of the body
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic()

        self.mp_face_mesh = mp.solutions.face_mesh  # FaceMesh for extracting face landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh()

        self.mp_drawing = mp.solutions.drawing_utils
        self.keypoints = {}

        # sets the previous rotations of the face to 0
        self.previous_pitch = 0
        self.previous_yaw = 0
        self.previous_roll = 0

        self.hip_midpoint = (0, 0)  # To store the current hip midpoint

    def process_frame(self, frame):
        # Convert the frame to RGB as Mediapipe works with RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect holistic landmarks
        results = self.holistic.process(rgb_frame)

        if results.face_landmarks:
            # Draw face landmarks without connections
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks
            )

            # Extract nose, left eye, and right eye landmarks for face rotation
            self.keypoints['nose'] = self._get_coordinates(results, 1)  # Nose landmark index
            self.keypoints['left_eye'] = self._get_coordinates(results, 33)  # Left eye landmark index
            self.keypoints['right_eye'] = self._get_coordinates(results, 133)  # Right eye landmark index
            # Draw body pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )
            # Extract left and right hip coordinates to calculate the midpoint
            left_hip = self._get_coordinates(results, 23)  # Left hip landmark index
            right_hip = self._get_coordinates(results, 24)  # Right hip landmark index
            self.hip_midpoint = self._calculate_midpoint(left_hip, right_hip)

            # Extract necessary coordinates for calculating angles
            left_shoulder = self._get_coordinates(results, 11)
            left_elbow = self._get_coordinates(results, 13)
            left_wrist = self._get_coordinates(results, 15)

            right_shoulder = self._get_coordinates(results, 12)
            right_elbow = self._get_coordinates(results, 14)
            right_wrist = self._get_coordinates(results, 16)
            
            # Calculate angles
            left_elbow_angle = self.calculate_elbow_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_body_angle = self.calculate_shoulder_body_angle(left_hip, left_shoulder, left_elbow)

            right_elbow_angle = self.calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_body_angle = self.calculate_shoulder_body_angle(right_hip, right_shoulder, right_elbow)

            # Display the calculated angles on the frame
            cv2.putText(
                frame, f"Left Elbow Angle: {left_elbow_angle:.2f}°",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Left Shoulder-Body Angle: {left_shoulder_body_angle:.2f}°",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # Display the calculated angles on the frame
            cv2.putText(
                frame, f"Right Elbow Angle: {right_elbow_angle:.2f}°",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Right Shoulder-Body Angle: {right_shoulder_body_angle:.2f}°",
                (10, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            # These Occur when a person cannot be detected
            fail_safe = (0, 0, 0, 0)
            left_elbow, left_hip, left_shoulder, left_wrist = fail_safe
            left_elbow_angle = 0
            left_shoulder_body_angle = 0
            right_elbow, right_hip, right_shoulder, right_wrist = fail_safe
            right_elbow_angle = 0
            right_shoulder_body_angle = 0
            self.keypoints['nose'] = fail_safe
            self.keypoints['right_eye'] = fail_safe
            self.keypoints['left_eye'] = fail_safe
            cv2.putText(
                frame, "GET BACK IN FRAME",
                (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 15
            )
            

        return frame

    def _get_coordinates(self, results, index):
        # Extract normalized coordinates for a specific landmark
        h, w, d = frame.shape
        landmark = results.face_landmarks.landmark[index]
        return int(landmark.x * w), int(landmark.y * h), landmark.z * d
    

    def calculate_elbow_angle(self, shoulder, elbow, wrist):
        # Convert points to numpy arrays for vector calculations
        shoulder = np.array(shoulder)
        elbow = np.array(elbow)
        wrist = np.array(wrist)

        # Calculate vectors
        shoulder_to_elbow = elbow - shoulder
        elbow_to_wrist = wrist - elbow

        # Calculate dot product and magnitudes
        dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
        magnitude_1 = np.linalg.norm(shoulder_to_elbow)
        magnitude_2 = np.linalg.norm(elbow_to_wrist)

        # Calculate the angle in radians, then convert to degrees
        angle = np.arccos(dot_product / (magnitude_1 * magnitude_2))
        return np.degrees(angle)

    def calculate_shoulder_body_angle(self, hip, shoulder, elbow):
        # Convert points to numpy arrays for vector calculations
        hip = np.array(hip)
        shoulder = np.array(shoulder)
        elbow = np.array(elbow)

        # Calculate vectors
        hip_to_shoulder = shoulder - hip
        shoulder_to_elbow = elbow - shoulder

        # Calculate dot product and magnitudes
        dot_product = np.dot(hip_to_shoulder, shoulder_to_elbow)
        magnitude_1 = np.linalg.norm(hip_to_shoulder)
        magnitude_2 = np.linalg.norm(shoulder_to_elbow)

        # Calculate the angle in radians, then convert to degrees
        angle = np.arccos(dot_product / (magnitude_1 * magnitude_2))
        return np.degrees(angle)
    

    def _calculate_midpoint(self, left_hip, right_hip):
        # Calculate the midpoint between two coordinates
        midpoint_x = (left_hip[0] + right_hip[0]) / 2
        midpoint_y = (left_hip[1] + right_hip[1]) / 2
        midpoint_z = (left_hip[2] + right_hip[2]) / 2
        return (midpoint_x, midpoint_y, midpoint_z)

    def calculate_face_rotation(self):
        if 'nose' in self.keypoints and 'left_eye' in self.keypoints and 'right_eye' in self.keypoints:
            nose = np.array(self.keypoints['nose'])
            left_eye = np.array(self.keypoints['left_eye'])
            right_eye = np.array(self.keypoints['right_eye'])

            # Calculate face yaw (left-right rotation)
            yaw = np.degrees(np.arctan2(right_eye[0] - left_eye[0], right_eye[2] - left_eye[2]))

            # Calculate face roll (tilt to the side)
            roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

            # Calculate face pitch (up-down rotation)
            pitch = np.degrees(np.arctan2(nose[2], np.linalg.norm(nose[:2] - left_eye[:2])))

            return {'yaw': yaw, 'roll': roll, 'pitch': pitch}
        return {}

    def release(self):
        # Clean up resources
        self.holistic.close()
        self.face_mesh.close()

    def count_head_turn(self, roll, yaw, pitch):
        global head_score
        threshold = 2  # Adjust this value to capture larger head turns (in degrees)
        if abs(self.previous_pitch - pitch) > threshold or \
        abs(self.previous_roll - roll) > threshold or \
        abs(self.previous_yaw - yaw) > threshold:
            head_score += 1
            self.previous_pitch = pitch
            self.previous_roll = roll
            self.previous_yaw = yaw
    

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Open webcam
    tracker = BodyTracker()

    while cap.isOpened():
        stime = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite("output_image.jpg", frame)

        # Process the frame
        frame = tracker.process_frame(frame)

        # Calculate face rotation
        face_metrics = tracker.calculate_face_rotation()

        if face_metrics:
            cv2.putText(
                frame, f"Yaw: {face_metrics['yaw']:.2f} degrees",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Roll: {face_metrics['roll']:.2f} degrees",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Pitch: {face_metrics['pitch']:.2f} degrees",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            # Handle case when no face is detected
            cv2.putText(
                frame, "No face detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            print("No face metrics available.")


        # Display the frame
        cv2.imshow("Holistic Tracker", frame)

        # Tracking head turns
        tracker.count_head_turn(face_metrics['roll'], face_metrics['yaw'], face_metrics['pitch'])

        # Position tracking logic
        if get_pos:
            curr_pos = tracker.hip_midpoint[0]  # Use the x-coordinate of the hip midpoint
            get_pos = False
            timer_active = True
            start_time = time.time()  # Start the timer
            head_active = True

        if timer_active:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 10:  # Timer ends after 10 seconds
                if curr_pos - 200 <= tracker.hip_midpoint[0] <= curr_pos + 200:
                    body_score += 1
                    print(f"Score: {body_score}")
                else:
                    print("Position moved, resetting.")
                get_pos = True
                timer_active = False

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("total time" + str(time.time() - stime))
            break

        if os.path.exists("output_image.jpg"):
            os.remove("output_image.jpg")
       

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()