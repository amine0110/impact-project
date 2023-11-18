import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./test_videos/left_right.mp4')

is_person_standing = False
was_person_standing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        threshold = 0.1 

        # Check if these points are roughly horizontally aligned (lying down)
        if abs(max(shoulder) - min(ankle)) < threshold:  
            is_person_standing = False
        else:
            is_person_standing = True

        # Detect transition from standing to lying down
        if was_person_standing and not is_person_standing:
            print("The person has transitioned from standing to lying down.")
            was_person_standing = False
        elif is_person_standing:
            was_person_standing = True

    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit with ESC key
        break

cap.release()
cv2.destroyAllWindows()
