import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # read frame
    _, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame for pose detection
    pose_results = pose.process(frame_rgb)
    # print(pose_results.pose_landmarks)
    nose = pose_results.pose_landmarks.landmark[0]

    cv2.circle(frame, (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])), 2, (255, 0, 0))

    # draw skeletlon on the frame
    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # display the frame
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) == ord('q'):
        break
print(nose)
cap.release()
cv2.destroyAllWindows()