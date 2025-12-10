from math import degrees, atan2
import cv2
import numpy as np
import arcade
import mediapipe as mp


class Vector:
    def __init__(self, x1, y1, x2, y2):
        self.x = x2 - x1
        self.y = y2 - y1

    def length(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __mul__(self, other: "Vector"): #scalar
        return self.x * other.x + self.y * other.y

    def cosine(self, other: "Vector"):
        return self * other / self.length() / other.length()

def is_straight(vector1: "Vector", vector2: "Vector") -> bool:
    if Vector.cosine(vector1, vector2) > 0.6:
        return True
    return False

def cnt_angle(vector1: "Vector", vector2: "Vector"):
    rangle_1 = atan2(vector1.y, vector1.x)
    rangle_2 = atan2(vector2.y, vector2.x)
    angle = degrees(rangle_2 - rangle_1)
    return angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

vector_north = Vector(0, 0, 0, -100)
south_vector = Vector(0, 100, 0, 0)
east_vector = Vector(100, 0, 0, 0)
west_vector = Vector(0, 0, 100, 0)
with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            assert RuntimeError("No video")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            x1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            y1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            x2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
            y2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            x3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
            y3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            vector1 = Vector(x1, y1, x2, y2)
            vector2 = Vector(x2, y2, x3, y3)
            # if is_straight(vector1, vector2):
            #     print("straight")
            # else:
            #     print("blocked")

            vector_hand = Vector(x1, y1, x3, y3)
            azimuth = cnt_angle(vector_hand, vector_north)
            azimuth = (azimuth + 360) % 360
            direction = ""
            if azimuth < 45 or azimuth > 315: direction = "north"
            elif azimuth < 135: direction = "east"
            elif azimuth < 225: direction = "south"
            else: direction = "west"

            text_cord = (50, 50)
            cv2.putText(frame, direction, text_cord, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
            print("angle:", (azimuth + 360) % 360)

        cv2.imshow("", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
