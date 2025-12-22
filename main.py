from math import degrees, atan2
import cv2
import numpy as np
import random
import time
import mediapipe as mp

class Vector:
    def __init__(self, x1, y1, x2, y2):
        self.x = x2 - x1
        self.y = y2 - y1

    def length(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __mul__(self, other: "Vector"):  # scalar
        return self.x * other.x + self.y * other.y

    def cosine(self, other: "Vector"):
        return self * other / self.length() / other.length()


def is_straight(vector1: "Vector", vector2: "Vector") -> bool:
    if Vector.cosine(vector1, vector2) > 0.5:
        return True
    return False


def cnt_angle(vector1: "Vector", vector2: "Vector"):
    rangle_1 = atan2(vector1.y, vector1.x)
    rangle_2 = atan2(vector2.y, vector2.x)
    angle = degrees(rangle_2 - rangle_1)
    return angle


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

vector_north = Vector(0, 0, 0, -100)
cur_time = time.time()
possible_directions = ["north", "south", "east", "west"]
direction_right = "wrong"
direction_left = "wrong"
cur_direction = random.choice(possible_directions)
STATE_MENU = 0
STATE_GAME = 1
STATE_RESULT = 2
ROUND_TIME = 2.5
RESULT_TIME = 4
state = STATE_MENU
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
Background = cv2.imread("background.png", cv2.IMREAD_COLOR)
is_backgroung_changed = False

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        ret, frame = cap.read()
        if not ret:
            assert RuntimeError("No video")

        if not is_backgroung_changed:
            is_backgroung_changed = True
            Background = cv2.resize(Background, (frame.shape[1], frame.shape[0]))

        if state == STATE_MENU:
            frame = Background.copy()
            cv2.putText(frame, "Press space to start", (120, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, BLACK, 3)
            cv2.putText(frame, "Press Q to quit", (120, 350),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, BLACK, 2)

            if key == ord(" "):
                state = STATE_GAME
                cur_time = time.time()
                cur_direction = random.choice(possible_directions)
                direction_right = direction_left = "wrong"
        elif state == STATE_GAME:
            if key == ord(" "):
                state = STATE_MENU

            remaining_time = ROUND_TIME - time.time() + cur_time

            cv2.putText(frame, f"SHOW: {cur_direction.upper()}", (200, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            cv2.putText(frame, f"time left: {remaining_time:.1f}", (300, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                # mp_drawing.draw_landmarks(
                #     frame,
                #     results.pose_landmarks,
                #     mp_pose.POSE_CONNECTIONS
                # )

                x_right1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                y_right1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                x_right2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
                y_right2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                x_right3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
                y_right3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
                vector_rigth1 = Vector(x_right1, y_right1, x_right2, y_right2)
                vector_right2 = Vector(x_right2, y_right2, x_right3, y_right3)

                x_left1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                y_left1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                x_left2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
                y_left2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
                x_left3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
                y_left3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                vector_left1 = Vector(x_left1, y_left1, x_left2, y_left2)
                vector_left2 = Vector(x_left2, y_left2, x_left3, y_left3)

                vector_right_hand = Vector(x_right1, y_right1, x_right3, y_right3)
                vector_left_hand = Vector(x_left1, y_left1, x_left3, y_left3)
                azimuth_right = cnt_angle(vector_right_hand, vector_north)
                azimuth_left = cnt_angle(vector_left_hand, vector_north)
                azimuth_right = (azimuth_right + 360) % 360
                azimuth_left = (azimuth_left + 360) % 360
                direction_right = "wrong"
                direction_left = "wrong"

                if is_straight(vector_rigth1, vector_right2):
                    if azimuth_right < 45 or azimuth_right > 315:
                        direction_right = "north"
                    elif azimuth_right < 135:
                        direction_right = "east"
                    elif azimuth_right < 225:
                        direction_right = "south"
                    else:
                        direction_right = "west"

                if is_straight(vector_left1, vector_left2):
                    if azimuth_left < 45 or azimuth_left > 315:
                        direction_left = "north"
                    elif azimuth_left < 135:
                        direction_left = "east"
                    elif azimuth_left < 225:
                        direction_left = "south"
                    else:
                        direction_left = "west"

            OK_RESULT = "OK!"
            if remaining_time <= 0:
                if direction_left == cur_direction == direction_right == "north" or \
                        direction_left == cur_direction == direction_right == "south":
                    result_text = OK_RESULT
                elif direction_left == cur_direction == "west" and direction_right != "east" \
                        or direction_right == cur_direction == "east" and direction_left != "west":
                    result_text = OK_RESULT
                else:
                    result_text = "WRONG!"

                state = STATE_RESULT
                state_time = time.time()
        elif state == STATE_RESULT:
            color = (0, 255, 0) if result_text == OK_RESULT else (0, 0, 255)

            cv2.putText(frame, result_text,
                        (260, 260), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)

            if time.time() - cur_time >= RESULT_TIME:
                state = STATE_GAME
                cur_time = time.time()
                cur_direction = random.choice(possible_directions)
                direction_right = direction_left = "wrong"

        cv2.imshow("", frame)

cap.release()
cv2.destroyAllWindows()
