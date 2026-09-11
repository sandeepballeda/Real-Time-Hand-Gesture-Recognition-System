import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time
import os
import sys
from collections import deque

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks

MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "gesture_rf.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "data", "models", "label_encoder.pkl")

clf = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ---------------------------------------------------------
# Existing gesture-control settings
# ---------------------------------------------------------
PRED_HISTORY_SIZE = 3
pred_history = deque(maxlen=PRED_HISTORY_SIZE)

COMMAND_COOLDOWN = 0.6
last_command_time = 0

prev_wrist_y = None
SCROLL_THRESHOLD = 30
SCROLL_AMOUNT = 5

last_gesture_for_command = None


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def majority_vote(history):
    if not history:
        return None
    values, counts = np.unique(history, return_counts=True)
    return values[np.argmax(counts)]


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) -
                                np.array(p2, dtype=np.float32)))


def hand_scale(coords):
    """Approximate hand size using wrist-to-middle-MCP distance."""
    return max(distance(coords[0], coords[9]), 1.0)


def finger_states(coords):
    """
    Return whether thumb/index/middle/ring/pinky are extended.

    For the four non-thumb fingers, compare fingertip distance from
    the wrist with the PIP-joint distance from the wrist. This works
    better than relying only on image Y coordinates.

    Thumb uses the thumb tip vs thumb IP distance from the wrist.
    """
    wrist = coords[0]

    states = {}

    # Index, middle, ring, pinky: TIP should be farther from wrist
    # than the corresponding PIP joint when extended.
    for name, tip, pip in [
        ("INDEX", 8, 6),
        ("MIDDLE", 12, 10),
        ("RING", 16, 14),
        ("PINKY", 20, 18),
    ]:
        states[name] = distance(coords[tip], wrist) > distance(coords[pip], wrist) * 1.10

    # Thumb: compare tip with IP joint.
    states["THUMB"] = distance(coords[4], wrist) > distance(coords[3], wrist) * 1.12

    return states


def describe_fingers(states):
    """Return a human-readable finger/gesture description."""
    extended = [name for name, is_up in states.items() if is_up]

    if len(extended) == 0:
        return "FIST"

    if len(extended) == 5:
        return "OPEN_PALM"

    if len(extended) == 1:
        return f"{extended[0]}_UP"

    if extended == ["INDEX", "MIDDLE"]:
        return "TWO_FINGERS"

    if extended == ["INDEX", "MIDDLE", "RING"]:
        return "THREE_FINGERS"

    if extended == ["INDEX", "MIDDLE", "RING", "PINKY"]:
        return "FOUR_FINGERS"

    return "+".join(extended)


def is_open_palm(coords):
    states = finger_states(coords)
    return sum(states.values()) >= 4


def palm_center(coords):
    """Center of the palm using wrist and MCP landmarks."""
    ids = [0, 5, 9, 13, 17]
    pts = np.array([coords[i] for i in ids], dtype=np.float32)
    return tuple(np.mean(pts, axis=0))


def are_hands_joined(coords1, coords2):
    """
    Detect two hands that are close together.

    The threshold is scaled by hand size so it works at different
    camera distances.
    """
    center1 = palm_center(coords1)
    center2 = palm_center(coords2)

    palm_distance = distance(center1, center2)
    scale = (hand_scale(coords1) + hand_scale(coords2)) / 2.0

    # Tune between ~1.25 and ~1.8 if necessary.
    return palm_distance < scale * 1.45

def send_command(gesture, wrist_y):
    global last_command_time, prev_wrist_y, last_gesture_for_command

    now = time.time()

    if now - last_command_time < COMMAND_COOLDOWN:
        return

    # FIST = scrolling
    if gesture == "FIST":
        if prev_wrist_y is not None and wrist_y is not None:
            dy = wrist_y - prev_wrist_y

            if dy > SCROLL_THRESHOLD:
                print("Scroll down")
                pyautogui.scroll(-SCROLL_AMOUNT)
                last_command_time = now

            elif dy < -SCROLL_THRESHOLD:
                print("Scroll up")
                pyautogui.scroll(SCROLL_AMOUNT)
                last_command_time = now

        prev_wrist_y = wrist_y
        return

    prev_wrist_y = None

    # Existing arrow gestures require two consecutive detections.
    if gesture in ["INDEX_RIGHT", "INDEX_LEFT", "INDEX_UP", "INDEX_DOWN"]:
        if gesture != last_gesture_for_command:
            last_gesture_for_command = gesture
            return
    else:
        last_gesture_for_command = None

    if gesture == "INDEX_RIGHT":
        print("Command: RIGHT_ARROW")
        pyautogui.press("right")
        last_command_time = now

    elif gesture == "INDEX_LEFT":
        print("Command: LEFT_ARROW")
        pyautogui.press("left")
        last_command_time = now

    elif gesture == "INDEX_UP":
        print("Command: UP_ARROW")
        pyautogui.press("up")
        last_command_time = now

    elif gesture == "INDEX_DOWN":
        print("Command: DOWN_ARROW")
        pyautogui.press("down")
        last_command_time = now


# ---------------------------------------------------------
# Main realtime system
# ---------------------------------------------------------
def run_realtime_control():
    global prev_wrist_y, last_gesture_for_command

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("!!! Camera could not be opened!")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,                 # IMPORTANT: two hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("▶ Upgraded realtime gesture control started.")
        print("   One hand: finger recognition + existing ML gestures")
        print("   Two hands: HANDS_JOINED detection")
        print("   Press 'q' or ESC in the video window to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("!! Frame could not be read, exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                all_hands = []

                # ---------------------------------------------
                # Collect all detected hands
                # ---------------------------------------------
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.append((
                            int(lm.x * w),
                            int(lm.y * h)
                        ))

                    all_hands.append(coords)

                # ---------------------------------------------
                # TWO HANDS
                # ---------------------------------------------
                if len(all_hands) >= 2:
                    coords1 = all_hands[0]
                    coords2 = all_hands[1]

                    if are_hands_joined(coords1, coords2):
                        gesture_text = "HANDS_JOINED"

                        # Prevent accidental keyboard commands.
                        pred_history.clear()
                        prev_wrist_y = None
                        last_gesture_for_command = None

                        cv2.putText(
                            frame,
                            "HANDS JOINED",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            "2 HANDS DETECTED",
                            (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )

                    else:
                        gesture_text = "TWO_HANDS"

                        cv2.putText(
                            frame,
                            "TWO HANDS",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            2
                        )

                        # Show finger states for each hand.
                        for idx, coords in enumerate(all_hands[:2]):
                            states = finger_states(coords)
                            description = describe_fingers(states)

                            cv2.putText(
                                frame,
                                f"Hand {idx + 1}: {description}",
                                (10, 75 + idx * 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (255, 255, 255),
                                2
                            )

                    # Do not run the one-hand ML command logic
                    # while two hands are present.
                    cv2.imshow("Realtime Gesture Control", frame)

                # ---------------------------------------------
                # ONE HAND
                # ---------------------------------------------
                else:
                    coords = all_hands[0]
                    wrist_y = coords[0][1]

                    # Finger-state detection
                    states = finger_states(coords)
                    finger_description = describe_fingers(states)

                    # Existing ML model prediction
                    norm_coords_flat = normalize_landmarks(coords)
                    feat = norm_coords_flat.reshape(1, -1)

                    probs = clf.predict_proba(feat)[0]
                    class_idx = int(np.argmax(probs))

                    predicted_label = label_encoder.inverse_transform(
                        [class_idx]
                    )[0]

                    predicted_proba = float(probs[class_idx])

                    pred_history.append(predicted_label)
                    smooth_label = majority_vote(pred_history)

                    # Keep existing behavior: open palm = IDLE.
                    if is_open_palm(coords):
                        smooth_label = "IDLE"
                        prev_wrist_y = None
                        last_gesture_for_command = None

                    # Display ML gesture
                    cv2.putText(
                        frame,
                        f"ML: {smooth_label} ({predicted_proba * 100:.1f}%)",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2
                    )

                    # Display individual finger interpretation
                    cv2.putText(
                        frame,
                        f"Fingers: {finger_description}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    # Display each finger state
                    state_text = " ".join(
                        f"{name}:{'UP' if states[name] else 'DOWN'}"
                        for name in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
                    )

                    cv2.putText(
                        frame,
                        state_text,
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        2
                    )

                    # Existing keyboard controls
                    if smooth_label != "IDLE":
                        send_command(smooth_label, wrist_y)

            else:
                pred_history.clear()
                prev_wrist_y = None
                last_gesture_for_command = None

                cv2.putText(
                    frame,
                    "NO HAND DETECTED",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Realtime Gesture Control", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("⏹ Exiting realtime control.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_control()
