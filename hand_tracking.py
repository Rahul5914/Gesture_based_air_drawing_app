"""
Hand tracking utility using the MediaPipe Tasks API (mediapipe >= 0.10).
Detects hand landmarks, finger states, and gesture recognition.

The hand landmark model is downloaded automatically on first run and cached
at ~/.cache/mediapipe/hand_landmarker.task
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import Optional, Tuple, List

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ── Model download ─────────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
_MODEL_PATH = os.path.join(_CACHE_DIR, "hand_landmarker.task")


def _ensure_model() -> str:
    """Download the MediaPipe hand landmark model if not already cached."""
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        print(f"[HandTracker] Downloading hand landmark model → {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[HandTracker] Download complete.")
    return _MODEL_PATH


# ── Connection pairs for drawing the hand skeleton ────────────────────────────
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandTracker:
    """
    Wraps the MediaPipe Tasks HandLandmarker for real-time landmark detection,
    gesture classification, and optional HUD drawing.
    """

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]

    GESTURE_DRAW  = "DRAW"
    GESTURE_STOP  = "STOP"
    GESTURE_CLEAR = "CLEAR"
    GESTURE_NONE  = "NONE"

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.75,
    ):
        model_path = _ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._timestamp_ms: int = 0

        self._result = None
        self.landmark_list: List[Tuple[int, int]] = []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run hand detection on a BGR frame. Updates landmark_list."""
        self._timestamp_ms += 33

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._result = self._landmarker.detect_for_video(
            mp_image, self._timestamp_ms
        )
        self._build_landmark_list(frame)
        return frame

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw hand skeleton connections and landmark dots."""
        if not self.landmark_list:
            return frame

        for (a, b) in _HAND_CONNECTIONS:
            if a < len(self.landmark_list) and b < len(self.landmark_list):
                cv2.line(frame, self.landmark_list[a], self.landmark_list[b],
                         (80, 200, 120), 2, cv2.LINE_AA)

        for i, pt in enumerate(self.landmark_list):
            color = (255, 255, 255) if i in self.FINGER_TIPS else (100, 220, 100)
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (30, 30, 30), 1, cv2.LINE_AA)

        return frame

    def get_finger_states(self) -> List[bool]:
        """Return [Thumb, Index, Middle, Ring, Pinky] up/down booleans."""
        if len(self.landmark_list) < 21:
            return [False] * 5

        states: List[bool] = []

        wrist   = self.landmark_list[0]
        mid_mcp = self.landmark_list[9]
        thumb_tip = self.landmark_list[4]
        thumb_ip  = self.landmark_list[3]

        if wrist[0] < mid_mcp[0]:
            states.append(thumb_tip[0] > thumb_ip[0])
        else:
            states.append(thumb_tip[0] < thumb_ip[0])

        for tip_id, pip_id in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            states.append(
                self.landmark_list[tip_id][1] < self.landmark_list[pip_id][1]
            )

        return states

    def get_index_finger_tip(self) -> Optional[Tuple[int, int]]:
        if len(self.landmark_list) >= 9:
            return self.landmark_list[8]
        return None

    def get_gesture(self) -> str:
        if not self.landmark_list:
            return self.GESTURE_NONE

        s = self.get_finger_states()
        fingers_up = sum(s[1:])

        if fingers_up >= 4:
            return self.GESTURE_CLEAR
        if s[1] and s[2] and not s[3] and not s[4]:
            return self.GESTURE_STOP
        if s[1] and not s[2] and not s[3] and not s[4]:
            return self.GESTURE_DRAW
        return self.GESTURE_NONE

    def hand_detected(self) -> bool:
        return len(self.landmark_list) == 21

    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.landmark_list:
            return None
        xs = [p[0] for p in self.landmark_list]
        ys = [p[1] for p in self.landmark_list]
        return min(xs), min(ys), max(xs), max(ys)

    def close(self) -> None:
        self._landmarker.close()

    def _build_landmark_list(self, frame: np.ndarray) -> None:
        self.landmark_list = []
        h, w = frame.shape[:2]
        if not self._result or not self._result.hand_landmarks:
            return
        for lm in self._result.hand_landmarks[0]:
            self.landmark_list.append((int(lm.x * w), int(lm.y * h)))
