"""
AI Air Drawing App
==================
Draw in the air using your index finger, tracked by MediaPipe.

Gestures:
  ☝  Index only        → Draw
  ✌  Index + Middle    → Stop / lift pen
  ✋  Open hand (4-5f)  → Clear canvas
"""

import time
import sys
import os

import cv2
import numpy as np
import streamlit as st

# ── Path setup (so `utils` resolves whether run from project root or anywhere)
sys.path.insert(0, os.path.dirname(__file__))

from utils.hand_tracking import HandTracker
from utils.drawing_utils import (
    Canvas,
    StrokeManager,
    hex_to_bgr,
    draw_hud,
    bgr_to_hex,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Air Drawing App",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS – dark neon theme ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        background-color: #080c14 !important;
        color: #c8d8f0 !important;
        font-family: 'Share Tech Mono', monospace;
    }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 2px; }
    h1 { color: #00f5c4 !important; text-shadow: 0 0 18px #00f5c480; font-size: 1.8rem !important; }
    h2 { color: #7eb8ff !important; font-size: 1.1rem !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1220 !important;
        border-right: 1px solid #1a2a44;
    }

    /* Buttons */
    div.stButton > button {
        background: transparent;
        border: 1px solid #00f5c4;
        color: #00f5c4;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 1px;
        border-radius: 4px;
        padding: 0.45rem 1rem;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: #00f5c420;
        box-shadow: 0 0 12px #00f5c460;
    }
    div.stButton > button:active { background: #00f5c440; }

    /* Download button (matches clear btn style) */
    div.stDownloadButton > button {
        background: transparent;
        border: 1px solid #7eb8ff;
        color: #7eb8ff;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 1px;
        border-radius: 4px;
        padding: 0.45rem 1rem;
        transition: all 0.2s;
    }
    div.stDownloadButton > button:hover {
        background: #7eb8ff20;
        box-shadow: 0 0 12px #7eb8ff50;
    }

    /* Slider */
    .stSlider > div > div > div > div { background: #00f5c4 !important; }

    /* Color picker */
    .stColorPicker label { color: #c8d8f0 !important; }

    /* Info boxes */
    .stInfo { background: #0d1a2e !important; border-left: 3px solid #00f5c4; color: #a0c4e8 !important; }
    .stSuccess { border-left: 3px solid #00f5c4; }
    .stWarning { border-left: 3px solid #ffcc44; }

    /* Image / webcam frame container */
    div[data-testid="stImage"] img {
        border: 1px solid #1a2a44;
        border-radius: 6px;
    }

    /* Metric */
    div[data-testid="metric-container"] {
        background: #0d1220;
        border: 1px solid #1a2a44;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    div[data-testid="metric-container"] label { color: #7eb8ff !important; font-size: 0.7rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00f5c4 !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.4rem !important;
    }

    hr { border-color: #1a2a44; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✏️ Air Drawing App")
    st.markdown("---")

    st.markdown("### 🎮 Gesture Guide")
    st.markdown(
        """
        | Gesture | Action |
        |---------|--------|
        | ☝️ Index only | **Draw** |
        | ✌️ Index + Middle | **Pen Up** |
        | ✋ Open hand | **Clear** |
        """
    )
    st.markdown("---")

    st.markdown("### ⚙️ Drawing Controls")
    draw_color_hex = st.color_picker("Pen Colour", "#00f5c4")
    brush_size = st.slider("Brush Thickness", 2, 30, 6, 1)
    enable_glow = st.toggle("Neon Glow Effect ✨", value=True)

    st.markdown("---")
    st.markdown("### 📷 Camera")
    cam_index = st.number_input("Camera Index", 0, 5, 0, 1)

    st.markdown("---")
    st.markdown("### 🌈 Quick Colours")
    quick_cols = st.columns(5)
    quick_colors = {
        "🟢": "#00f5c4",
        "🔵": "#4da6ff",
        "🟣": "#bf5fff",
        "🔴": "#ff4466",
        "🟡": "#ffdd00",
    }
    if "qcolor" not in st.session_state:
        st.session_state.qcolor = None

    for idx, (emoji, hx) in enumerate(quick_colors.items()):
        with quick_cols[idx]:
            if st.button(emoji, key=f"qc_{idx}"):
                st.session_state.qcolor = hx

    if st.session_state.qcolor:
        draw_color_hex = st.session_state.qcolor

    st.markdown("---")
    st.caption("Built with MediaPipe · OpenCV · Streamlit")

# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "running": False,
        "canvas": None,
        "stroke_mgr": None,
        "tracker": None,
        "cap": None,
        "frame_count": 0,
        "fps": 0.0,
        "t0": time.time(),
        "last_gesture": "NONE",
        "gesture_hold": 0,       # Frames gesture has been held
        "download_png": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# ✏️ AI Air Drawing App")
st.markdown(
    "<p style='color:#7eb8ff;font-size:0.85rem;margin-top:-0.5rem;'>"
    "Draw in the air using your index finger — powered by MediaPipe hand tracking</p>",
    unsafe_allow_html=True,
)

# ── Top controls row ──────────────────────────────────────────────────────────
col_start, col_clear, col_dl, col_fps, col_gesture = st.columns([1.2, 1.2, 1.2, 1, 1.5])

with col_start:
    start_label = "⏹  Stop Webcam" if st.session_state.running else "▶  Start Webcam"
    if st.button(start_label, use_container_width=True):
        if st.session_state.running:
            # ── Stop ──────────────────────────────────────────────────────────
            st.session_state.running = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            if st.session_state.tracker:
                st.session_state.tracker.close()
                st.session_state.tracker = None
        else:
            # ── Start ─────────────────────────────────────────────────────────
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error("⚠️ Could not open webcam. Check the Camera Index in the sidebar.")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)

                st.session_state.cap = cap
                st.session_state.tracker = HandTracker(
                    max_hands=1,
                    detection_confidence=0.75,
                    tracking_confidence=0.75,
                )
                st.session_state.stroke_mgr = StrokeManager()
                st.session_state.running = True
                st.session_state.frame_count = 0
                st.session_state.t0 = time.time()
                st.session_state.download_png = None

                # Canvas size will be set after first frame; placeholder for now
                st.session_state.canvas = None

with col_clear:
    if st.button("🗑  Clear Canvas", use_container_width=True):
        if st.session_state.canvas:
            st.session_state.canvas.clear()
            if st.session_state.stroke_mgr:
                st.session_state.stroke_mgr.end_stroke()

with col_dl:
    if st.session_state.download_png:
        st.download_button(
            "⬇  Download",
            data=st.session_state.download_png,
            file_name="air_drawing.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("⬇  Download", disabled=True, use_container_width=True)

with col_fps:
    st.metric("FPS", f"{st.session_state.fps:.0f}")

with col_gesture:
    g_icon = {
        "DRAW": "☝ Drawing",
        "STOP": "✌ Pen Up",
        "CLEAR": "✋ Clearing",
        "NONE": "— No Gesture",
    }.get(st.session_state.last_gesture, "—")
    st.metric("Gesture", g_icon)

# ── Main feed ─────────────────────────────────────────────────────────────────
feed_placeholder = st.empty()

if not st.session_state.running:
    # Idle splash
    feed_placeholder.markdown(
        """
        <div style="
            height:420px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            border:1px solid #1a2a44;
            border-radius:8px;
            background:#0d1220;
            color:#2a4060;
            font-family:'Orbitron',sans-serif;
            letter-spacing:2px;
        ">
            <div style="font-size:4rem; margin-bottom:1rem;">✏️</div>
            <div style="font-size:1.1rem;">Press <span style="color:#00f5c4">▶ Start Webcam</span> to begin</div>
            <div style="font-size:0.7rem; margin-top:0.5rem; color:#1a3050;">
                Make sure your browser has camera permission
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ── Frame loop ────────────────────────────────────────────────────────────
    CLEAR_HOLD_FRAMES = 8   # Hold "open hand" for this many frames to clear

    while st.session_state.running:
        cap: cv2.VideoCapture = st.session_state.cap
        if cap is None or not cap.isOpened():
            st.warning("Webcam disconnected.")
            st.session_state.running = False
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read frame.")
            break

        # Mirror horizontally for natural feel
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ── Lazy canvas init (now we know frame dimensions) ───────────────────
        if st.session_state.canvas is None:
            st.session_state.canvas = Canvas(w, h)
        else:
            st.session_state.canvas.resize(w, h)

        canvas: Canvas = st.session_state.canvas
        tracker: HandTracker = st.session_state.tracker
        stroke_mgr: StrokeManager = st.session_state.stroke_mgr

        # ── Hand tracking ─────────────────────────────────────────────────────
        tracker.process_frame(frame)
        gesture = tracker.get_gesture()
        index_tip = tracker.get_index_finger_tip()

        color_bgr = hex_to_bgr(draw_color_hex)

        # ── Gesture-based color switching (bonus) ─────────────────────────────
        # Color changes via quick-color buttons are already reflected through
        # `draw_color_hex` which updates every Streamlit re-run.

        # ── Gesture logic ─────────────────────────────────────────────────────
        if gesture == "DRAW" and index_tip:
            if not stroke_mgr.is_drawing():
                stroke_mgr.start_stroke()
            stroke_mgr.add_point(index_tip, canvas, color_bgr, brush_size, enable_glow)
            st.session_state.gesture_hold = 0

        elif gesture == "STOP":
            if stroke_mgr.is_drawing():
                stroke_mgr.end_stroke()
            st.session_state.gesture_hold = 0

        elif gesture == "CLEAR":
            if stroke_mgr.is_drawing():
                stroke_mgr.end_stroke()
            st.session_state.gesture_hold += 1
            if st.session_state.gesture_hold >= CLEAR_HOLD_FRAMES:
                canvas.clear()
                st.session_state.gesture_hold = 0

        else:  # NONE
            if stroke_mgr.is_drawing():
                stroke_mgr.end_stroke()
            st.session_state.gesture_hold = 0

        # ── Draw hand skeleton ────────────────────────────────────────────────
        if tracker.hand_detected():
            tracker.draw_landmarks(frame)

        # ── Composite canvas on frame ─────────────────────────────────────────
        frame = canvas.composite_on_frame(frame)

        # ── HUD overlay ───────────────────────────────────────────────────────
        frame = draw_hud(frame, gesture, st.session_state.fps, color_bgr, brush_size)

        # Draw cursor dot at finger tip
        if index_tip and gesture == "DRAW":
            b, g_ch, r = color_bgr
            cv2.circle(frame, index_tip, brush_size + 4, (b, g_ch, r), 2, cv2.LINE_AA)
            cv2.circle(frame, index_tip, brush_size, (255, 255, 255), -1, cv2.LINE_AA)

        # ── FPS calculation ───────────────────────────────────────────────────
        st.session_state.frame_count += 1
        elapsed = time.time() - st.session_state.t0
        if elapsed >= 1.0:
            st.session_state.fps = st.session_state.frame_count / elapsed
            st.session_state.frame_count = 0
            st.session_state.t0 = time.time()

        # ── Update session state for metrics ─────────────────────────────────
        st.session_state.last_gesture = gesture

        # ── Prepare download PNG (save current composite) ────────────────────
        st.session_state.download_png = canvas.to_png_bytes()

        # ── Render frame to Streamlit ─────────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Small sleep to yield to Streamlit scheduler
        time.sleep(0.005)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:0.7rem;color:#2a4060;'>"
    "AI Air Drawing App · MediaPipe + OpenCV + Streamlit · "
    "Raise your ✋ and wave to clear the canvas!"
    "</p>",
    unsafe_allow_html=True,
)
