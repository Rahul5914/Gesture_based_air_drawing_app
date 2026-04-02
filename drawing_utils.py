"""
Drawing utilities for the Air Drawing App.
Handles canvas management, smooth line interpolation, and neon glow effects.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import deque


# ── Type aliases ──────────────────────────────────────────────────────────────
Point = Tuple[int, int]
Color = Tuple[int, int, int]          # BGR
ColorRGBA = Tuple[int, int, int, int]


# ── Canvas Manager ────────────────────────────────────────────────────────────

class Canvas:
    """
    Maintains a transparent drawing layer that is composited on top of the
    webcam feed.  All drawing is done on an RGBA canvas so transparency is
    preserved when overlaying on the video stream.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._canvas: np.ndarray = np.zeros((height, width, 4), dtype=np.uint8)

    # ── Public API ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Wipe the entire canvas."""
        self._canvas[:] = 0

    def get_canvas(self) -> np.ndarray:
        """Return the current RGBA canvas (read-only view)."""
        return self._canvas

    def resize(self, width: int, height: int) -> None:
        """Resize canvas, preserving existing content as best as possible."""
        if width == self.width and height == self.height:
            return
        new_canvas = np.zeros((height, width, 4), dtype=np.uint8)
        # Copy overlapping region
        h = min(height, self.height)
        w = min(width, self.width)
        new_canvas[:h, :w] = self._canvas[:h, :w]
        self._canvas = new_canvas
        self.width = width
        self.height = height

    def draw_segment(
        self,
        pt1: Point,
        pt2: Point,
        color_bgr: Color,
        thickness: int,
        glow: bool = True,
    ) -> None:
        """
        Draw a smooth line segment from pt1 to pt2 with optional neon glow.
        """
        if glow:
            self._draw_neon_segment(pt1, pt2, color_bgr, thickness)
        else:
            cv2.line(self._canvas, pt1, pt2, (*color_bgr, 255), thickness, cv2.LINE_AA)

    def draw_dot(
        self,
        pt: Point,
        color_bgr: Color,
        thickness: int,
        glow: bool = True,
    ) -> None:
        """Draw a single dot (for single-frame contacts)."""
        if glow:
            self._draw_neon_dot(pt, color_bgr, thickness)
        else:
            cv2.circle(self._canvas, pt, thickness // 2, (*color_bgr, 255), -1, cv2.LINE_AA)

    def composite_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Alpha-blend the canvas onto the BGR video frame.
        Returns a new BGR frame.
        """
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        alpha = self._canvas[:, :, 3:4].astype(np.float32) / 255.0
        canvas_bgr = self._canvas[:, :, :3].astype(np.float32)
        frame_f = frame.astype(np.float32)

        blended = frame_f * (1.0 - alpha) + canvas_bgr * alpha
        return blended.clip(0, 255).astype(np.uint8)

    def to_png_bytes(self) -> bytes:
        """Encode the canvas as a PNG byte string (for download)."""
        _, buf = cv2.imencode(".png", self._canvas)
        return buf.tobytes()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_neon_segment(
        self, pt1: Point, pt2: Point, color_bgr: Color, thickness: int
    ) -> None:
        """
        Multi-pass neon glow:
        1. Wide, low-opacity outer glow
        2. Medium mid-glow
        3. Bright core line
        """
        b, g, r = color_bgr

        # Outer glow (wide, ~20 % opacity)
        glow_thick = max(thickness * 4, 8)
        overlay = np.zeros_like(self._canvas)
        cv2.line(overlay, pt1, pt2, (b, g, r, 50), glow_thick, cv2.LINE_AA)
        self._alpha_blend_layer(overlay)

        # Mid glow (medium, ~50 % opacity)
        mid_thick = max(thickness * 2, 4)
        overlay = np.zeros_like(self._canvas)
        cv2.line(overlay, pt1, pt2, (b, g, r, 130), mid_thick, cv2.LINE_AA)
        self._alpha_blend_layer(overlay)

        # Core line (full opacity, white-tinted for "hot" look)
        core_color = self._tint_white(b, g, r, 0.35)
        cv2.line(self._canvas, pt1, pt2, (*core_color, 255), thickness, cv2.LINE_AA)

    def _draw_neon_dot(self, pt: Point, color_bgr: Color, thickness: int) -> None:
        b, g, r = color_bgr
        radius = max(thickness // 2, 2)

        overlay = np.zeros_like(self._canvas)
        cv2.circle(overlay, pt, radius * 3, (b, g, r, 50), -1, cv2.LINE_AA)
        self._alpha_blend_layer(overlay)

        overlay = np.zeros_like(self._canvas)
        cv2.circle(overlay, pt, radius * 2, (b, g, r, 130), -1, cv2.LINE_AA)
        self._alpha_blend_layer(overlay)

        core_color = self._tint_white(b, g, r, 0.35)
        cv2.circle(self._canvas, pt, radius, (*core_color, 255), -1, cv2.LINE_AA)

    def _alpha_blend_layer(self, layer: np.ndarray) -> None:
        """Blend `layer` (RGBA) onto self._canvas using over-compositing."""
        src_a = layer[:, :, 3:4].astype(np.float32) / 255.0
        dst_a = self._canvas[:, :, 3:4].astype(np.float32) / 255.0

        out_a = src_a + dst_a * (1.0 - src_a)
        # Avoid division by zero
        safe_out = np.where(out_a > 0, out_a, 1.0)

        for c in range(3):
            self._canvas[:, :, c] = np.clip(
                (layer[:, :, c] * src_a[:, :, 0]
                 + self._canvas[:, :, c] * dst_a[:, :, 0] * (1.0 - src_a[:, :, 0]))
                / safe_out[:, :, 0],
                0, 255
            ).astype(np.uint8)

        self._canvas[:, :, 3] = np.clip(out_a[:, :, 0] * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _tint_white(b: int, g: int, r: int, amount: float) -> Tuple[int, int, int]:
        """Blend a colour toward white by `amount` (0–1)."""
        b2 = int(b + (255 - b) * amount)
        g2 = int(g + (255 - g) * amount)
        r2 = int(r + (255 - r) * amount)
        return (b2, g2, r2)


# ── Stroke Manager ────────────────────────────────────────────────────────────

class StrokeManager:
    """
    Manages the current drawing stroke with Catmull-Rom spline smoothing.
    Keeps a small rolling buffer of recent points so lines are silky-smooth.
    """

    BUFFER_SIZE = 8   # How many recent points to keep for smoothing

    def __init__(self) -> None:
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._drawing: bool = False
        self._prev_point: Optional[Point] = None

    def start_stroke(self) -> None:
        self._drawing = True
        self._buffer.clear()
        self._prev_point = None

    def end_stroke(self) -> None:
        self._drawing = False
        self._buffer.clear()
        self._prev_point = None

    def is_drawing(self) -> bool:
        return self._drawing

    def add_point(
        self,
        point: Point,
        canvas: Canvas,
        color_bgr: Color,
        thickness: int,
        glow: bool = True,
    ) -> None:
        """
        Add a new point to the stroke buffer and draw to the canvas.
        Uses the smoothed midpoint technique for fluid curves.
        """
        if not self._drawing:
            return

        self._buffer.append(point)

        if len(self._buffer) < 2:
            canvas.draw_dot(point, color_bgr, thickness, glow)
            self._prev_point = point
            return

        # Smooth using rolling midpoints (chaikin-style)
        smooth_pts = self._smooth_buffer()

        if self._prev_point is not None:
            pt1 = self._prev_point
            pt2 = smooth_pts[-1]
            canvas.draw_segment(pt1, pt2, color_bgr, thickness, glow)

        self._prev_point = smooth_pts[-1]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _smooth_buffer(self) -> List[Point]:
        """
        Generate smoothed points from the buffer using midpoint averaging.
        Returns a list where the last element is the smoothed current tip.
        """
        pts = list(self._buffer)
        if len(pts) == 1:
            return pts

        smooth = []
        for i in range(len(pts) - 1):
            mx = (pts[i][0] + pts[i + 1][0]) // 2
            my = (pts[i][1] + pts[i + 1][1]) // 2
            smooth.append((mx, my))
        smooth.append(pts[-1])
        return smooth


# ── Helper: hex/rgb conversion ────────────────────────────────────────────────

def hex_to_bgr(hex_color: str) -> Color:
    """Convert '#RRGGBB' to (B, G, R) tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def bgr_to_hex(bgr: Color) -> str:
    """Convert (B, G, R) tuple to '#RRGGBB'."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# ── HUD overlay ───────────────────────────────────────────────────────────────

GESTURE_COLORS = {
    "DRAW":  (0, 255, 120),   # Green
    "STOP":  (0, 180, 255),   # Orange-yellow
    "CLEAR": (0, 60, 255),    # Red
    "NONE":  (180, 180, 180), # Gray
}

GESTURE_ICONS = {
    "DRAW":  "☝  DRAWING",
    "STOP":  "✌  PEN UP",
    "CLEAR": "✋  CLEARING",
    "NONE":  "   —",
}


def draw_hud(
    frame: np.ndarray,
    gesture: str,
    fps: float,
    color_bgr: Color,
    thickness: int,
) -> np.ndarray:
    """
    Render a minimal heads-up display (gesture label, FPS, colour swatch)
    onto the video frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 48), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Gesture label
    label = GESTURE_ICONS.get(gesture, "")
    g_color = GESTURE_COLORS.get(gesture, (200, 200, 200))
    cv2.putText(
        frame, label, (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, g_color, 2, cv2.LINE_AA
    )

    # FPS counter
    fps_label = f"FPS: {fps:.0f}"
    cv2.putText(
        frame, fps_label, (w - 110, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA
    )

    # Colour + thickness swatch (bottom-left corner)
    swatch_x, swatch_y = 12, h - 20
    r = max(thickness, 6)
    b, g, rv = color_bgr
    # Glow ring
    cv2.circle(frame, (swatch_x + r, swatch_y), r + 4, (b // 3, g // 3, rv // 3), -1)
    cv2.circle(frame, (swatch_x + r, swatch_y), r, (b, g, rv), -1)

    return frame
