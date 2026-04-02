# ✏️ AI Air Drawing App

Draw in the air using your **index finger** — tracked live by **MediaPipe** hand tracking, rendered in real-time with neon glow effects through **OpenCV** and **Streamlit**.

---

## 🎥 Demo

> *(Add a screen-recording GIF or screenshot here)*

---

## ✨ Features

| Feature | Details |
|---|---|
| Real-time hand tracking | MediaPipe Hands (single hand, 21 landmarks) |
| Gesture recognition | Draw / Pen Up / Clear canvas |
| Neon glow drawing | Multi-pass alpha-blend glow effect |
| Smooth strokes | Rolling midpoint / Chaikin-style smoothing |
| HUD overlay | Gesture label + FPS counter + colour swatch |
| Colour picker | Full HSV colour picker in sidebar |
| Quick colours | 5 one-click neon palette swatches |
| Brush thickness | 2 – 30 px slider |
| Download drawing | PNG export of the current canvas |
| FPS counter | Live calculation, shown in UI metric + HUD |

---

## 🤌 Gesture Guide

| Hand gesture | Action |
|---|---|
| ☝️ **Index finger only** | Draw — follow your fingertip |
| ✌️ **Index + Middle finger** | Pen Up — lift the pen without clearing |
| ✋ **Open hand (4 – 5 fingers)** | Hold for ~8 frames → **Clear canvas** |

---

## 📁 Project Structure

```
gesture-drawing-app/
│
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── utils/
    ├── __init__.py
    ├── hand_tracking.py    ← MediaPipe wrapper + gesture logic
    └── drawing_utils.py    ← Canvas, StrokeManager, neon glow, HUD
```

---

## 🚀 Run Locally

### Prerequisites

- Python **3.9 – 3.11** (MediaPipe is not yet compatible with 3.12+)
- A working webcam

### 1 – Clone / download

```bash
git clone https://github.com/YOUR_USERNAME/gesture-drawing-app.git
cd gesture-drawing-app
```

### 2 – Create a virtual environment (recommended)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> **Linux users:** you may need extra system libraries for OpenCV:
> ```bash
> sudo apt-get install -y libgl1 libglib2.0-0
> ```

### 4 – Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

Click **▶ Start Webcam**, allow camera access, and start drawing! 🎨

---

## ☁️ Deploy on Streamlit Cloud

1. **Push this repo to GitHub** (make sure `requirements.txt` is at the root).

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → *New app*.

3. Choose your repo, branch (`main`), and set **Main file path** to `app.py`.

4. Click **Deploy**.

> ⚠️ **Important:** Streamlit Cloud does **not** have a physical webcam attached.  
> The app will warn you that the camera cannot be opened.  
> For real webcam usage, run the app **locally** or on a machine with a camera (e.g., a Linux VM with a virtual webcam / v4l2loopback).  
> Alternatively, integrate `streamlit-webrtc` for browser-based webcam streaming — see the note below.

### Optional: streamlit-webrtc (for true cloud deployment)

For a cloud-deployable version that streams the webcam **from the user's browser**, replace the `cv2.VideoCapture` loop in `app.py` with a `streamlit-webrtc` `VideoTransformerBase` component.  Add to `requirements.txt`:

```
streamlit-webrtc>=0.47.1
aiortc>=1.6.0
```

---

## 🛠 Configuration

| Sidebar control | Default | Notes |
|---|---|---|
| Pen Colour | `#00f5c4` (neon teal) | Full hex colour picker |
| Brush Thickness | 6 | Range 2 – 30 px |
| Neon Glow Effect | On | Toggleable |
| Camera Index | 0 | Try 1/2 for external webcams |

---

## 🧠 Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[MediaPipe](https://mediapipe.dev)** — Hand landmark detection
- **[OpenCV](https://opencv.org)** — Frame processing, drawing, compositing
- **[NumPy](https://numpy.org)** — Array operations for alpha blending

---

## 📸 Screenshots

> *(Add screenshots here — draw a few strokes and take a screenshot!)*

---

## 🤝 Contributing

PRs welcome! Feel free to open issues for bugs or feature requests.

---

## 📄 Licence

MIT — do whatever you like with it.
