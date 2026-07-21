# Sign Language Recognition (OpenCV + MediaPipe)

Real-time hand sign recognition using your webcam. Detects hand landmarks with
MediaPipe Hands, then matches them against gestures you've recorded yourself —
no pre-trained gesture dataset required.

## How it works

Unlike face recognition (which compares against photos), signs are defined by
hand *shape*, not appearance. This project:

1. Captures your hand's 21 landmark points via MediaPipe
2. Normalizes them (translation + scale invariant, anchored on the wrist) so
   it works regardless of hand size or distance from the camera
3. Matches new frames against your recorded gestures using nearest-neighbor
   distance on those normalized landmarks

## Files

| File | Purpose |
|---|---|
| `simple_signrec.py` | Core engine — landmark detection, normalization, matching |
| `collect_signs.py` | Interactive tool to record your own gesture samples |
| `main.py` | Live webcam recognition loop |
| `encodings/` | Auto-created folder storing recorded gesture data (`.npy` files) |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install opencv-python mediapipe numpy
```

> **Note:** MediaPipe works best on Python 3.10–3.11. If you hit
> `AttributeError: module 'mediapipe' has no attribute 'solutions'`, try a
> clean venv with Python 3.10/3.11, or pin `pip install mediapipe==0.10.9`.

## Usage

### 1. Record your gestures

```bash
python collect_signs.py
```

- Enter a label when prompted (e.g. `hello`, `yes`, `thank_you`)
- Position your hand in frame, press **SPACE** to capture a sample
  (aim for 15–30 samples per gesture, varying angle/position/distance each time)
- Press **ESC** to finish that gesture and move to the next
- Leave the label empty and press Enter to quit

Samples are saved to `encodings/<label>_<index>.npy`.

### 2. Run recognition

```bash
python main.py
```

Opens your webcam, draws a box and predicted label over each detected hand.
Press **ESC** to quit.

## Tuning accuracy

`distance_threshold` in `main.py` controls how strict matching is:

```python
ssr = SimpleSignRec(distance_threshold=0.15)
```

- **Lower** (e.g. `0.1`) → stricter, may reject valid signs from slightly
  different angles
- **Higher** (e.g. `0.25–0.3`) → more flexible, but similar-looking signs may
  get confused with each other

## Tips for better accuracy

- Record more samples per gesture, with real variation in angle, hand
  position, and distance from camera — not just the same pose repeated
- Keep gestures visually distinct from one another; near-identical hand
  shapes will be harder to tell apart
- If multiple people will use the system, consider recording samples from
  each person

## Known limitations

- This is a **static, single-frame** matcher — it recognizes hand *shapes*,
  not motion. True sign language often involves movement and two-hand
  coordination.
- For a larger vocabulary or motion-based signs, you'd want to record
  landmark *sequences* over time and train a temporal model (e.g. LSTM)
  instead of frame-by-frame nearest-neighbor matching.
