print('Controls: ESC to quit, SPACE to freeze/unfreeze the camera (frees the webcam for other apps)')
import sys
import cv2
import numpy as np
import os
import time
from simple_signrec import SimpleSignRec
import subprocess

WINDOW_NAME = 'Sign Language'
PANEL_HEIGHT = 720  # height of each side of the split screen
HOLD_MS = 500  # a sign must stay the same this long before it is shown/spoken


def resource_path(name):
    """Resolve a data file whether running from source or inside a PyInstaller EXE."""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def fit_to_height(image, height):
    """Resize an image to the given height, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = height / h
    return cv2.resize(image, (int(w * scale), height))


def load_signs_panel(path, height):
    """Load the left-side reference image, or a placeholder if missing."""
    img = cv2.imread(path)
    if img is None:
        print("Warning: '{}' not found, left side will be blank.".format(path))
        img = np.zeros((height, height, 3), dtype=np.uint8)
        cv2.putText(img, "signs.jpg not found", (40, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img
    return fit_to_height(img, height)


signs_panel = load_signs_panel(resource_path('signs.jpg'), PANEL_HEIGHT)

cap = cv2.VideoCapture(0)

ssr = SimpleSignRec(distance_threshold=0.32)
model_path = resource_path('signs_model.joblib')
npz_path = resource_path('signs_model.npz')
if os.path.exists(model_path):
    ssr.load_svm(model_path)                           # trained SVM model
elif os.path.exists(npz_path):
    ssr.load_model(npz_path)                           # compact centroid model
else:
    ssr.load_encoding_signs("encodings")  # folder created by collect_signs.py

# Pre-generate speech audio for all known signs in the background, so
# speaking a sign is a local file playback instead of an online TTS call.
ssr.warm_up_tts()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(WINDOW_NAME, PANEL_HEIGHT + signs_panel.shape[1], PANEL_HEIGHT)

frozen = False
frozen_frame = None  # last displayed frame, shown while frozen
stable_signs = set()  # signs held long enough: shown on screen + spoken
pending_since = {}    # sign -> when it first appeared continuously
while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror view, feels more natural

        hand_boxes, hand_names = ssr.detect_known_signs(frame)

        now = time.time()
        # Track each sign independently, so two hands showing different
        # signs are both held/committed even if MediaPipe reorders the
        # hands between frames (which reset a single global timer).
        seen = {name for name in hand_names if name}
        for sign in list(pending_since):  # forget signs no longer in frame
            if sign not in seen:
                del pending_since[sign]
                stable_signs.discard(sign)
        # Commit + speak each sign once it has been held long enough.
        for sign in seen:
            pending_since.setdefault(sign, now)
            if (sign not in stable_signs and
                    (now - pending_since[sign]) * 1000 >= HOLD_MS):
                stable_signs.add(sign)
                ssr.speak(sign)

        for box, name in zip(hand_boxes, hand_names):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 0, 0), 4)
            # Only label the sign once it has stabilised.
            if name and name in stable_signs:
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 200), 2)

        frozen_frame = frame.copy()
    else:
        h, w = frozen_frame.shape[:2]
        frame = np.zeros((h, w, 3), dtype=np.uint8)  # black screen while frozen
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 150), -1)
        cv2.putText(frame, "FROZEN - camera released, press SPACE to resume",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    camera_panel = fit_to_height(frame, PANEL_HEIGHT)
    display = np.hstack([signs_panel, camera_panel])

    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    # Stop if the user clicked the window's close (X) button.
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord(' '):
        frozen = not frozen
        if frozen:
            # Release the webcam so other programs can use it.
            cap.release()
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Could not reopen the webcam.")
                break
subprocess.run("cls", shell=True)
cap.release()
cv2.destroyAllWindows()
