"""
Run this first, once per gesture you want to recognize.

Usage:
    python collect_signs.py

Controls:
    - Type the gesture label when prompted (e.g. "hello", "yes", "thank_you")
    - Press SPACE to capture a sample while your hand is in frame
      (capture ~15-20 samples per gesture, varying hand angle/position slightly)
    - Press ESC to finish that gesture and move to the next, or quit

Samples are saved as .npy files into the "encodings" folder as:
    <label>_<index>.npy
which simple_signrec.py's load_encoding_signs() reads automatically.
"""

import cv2
import os
import numpy as np
from simple_signrec import SimpleSignRec

OUTPUT_DIR = "encodings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sfr = SimpleSignRec()
cap = cv2.VideoCapture(0)

while True:
    label = input("Enter gesture label (or leave empty to quit): ").strip()
    if not label:
        break

    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(label + "_")]
    sample_idx = len(existing)

    print(f"Recording '{label}'. SPACE = capture, ESC = done with this gesture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = sfr.hands.process(rgb_frame)

        display = frame.copy()
        current_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            current_landmarks = hand_landmarks
            sfr.mp_draw.draw_landmarks(
                display, hand_landmarks, sfr.mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(display, f"Label: {label}  Samples: {sample_idx}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Signs", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and current_landmarks is not None:  # SPACE
            vec = sfr.normalize_landmarks(current_landmarks)
            save_path = os.path.join(OUTPUT_DIR, f"{label}_{sample_idx}.npy")
            np.save(save_path, vec)
            print("Saved:", save_path)
            sample_idx += 1

cap.release()
cv2.destroyAllWindows()
print("Done collecting. Encodings saved in:", OUTPUT_DIR)
