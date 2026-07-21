print('Controls: ESC to quit, SPACE to freeze/unfreeze the camera (frees the webcam for other apps)')
import cv2
from simple_signrec import SimpleSignRec
import subprocess

cap = cv2.VideoCapture(0)

ssr = SimpleSignRec(distance_threshold=0.32)
ssr.load_encoding_signs("encodings")  # folder created by collect_signs.py

speech = ''
frozen = False
frozen_frame = None  # last displayed frame, shown while frozen
while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror view, feels more natural

        hand_boxes, hand_names = ssr.detect_known_signs(frame)
        for box, name in zip(hand_boxes, hand_names):
            speech = name
            x1, y1, x2, y2 = box
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 0, 0), 4)

        frozen_frame = frame.copy()
        ssr.speak(speech)
    else:
        frame = frozen_frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 150), -1)
        cv2.putText(frame, "FROZEN - camera released, press SPACE to resume",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
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