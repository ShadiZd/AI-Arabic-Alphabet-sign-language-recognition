import cv2
import os
import glob
import sys
import numpy as np
import mediapipe as mp
import joblib
from gtts import gTTS
import pygame
import threading
from collections import deque

try:
    # Ensures PyInstaller bundles the sklearn pieces the SVM model needs.
    from sklearn.pipeline import Pipeline  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    from sklearn.calibration import CalibratedClassifierCV  # noqa: F401
    from sklearn.svm import SVC  # noqa: F401
except ImportError:
    pass


class SimpleSignRec:
    """
    Hand sign recognizer, structured similarly to SimpleFacerec:
      - load_encoding_signs(path)  -> loads known gesture "encodings" from disk
      - detect_known_signs(frame)  -> returns hand boxes + predicted labels

    Instead of face photos, "encodings" here are normalized hand landmark
    vectors saved as .npy files (one or more per gesture, see collect_signs.py).
    File naming convention: <label>_<index>.npy  e.g. hello_0.npy, hello_1.npy
    """

    def __init__(self, distance_threshold=0.15, reference_hand='Right'):
        self.known_encodings = []   # list of normalized landmark vectors (63,)
        self.known_labels = []      # matching label for each encoding
        self.distance_threshold = distance_threshold  # lower = stricter match
        # Hand the encodings were recorded with. A hand of the other type is
        # mirrored before comparison, so the same sign matches on both hands.
        self.reference_hand = reference_hand
        # Below this SVM probability a sign is treated as "Unknown".
        self.confidence_threshold = 0.45
        self.model = None  # SVM set by load_svm(); cluster path used otherwise

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils
        pygame.mixer.init()
        pygame.mixer.music.set_volume(1)
        self.language = 'ar'
        self.last_word = None
        self.is_speaking = False
        self.speech_queue = deque()  # words waiting to be spoken, in order
        self.lock = threading.Lock()
        self.gen_lock = threading.Lock()  # serializes TTS generation
        self.tts_dir = self._default_tts_dir()

    # ---------- Loading known gestures ----------

    def load_encoding_signs(self, encodings_path):
        """
        Load pre-recorded landmark encodings from a folder of .npy files.
        Each file should contain a single normalized (21, 3) or (63,) array.
        """
        files = glob.glob(os.path.join(encodings_path, "*.npy"))
        print("{} sign encodings found.".format(len(files)))

        for file_path in files:
            basename = os.path.basename(file_path)
            filename, _ = os.path.splitext(basename)
            # label_0.npy -> label
            label = filename.rsplit("_", 1)[0] if "_" in filename else filename

            vec = np.load(file_path)
            vec = vec.flatten()
            self.known_encodings.append(vec)
            self.known_labels.append(label)

        print(len(set(self.known_labels)),"Sign encodings loaded:", sorted(set(self.known_labels)))

    def load_model(self, model_path):
        """
        Load a compact centroid model saved by build_model.py (a single
        .npz with one normalized centroid per gesture) instead of the
        whole folder of per-sample encodings.
        """
        data = np.load(model_path, allow_pickle=True)
        self.known_labels = list(data["labels"])
        self.known_encodings = list(data["centroids"])
        print("{} Sign classes loaded from:".format(len(self.known_labels)), model_path)

    def load_svm(self, model_path):
        """
        Load a trained SVM model (saved by build_model.py with joblib) and
        use it, with probabilities, instead of the raw encodings/centroids.
        """
        try:
            self.model = joblib.load(model_path)
        except ModuleNotFoundError:
            print("Missing dependency: scikit-learn is required to load the "
                  "SVM model (signs_model.joblib).")
            print("Install it with:  pip install scikit-learn")
            raise
        print("SVM model loaded: {} classes from {}".format(
            len(self.model.classes_), model_path))

    @property
    def labels(self):
        """All sign labels the recognizer currently knows about."""
        if self.model is not None:
            return list(self.model.classes_)
        return sorted(set(self.known_labels))

    # ---------- Landmark normalization ----------

    @staticmethod
    def normalize_landmarks(hand_landmarks, mirror=False):
        """
        Convert MediaPipe landmarks into a translation/scale-invariant vector.
        Anchors on the wrist (landmark 0) and scales by the hand size,
        so it works regardless of hand position/distance from camera.

        If mirror=True, the hand is reflected horizontally (x -> -x around the
        wrist), turning a left hand into its right-hand equivalent so both
        hands match the same recorded encoding.
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = pts[0]
        pts = pts - wrist  # translate so wrist is origin

        if mirror:
            pts[:, 0] = -pts[:, 0]  # reflect across the vertical axis

        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 0:
            pts = pts / scale  # scale-invariant

        return pts.flatten()  # shape (63,)

    # ---------- Detection / classification ----------

    def detect_known_signs(self, frame):
        """
        Detect hands in frame, classify each against known encodings.
        Returns:
            hand_boxes: list of [x1, y1, x2, y2] in original frame coords
            hand_names: list of predicted labels (or "Unknown")
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_boxes = []
        hand_names = []

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            # MediaPipe reports "Left"/"Right" for each detected hand, in the
            # same order as multi_hand_landmarks.
            handedness = []
            if results.multi_handedness:
                handedness = [
                    hd.classification[0].label for hd in results.multi_handedness
                ]

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Bounding box from landmark pixel coords
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))

                # Mirror the hand if it differs from the hand the encodings
                # were recorded with, so the sign matches on either hand.
                detected_hand = handedness[i] if i < len(handedness) else self.reference_hand
                mirror = detected_hand != self.reference_hand

                name = ""
                if self.model is not None or self.known_encodings:
                    query = self.normalize_landmarks(hand_landmarks, mirror=mirror)
                if self.model is not None:
                    proba = self.model.predict_proba([query])[0]
                    best = int(np.argmax(proba))
                    if proba[best] >= self.confidence_threshold:
                        name = self.model.classes_[best]
                elif self.known_encodings:
                    distances = [
                        np.linalg.norm(query - known)
                        for known in self.known_encodings
                    ]
                    best_idx = int(np.argmin(distances))
                    if distances[best_idx] < self.distance_threshold:
                        name = self.known_labels[best_idx]

                hand_boxes.append([x1, y1, x2, y2])
                hand_names.append(name)

        return hand_boxes, hand_names

    def draw_landmarks(self, frame, results_landmarks):
        """Optional: draw MediaPipe hand skeleton on frame for debugging."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame
    
    def _speak(self, word):
        """Speak one word (called by the background worker)."""
        path = self._synthesize(word)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(20)
        pygame.mixer.music.unload()

    def _synthesize(self, word):
        """Return the cached clip for a word, generating it with gTTS if missing."""
        with self.gen_lock:
            path = self._cached_path(word)
            if not os.path.exists(path):
                os.makedirs(self.tts_dir, exist_ok=True)
                gTTS(text=word, lang=self.language).save(path)
            return path

    def _cached_path(self, word):
        """Cache file path for a word (illegal filename characters removed)."""
        safe = "".join(c for c in word if c not in '\\/:*?"<>|')
        return os.path.join(self.tts_dir, safe + ".mp3")

    @staticmethod
    def _default_tts_dir():
        """Folder for cached audio clips (writable location next to exe/script)."""
        if getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "tts_cache")

    def warm_up_tts(self, labels=None):
        """Pre-generate speech audio for known labels in the background."""
        if labels is None:
            labels = self.labels
        todo = [lbl for lbl in labels
                if lbl and not os.path.exists(self._cached_path(lbl))]
        if not todo:
            print("Speech audio cache already complete:", self.tts_dir)
            return
        print("Pre-generating speech audio for {} signs -> {}".format(
            len(todo), self.tts_dir))
        threading.Thread(target=self._warm_up, args=(todo,), daemon=True).start()

    def _warm_up(self, labels):
        for label in labels:
            try:
                self._synthesize(label)
            except Exception as e:
                print("TTS pre-generation failed for '{}': {}".format(label, e))
        print("Speech audio cache ready:", self.tts_dir)

    def _speak_worker(self):
        """Drain the speech queue one word at a time, then shut down."""
        try:
            while True:
                with self.lock:
                    if not self.speech_queue:
                        return
                    word = self.speech_queue.popleft()
                try:
                    self._speak(word)
                except Exception as e:  # keep speaking the rest on TTS errors
                    print("TTS error while speaking '{}': {}".format(word, e))
        finally:
            with self.lock:
                self.is_speaking = False

    def speak(self, word):
        word = word.strip()
        if not word:
            return
        with self.lock:
            # Skip if it's the same as the previous queued word
            if word == self.last_word:
                return
            self.last_word = word
            self.speech_queue.append(word)
            # If a worker is already speaking, it will pick this word up
            # when it finishes the current one.
            if self.is_speaking:
                return
            self.is_speaking = True
        threading.Thread(
            target=self._speak_worker,
            daemon=True
        ).start()