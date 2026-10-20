"""
Train an SVM classifier from the per-sample encodings.

Usage:
    python build_model.py

Reads every .npy encoding in the "encodings" folder, trains a multi-class
RBF SVM that can output probabilities, and saves a compact model file.
Recognition (main.py) loads it via simple_signrec's load_svm().

Output: signs_model.joblib (tens of KB instead of the whole dataset).
"""

import os
import glob
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

INPUT_DIR = "encodings"
OUTPUT_FILE = "signs_model.joblib"
TEST_SIZE = 0.2


def ascii_safe(vals):
    return [v.encode('ascii', 'backslashreplace').decode() for v in vals]


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.npy"))
    if not files:
        print("No encodings found in:", INPUT_DIR)
        return

    X, y = [], []
    for file_path in files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # Same naming rule as simple_signrec.py: <label>_<index>
        label = filename.rsplit("_", 1)[0] if "_" in filename else filename
        X.append(np.load(file_path).flatten())
        y.append(label)

    X = np.stack(X)
    y = np.array(y)
    print("Samples:", len(files), "Classes:", len(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

    model = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(kernel='rbf', C=100, gamma='scale'), ensemble=False
        ),
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("Test accuracy: {:.1%} on {} held-out samples".format(acc, len(y_test)))

    # Per-class mean confidence on the held-out set, to help pick the
    # confidence threshold used for rejecting unknown signs.
    proba = model.predict_proba(X_test)
    for i, cls in enumerate(model.classes_):
        cond = y_test == cls
        if cond.sum():
            mean_conf = proba[cond][:, i].mean()
            print("  {:10s} mean conf: {:.2f}".format(ascii_safe([cls])[0], mean_conf))

    joblib.dump(model, OUTPUT_FILE)
    print("Saved model ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()