#!/usr/bin/env python3
"""
Spot the Fake Photo - one-line predictor.

Usage:
    python predict.py path/to/image.jpg

Prints a single float to stdout in [0, 1]:
    0 = REAL photo
    1 = PHOTO OF A SCREEN (recapture)

A human-readable label + latency is printed to stderr so stdout stays
clean for piping, e.g.:

    score=$(python predict.py photo.jpg)
"""
import os
import sys
import time

import joblib

from features import extract_features, load_image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, "models", "pipeline_v3.pkl")

_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.isfile(_MODEL_PATH):
            print(f"Error: model not found at {_MODEL_PATH}", file=sys.stderr)
            sys.exit(1)
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict(image_path: str) -> float:
    """Load an image, extract features, run the trained pipeline. Returns a
    float in [0, 1] where higher = more likely a screen/recapture."""
    img = load_image(image_path)
    feats = extract_features(img)
    model = _get_model()
    score = model.predict_proba([feats])[0][1]
    return float(score)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    score = predict(image_path)
    t1 = time.perf_counter()

    print(f"{score:.4f}")
    label = "SCREEN / RECAPTURE" if score >= 0.5 else "REAL"
    print(f"# {label}  ({(t1 - t0) * 1000:.1f} ms)", file=sys.stderr)


if __name__ == "__main__":
    main()