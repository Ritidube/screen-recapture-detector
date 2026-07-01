"""
Spot the Fake Photo — Recapture Scanner (Streamlit edition)

One process, no separate backend: the upload widget, the feature
extraction (features.py — same file predict.py uses), and the trained
LightGBM cascade all live here. Deploy this single file + features.py +
models/pipeline_v3.pkl to a Streamlit HF Space and you're done.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
import os
import time

import joblib
import numpy as np
import streamlit as st
from PIL import Image

from features import extract_features, load_image

# ── Config ──────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, "models", "pipeline_v3.pkl")
_THRESHOLD = 0.5

st.set_page_config(
    page_title="Spot the Fake Photo — Recapture Scanner",
    page_icon="🕵️",
    layout="centered",
)

# ── Styling (dark cyberpunk-ish theme, matches the old HTML demo) ───
st.markdown(
    """
    <style>
    :root{
        --bg: #0b0d10; --panel: #13161b; --line: #232830;
        --ink: #e8ecef; --dim: #8a93a0; --real: #34d399; --screen: #f5a623;
    }
    .stApp{ background: var(--bg); color: var(--ink); }
    .eyebrow{
        font-family: 'JetBrains Mono', monospace; font-size: 11px;
        letter-spacing: .18em; text-transform: uppercase; color: var(--dim);
        margin-bottom: 6px;
    }
    .eyebrow .dot{
        display:inline-block; width:6px; height:6px; border-radius:50%;
        background: var(--real); box-shadow: 0 0 8px var(--real); margin-right:6px;
    }
    h1 span{ color: var(--screen); }
    .verdict{ font-size: 26px; font-weight: 700; margin: 4px 0 2px; }
    .verdict.real{ color: var(--real); }
    .verdict.screen{ color: var(--screen); }
    .score-num{
        font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--dim);
    }
    .meta-row{
        font-family: 'JetBrains Mono', monospace; font-size: 12px;
        color: var(--dim); margin-top: 10px;
    }
    .meta-row b{ color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="eyebrow"><span class="dot"></span>RECAPTURE DETECTOR · ON-DEVICE FEATURES</p>',
    unsafe_allow_html=True,
)
st.markdown("## Is that real, or a photo of a <span>screen</span>?", unsafe_allow_html=True)
st.caption(
    "Upload an image. It's analyzed for wavelet, moiré, noise, chromatic-aberration "
    "and glare signatures that distinguish a genuine photo from a screen recapture — "
    "no cloud vision model, just classic signal processing + a LightGBM cascade."
)


# ── Model loading (cached across reruns, loaded once per session) ───
@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.isfile(_MODEL_PATH):
        return None, f"Model not found at {_MODEL_PATH}. Did you commit models/pipeline_v3.pkl?"
    try:
        model = joblib.load(_MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


model, model_error = get_model()

if model_error:
    st.error(f"⚠️ {model_error}")
    st.stop()

# ── Upload + predict ─────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is not None:
    col1, col2 = st.columns([1, 1])

    # Save to a temp path so we can reuse features.load_image (cv2.imread)
    tmp_path = os.path.join(_THIS_DIR, f"_tmp_upload_{uploaded.file_id if hasattr(uploaded, 'file_id') else 'img'}.jpg")
    # Use a stable temp filename derived from the uploaded bytes to avoid collisions
    tmp_path = os.path.join(_THIS_DIR, "_tmp_upload.jpg")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with col1:
        st.image(Image.open(uploaded), caption="Preview", use_container_width=True)

    try:
        with st.spinner("Analyzing…"):
            t0 = time.perf_counter()
            img = load_image(tmp_path)
            feats = extract_features(img)
            score = float(model.predict_proba([feats])[0][1])
            t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 1)
        is_screen = score >= _THRESHOLD

        with col2:
            verdict_label = "SCREEN / RECAPTURE" if is_screen else "REAL PHOTO"
            verdict_class = "screen" if is_screen else "real"
            st.markdown(
                f'<div class="verdict {verdict_class}">{verdict_label}</div>'
                f'<div class="score-num">score {score:.2f}</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0.0), 1.0))
            st.markdown(
                f'<div class="meta-row">latency <b>{latency_ms} ms</b> &nbsp;·&nbsp; '
                f'backend <b>local (streamlit)</b></div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.info("No image uploaded yet.")

st.markdown("---")
st.caption(
    "Runs the same `features.py` extraction pipeline used by `predict.py`, in-process — "
    "no separate API call. No image is stored between runs; each upload is written to a "
    "temp file for OpenCV to read and deleted right after scoring."
)
