"""
# FastAPI backend for the Spot the Fake Photo live demo.

# Endpoints:
#     GET  /health   -> {"status": "ok"}
#     POST /predict  -> multipart/form-data with field "image"
#                        returns {"score": float, "label": str, "latency_ms": float}

# Run locally:
#     uvicorn backend.app:app --reload --port 8000

# The model is loaded once at startup and reused across requests (no reload
# per-request), so steady-state latency matches predict.py's per-image cost.
# """
# import os
# import sys
# import tempfile
# import time

# import joblib
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware

# # Make the repo root importable so we can reuse features.py (the same
# # extractor predict.py uses) instead of duplicating it.
# _BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# _REPO_ROOT = os.path.dirname(_BACKEND_DIR)
# sys.path.insert(0, _REPO_ROOT)

# from features import extract_features, load_image  # noqa: E402

# _MODEL_PATH = os.path.join(_REPO_ROOT, "models", "pipeline_v3.pkl")
# _ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# app = FastAPI(title="Spot the Fake Photo API", version="1.0")

# # Tighten allow_origins to your deployed frontend's URL before going to
# # production (e.g. ["https://yourname.github.io"]).
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# _model = None


# def _get_model():
#     global _model
#     if _model is None:
#         if not os.path.isfile(_MODEL_PATH):
#             raise RuntimeError(f"Model not found at {_MODEL_PATH}")
#         _model = joblib.load(_MODEL_PATH)
#     return _model


# @app.on_event("startup")
# def _warm_up():
#     """Load the model once at boot instead of on the first request."""
#     _get_model()


# @app.get("/health")
# def health():
#     return {"status": "ok"}


# @app.post("/predict")
# async def predict(image: UploadFile = File(...)):
#     if image.content_type not in _ALLOWED_TYPES:
#         raise HTTPException(400, f"Unsupported content type: {image.content_type}")

#     suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
#     tmp_path = None
#     try:
#         with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
#             tmp.write(await image.read())
#             tmp_path = tmp.name

#         t0 = time.perf_counter()
#         img = load_image(tmp_path)
#         feats = extract_features(img)
#         model = _get_model()
#         score = float(model.predict_proba([feats])[0][1])
#         t1 = time.perf_counter()
#     except Exception as e:
#         raise HTTPException(400, f"Could not process image: {e}")
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)

#     return {
#         "score": round(score, 4),
#         "label": "screen" if score >= 0.5 else "real",
#         "latency_ms": round((t1 - t0) * 1000, 1),
#     }

"""
Plain FastAPI backend for the Spot the Fake Photo live demo.

Endpoints:
    GET  /        -> redirects to /docs (so the bare URL doesn't 404)
    GET  /health  -> {"status": "ok", "model_loaded": bool}
    POST /predict -> multipart/form-data with field "image"
                      returns {"score": float, "label": str, "latency_ms": float}

Run it:
    cd backend
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
    # or, from the repo root:
    uvicorn backend.app:app --reload --port 8000

Then open http://127.0.0.1:8000/docs to test /predict directly, or point
frontend/index.html's "API" field at this URL.

Two things make this safe to run anywhere (your own machine, a VPS, a
container) without surprises:
  - Startup never raises. If models/pipeline_v3.pkl is missing or fails to
    load, the server still starts; /predict just returns a clear 503 until
    the model is in place, and /health reports model_loaded + model_error
    so you can see why.
  - The model is loaded once at startup (when available) and reused across
    requests, so steady-state latency matches predict.py's per-image cost.
"""
import os
import sys
import tempfile
import time

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Make the repo root importable so we can reuse features.py (the same
# extractor predict.py uses) instead of duplicating it. Works whether this
# file is run as `backend.app` (from repo root) or `app` (from backend/).
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BACKEND_DIR)
for _p in (_REPO_ROOT, _BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features import extract_features, load_image  # noqa: E402

_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "pipeline_v3.pkl")
_ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# Comma-separated list of allowed origins, e.g.
#   FRONTEND_ORIGIN="https://yourname.github.io,http://localhost:5500"
# Defaults to "*" so the demo works out of the box; tighten this once you
# have a real deployed frontend URL.
_origins_env = os.environ.get("FRONTEND_ORIGIN", "*").strip()
_ALLOWED_ORIGINS = ["*"] if _origins_env in ("", "*") else [o.strip() for o in _origins_env.split(",")]

app = FastAPI(title="Spot the Fake Photo API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_error = None


def _get_model():
    """Load and cache the model. Raises only when actually called (i.e. on
    a /predict request), never at import or startup time."""
    global _model, _model_error
    if _model is None:
        if not os.path.isfile(_MODEL_PATH):
            _model_error = f"Model file not found at {_MODEL_PATH}. Did you commit models/pipeline_v3.pkl?"
            raise RuntimeError(_model_error)
        try:
            _model = joblib.load(_MODEL_PATH)
            _model_error = None
        except Exception as e:
            _model_error = f"Failed to load model: {e}"
            raise RuntimeError(_model_error)
    return _model


@app.on_event("startup")
def _warm_up():
    """Try to load the model once at boot so the first real request is
    fast. Deliberately swallows errors -- a missing/corrupt model should
    not take the whole service down or fail the platform's health check.
    The actual error surfaces from /health and /predict instead."""
    try:
        _get_model()
    except Exception:
        pass  # already recorded in _model_error; service stays up


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_error": _model_error}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in _ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported content type: {image.content_type}")

    # Fail fast and clearly if the model isn't available, rather than a
    # generic 400 that looks like a bad upload.
    try:
        model = _get_model()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            contents = await image.read()
            if not contents:
                raise ValueError("Uploaded file is empty")
            tmp.write(contents)
            tmp_path = tmp.name

        t0 = time.perf_counter()
        img = load_image(tmp_path)
        feats = extract_features(img)
        score = float(model.predict_proba([feats])[0][1])
        t1 = time.perf_counter()
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "score": round(score, 4),
        "label": "screen" if score >= 0.5 else "real",
        "latency_ms": round((t1 - t0) * 1000, 1),
    }