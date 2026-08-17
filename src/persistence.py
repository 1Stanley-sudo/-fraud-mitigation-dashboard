"""
Saves fitted models to disk so the app starts without retraining.

The cache key covers every setting that affects training, so changing a
sidebar control forces a retrain instead of reusing a stale model.

This also matches how models are deployed in practice: training happens
offline once, and the scoring service loads the fitted result.
"""

import hashlib
import json
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_store"
MODEL_DIR.mkdir(exist_ok=True)


def cache_key(**settings) -> str:
    """Short hash of the training settings."""
    blob = json.dumps(settings, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _path(key: str) -> Path:
    return MODEL_DIR / f"bundle_{key}.joblib"


def save_bundle(key: str, bundle: dict) -> Path:
    """Save the fitted models, baseline stats and column list."""
    path = _path(key)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(key: str) -> dict | None:
    """Load the saved bundle for these settings, or None if there is none."""
    path = _path(key)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        # A corrupt or version-mismatched file should not break the app,
        # so fall back to retraining.
        return None


def clear_store() -> int:
    """Delete all saved model bundles and return how many were removed."""
    removed = 0
    for f in MODEL_DIR.glob("bundle_*.joblib"):
        f.unlink()
        removed += 1
    return removed
