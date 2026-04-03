"""
Load_model.py  (project root — e:\\Projects\\Gioblastoma\\Load_model.py)
────────────────────────────────────────────────────────────────────────
Loads the trained 3-D U-Net from a Keras .h5 / SavedModel checkpoint.

Usage
-----
    from Load_model import load_model
    model = load_model("/path/to/model.h5")
    pred  = model.predict(...)

The function is deliberately simple: it caches the model in a module-level
variable so that repeat calls within the same process are free.
"""

import os
import threading

import tensorflow as tf

# ── Module-level cache ────────────────────────────────────────────────────────
_model = None
_lock  = threading.Lock()


def _build_model_candidates(model_path: str | None) -> list[str]:
    """
    Build an ordered list of candidate model paths.

    This makes loading resilient to stale relative paths like
    '../Tumor Model/model.h5' by checking equivalent absolute locations
    and preferred .keras checkpoints in the same directory.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    neuroassist_root = os.path.join(project_root, "neuroassist")
    search_bases = [os.getcwd(), project_root, neuroassist_root]

    candidates: list[str] = []

    def add(path: str | None) -> None:
        if not path:
            return
        normalized = os.path.normpath(path)
        if normalized not in candidates:
            candidates.append(normalized)

    raw = (model_path or "").strip()
    if raw:
        raw = os.path.expandvars(os.path.expanduser(raw))
        add(raw)

        if not os.path.isabs(raw):
            for base in search_bases:
                add(os.path.join(base, raw))

        # If a stale filename was provided, try preferred sibling checkpoints.
        snapshot = list(candidates)
        for candidate in snapshot:
            folder = os.path.dirname(candidate)
            if folder:
                add(os.path.join(folder, "best_attention_unet_v2.keras"))
                add(os.path.join(folder, "model.keras"))
                add(os.path.join(folder, "model.h5"))

    # Conventional locations (preferred first)
    for base in [project_root, neuroassist_root]:
        add(os.path.join(base, "Tumor Model", "best_attention_unet_v2.keras"))
        add(os.path.join(base, "Tumor Model", "model.keras"))
        add(os.path.join(base, "Tumor Model", "model.h5"))
        add(os.path.join(base, "Tumor Model"))  # SavedModel dir

    return candidates


def resolve_model_path(model_path: str | None = None) -> str | None:
    """
    Return the first existing model path from known candidates, else None.
    """
    for candidate in _build_model_candidates(model_path):
        if os.path.exists(candidate):
            return candidate
    return None


def load_model(model_path: str | None = None) -> tf.keras.Model:
    """
    Load (or return cached) 3-D U-Net model.

    Parameters
    ----------
    model_path : str
        Path to the saved model file (.h5 or SavedModel directory).
        Defaults to the MODEL_PATH env-var or the 'Tumor Model/' folder
        in the project root.

    Returns
    -------
    tf.keras.Model  — compiled U-Net ready for .predict()

    Raises
    ------
    FileNotFoundError if no model file is found.
    """
    global _model

    # Fast path: already loaded
    if _model is not None:
        return _model

    with _lock:
        # Double-checked locking
        if _model is not None:
            return _model

        # Resolve path
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH", "")

        requested_path = model_path
        model_path = resolve_model_path(model_path)

        if not model_path:
            checked = "\n".join(f"  - {c}" for c in _build_model_candidates(requested_path)[:8])
            raise FileNotFoundError(
                f"Model not found at: {requested_path}\n"
                "Please set MODEL_PATH in .env or place the model in "
                "'Tumor Model/best_attention_unet_v2.keras'.\n"
                f"Checked:\n{checked}"
            )

        if requested_path and os.path.normpath(requested_path) != os.path.normpath(model_path):
            print(f"[Load_model] Requested path unavailable, using fallback: {model_path}")

        print(f"[Load_model] Loading model from: {model_path}")

        # TF2 recommended loaders
        if os.path.isdir(model_path):
            # SavedModel format
            _model = tf.saved_model.load(model_path)
        else:
            # H5 / Keras format — compile=False avoids needing optimizer state
            _model = tf.keras.models.load_model(model_path, compile=False)

        print(f"[Load_model] Model loaded successfully.")
        return _model


def unload_model() -> None:
    """Release the cached model (useful for testing or memory management)."""
    global _model
    with _lock:
        _model = None
    tf.keras.backend.clear_session()
