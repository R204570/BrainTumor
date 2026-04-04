"""
config.py - NeuroAssist configuration
"""

from __future__ import annotations

import os

from dotenv import dotenv_values, load_dotenv

# Load local .env first, then project-root .env as fallback.
_LOCAL_DOTENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
_ROOT_DOTENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_LOCAL_DOTENV_PATH)
load_dotenv(dotenv_path=_ROOT_DOTENV_PATH)

_DOTENV_LOCAL = dotenv_values(_LOCAL_DOTENV_PATH)
_DOTENV_ROOT = dotenv_values(_ROOT_DOTENV_PATH)


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _env_with_fallback(*keys: str, default: str = "") -> str:
    # 1) Environment vars (highest priority)
    env_val = _first_nonempty(*(os.environ.get(k) for k in keys))
    if env_val:
        return env_val

    # 2) local neuroassist/.env
    local_val = _first_nonempty(*(_DOTENV_LOCAL.get(k) for k in keys))
    if local_val:
        return local_val

    # 3) project root .env
    root_val = _first_nonempty(*(_DOTENV_ROOT.get(k) for k in keys))
    if root_val:
        return root_val

    return default


class Config:
    # Flask
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    DEBUG: bool = os.environ.get("DEBUG", "True").lower() == "true"

    # PostgreSQL
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:Admin%40123@localhost:5433/neuroassist",
    )
    AUTO_BOOTSTRAP_DB: bool = os.environ.get("AUTO_BOOTSTRAP_DB", "True").lower() == "true"

    # LLM provider
    LLM_PROVIDER: str = _env_with_fallback("LLM_PROVIDER", default="groq").lower()
    GROQ_API_KEY: str = _env_with_fallback("GROQ_API_KEY")
    GROQ_MODEL: str = _env_with_fallback("GROQ_MODEL", default="llama-3.3-70b-versatile")
    GROQ_BASE_URL: str = _env_with_fallback("GROQ_BASE_URL", default="https://api.groq.com/openai/v1")

    # Legacy local-model settings retained only for optional fallback/debugging
    OLLAMA_MODEL: str = _env_with_fallback("OLLAMA_MODEL", default="llama3.1:8b")
    OLLAMA_BASE_URL: str = _env_with_fallback("OLLAMA_BASE_URL", default="http://localhost:11434")
    OLLAMA_NUM_CTX: int = int(_env_with_fallback("OLLAMA_NUM_CTX", default="8192"))
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
    LLM_TOP_P: float = float(os.environ.get("LLM_TOP_P", "0.9"))
    LLM_MAX_OUTPUT_TOKENS: int = int(_env_with_fallback("LLM_MAX_OUTPUT_TOKENS", default="2048"))
    LLM_TIMEOUT_SECONDS: int = int(_env_with_fallback("LLM_TIMEOUT_SECONDS", default="3600"))
    LLM_MAX_INTERVIEW_TURNS: int = int(_env_with_fallback("LLM_MAX_INTERVIEW_TURNS", default="15"))
    LLM_USE_FULL_SPEC_PROMPT: bool = _env_with_fallback("LLM_USE_FULL_SPEC_PROMPT", default="False").lower() == "true"
    LLM_SYSTEM_PROMPT_MAX_CHARS: int = int(_env_with_fallback("LLM_SYSTEM_PROMPT_MAX_CHARS", default="4000"))

    # Context and RAG settings - NO LIMITS
    CONTEXT_HISTORY_LIMIT: int = int(_env_with_fallback("CONTEXT_HISTORY_LIMIT", default="1000"))
    RAG_TOP_K: int = int(_env_with_fallback("RAG_TOP_K", default="10"))
    RAG_CHUNK_CHAR_LIMIT: int = int(_env_with_fallback("RAG_CHUNK_CHAR_LIMIT", default="2000"))

    # Sentence embeddings
    EMBED_MODEL: str = _env_with_fallback("EMBED_MODEL", default="sentence-transformers/all-MiniLM-L6-v2")

    # Upload paths (resolved relative to this file)
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER: str = os.path.join(BASE_DIR, "uploads")
    ANNOTATED_FOLDER: str = os.path.join(BASE_DIR, "static", "annotated")
    MAX_CONTENT_LENGTH: int = 500 * 1024 * 1024  # 500 MB

    # Model path (3D U-Net weights)
    _MODEL_DEFAULT: str = os.path.join(
        os.path.dirname(BASE_DIR), "Tumor Model", "best_attention_unet_v2.keras"
    )
    _MODEL_ENV: str = (os.environ.get("MODEL_PATH") or "").strip()
    if _MODEL_ENV:
        # Resolve relative paths against neuroassist/ so .env values are stable
        # regardless of where `python app.py` is launched from.
        _resolved_model: str = (
            _MODEL_ENV
            if os.path.isabs(_MODEL_ENV)
            else os.path.normpath(os.path.join(BASE_DIR, _MODEL_ENV))
        )
        # If an outdated env path is present, gracefully fall back to default.
        MODEL_PATH: str = _resolved_model if os.path.exists(_resolved_model) else _MODEL_DEFAULT
    else:
        MODEL_PATH: str = _MODEL_DEFAULT
