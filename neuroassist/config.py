"""
config.py - NeuroAssist configuration
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load one shared project-level .env for the entire repository.
_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE_DIR.parent
_ROOT_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ROOT_DOTENV_PATH)


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _env(*keys: str, default: str = "") -> str:
    return _first_nonempty(*(os.environ.get(key) for key in keys)) or default


class Config:
    # Flask
    SECRET_KEY: str = _env("SECRET_KEY", default="dev-secret-key-change-me")
    DEBUG: bool = _env("DEBUG", default="True").lower() == "true"

    # PostgreSQL
    DATABASE_URL: str = _env(
        "DATABASE_URL",
        default="postgresql://postgres:Admin%40123@localhost:5433/neuroassist",
    )
    AUTO_BOOTSTRAP_DB: bool = _env("AUTO_BOOTSTRAP_DB", default="True").lower() == "true"

    # LLM provider
    LLM_PROVIDER: str = _env("LLM_PROVIDER", default="groq").lower()
    GROQ_API_KEY: str = _env("GROQ_API_KEY")
    GROQ_MODEL: str = _env("GROQ_MODEL", default="llama-3.3-70b-versatile")
    GROQ_BASE_URL: str = _env("GROQ_BASE_URL", default="https://api.groq.com/openai/v1")

    # Legacy local-model settings retained only for optional fallback/debugging
    OLLAMA_MODEL: str = _env("OLLAMA_MODEL", default="llama3.1:8b")
    OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", default="http://localhost:11434")
    OLLAMA_NUM_CTX: int = int(_env("OLLAMA_NUM_CTX", default="8192"))
    LLM_TEMPERATURE: float = float(_env("LLM_TEMPERATURE", default="0.3"))
    LLM_TOP_P: float = float(_env("LLM_TOP_P", default="0.9"))
    LLM_MAX_OUTPUT_TOKENS: int = int(_env("LLM_MAX_OUTPUT_TOKENS", default="2048"))
    LLM_TIMEOUT_SECONDS: int = int(_env("LLM_TIMEOUT_SECONDS", default="3600"))
    LLM_MAX_INTERVIEW_TURNS: int = int(_env("LLM_MAX_INTERVIEW_TURNS", default="15"))
    LLM_USE_FULL_SPEC_PROMPT: bool = _env("LLM_USE_FULL_SPEC_PROMPT", default="False").lower() == "true"
    LLM_SYSTEM_PROMPT_MAX_CHARS: int = int(_env("LLM_SYSTEM_PROMPT_MAX_CHARS", default="4000"))

    # Context and RAG settings
    CONTEXT_HISTORY_LIMIT: int = int(_env("CONTEXT_HISTORY_LIMIT", default="1000"))
    RAG_TOP_K: int = int(_env("RAG_TOP_K", default="10"))
    RAG_CHUNK_CHAR_LIMIT: int = int(_env("RAG_CHUNK_CHAR_LIMIT", default="2000"))

    # Sentence embeddings
    EMBED_MODEL: str = _env("EMBED_MODEL", default="sentence-transformers/all-MiniLM-L6-v2")

    # Upload paths (resolved relative to this file)
    BASE_DIR: str = str(_BASE_DIR)
    PROJECT_ROOT: str = str(_PROJECT_ROOT)
    UPLOAD_FOLDER: str = str(_BASE_DIR / "uploads")
    ANNOTATED_FOLDER: str = str(_BASE_DIR / "static" / "annotated")
    MAX_CONTENT_LENGTH: int = 500 * 1024 * 1024  # 500 MB

    # Model path (3D U-Net weights)
    _MODEL_DEFAULT_PATH = _PROJECT_ROOT / "Tumor Model" / "best_attention_unet_v2.keras"
    _MODEL_DEFAULT: str = str(_MODEL_DEFAULT_PATH)
    _MODEL_ENV: str = _env("MODEL_PATH")
    if _MODEL_ENV:
        _model_candidates: list[Path] = []
        _raw_model_path = Path(_MODEL_ENV).expanduser()
        if _raw_model_path.is_absolute():
            _model_candidates.append(_raw_model_path)
        else:
            # Accept both new root-relative values and older neuroassist-relative values.
            _model_candidates.append((_PROJECT_ROOT / _raw_model_path).resolve())
            _model_candidates.append((_BASE_DIR / _raw_model_path).resolve())

        MODEL_PATH: str = _MODEL_DEFAULT
        for candidate in _model_candidates:
            if candidate.exists():
                MODEL_PATH = str(candidate)
                break
    else:
        MODEL_PATH: str = _MODEL_DEFAULT
