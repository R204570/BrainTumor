"""
services/nvidia_memory.py

DISABLED - No longer using NVIDIA API for memory compaction via local Ollama.
This module is kept as a stub for backward compatibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compact_qa_pairs(qa_pairs: list[dict[str, Any]]) -> str:
    """
    Stub function - returns raw history instead of compacting.
    
    With VE Ollama model, context is passed directly to LLM.
    No need for external memory compaction service.
    """
    logger.debug("[nvidia_memory.py] compact_qa_pairs STUB (no-op)")
    if not qa_pairs:
        return ""
    return json.dumps(qa_pairs, ensure_ascii=False)[:3000]
