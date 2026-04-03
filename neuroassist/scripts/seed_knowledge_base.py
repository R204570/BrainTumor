"""
scripts/seed_knowledge_base.py

Seeds knowledge_chunks table with curated WHO 2021 / VASARI / RANO reference
chunks and sentence embeddings.

Usage:
    python -m scripts.seed_knowledge_base
    python -m scripts.seed_knowledge_base --append
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

from config import Config
from db.connection import execute_query, init_pool
from services.embeddings import encode_texts


@dataclass
class SeedChunk:
    source: str
    chunk_text: str
    metadata: dict


SEED_CHUNKS: list[SeedChunk] = [
    SeedChunk(
        source="WHO_2021",
        chunk_text=(
            "Glioblastoma, IDH-wildtype (CNS WHO grade 4) is supported by necrosis or microvascular "
            "proliferation, and molecular markers such as TERT promoter mutation, EGFR amplification, "
            "or combined chromosome 7 gain / chromosome 10 loss."
        ),
        metadata={"topic": "grading", "domain": "glioblastoma"},
    ),
    SeedChunk(
        source="WHO_2021",
        chunk_text=(
            "Astrocytoma, IDH-mutant is assigned grade 4 if CDKN2A/B homozygous deletion is present, "
            "even when necrosis or microvascular proliferation are absent."
        ),
        metadata={"topic": "grading", "domain": "astrocytoma"},
    ),
    SeedChunk(
        source="WHO_2021",
        chunk_text=(
            "Oligodendroglioma diagnosis requires both IDH mutation and whole-arm 1p/19q codeletion. "
            "Mitotic activity and anaplasia support higher grade assignment."
        ),
        metadata={"topic": "grading", "domain": "oligodendroglioma"},
    ),
    SeedChunk(
        source="WHO_2021",
        chunk_text=(
            "Diffuse midline glioma with H3 K27 alteration is considered CNS WHO grade 4 regardless "
            "of classic lower-grade histologic appearance."
        ),
        metadata={"topic": "grading", "domain": "midline_glioma"},
    ),
    SeedChunk(
        source="WHO_2021",
        chunk_text=(
            "MGMT promoter methylation is prognostic and predictive for alkylating chemotherapy benefit, "
            "especially in glioblastoma treatment planning."
        ),
        metadata={"topic": "treatment", "domain": "molecular"},
    ),
    SeedChunk(
        source="VASARI_Manual",
        chunk_text=(
            "Ring enhancement with central non-enhancement often corresponds to necrotic high-grade disease, "
            "while extensive peritumoral FLAIR abnormality reflects edema and infiltrative burden."
        ),
        metadata={"topic": "imaging", "domain": "vasari"},
    ),
    SeedChunk(
        source="VASARI_Manual",
        chunk_text=(
            "Crossing the midline and multifocal disease are adverse imaging features associated with "
            "lower resectability and poorer outcomes."
        ),
        metadata={"topic": "imaging", "domain": "vasari"},
    ),
    SeedChunk(
        source="VASARI_Manual",
        chunk_text=(
            "Ependymal or subependymal spread raises concern for CSF dissemination and should prompt "
            "additional neuraxis evaluation."
        ),
        metadata={"topic": "imaging", "domain": "vasari"},
    ),
    SeedChunk(
        source="RANO_Criteria",
        chunk_text=(
            "Anti-VEGF therapy can reduce contrast enhancement without true antitumor effect, creating "
            "pseudoresponse and underestimating active enhancing tumor volume."
        ),
        metadata={"topic": "response_assessment", "domain": "rano"},
    ),
    SeedChunk(
        source="RANO_Criteria",
        chunk_text=(
            "Clinical status and corticosteroid dose are required context when interpreting imaging response "
            "to avoid overcalling progression or response."
        ),
        metadata={"topic": "response_assessment", "domain": "rano"},
    ),
    SeedChunk(
        source="WHO_2021",
        chunk_text=(
            "Short symptom duration, rapidly progressive neurologic decline, and high perfusion support a "
            "higher-grade phenotype when molecular testing is pending."
        ),
        metadata={"topic": "clinical", "domain": "triage"},
    ),
    SeedChunk(
        source="WHO_2021",
        chunk_text=(
            "Karnofsky performance status and extent of resection are key determinants of prognosis and "
            "therapy candidacy in diffuse gliomas."
        ),
        metadata={"topic": "prognosis", "domain": "clinical"},
    ),
]


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _embedding_storage_mode() -> str:
    row = execute_query(
        """
        SELECT data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'knowledge_chunks'
          AND column_name = 'embedding'
        """,
        fetch="one",
    )
    if not row:
        raise RuntimeError("knowledge_chunks table not found. Run: python -m db.init_db")

    udt_name = str(row.get("udt_name") or "")
    return "vector" if udt_name == "vector" else "array"


def seed_knowledge_base(append: bool = False) -> int:
    init_pool()

    if not append:
        execute_query(
            "DELETE FROM knowledge_chunks WHERE source = ANY(%s)",
            (["WHO_2021", "VASARI_Manual", "RANO_Criteria"],),
        )

    texts = [c.chunk_text for c in SEED_CHUNKS]
    vectors = encode_texts(texts)

    mode = _embedding_storage_mode()

    inserted = 0
    for chunk, vector in zip(SEED_CHUNKS, vectors):
        metadata_json = json.dumps(chunk.metadata)

        if mode == "vector":
            execute_query(
                """
                INSERT INTO knowledge_chunks (source, chunk_text, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s::jsonb)
                """,
                (chunk.source, chunk.chunk_text, _vector_literal(vector), metadata_json),
            )
        else:
            execute_query(
                """
                INSERT INTO knowledge_chunks (source, chunk_text, embedding, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (chunk.source, chunk.chunk_text, [float(x) for x in vector], metadata_json),
            )

        inserted += 1

    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed NeuroAssist knowledge_chunks")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append chunks instead of replacing existing WHO/VASARI/RANO seed rows",
    )
    args = parser.parse_args()

    total = seed_knowledge_base(append=args.append)
    print(f"[seed_knowledge_base] inserted {total} chunks into knowledge_chunks")


if __name__ == "__main__":
    main()
