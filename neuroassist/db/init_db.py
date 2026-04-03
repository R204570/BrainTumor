"""
db/init_db.py

Creates the configured PostgreSQL database if missing and applies schema.sql.

Usage:
    python -m db.init_db
or:
    python db/init_db.py
"""

from __future__ import annotations

from pathlib import Path

from psycopg2 import OperationalError

from db.connection import bootstrap_database


if __name__ == "__main__":
    schema = Path(__file__).with_name("schema.sql")
    try:
        bootstrap_database(schema_path=str(schema))
        print(f"[NeuroAssist] Database is ready. Applied schema from: {schema}")
    except OperationalError as exc:
        print("[NeuroAssist] Database bootstrap failed.")
        print("Please verify DATABASE_URL credentials and that PostgreSQL is running.")
        print(f"Details: {exc}")
        raise SystemExit(1)
