"""
db/connection.py - psycopg2 connection pool and DB bootstrap helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlparse

import psycopg2
from psycopg2 import InterfaceError, OperationalError, extras, pool, sql

from config import Config

_pool: pool.ThreadedConnectionPool | None = None


def _parse_database_url(database_url: str) -> dict:
    """Parse postgresql:// URL into psycopg2 kwargs."""
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise ValueError("DATABASE_URL must use postgresql:// scheme")

    query = dict(parse_qsl(parsed.query))
    return {
        "user": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": unquote((parsed.path or "/").lstrip("/") or "postgres"),
        **query,
    }


def init_pool(min_conn: int = 1, max_conn: int = 10, force: bool = False) -> None:
    """Initialize global connection pool."""
    global _pool
    if _pool is not None and not force:
        return

    if _pool is not None and force:
        _pool.closeall()
        _pool = None

    _pool = pool.ThreadedConnectionPool(
        min_conn,
        max_conn,
        dsn=Config.DATABASE_URL,
        cursor_factory=extras.RealDictCursor,
    )


def get_conn():
    """Get one pooled DB connection."""
    global _pool
    if _pool is None:
        init_pool()
    return _pool.getconn()


def release_conn(conn) -> None:
    """Return a connection to the pool."""
    global _pool
    if _pool is not None and conn is not None:
        _pool.putconn(conn)


def execute_query(sql: str, params=None, fetch: str = "none"):
    """
    Execute SQL and optionally fetch rows.

    fetch: 'none' | 'one' | 'all'
    """
    conn = None
    conn_released = False
    try:
        conn = get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None
    except (OperationalError, InterfaceError):
        # One reconnect attempt for transient DB disconnects.
        if conn is not None:
            try:
                release_conn(conn)
                conn_released = True
            except Exception:
                pass
        init_pool(force=True)
        retry_conn = get_conn()
        try:
            with retry_conn:
                with retry_conn.cursor() as cur:
                    cur.execute(sql, params)
                    if fetch == "one":
                        return cur.fetchone()
                    if fetch == "all":
                        return cur.fetchall()
                    return None
        finally:
            release_conn(retry_conn)
    finally:
        if conn is not None and not conn_released:
            release_conn(conn)


def run_schema(schema_path: str | None = None) -> None:
    """Apply the SQL schema file to the configured DATABASE_URL."""
    if schema_path is None:
        schema_path = str(Path(__file__).with_name("schema.sql"))

    with open(schema_path, "r", encoding="utf-8") as schema_file:
        ddl = schema_file.read()
    ddl = ddl.lstrip("\ufeff")

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
    finally:
        release_conn(conn)


def ensure_schema_compatibility() -> None:
    """Apply lightweight ALTERs for older local databases."""
    statements = [
        "ALTER TABLE diagnostic_reports ALTER COLUMN who_grade_predicted TYPE VARCHAR(20)",
        "ALTER TABLE diagnostic_reports ALTER COLUMN survival_category TYPE VARCHAR(50)",
        "ALTER TABLE diagnostic_reports ALTER COLUMN estimated_median_months TYPE VARCHAR(50)",
    ]

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
    finally:
        release_conn(conn)


def ensure_database_exists(database_url: str | None = None) -> bool:
    """
    Ensure the target database exists.

    Returns True when DB already existed or was created successfully.
    """
    database_url = database_url or Config.DATABASE_URL
    db_cfg = _parse_database_url(database_url)
    target_db = db_cfg["dbname"]

    admin_cfg = dict(db_cfg)
    admin_cfg["dbname"] = os.environ.get("POSTGRES_ADMIN_DB", "postgres")

    # Try connecting directly first; if that works, DB already exists.
    try:
        conn = psycopg2.connect(**db_cfg)
        conn.close()
        return True
    except OperationalError:
        pass

    admin_conn = psycopg2.connect(**admin_cfg)
    admin_conn.autocommit = True
    try:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
    finally:
        admin_conn.close()

    return True


def bootstrap_database(schema_path: str | None = None) -> None:
    """Create database if needed, initialize pool, then apply schema."""
    ensure_database_exists()
    init_pool(force=True)
    run_schema(schema_path=schema_path)
    ensure_schema_compatibility()
