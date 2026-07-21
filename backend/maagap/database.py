"""Supabase (Postgres) persistence layer.

Replaces the CSV/JSON file exports with real database writes so the Next.js
API routes can query live data instead of reading files. Schema is defined
in ``backend/supabase/schema.sql`` and must be applied once via the Supabase
SQL Editor before running this module.

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in ``backend/.env``.
"""

import json
import os
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from .logger import get_logger

logger = get_logger(__name__)

# Child-to-parent order for deletes (respects FK constraints);
# reversed for inserts (parent-to-child).
TABLE_ORDER = [
    "risk_alerts",
    "assignments",
    "predictions",
    "inspection_logs",
    "external_context",
    "projects",
    "inspectors",
    "contractors",
]

_BATCH_SIZE = 500


def get_client() -> Optional[Client]:
    """Create a Supabase client from backend/.env, or None if unconfigured."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.warning("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set; skipping database sync.")
        return None

    # The client appends /rest/v1 itself; normalize in case the project URL
    # was copied with a REST path already attached.
    url = url.rstrip("/")
    for suffix in ("/rest/v1", "/rest"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break

    return create_client(url, key)


def _df_to_records(df: pd.DataFrame) -> List[dict]:
    """Convert a DataFrame to JSON-safe records (NaN -> null, numpy -> native,
    dates -> ISO strings) via a JSON round-trip."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def fetch_table(client: Client, table: str, columns: str = "*") -> List[dict]:
    """Fetch all rows from a table (used to snapshot state before overwrite,
    e.g. diffing risk tiers across pipeline runs for alerts)."""
    try:
        resp = client.table(table).select(columns).execute()
        return resp.data or []
    except Exception as e:
        logger.warning(f"Could not fetch table '{table}' (may not exist yet): {e}")
        return []


def _delete_all(client: Client, table: str, pk_col: str) -> None:
    client.table(table).delete().neq(pk_col, "__none__").execute()


def _insert_batches(client: Client, table: str, records: List[dict]) -> None:
    for i in range(0, len(records), _BATCH_SIZE):
        batch = records[i:i + _BATCH_SIZE]
        client.table(table).insert(batch).execute()


def sync_table(client: Client, table: str, df: pd.DataFrame, pk_col: str) -> None:
    """Replace all rows in a table with the contents of df (delete-then-insert),
    mirroring the CSV export's "regenerate from scratch" semantics."""
    records = _df_to_records(df)
    _delete_all(client, table, pk_col)
    if records:
        _insert_batches(client, table, records)
    logger.info(f"Synced table '{table}': {len(records)} rows")


PK_COLUMNS = {
    "contractors": "contractor_id",
    "inspectors": "inspector_id",
    "projects": "project_id",
    "external_context": "context_id",
    "inspection_logs": "log_id",
    "predictions": "prediction_id",
    "assignments": "assignment_id",
    "risk_alerts": "id",
}


def sync_all(client: Client, tables: dict) -> None:
    """Sync all MAAGAP tables in FK-safe order.

    ``tables`` maps table name -> DataFrame (a subset is fine; missing
    tables are skipped).
    """
    # Delete children first, then parents.
    for name in TABLE_ORDER:
        if name in tables:
            _delete_all(client, name, PK_COLUMNS[name])

    # Insert parents first, then children.
    for name in reversed(TABLE_ORDER):
        if name in tables:
            records = _df_to_records(tables[name])
            if records:
                _insert_batches(client, name, records)
            logger.info(f"Synced table '{name}': {len(records)} rows")
