from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sequences (
  name TEXT PRIMARY KEY,
  value INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS directions (
  id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  owner TEXT NOT NULL,
  status TEXT NOT NULL,
  score REAL NOT NULL DEFAULT 0,
  success_criteria TEXT NOT NULL,
  abandon_criteria TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  direction_slug TEXT NOT NULL REFERENCES directions(slug),
  title TEXT NOT NULL,
  kind TEXT NOT NULL,
  description TEXT NOT NULL,
  owner TEXT NOT NULL,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 2,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  direction_slug TEXT NOT NULL REFERENCES directions(slug),
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  range_start INTEGER NOT NULL,
  range_end INTEGER NOT NULL,
  kernel TEXT NOT NULL,
  owner TEXT NOT NULL,
  checkpoint_json TEXT NOT NULL DEFAULT '{}',
  metrics_json TEXT NOT NULL DEFAULT '{}',
  summary TEXT NOT NULL DEFAULT '',
  code_version TEXT NOT NULL,
  hardware TEXT NOT NULL,
  checksum TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS claims (
  id TEXT PRIMARY KEY,
  direction_slug TEXT NOT NULL REFERENCES directions(slug),
  title TEXT NOT NULL,
  statement TEXT NOT NULL,
  status TEXT NOT NULL,
  owner TEXT NOT NULL,
  dependencies_json TEXT NOT NULL DEFAULT '[]',
  notes TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_run_links (
  claim_id TEXT NOT NULL REFERENCES claims(id),
  run_id TEXT NOT NULL REFERENCES runs(id),
  relation TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (claim_id, run_id, relation)
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  checksum TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  run_id TEXT REFERENCES runs(id),
  claim_id TEXT REFERENCES claims(id),
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  status TEXT NOT NULL,
  hardware TEXT NOT NULL,
  capabilities_json TEXT NOT NULL DEFAULT '[]',
  current_run_id TEXT REFERENCES runs(id),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_heartbeat_at TEXT
);

CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  direction_slug TEXT NOT NULL REFERENCES directions(slug),
  title TEXT NOT NULL,
  authors TEXT NOT NULL DEFAULT '',
  year TEXT NOT NULL DEFAULT '',
  url TEXT NOT NULL DEFAULT '',
  source_type TEXT NOT NULL,
  claim_type TEXT NOT NULL,
  review_status TEXT NOT NULL,
  map_variant TEXT NOT NULL DEFAULT 'unspecified',
  summary TEXT NOT NULL DEFAULT '',
  notes TEXT NOT NULL DEFAULT '',
  fallacy_tags_json TEXT NOT NULL DEFAULT '[]',
  rubric_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def ensure_schema_migrations(connection: sqlite3.Connection) -> None:
    source_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(sources)").fetchall()
    }
    if source_columns and "map_variant" not in source_columns:
        connection.execute(
            "ALTER TABLE sources ADD COLUMN map_variant TEXT NOT NULL DEFAULT 'unspecified'"
        )
