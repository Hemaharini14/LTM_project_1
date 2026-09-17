"""
SQLite persistence for the web app: user accounts and saved trip records
(flight-delay checks and budget-trip plans). Plain sqlite3 - no extra
dependency needed since it ships with Python.

WAL journal mode + a generous busy_timeout are set on every connection so
concurrent requests (the Flask dev server can be multi-threaded, and the
debugger keeps a failed request's connection alive until dismissed) don't
immediately raise "database is locked" - they wait briefly for the other
connection to finish instead. Every connection is opened via _connection()
so it's guaranteed to close even if the query raises, which is what
actually caused connections (and their locks) to leak before.
"""
import os
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_data.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


@contextmanager
def _connection():
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


def _ensure_column(conn, table: str, column: str, coltype: str):
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    with _connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS flight_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                inputs_json TEXT NOT NULL,
                delay_probability REAL,
                risk_label TEXT,
                recommendation_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS budget_trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                origin TEXT,
                destination TEXT,
                total_budget REAL,
                days INTEGER,
                inputs_json TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        _ensure_column(conn, "users", "is_admin", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "users", "preferred_currency", "TEXT NOT NULL DEFAULT 'USD'")
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_user_currency(user_id: int, currency: str):
    with _connection() as conn:
        conn.execute("UPDATE users SET preferred_currency = ? WHERE id = ?", (currency, user_id))
        conn.commit()


# ---------------------------------------------------------------- users

def create_user(name: str, email: str, password_hash: str, is_admin: bool = False) -> int:
    with _connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at, is_admin) VALUES (?, ?, ?, ?, ?)",
            (name, email.lower().strip(), password_hash, _now(), int(is_admin)),
        )
        conn.commit()
        return cur.lastrowid


def get_all_users():
    with _connection() as conn:
        return conn.execute("SELECT * FROM users ORDER BY id").fetchall()


def get_activity_stats():
    with _connection() as conn:
        return {
            "user_count": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            "flight_check_count": conn.execute("SELECT COUNT(*) FROM flight_checks").fetchone()[0],
            "budget_trip_count": conn.execute("SELECT COUNT(*) FROM budget_trips").fetchone()[0],
            "high_risk_count": conn.execute(
                "SELECT COUNT(*) FROM flight_checks WHERE risk_label IN ('High','Moderate')"
            ).fetchone()[0],
        }


def get_all_flight_checks(limit: int = 30):
    with _connection() as conn:
        return conn.execute(
            """SELECT flight_checks.*, users.name AS user_name, users.email AS user_email
               FROM flight_checks JOIN users ON users.id = flight_checks.user_id
               ORDER BY flight_checks.id DESC LIMIT ?""", (limit,)
        ).fetchall()


def get_all_budget_trips(limit: int = 30):
    with _connection() as conn:
        return conn.execute(
            """SELECT budget_trips.*, users.name AS user_name, users.email AS user_email
               FROM budget_trips JOIN users ON users.id = budget_trips.user_id
               ORDER BY budget_trips.id DESC LIMIT ?""", (limit,)
        ).fetchall()


def get_flight_check_by_id(check_id: int):
    with _connection() as conn:
        return conn.execute("SELECT * FROM flight_checks WHERE id = ?", (check_id,)).fetchone()


def get_budget_trip_by_id(trip_id: int):
    with _connection() as conn:
        return conn.execute("SELECT * FROM budget_trips WHERE id = ?", (trip_id,)).fetchone()


def get_user_by_email(email: str):
    with _connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()


def get_user_by_id(user_id: int):
    with _connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


# ---------------------------------------------------------------- trips

def save_flight_check(user_id: int, inputs: dict, delay_probability: float,
                       risk_label: str, recommendation: dict | None = None) -> int:
    with _connection() as conn:
        cur = conn.execute(
            """INSERT INTO flight_checks
               (user_id, inputs_json, delay_probability, risk_label, recommendation_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, json.dumps(inputs), delay_probability, risk_label,
             json.dumps(recommendation) if recommendation else None, _now()),
        )
        conn.commit()
        return cur.lastrowid


def save_budget_trip(user_id: int, inputs: dict, plan: dict) -> int:
    with _connection() as conn:
        cur = conn.execute(
            """INSERT INTO budget_trips
               (user_id, origin, destination, total_budget, days, inputs_json, plan_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, inputs.get("origin_airport"), inputs.get("destination_airport"),
             inputs.get("total_budget"), inputs.get("days"),
             json.dumps(inputs), json.dumps(plan), _now()),
        )
        conn.commit()
        return cur.lastrowid


def get_recent_trips(user_id: int, limit: int = 5):
    with _connection() as conn:
        checks = conn.execute(
            "SELECT * FROM flight_checks WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall()
        trips = conn.execute(
            "SELECT * FROM budget_trips WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall()
        return checks, trips
