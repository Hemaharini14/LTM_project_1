"""
SQLite persistence for the web app: user accounts and saved trip records
(flight-delay checks and budget-trip plans). Plain sqlite3 - no extra
dependency needed since it ships with Python.
"""
import os
import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_data.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_column(conn, table: str, column: str, coltype: str):
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    conn = get_db()
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
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- users

def create_user(name: str, email: str, password_hash: str, is_admin: bool = False) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO users (name, email, password_hash, created_at, is_admin) VALUES (?, ?, ?, ?, ?)",
        (name, email.lower().strip(), password_hash, _now(), int(is_admin)),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_all_users():
    conn = get_db()
    rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()
    return rows


def get_activity_stats():
    conn = get_db()
    stats = {
        "user_count": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        "flight_check_count": conn.execute("SELECT COUNT(*) FROM flight_checks").fetchone()[0],
        "budget_trip_count": conn.execute("SELECT COUNT(*) FROM budget_trips").fetchone()[0],
        "high_risk_count": conn.execute(
            "SELECT COUNT(*) FROM flight_checks WHERE risk_label IN ('High','Moderate')"
        ).fetchone()[0],
    }
    conn.close()
    return stats


def get_all_flight_checks(limit: int = 30):
    conn = get_db()
    rows = conn.execute(
        """SELECT flight_checks.*, users.name AS user_name, users.email AS user_email
           FROM flight_checks JOIN users ON users.id = flight_checks.user_id
           ORDER BY flight_checks.id DESC LIMIT ?""", (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_all_budget_trips(limit: int = 30):
    conn = get_db()
    rows = conn.execute(
        """SELECT budget_trips.*, users.name AS user_name, users.email AS user_email
           FROM budget_trips JOIN users ON users.id = budget_trips.user_id
           ORDER BY budget_trips.id DESC LIMIT ?""", (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_flight_check_by_id(check_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM flight_checks WHERE id = ?", (check_id,)).fetchone()
    conn.close()
    return row


def get_budget_trip_by_id(trip_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM budget_trips WHERE id = ?", (trip_id,)).fetchone()
    conn.close()
    return row


def get_user_by_email(email: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------- trips

def save_flight_check(user_id: int, inputs: dict, delay_probability: float,
                       risk_label: str, recommendation: dict | None = None) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO flight_checks
           (user_id, inputs_json, delay_probability, risk_label, recommendation_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, json.dumps(inputs), delay_probability, risk_label,
         json.dumps(recommendation) if recommendation else None, _now()),
    )
    conn.commit()
    check_id = cur.lastrowid
    conn.close()
    return check_id


def save_budget_trip(user_id: int, inputs: dict, plan: dict) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO budget_trips
           (user_id, origin, destination, total_budget, days, inputs_json, plan_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, inputs.get("origin_airport"), inputs.get("destination_airport"),
         inputs.get("total_budget"), inputs.get("days"),
         json.dumps(inputs), json.dumps(plan), _now()),
    )
    conn.commit()
    trip_id = cur.lastrowid
    conn.close()
    return trip_id


def get_recent_trips(user_id: int, limit: int = 5):
    conn = get_db()
    checks = conn.execute(
        "SELECT * FROM flight_checks WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
    ).fetchall()
    trips = conn.execute(
        "SELECT * FROM budget_trips WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
    ).fetchall()
    conn.close()
    return checks, trips
