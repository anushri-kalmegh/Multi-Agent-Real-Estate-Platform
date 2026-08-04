"""Persistent application data, authentication and saved user activity."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.config import APP_DATABASE_PATH


PBKDF2_ITERATIONS = 310_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connection() -> sqlite3.Connection:
    db = sqlite3.connect(APP_DATABASE_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    return db


def initialize_app_database() -> None:
    with connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_iterations INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                message_type TEXT NOT NULL DEFAULT 'text',
                payload_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conversation_user_thread
                ON conversations(user_id, thread_id, created_at);
            CREATE TABLE IF NOT EXISTS saved_searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                query TEXT NOT NULL,
                requirements_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS favourites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                property_id TEXT NOT NULL,
                property_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, property_id)
            );
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL,
                query TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reports_user_created
                ON reports(user_id, created_at DESC);
            """
        )


def _validate_account(email: str, display_name: str, password: str) -> None:
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        raise ValueError("Enter a valid email address.")
    if len(display_name.strip()) < 2:
        raise ValueError("Display name must contain at least two characters.")
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise ValueError("Password must be at least 8 characters and include a letter and number.")


def _password_digest(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations).hex()


def create_user(email: str, display_name: str, password: str) -> dict:
    _validate_account(email, display_name, password)
    salt = secrets.token_bytes(32)
    digest = _password_digest(password, salt)
    try:
        with connection() as db:
            cursor = db.execute(
                """
                INSERT INTO users(email, display_name, password_hash, password_salt,
                                  password_iterations, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    email.strip().casefold(), display_name.strip(), digest, salt.hex(),
                    PBKDF2_ITERATIONS, utc_now(),
                ),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as error:
        raise ValueError("An account with this email already exists.") from error
    return {"id": user_id, "email": email.strip().casefold(), "display_name": display_name.strip()}


def authenticate(email: str, password: str) -> dict | None:
    with connection() as db:
        row = db.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE AND is_active = 1",
            (email.strip(),),
        ).fetchone()
    if not row:
        return None
    actual = _password_digest(password, bytes.fromhex(row["password_salt"]), row["password_iterations"])
    if not hmac.compare_digest(actual, row["password_hash"]):
        return None
    return {"id": row["id"], "email": row["email"], "display_name": row["display_name"]}


def create_session(user_id: int, days: int = 7) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = datetime.now(timezone.utc)
    with connection() as db:
        db.execute(
            "INSERT INTO sessions(user_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (user_id, token_hash, now.isoformat(), (now + timedelta(days=days)).isoformat()),
        )
    return token


def revoke_session(token: str) -> None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with connection() as db:
        db.execute(
            "UPDATE sessions SET revoked_at = ? WHERE token_hash = ?",
            (utc_now(), token_hash),
        )


def session_user(token: str) -> dict | None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with connection() as db:
        row = db.execute(
            """
            SELECT u.id, u.email, u.display_name
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.revoked_at IS NULL
              AND s.expires_at > ? AND u.is_active = 1
            """,
            (token_hash, utc_now()),
        ).fetchone()
    return dict(row) if row else None


def save_message(
    user_id: int, thread_id: str, role: str, content: str,
    message_type: str = "text", payload: dict | None = None,
) -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO conversations(user_id, thread_id, role, content,
                                      message_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, thread_id, role, content, message_type,
             json.dumps(payload, default=str) if payload is not None else None, utc_now()),
        )


def clear_conversation_history(user_id: int, thread_id: str | None = None) -> int:
    with connection() as db:
        if thread_id:
            cursor = db.execute(
                "DELETE FROM conversations WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id),
            )
        else:
            cursor = db.execute(
                "DELETE FROM conversations WHERE user_id = ?", (user_id,)
            )
        return cursor.rowcount


def save_search(user_id: int, query: str, requirements: dict) -> int:
    with connection() as db:
        cursor = db.execute(
            "INSERT INTO saved_searches(user_id, query, requirements_json, created_at) VALUES (?, ?, ?, ?)",
            (user_id, query, json.dumps(requirements, default=str), utc_now()),
        )
        return int(cursor.lastrowid)


def add_favourite(user_id: int, property_data: dict) -> bool:
    try:
        with connection() as db:
            db.execute(
                "INSERT INTO favourites(user_id, property_id, property_json, created_at) VALUES (?, ?, ?, ?)",
                (user_id, property_data["property_id"], json.dumps(property_data, default=str), utc_now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def remove_favourite(user_id: int, property_id: str) -> None:
    with connection() as db:
        db.execute(
            "DELETE FROM favourites WHERE user_id = ? AND property_id = ?",
            (user_id, property_id),
        )


def list_favourites(user_id: int) -> list[dict]:
    with connection() as db:
        rows = db.execute(
            "SELECT property_json, created_at FROM favourites WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [{**json.loads(row["property_json"]), "saved_at": row["created_at"]} for row in rows]


def save_report(user_id: int, thread_id: str, query: str, report: dict) -> int:
    with connection() as db:
        cursor = db.execute(
            "INSERT INTO reports(user_id, thread_id, query, report_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, thread_id, query, json.dumps(report, default=str), utc_now()),
        )
        return int(cursor.lastrowid)


def list_reports(user_id: int, limit: int = 20) -> list[dict]:
    with connection() as db:
        rows = db.execute(
            "SELECT id, query, report_json, created_at FROM reports WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [
        {"id": row["id"], "query": row["query"], "report": json.loads(row["report_json"]),
         "created_at": row["created_at"]}
        for row in rows
    ]


initialize_app_database()
