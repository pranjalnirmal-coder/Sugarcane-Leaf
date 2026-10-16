"""Local SQLite account and prediction-history storage."""
from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import DATABASE_PATH

ITERATIONS = 310_000


@contextmanager
def _connect(db_path: Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    with _connect(db_path) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                image_name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                uncertain INTEGER NOT NULL DEFAULT 0,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)


def create_user(username: str, password: str, db_path: Path = DATABASE_PATH) -> int:
    normalized = username.strip().lower()
    if len(normalized) < 3 or len(normalized) > 40:
        raise ValueError("Username must contain 3 to 40 characters.")
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    try:
        with _connect(db_path) as connection:
            cursor = connection.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                                        (normalized, digest, salt))
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already registered.") from exc


def authenticate(username: str, password: str, db_path: Path = DATABASE_PATH) -> int | None:
    normalized = username.strip().lower()
    with _connect(db_path) as connection:
        row = connection.execute("SELECT id, password_hash, salt FROM users WHERE username = ?",
                                 (normalized,)).fetchone()
    if row is None:
        return None
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), row["salt"], ITERATIONS)
    return int(row["id"]) if hmac.compare_digest(candidate, row["password_hash"]) else None


def save_prediction(user_id: int, image_name: str, class_name: str, confidence: float,
                    uncertain: bool, model_version: str, db_path: Path = DATABASE_PATH) -> int:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO predictions (user_id, image_name, class_name, confidence, uncertain, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, image_name[:255], class_name, float(confidence), int(uncertain), model_version))
        return int(cursor.lastrowid)


def get_history(user_id: int, limit: int = 50, db_path: Path = DATABASE_PATH) -> list[dict]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT image_name, class_name, confidence, uncertain, model_version, created_at "
            "FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(row) for row in rows]
