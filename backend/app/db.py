"""Database setup.

SQLite by default (a file next to the project, zero setup), but everything goes
through SQLAlchemy so pointing ``FEDFL_DATABASE_URL`` at Postgres is a one-line
change. Sessions are created per request, and per background job, because SQLite
connections are not shared across threads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "app.db"


class Base(DeclarativeBase):
    """Declarative base for every table."""


def database_url() -> str:
    env = os.environ.get("FEDFL_DATABASE_URL")
    if env:
        return env
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


def make_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    is_sqlite = url.startswith("sqlite")
    kwargs: dict = {
        "echo": False,
        "future": True,
        "connect_args": {"check_same_thread": False, "timeout": 30} if is_sqlite else {},
    }
    is_memory = is_sqlite and (":memory:" in url or url.rstrip("/") in ("sqlite:", "sqlite:/"))
    if is_memory:
        # an in-memory SQLite database belongs to one connection, so every
        # session must share that single connection or they see no tables
        kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
            cur = dbapi_connection.cursor()
            # WAL lets the background training thread write while a request reads.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    from . import models  # noqa: F401  (import registers the tables)

    Base.metadata.create_all(engine)


def get_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """FastAPI dependency factory bound to a specific session factory."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
