"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Iterator

from fastapi import Request
from sqlalchemy.orm import Session, sessionmaker


def get_session_factory(request: Request) -> sessionmaker[Session]:
    return request.app.state.session_factory


def get_db(request: Request) -> Iterator[Session]:
    """One session per request, closed even if the handler raises."""
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()
