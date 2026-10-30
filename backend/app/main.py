"""FastAPI application factory.

``create_app`` takes an optional database URL so tests can spin up an isolated
in-memory SQLite database per test file without touching the real one.

Layout: the JSON API lives under ``/api``, and the single-page frontend is
served from ``frontend/`` at the root, so ``uvicorn app.main:app`` gives you the
whole product on one port -- no separate dev server, no CORS setup.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .db import PROJECT_ROOT, init_db, make_engine, make_session_factory
from .routers import meta, models, predict, runs
from .services.registry_service import ServiceError

__version__ = "1.0.0"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def create_app(database_url: str | None = None) -> FastAPI:
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    init_db(engine)

    app = FastAPI(
        title="Federated Learning MLOps API",
        version=__version__,
        description=(
            "Federated learning with a small MLOps workflow: train across "
            "simulated clients, track runs, register and promote models, serve "
            "predictions. Educational demo on the WDBC dataset."
        ),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.engine = engine
    app.state.session_factory = session_factory

    # The frontend is served from the same origin, so CORS is only for people
    # running a separate dev server against this API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        """A violated rule (unmeasured model, blocked promotion) is a 409."""
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    app.include_router(meta.router)
    app.include_router(runs.router)
    app.include_router(models.router)
    app.include_router(predict.router)

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
