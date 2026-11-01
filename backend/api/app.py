from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import engine, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine._load_model("ru")
    except Exception as e:
        print(f"Warning during preloading model: {e}")
    yield
    engine._models.clear()
    engine._tokenizers.clear()


def create_app() -> FastAPI:
    application = FastAPI(
        title="VeriText AI Detector API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Include API router first so /api/* routes take precedence
    application.include_router(router)

    # Mount static frontend files if web/public exists
    project_root = Path(__file__).resolve().parent.parent.parent
    web_public = project_root / "web" / "public"
    if web_public.exists() and (web_public / "index.html").exists():
        application.mount("/", StaticFiles(directory=str(web_public), html=True), name="static")

    return application


app = create_app()
