from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    application.include_router(router)
    return application


app = create_app()
