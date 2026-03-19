from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="MacMemory Local API",
        description="Local semantic search API for text, images, and PDFs.",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()

