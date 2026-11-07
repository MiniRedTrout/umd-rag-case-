from fastapi import FastAPI
from loguru import logger

from src.api.api import api_router
from src.api.lifespan import lifespan
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os


app = FastAPI(title="CHAT API V1", lifespan=lifespan, debug=False)
app.include_router(api_router)

logger.info("Application initialization complete")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


def start_service():
    import os
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    start_service()