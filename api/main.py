"""
CCTV Surveillance — FastAPI Backend

Provides REST endpoints for AI-powered detection models,
authentication, and health monitoring.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.routers import auth, detection, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info(
        "Starting CCTV Surveillance API — docs at http://%s:%s/docs",
        settings.HOST,
        settings.PORT,
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="CCTV Surveillance API",
    description="AI-powered detection backend for CCTV surveillance. "
    "Supports vehicle, license plate, fire/smoke, accident, face, and gunshot detection.",
    version="0.1.0",
    lifespan=lifespan,
    contact={
        "name": "safe-cam",
        "url": "https://github.com/safe-cam/cctv-surveillance",
    },
)

# ------------------------------------------------------------------
#  CORS — allow JavaScript calls from any origin
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
#  Global exception handler
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
    )


# ------------------------------------------------------------------
#  Routers
# ------------------------------------------------------------------
app.include_router(health.router)
app.include_router(detection.router)
app.include_router(auth.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "CCTV Surveillance API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


# ------------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
        reload=True,
    )
