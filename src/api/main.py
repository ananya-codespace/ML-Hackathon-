
import logging

from fastapi import FastAPI, Request

from src.api.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    LOG_LEVEL
)

from src.api.pipeline import load_artifacts
from src.api.routes import router


# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)


# Register API routes
app.include_router(router)


# Load the model and HMM settings once, before the first request
@app.on_event("startup")
def startup():
    load_artifacts()


# Log incoming API requests
@app.middleware("http")
async def log_requests(request: Request, call_next):

    logger.info(
        "Incoming request from %s: %s %s",
        request.client.host if request.client else "unknown",
        request.method,
        request.url.path
    )

    response = await call_next(request)

    logger.info(
        "Response status: %s",
        response.status_code
    )

    return response


# Home endpoint
@app.get("/", tags=["General"])
def home():

    return {
        "message": "Welcome to PathSnap API",
        "status": "running"
    }


# Health check endpoint
@app.get("/health", tags=["General"])
def health_check():

    return {
        "status": "healthy",
        "service": "PathSnap Backend"
    }
