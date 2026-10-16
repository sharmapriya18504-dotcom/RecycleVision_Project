import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.services.classifier import classifier
from backend.app.api.v1.endpoints import (
    predict,
    model,
    analytics,
    history,
    admin,
    chat
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recyclevision")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
origins = settings.ALLOWED_ORIGINS
if isinstance(origins, str):
    origins = [origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global safe exception handler: never leak server internals, local paths, or stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing your request. Please try again later.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

@app.get("/", tags=["General"], summary="API Root")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "tagline": "Smart Waste Classification & Recycling Analytics",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": "/health",
        "api_v1_prefix": settings.API_V1_STR
    }

@app.get("/health", tags=["General"], summary="Health Check")
def health_check():
    """
    Monitors backend service health, model availability, and timestamp.
    Used by Render / hosting providers for liveness probes.
    """
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": classifier.is_loaded,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Register API v1 routers
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}", tags=["Inference"])
app.include_router(model.router, prefix=f"{settings.API_V1_STR}/model", tags=["Model"])
app.include_router(analytics.router, prefix=f"{settings.API_V1_STR}/analytics", tags=["Analytics"])
app.include_router(history.router, prefix=f"{settings.API_V1_STR}/predictions", tags=["History"])
app.include_router(admin.router, prefix=f"{settings.API_V1_STR}/admin", tags=["Admin"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["RecycleBot Chat"])

