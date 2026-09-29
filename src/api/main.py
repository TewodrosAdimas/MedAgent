"""
MedAgent FastAPI Application Entrypoint.

Starts the clinical AI inference and multi-agent decision support web service.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import get_model_cache, router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MedAgent.API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up neural networks and agent pipelines on application startup."""
    logger.info("Initializing MedAgent neural models and agent orchestration pipeline...")
    get_model_cache()
    logger.info("MedAgent inference engine ready.")
    yield
    logger.info("Shutting down MedAgent service.")


app = FastAPI(
    title="MedAgent Clinical AI Platform",
    description="Multimodal Medical AI Agent for Explainable Clinical Decision Support (ECG, Chest X-Ray, Multi-Agent RAG).",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local dashboards and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    """Service status and interactive documentation links."""
    return {
        "service": "MedAgent Clinical Decision Support API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "predict_ecg": "/api/v1/predict/ecg",
            "predict_xray": "/api/v1/predict/xray",
            "predict_multimodal": "/api/v1/predict/multimodal",
            "agent_summary": "/api/v1/agent/clinical-summary",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
