"""
FastAPI Server Configuration
Imports and configures the backend routes
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Import backend routes
from src.backend import app as backend_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Precision Agriculture API",
    description="AI-Powered Agricultural Decision Support System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount backend routes
app.mount("/api", backend_app)

# Serve frontend static files
if os.path.exists("frontend/src"):
    app.mount("/static", StaticFiles(directory="frontend/src"), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML"""
        return FileResponse("frontend/src/index.html")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Server health check"""
    return {
        "status": "healthy",
        "message": "Precision Agriculture API is running",
        "version": "1.0.0"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    logger.info("=" * 70)
    logger.info("🌱 Precision Agriculture System Starting...")
    logger.info("=" * 70)
    logger.info("✅ Server initialized successfully")
    logger.info("📍 API available at: http://localhost:8000")
    logger.info("📚 Documentation: http://localhost:8000/docs")
    logger.info("=" * 70)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown"""
    logger.info("🛑 Server shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
