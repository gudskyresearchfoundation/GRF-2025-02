"""
FastAPI Server Configuration
FIXED: Works with Lightning.ai deployment and custom domains
Handles both local and deployed environments
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import os
import sys

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pre-load all models BEFORE creating the app
logger.info("=" * 70)
logger.info("🌱 PRECISION AGRICULTURE SYSTEM - INITIALIZING")
logger.info("=" * 70)
logger.info("📥 Pre-loading all AI/ML models...")
logger.info("⏳ This may take 1-2 minutes on first run (downloading models)...")
logger.info("")

try:
    # Import model getters
    from src.models.disease_detection import get_disease_detector
    from src.models.weather_service import get_weather_service
    from src.models.soil_model import get_soil_analyzer
    from src.models.yield_prediction import get_yield_predictor
    from src.models.irrigation import get_irrigation_optimizer
    from src.models.pest_risk_assessor import get_pest_risk_assessor
    from src.report_generation import get_report_generator
    from src.explainability import get_explainer
    
    # Load Model 1: Disease Detection
    logger.info("1/8 📸 Loading Disease Detection Model...")
    get_disease_detector()
    logger.info("    ✅ Disease Detector ready")
    
    # Load Model 2: Weather Service
    logger.info("2/8 🌤️  Loading Weather Service...")
    get_weather_service()
    logger.info("    ✅ Weather Service ready")
    
    # Load Model 3: Soil Analyzer
    logger.info("3/8 🌱 Loading Soil Analyzer...")
    get_soil_analyzer()
    logger.info("    ✅ Soil Analyzer ready")
    
    # Load Model 4: Yield Predictor
    logger.info("4/8 📈 Loading Yield Predictor...")
    get_yield_predictor()
    logger.info("    ✅ Yield Predictor ready")
    
    # Load Model 5: Irrigation Optimizer
    logger.info("5/8 💧 Loading Irrigation Optimizer...")
    get_irrigation_optimizer()
    logger.info("    ✅ Irrigation Optimizer ready")
    
    # Load Model 6: Pest Risk Assessor
    logger.info("6/8 🐛 Loading Pest Risk Assessor...")
    get_pest_risk_assessor()
    logger.info("    ✅ Pest Risk Assessor ready")
    
    # Load Report Generator
    logger.info("7/8 📄 Loading Report Generator...")
    try:
        get_report_generator()
        logger.info("    ✅ Report Generator ready")
    except Exception as e:
        logger.warning(f"    ⚠️  Report Generator not available: {e}")
    
    # Load Explainability Module
    logger.info("8/8 🔍 Loading Explainability Module...")
    try:
        get_explainer()
        logger.info("    ✅ Explainer ready")
    except Exception as e:
        logger.warning(f"    ⚠️  Explainer not available: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ ALL MODELS LOADED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("")

except Exception as e:
    logger.error("=" * 70)
    logger.error(f"❌ CRITICAL ERROR: Failed to load models")
    logger.error(f"Error: {str(e)}")
    logger.error("=" * 70)
    logger.error("")
    logger.error("Troubleshooting:")
    logger.error("1. Check internet connection (models download on first run)")
    logger.error("2. Ensure all dependencies installed: pip install -r requirements.txt")
    logger.error("3. Check transformers version: pip install --upgrade transformers")
    logger.error("")
    sys.exit(1)

# Import backend routes (models are already loaded above)
from src.backend import app as backend_app

# Create FastAPI app with custom docs URLs that work with proxies
app = FastAPI(
    title="Precision Agriculture API",
    description="AI-Powered Agricultural Decision Support System",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url="/api/openapi.json"  # Custom OpenAPI URL
)

# CORS Configuration - Allow all origins for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount backend routes
app.mount("/api", backend_app)

# Serve frontend static files if available
if os.path.exists("frontend/src"):
    app.mount("/static", StaticFiles(directory="frontend/src"), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML"""
        return FileResponse("frontend/src/index.html")
else:
    @app.get("/")
    async def root_redirect():
        """Redirect to API docs if no frontend"""
        return RedirectResponse(url="/docs")

# Custom Swagger UI with dynamic base URL
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    """Custom Swagger UI that works with proxies and deployments"""
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

# Custom ReDoc with dynamic base URL
@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html(request: Request):
    """Custom ReDoc that works with proxies and deployments"""
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title=f"{app.title} - ReDoc",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Server health check"""
    return {
        "status": "healthy",
        "message": "Precision Agriculture API is running",
        "version": "1.0.0",
        "models_loaded": True,
        "environment": os.getenv("ENVIRONMENT", "production")
    }

# API Info endpoint
@app.get("/api-info")
async def api_info(request: Request):
    """Get API information with correct URLs for current deployment"""
    base_url = str(request.base_url).rstrip('/')
    
    return {
        "title": "Precision Agriculture API",
        "version": "1.0.0",
        "status": "online",
        "base_url": base_url,
        "endpoints": {
            "docs": f"{base_url}/docs",
            "redoc": f"{base_url}/redoc",
            "openapi": f"{base_url}/api/openapi.json",
            "health": f"{base_url}/health",
            "api": f"{base_url}/api"
        },
        "features": [
            "Disease Detection (AI)",
            "Weather Forecasting",
            "Soil Health Analysis",
            "Yield Prediction",
            "Irrigation Scheduling",
            "Pest Risk Assessment",
            "AI Report Generation",
            "Model Explainability"
        ]
    }

# Startup event (models already loaded, just show message)
@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    logger.info("=" * 70)
    logger.info("🚀 SERVER READY!")
    logger.info("=" * 70)
    
    # Detect environment
    env = os.getenv("ENVIRONMENT", "unknown")
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "localhost")
    
    # Try to detect Lightning.ai or other cloud platforms
    if os.getenv("LIGHTNING_CLOUD_APP_ID"):
        logger.info("☁️  Running on Lightning.ai Cloud")
        logger.info("📍 Access your app via Lightning.ai provided URL")
    elif os.getenv("RAILWAY_ENVIRONMENT"):
        logger.info("☁️  Running on Railway")
    elif os.getenv("RENDER"):
        logger.info("☁️  Running on Render")
    else:
        logger.info(f"📍 Local: http://{host}:{port}")
    
    logger.info("📚 API Docs: /docs")
    logger.info("📊 ReDoc: /redoc")
    logger.info("❤️  Health: /health")
    logger.info("ℹ️  Info: /api-info")
    logger.info("=" * 70)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown"""
    logger.info("=" * 70)
    logger.info("🛑 Server shutting down...")
    logger.info("=" * 70)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("🌐 Starting uvicorn server...")
    logger.info(f"📍 Host: {host}, Port: {port}")
    logger.info("")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        proxy_headers=True,  # Important for deployment behind proxy
        forwarded_allow_ips="*"  # Trust proxy headers
    )