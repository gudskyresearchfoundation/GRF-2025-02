"""
Backend API Routes
FastAPI routes connecting all 7 models + Report Generation + XAI + Balaramaji Chat
UPDATED: Added Balaramaji Divine Assistant with Qwen 2.5 via Ollama
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import logging
import os
from PIL import Image
import io

# Import model getters (models will already be loaded by server.py)
from src.models.disease_detection import get_disease_detector
from src.models.weather_service import get_weather_service
from src.models.soil_model import get_soil_analyzer
from src.models.yield_prediction import get_yield_predictor
from src.models.irrigation import get_irrigation_optimizer
from src.models.pest_risk_assessor import get_pest_risk_assessor
from src.models.balaramaji import get_balaramaji_assistant  # ✅ NEW

# Import report generation and explainability
from src.report_generation import get_report_generator
from src.explainability import get_explainer

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Precision Agriculture Backend API",
    description="7 AI Models + Divine Chat Assistant + Report Generation + Explainability",
    version="2.0.0"  # ✅ UPDATED
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SoilData(BaseModel):
    """Soil analysis input data"""
    ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    organic_carbon: Optional[float] = None
    moisture: Optional[float] = None


class YieldData(BaseModel):
    """Yield prediction input data"""
    historical_yields: List[Dict]
    crop_type: str
    forecast_days: int = 120


class IrrigationData(BaseModel):
    """Irrigation calculation input data"""
    crop_type: str
    soil_moisture: float
    growth_stage: str
    rainfall_forecast_mm: float = 0
    temperature_celsius: float = 25


class PestData(BaseModel):
    """Pest risk assessment input data"""
    crop_type: str
    temperature: float
    humidity: float
    rainfall_7days: float
    previous_infestation: bool = False


class ChatMessage(BaseModel):
    """Chat message input for Balaramaji"""
    message: str
    analysis_context: Optional[Dict] = None
    chat_history: Optional[List[Dict]] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root - service information"""
    return {
        "service": "Precision Agriculture API",
        "version": "2.0.0",  # ✅ UPDATED
        "status": "online",
        "models": {
            "disease_detection": "MobileNetV2 (Hugging Face)",
            "weather": "Open-Meteo API",
            "soil_analysis": "Rule-based",
            "yield_prediction": "Prophet/Rule-based",
            "irrigation": "Rule-based (FAO guidelines)",
            "pest_risk": "Rule-based",
            "balaramaji_chat": "Qwen 2.5 14B/32B (Ollama)"  # ✅ NEW
        },
        "features": [
            "Disease Detection (AI)",
            "Weather Forecasting",
            "Soil Health Analysis",
            "Yield Prediction",
            "Irrigation Scheduling",
            "Pest Risk Assessment",
            "AI Report Generation",
            "Model Explainability (XAI)",
            "Divine Chat Assistant - Qwen 2.5 (Balaramaji)"  # ✅ NEW
        ],
        "endpoints": {
            "disease": "/predict/disease",
            "weather": "/weather",
            "soil": "/soil/analyze",
            "yield": "/yield/predict",
            "irrigation": "/irrigation/calculate",
            "pest": "/pest/assess",
            "report": "/report/generate",
            "explain": "/explain/*",
            "chat": "/chat/balaramaji"  # ✅ NEW
        },
        "ai_models": {  # ✅ NEW SECTION
            "local": ["Qwen 2.5 (Balaramaji - 14B/32B params)"],
            "cloud": ["MobileNetV2 (Disease Detection)"],
            "rule_based": ["Soil Analyzer", "Irrigation Optimizer", "Pest Risk Assessor"]
        }
    }


# ============================================================================
# MODEL 1: DISEASE DETECTION
# ============================================================================

@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    """
    Detect crop disease from plant leaf image

    Upload: Image file (JPG/PNG)
    Returns: Disease prediction with confidence
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image (JPG, PNG)"
            )

        contents = await file.read()

        detector = get_disease_detector()
        result = detector.predict(contents)

        logger.info(f"✅ Disease prediction: {result['crop']} - {result['disease']}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Disease prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/disease/supported-crops")
async def get_supported_crops():
    """Get list of crops supported by disease detection model"""
    try:
        detector = get_disease_detector()
        crops = detector.get_supported_crops()
        return {
            "supported_crops": crops,
            "total_crops": len(crops)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/disease/model-info")
async def get_disease_model_info():
    """Get disease detection model information"""
    try:
        detector = get_disease_detector()
        info = detector.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL 2: WEATHER SERVICE
# ============================================================================

@app.get("/weather")
async def get_weather(latitude: float, longitude: float, days: int = 7):
    """
    Get weather forecast for given coordinates

    Parameters:
        - latitude: Latitude (-90 to 90)
        - longitude: Longitude (-180 to 180)
        - days: Forecast days (1-16)

    Returns: Weather data with agricultural alerts
    """
    try:
        service = get_weather_service()

        weather_data = service.get_weather(latitude, longitude, days)
        alerts = service.get_agricultural_alerts(latitude, longitude, days)

        result = {
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "weather": weather_data,
            "agricultural_alerts": alerts,
            "forecast_days": days
        }

        logger.info(f"✅ Weather data fetched for ({latitude}, {longitude})")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌ Weather error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/current")
async def get_current_weather(latitude: float, longitude: float):
    """Get current weather conditions only"""
    try:
        service = get_weather_service()
        current = service.get_current_weather(latitude, longitude)

        logger.info(f"✅ Current weather fetched")
        return JSONResponse(content=current)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/search-location")
async def search_location(query: str):
    """Search for location by name"""
    try:
        service = get_weather_service()
        locations = service.search_location(query)

        return {
            "query": query,
            "locations": locations,
            "count": len(locations)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL 3: SOIL ANALYZER
# ============================================================================

@app.post("/soil/analyze")
async def analyze_soil(data: SoilData):
    """
    Analyze soil health and generate NPK recommendations

    Input: pH, Nitrogen, Phosphorus, Potassium, optional: organic carbon, moisture
    Returns: Health score, recommendations, NPK fertilizer requirements
    """
    try:
        analyzer = get_soil_analyzer()

        # Prepare analysis parameters
        params = {
            "ph": data.ph,
            "nitrogen": data.nitrogen,
            "phosphorus": data.phosphorus,
            "potassium": data.potassium
        }

        if data.organic_carbon is not None:
            params["organic_carbon"] = data.organic_carbon
        if data.moisture is not None:
            params["moisture"] = data.moisture

        result = analyzer.analyze(**params)

        logger.info(f"✅ Soil analysis: {result['health_status']} ({result['health_score']}/100)")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌ Soil analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL 4: YIELD PREDICTOR
# ============================================================================

@app.post("/yield/predict")
async def predict_yield(data: YieldData):
    """
    Predict crop yield based on historical data

    Input: Historical yields, crop type, forecast period
    Returns: Predicted yield with confidence intervals
    """
    try:
        predictor = get_yield_predictor()

        result = predictor.predict(
            historical_yields=data.historical_yields,
            crop_type=data.crop_type,
            forecast_days=data.forecast_days
        )

        logger.info(f"✅ Yield prediction: {result['predicted_average_yield']} kg/ha")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌ Yield prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL 5: IRRIGATION OPTIMIZER
# ============================================================================

@app.post("/irrigation/calculate")
async def calculate_irrigation(data: IrrigationData):
    """
    Calculate irrigation requirements

    Input: Crop, soil moisture, growth stage, weather
    Returns: Irrigation schedule and recommendations
    """
    try:
        optimizer = get_irrigation_optimizer()

        result = optimizer.calculate_irrigation(
            crop_type=data.crop_type,
            soil_moisture=data.soil_moisture,
            growth_stage=data.growth_stage,
            rainfall_forecast_mm=data.rainfall_forecast_mm,
            temperature_celsius=data.temperature_celsius
        )

        logger.info(f"✅ Irrigation calculated: {result['irrigation_amount_mm_per_day']} mm/day")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌ Irrigation calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL 6: PEST RISK ASSESSOR
# ============================================================================

@app.post("/pest/assess")
async def assess_pest_risk(data: PestData):
    """
    Assess pest risk based on environmental conditions

    Input: Crop, temperature, humidity, rainfall
    Returns: Risk level and preventive measures
    """
    try:
        assessor = get_pest_risk_assessor()

        result = assessor.assess_risk(
            crop_type=data.crop_type,
            temperature=data.temperature,
            humidity=data.humidity,
            rainfall_7days=data.rainfall_7days,
            previous_infestation=data.previous_infestation
        )

        logger.info(f"✅ Pest risk assessed: {result['risk_level']}")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌ Pest risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL 7: BALARAMAJI CHAT ASSISTANT (Qwen 2.5 via Ollama) ✅ NEW
# ============================================================================

@app.post("/chat/balaramaji")
async def chat_with_balaramaji(data: ChatMessage):
    """
    Chat with Balaramaji Divine Agricultural Assistant
    Powered by Qwen 2.5 (14B/32B parameters) via Ollama - 100% Local Inference
    
    Features:
    - Context-aware responses using all farm analysis results
    - Conversation memory with chat history
    - Sanskrit-infused divine wisdom + modern science
    - 100% local inference (no cloud, no API costs)
    - Privacy-focused (all data stays on your server)
    
    Request Body:
        - message: User's question (required)
        - analysis_context: All completed farm analyses (optional)
          Format: {
              "disease": {"data": {...}},
              "soil": {"data": {...}},
              "weather": {"data": {...}},
              "yield": {"data": {...}},
              "irrigation": {"data": {...}},
              "pest": {"data": {...}}
          }
        - chat_history: Previous conversation messages (optional)
          Format: [
              {"role": "user", "content": "message"},
              {"role": "assistant", "content": "response"}
          ]
    
    Returns:
        - response: Balaramaji's integrated divine guidance
        - model: "qwen2.5:14b" or "qwen2.5:32b" or "fallback"
        - source: "ollama" or "fallback"
        - has_context: boolean indicating if analysis context was provided
        - timestamp: ISO format timestamp
    
    Example Request:
        {
            "message": "What should I do about my crops?",
            "analysis_context": {
                "disease": {"data": {"crop": "Tomato", "disease": "Late Blight"}},
                "soil": {"data": {"ph": 5.2, "health_score": 65}}
            },
            "chat_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Blessed farmer..."}
            ]
        }
    """
    try:
        # Get Ollama host from environment (default: localhost)
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Get singleton assistant instance
        assistant = get_balaramaji_assistant(ollama_host=ollama_host)
        
        # Log request details for debugging
        context_keys = []
        if data.analysis_context:
            context_keys = list(data.analysis_context.keys())
        
        logger.info(f"💬 Balaramaji chat request")
        logger.info(f"   Message: {data.message[:50]}...")
        logger.info(f"   Context: {', '.join(context_keys) if context_keys else 'None'}")
        logger.info(f"   History: {len(data.chat_history) if data.chat_history else 0} messages")
        
        # Generate response with full context
        response = assistant.generate_response(
            user_message=data.message,
            analysis_context=data.analysis_context,
            chat_history=data.chat_history
        )
        
        # Log response metadata
        source = response.get('source', 'unknown')
        model = response.get('model', 'unknown')
        has_context = response.get('has_context', False)
        
        logger.info(f"✅ Balaramaji responded")
        logger.info(f"   Source: {source}")
        logger.info(f"   Model: {model}")
        logger.info(f"   Context used: {has_context}")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"❌ Balaramaji chat error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Chat service error: {str(e)}"
        )


@app.get("/chat/balaramaji/info")
async def get_balaramaji_info():
    """
    Get information about Balaramaji assistant and Qwen model
    
    Returns detailed information about:
    - Model specifications
    - Inference engine
    - Features and capabilities
    - Privacy and cost information
    """
    return {
        "name": "Balaramaji Divine Assistant",
        "description": "AI-powered agricultural guidance with divine wisdom",
        "persona": "Lord Balarama - God of Agriculture in Hindu mythology",
        "model": {
            "name": "Qwen 2.5",
            "variants": ["14B (16GB RAM)", "32B (24GB RAM)"],
            "current": os.getenv("BALARAMAJI_MODEL", "Auto-detected"),
            "parameters": "14-32 billion",
            "size": "8-20 GB",
            "context_window": "32K tokens",
            "languages": ["English", "Hindi", "Sanskrit"],
            "developer": "Alibaba Cloud"
        },
        "inference": {
            "engine": "Ollama",
            "type": "Local (On-premise)",
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "privacy": "100% - All data processed locally",
            "cost": "Free - No API fees",
            "offline_capable": True
        },
        "features": [
            "Context-aware responses from all 6 farm analyses",
            "Sanskrit-infused spiritual wisdom",
            "Modern scientific agricultural advice",
            "Natural conversation with memory (up to 8 messages)",
            "Specialized guidance for crops, soil, irrigation, pests",
            "Integrated analysis of multiple issues",
            "Priority-based action plans",
            "Intelligent fallback support for high availability"
        ],
        "capabilities": {
            "disease_advice": True,
            "soil_recommendations": True,
            "irrigation_planning": True,
            "pest_management": True,
            "yield_optimization": True,
            "weather_interpretation": True,
            "integrated_analysis": True,
            "conversation_memory": True
        },
        "supported_contexts": [
            "disease_detection",
            "soil_analysis",
            "weather_forecast",
            "yield_prediction",
            "irrigation_schedule",
            "pest_risk_assessment"
        ],
        "performance": {
            "with_gpu": "2-5 seconds per response",
            "cpu_only_14b": "20-30 seconds per response",
            "cpu_only_32b": "40-60 seconds per response",
            "recommended_hardware": "GPU with 16GB+ VRAM or 16GB+ RAM for CPU"
        },
        "status": "online",
        "version": "2.0-ollama-qwen"
    }


@app.get("/health/ollama")
async def check_ollama_health():
    """
    Health check for Ollama service
    Verifies Qwen 2.5 model is loaded and ready
    
    Returns:
    - status: healthy/degraded/unhealthy
    - ollama connection status
    - qwen model availability
    - loaded models list
    """
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        import requests
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            model_names = [m.get("name") for m in models]
            
            # Check for any Qwen 2.5 variant
            qwen_32b = any("qwen2.5:32b" in name for name in model_names)
            qwen_14b = any("qwen2.5:14b" in name for name in model_names)
            qwen_7b = any("qwen2.5:7b" in name for name in model_names)
            qwen_available = qwen_32b or qwen_14b or qwen_7b
            
            # Determine which model is being used
            active_model = None
            if qwen_32b:
                active_model = "qwen2.5:32b"
            elif qwen_14b:
                active_model = "qwen2.5:14b"
            elif qwen_7b:
                active_model = "qwen2.5:7b"
            
            return {
                "status": "healthy" if qwen_available else "degraded",
                "ollama": {
                    "connected": True,
                    "host": ollama_host,
                    "models_loaded": len(models),
                    "responsive": True
                },
                "qwen": {
                    "available": qwen_available,
                    "active_model": active_model,
                    "variants_installed": {
                        "32b": qwen_32b,
                        "14b": qwen_14b,
                        "7b": qwen_7b
                    }
                },
                "all_models": model_names,
                "message": f"Qwen 2.5 ready ({active_model})" if qwen_available else "No Qwen model found - run: ollama pull qwen2.5:14b",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "unhealthy",
                "ollama": {
                    "connected": False,
                    "error": f"HTTP {response.status_code}",
                    "host": ollama_host
                },
                "message": "Ollama service returned error",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama": {
                "connected": False,
                "error": str(e),
                "host": os.getenv("OLLAMA_HOST", "http://localhost:11434")
            },
            "message": "Ollama service unreachable - ensure Ollama is running: ollama serve",
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# AI REPORT GENERATION
# ============================================================================

@app.post("/report/generate")
async def generate_report(predictions: Dict = Body(...)):
    """
    Generate AI-powered comprehensive agricultural report

    Input: All model predictions
    Returns: Natural language report with recommendations
    """
    try:
        report_gen = get_report_generator()

        report = report_gen.generate_report(predictions)

        logger.info("✅ AI report generated")

        return JSONResponse(content={
            "status": "success",
            "report": report,
            "generated_at": report_gen.get_timestamp()
        })

    except Exception as e:
        logger.error(f"❌ Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXPLAINABILITY (XAI)
# ============================================================================

@app.post("/explain/disease")
async def explain_disease_prediction(file: UploadFile = File(...)):
    """
    Explain disease detection prediction using LIME

    Upload: Same image used for disease detection
    Returns: Explanation with highlighted regions
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        explainer = get_explainer()
        detector = get_disease_detector()

        explanation = explainer.explain_image_prediction(image, detector)

        logger.info("✅ Disease prediction explained")

        return JSONResponse(content=explanation)

    except Exception as e:
        logger.error(f"❌ Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/tabular")
async def explain_tabular_prediction(
    features: List[float] = Body(...),
    feature_names: List[str] = Body(...),
    model_type: str = Body(...)
):
    """
    Explain tabular model predictions using SHAP

    Input: Features and model type
    Returns: Feature importance explanation
    """
    try:
        explainer = get_explainer()

        explanation = explainer.explain_tabular_prediction(
            features=features,
            feature_names=feature_names,
            model_type=model_type
        )

        logger.info(f"✅ {model_type} prediction explained")

        return JSONResponse(content=explanation)

    except Exception as e:
        logger.error(f"❌ Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK & INFO
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        models_status = {
            "disease_detector": "loaded",
            "weather_service": "loaded",
            "soil_analyzer": "loaded",
            "yield_predictor": "loaded",
            "irrigation_optimizer": "loaded",
            "pest_assessor": "loaded",
            "balaramaji_assistant": "loaded",  # ✅ NEW
            "agro_intelligence": "loaded"
        }

        return {
            "status": "healthy",
            "models": models_status,
            "message": "All systems operational",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# STARTUP EVENT - Models already loaded in server.py
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize all models on startup
    NOTE: Models are pre-loaded in server.py before uvicorn starts
    This is just a verification step
    """
    logger.info("🔄 Verifying models are loaded...")
    
    try:
        # Just verify they're accessible (already loaded)
        get_disease_detector()
        get_weather_service()
        get_soil_analyzer()
        get_yield_predictor()
        get_irrigation_optimizer()
        get_pest_risk_assessor()
        
        # Try to load Balaramaji (may use fallback if Ollama not available)
        try:
            get_balaramaji_assistant()
            logger.info("✅ Balaramaji assistant verified")
        except Exception as e:
            logger.warning(f"⚠️  Balaramaji not available (will use fallback): {e}")
        
        # Try to load report generator
        try:
            get_report_generator()
        except:
            pass
        
        # Try to load explainer
        try:
            get_explainer()
        except:
            pass
        
        logger.info("✅ All models verified and ready")
        
    except Exception as e:
        logger.error(f"⚠️  Model verification failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)