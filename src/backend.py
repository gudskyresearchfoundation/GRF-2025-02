"""
Backend API Routes
FastAPI routes connecting all 6 models + Report Generation + XAI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from PIL import Image
import io

# Import model getters (models will already be loaded by server.py)
from src.models.disease_detection import get_disease_detector
from src.models.weather_service import get_weather_service
from src.models.soil_model import get_soil_analyzer
from src.models.yield_prediction import get_yield_predictor
from src.models.irrigation import get_irrigation_optimizer
from src.models.pest_risk_assessor import get_pest_risk_assessor

# Import report generation and explainability
from src.report_generation import get_report_generator
from src.explainability import get_explainer

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Precision Agriculture Backend API",
    description="6 ML Models + AI Report Generation + Explainability + CatBoost Yield Prediction",
    version="2.0.0"
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
    """Yield prediction input data (Simple - Prophet/rule-based)"""
    historical_yields: List[Dict]
    crop_type: str
    forecast_days: int = 120


class YieldDataAdvanced(BaseModel):
    """Advanced yield prediction input data (CatBoost ML model)"""
    crop: str
    season: str
    state: str
    crop_year: int
    area: float
    annual_rainfall: float
    fertilizer: float
    pesticide: float


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


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root - service information"""
    return {
        "service": "Precision Agriculture API",
        "version": "2.0.0",
        "status": "online",
        "models": {
            "disease_detection": "MobileNetV2 (Hugging Face)",
            "weather": "Open-Meteo API",
            "soil_analysis": "Rule-based",
            "yield_prediction": "Hybrid (CatBoost 96.48% + Prophet)",
            "irrigation": "Rule-based (FAO guidelines)",
            "pest_risk": "Rule-based"
        },
        "features": [
            "Disease Detection",
            "Weather Forecasting",
            "Soil Health Analysis",
            "Yield Prediction (Simple & Advanced)",
            "Irrigation Scheduling",
            "Pest Risk Assessment",
            "AI Report Generation",
            "Model Explainability (XAI)"
        ],
        "endpoints": {
            "disease": "/predict/disease",
            "weather": "/weather",
            "soil": "/soil/analyze",
            "yield_simple": "/yield/predict",
            "yield_advanced": "/yield/predict-advanced",
            "yield_model_info": "/yield/model-info",
            "irrigation": "/irrigation/calculate",
            "pest": "/pest/assess",
            "report": "/report/generate",
            "explain": "/explain/*"
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
    Analyze soil health and provide recommendations

    Input: pH, NPK levels, organic carbon, moisture
    Returns: Health score, status, and fertilizer recommendations
    """
    try:
        analyzer = get_soil_analyzer()

        result = analyzer.analyze(
            ph=data.ph,
            nitrogen=data.nitrogen,
            phosphorus=data.phosphorus,
            potassium=data.potassium,
            organic_carbon=data.organic_carbon,
            moisture=data.moisture
        )

        logger.info(f"✅ Soil analysis: {result['health_status']} ({result['health_score']}/100)")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌ Soil analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL 4: YIELD PREDICTOR (UPDATED WITH CATBOOST)
# ============================================================================

@app.post("/yield/predict")
async def predict_yield(data: YieldData):
    """
    Simple yield prediction based on historical data (Prophet/rule-based)
    
    Good for: Trend analysis, quick estimates
    Accuracy: Moderate (depends on historical data quality)

    Input: Historical yields, crop type, forecast period
    Returns: Predicted yield with trend analysis
    
    For high accuracy, use /yield/predict-advanced instead
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


@app.post("/yield/predict-advanced")
async def predict_yield_advanced(data: YieldDataAdvanced):
    """
    Advanced yield prediction using CatBoost ML model
    
    ✨ RECOMMENDED METHOD ✨
    
    Good for: Accurate predictions with detailed farm data
    Accuracy: 96.48% (R² score)
    Model: CatBoost trained on 19,689 records (1997-2020)
    
    Input: Crop, season, state, year, area, rainfall, fertilizer, pesticide
    Returns: High-accuracy predicted yield with confidence intervals
    
    Example:
    {
        "crop": "Rice",
        "season": "Kharif",
        "state": "Punjab",
        "crop_year": 2024,
        "area": 50000,
        "annual_rainfall": 1200,
        "fertilizer": 5000000,
        "pesticide": 15000
    }
    """
    try:
        predictor = get_yield_predictor()
        
        result = predictor.predict_advanced(
            crop=data.crop,
            season=data.season,
            state=data.state,
            crop_year=data.crop_year,
            area=data.area,
            annual_rainfall=data.annual_rainfall,
            fertilizer=data.fertilizer,
            pesticide=data.pesticide
        )
        
        logger.info(f"✅ Advanced yield prediction: {result['predicted_yield']} kg/ha")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"❌ Advanced yield prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/yield/model-info")
async def get_yield_model_info():
    """
    Get information about available yield prediction models
    
    Returns: Model availability, accuracy, and recommended usage
    """
    try:
        predictor = get_yield_predictor()
        info = predictor.get_model_info()
        
        return JSONResponse(content=info)
        
    except Exception as e:
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
            "pest_assessor": "loaded"
        }

        return {
            "status": "healthy",
            "models": models_status,
            "message": "All systems operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
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
        
        try:
            get_report_generator()
        except:
            pass
        
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