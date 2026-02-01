"""
Models Package
Contains all ML models for agricultural predictions
Updated: Added Balaramaji chat assistant
"""

from .disease_detection import DiseaseDetector, get_disease_detector
from .weather_service import WeatherService, get_weather_service
from .soil_model import SoilAnalyzer, get_soil_analyzer
from .yield_prediction import YieldPredictor, get_yield_predictor
from .irrigation import IrrigationOptimizer, get_irrigation_optimizer
from .pest_risk_assessor import PestRiskAssessor, get_pest_risk_assessor
from .balaramaji import BalaramajiAssistant, get_balaramaji_assistant

__all__ = [
    'DiseaseDetector',
    'WeatherService',
    'SoilAnalyzer',
    'YieldPredictor',
    'IrrigationOptimizer',
    'PestRiskAssessor',
    'BalaramajiAssistant',
    'get_disease_detector',
    'get_weather_service',
    'get_soil_analyzer',
    'get_yield_predictor',
    'get_irrigation_optimizer',
    'get_pest_risk_assessor',
    'get_balaramaji_assistant'
]

__version__ = "2.0.0"  # Updated with Balarama Ji integration