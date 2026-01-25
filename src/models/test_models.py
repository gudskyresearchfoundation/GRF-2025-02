"""
Test Suite for All 6 Models
Tests each model independently
"""

import sys
sys.path.append('..')

print("=" * 70)
print("🧪 TESTING ALL 6 MODELS")
print("=" * 70)

# Test Model 1: Disease Detection
print("\n📷 Test 1: Disease Detection Model")
try:
    from disease_detection import DiseaseDetector
    detector = DiseaseDetector()
    print(f"✅ Disease Detector initialized")
    print(f"   Supported crops: {len(detector.get_supported_crops())}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Model 2: Weather Service
print("\n🌤️  Test 2: Weather Service")
try:
    from weather_service import WeatherService
    weather = WeatherService()
    # Test with Delhi coordinates
    alerts = weather.get_agricultural_alerts(28.6139, 77.2090, days=3)
    print(f"✅ Weather Service working")
    print(f"   Alerts generated: {len(alerts)}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Model 3: Soil Analyzer
print("\n🌱 Test 3: Soil Analyzer")
try:
    from soil_model import SoilAnalyzer
    soil = SoilAnalyzer()
    result = soil.analyze(ph=6.5, nitrogen=70, phosphorus=25, potassium=150)
    print(f"✅ Soil Analyzer working")
    print(f"   Health Status: {result['health_status']}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Model 4: Yield Predictor
print("\n📈 Test 4: Yield Predictor")
try:
    from yield_prediction import YieldPredictor
    yield_pred = YieldPredictor()
    result = yield_pred.predict([], "wheat", forecast_days=120)
    print(f"✅ Yield Predictor working")
    print(f"   Predicted Yield: {result['predicted_average_yield']} kg/ha")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Model 5: Irrigation Optimizer
print("\n💧 Test 5: Irrigation Optimizer")
try:
    from irrigation import IrrigationOptimizer
    irrigation = IrrigationOptimizer()
    result = irrigation.calculate_irrigation("rice", 0.4, "vegetative", rainfall_forecast_mm=10)
    print(f"✅ Irrigation Optimizer working")
    print(f"   Irrigation needed: {result['irrigation_amount_mm_per_day']} mm/day")
except Exception as e:
    print(f"❌ Error: {e}")

# Test Model 6: Pest Risk Assessor
print("\n🐛 Test 6: Pest Risk Assessor")
try:
    from pest_risk_assessor import PestRiskAssessor
    pest = PestRiskAssessor()
    result = pest.assess_risk("rice", temperature=28, humidity=75, rainfall_7days=30)
    print(f"✅ Pest Risk Assessor working")
    print(f"   Risk Level: {result['risk_level']}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("✅ ALL MODEL TESTS COMPLETED!")
print("=" * 70)
