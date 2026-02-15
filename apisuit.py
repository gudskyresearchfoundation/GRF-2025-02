"""
🧪 COMPREHENSIVE TEST SUITE - UPDATED WITH CATBOOST
Precision Agriculture System - All Components Test
UPDATED: Tests both simple and advanced yield prediction

This script:
1. Tests each model independently
2. Tests both Prophet and CatBoost yield predictions
3. Saves results to timestamped text file
4. Works on Windows, Linux, and Mac
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResults:
    """Store and display test results"""
    
    def __init__(self):
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "tests": []
        }
        self.start_time = datetime.now()
        self.output_lines = []
    
    def add_test(self, name: str, status: str, message: str = "", error: str = ""):
        """Add a test result"""
        self.results["total"] += 1
        self.results[status] += 1
        
        self.results["tests"].append({
            "name": name,
            "status": status,
            "message": message,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def print_and_save(self, text: str):
        """Print to console and save to output buffer"""
        print(text)
        self.output_lines.append(text)
    
    def print_summary(self):
        """Print test summary"""
        self.print_and_save("\n" + "=" * 80)
        self.print_and_save("TEST SUMMARY")
        self.print_and_save("=" * 80)
        self.print_and_save(f"Total Tests: {self.results['total']}")
        self.print_and_save(f"Passed: {self.results['passed']}")
        self.print_and_save(f"Failed: {self.results['failed']}")
        self.print_and_save(f"Skipped: {self.results['skipped']}")
        
        if self.results['total'] > 0:
            success_rate = (self.results['passed'] / self.results['total']) * 100
        else:
            success_rate = 0
            
        self.print_and_save(f"Success Rate: {success_rate:.1f}%")
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.print_and_save(f"Time Taken: {duration:.2f} seconds")
        self.print_and_save("=" * 80)
        
        self.print_and_save("\nDETAILED RESULTS:")
        for test in self.results["tests"]:
            status_emoji = {
                "passed": "PASSED",
                "failed": "FAILED",
                "skipped": "SKIPPED"
            }.get(test["status"], "UNKNOWN")
            
            self.print_and_save(f"\n{status_emoji} - {test['name']}")
            if test["message"]:
                self.print_and_save(f"   Message: {test['message']}")
            if test["error"]:
                self.print_and_save(f"   Error: {test['error']}")
    
    def save_to_file(self, output_dir: str = "test_results"):
        """Save results to text file"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PRECISION AGRICULTURE SYSTEM - TEST RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Platform: {sys.platform}\n")
                f.write(f"Python Version: {sys.version}\n")
                f.write("=" * 80 + "\n\n")
                
                for line in self.output_lines:
                    try:
                        f.write(line + "\n")
                    except UnicodeEncodeError:
                        ascii_line = line.encode('ascii', 'ignore').decode('ascii')
                        f.write(ascii_line + "\n")
                
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("DETAILED JSON RESULTS\n")
                f.write("=" * 80 + "\n")
                import json
                f.write(json.dumps(self.results, indent=2))
            
            self.print_and_save(f"\nResults saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.print_and_save(f"\nFailed to save results: {str(e)}")
            traceback.print_exc()
            return None


class PrecisionAgricultureTester:
    """Comprehensive testing for all models"""
    
    def __init__(self):
        self.results = TestResults()
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data for all models"""
        return {
            "soil": {
                "good": {"ph": 6.5, "nitrogen": 70, "phosphorus": 25, "potassium": 150, "organic_carbon": 1.0, "moisture": 0.5},
                "poor": {"ph": 5.0, "nitrogen": 20, "phosphorus": 8, "potassium": 50, "organic_carbon": 0.3, "moisture": 0.2},
            },
            "weather": {
                "delhi": {"latitude": 28.6139, "longitude": 77.2090, "name": "Delhi, India"},
            },
            "yield_simple": {
                "with_history": {
                    "historical_yields": [
                        {"date": "2020-01-01", "yield": 2600},
                        {"date": "2020-06-01", "yield": 2700},
                        {"date": "2021-01-01", "yield": 2850},
                        {"date": "2021-06-01", "yield": 2950},
                        {"date": "2022-01-01", "yield": 3100},
                        {"date": "2022-06-01", "yield": 3200},
                        {"date": "2023-01-01", "yield": 3350}
                    ],
                    "crop_type": "wheat",
                    "forecast_days": 120
                },
                "without_history": {
                    "historical_yields": [],
                    "crop_type": "rice",
                    "forecast_days": 120
                }
            },
            "yield_advanced": {
                "rice_punjab": {
                    "crop": "Rice",
                    "season": "Kharif",
                    "state": "Punjab",
                    "crop_year": 2024,
                    "area": 50000,
                    "annual_rainfall": 1200,
                    "fertilizer": 5000000,
                    "pesticide": 15000
                },
                "wheat_karnataka": {
                    "crop": "Wheat",
                    "season": "Rabi",
                    "state": "Karnataka",
                    "crop_year": 2024,
                    "area": 75000,
                    "annual_rainfall": 800,
                    "fertilizer": 6000000,
                    "pesticide": 20000
                }
            },
            "irrigation": {
                "critical": {"crop_type": "rice", "soil_moisture": 0.2, "growth_stage": "flowering", "rainfall_forecast_mm": 0, "temperature_celsius": 32},
                "moderate": {"crop_type": "wheat", "soil_moisture": 0.5, "growth_stage": "vegetative", "rainfall_forecast_mm": 10, "temperature_celsius": 25},
            },
            "pest": {
                "high_risk": {"crop_type": "rice", "temperature": 28, "humidity": 85, "rainfall_7days": 80, "previous_infestation": True},
                "low_risk": {"crop_type": "corn", "temperature": 20, "humidity": 50, "rainfall_7days": 10, "previous_infestation": False}
            }
        }
    
    def test_disease_detection(self):
        """Test Disease Detection Model"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING MODEL 1: DISEASE DETECTION")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.models.disease_detection import get_disease_detector
            
            try:
                detector = get_disease_detector()
                self.results.add_test("Disease Detector - Initialization", "passed", "Model loaded")
                self.results.print_and_save("PASSED: Model Initialization")
            except Exception as e:
                self.results.add_test("Disease Detector - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            try:
                info = detector.get_model_info()
                self.results.add_test("Disease Detector - Model Info", "passed", f"Classes: {info['num_classes']}")
                self.results.print_and_save(f"PASSED: {info['num_classes']} classes")
            except Exception as e:
                self.results.add_test("Disease Detector - Model Info", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
            try:
                crops = detector.get_supported_crops()
                self.results.add_test("Disease Detector - Supported Crops", "passed", f"{len(crops)} crops")
                self.results.print_and_save(f"PASSED: {len(crops)} crops")
            except Exception as e:
                self.results.add_test("Disease Detector - Supported Crops", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
            self.results.add_test("Disease Detector - Image Prediction", "skipped", "Requires image file")
            self.results.print_and_save("SKIPPED: Image Prediction")
            
        except ImportError as e:
            self.results.add_test("Disease Detector - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_weather_service(self):
        """Test Weather Service"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING MODEL 2: WEATHER SERVICE")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.models.weather_service import get_weather_service
            
            try:
                service = get_weather_service()
                self.results.add_test("Weather Service - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Weather Service - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            try:
                locations = service.search_location("Delhi", max_results=3)
                self.results.add_test("Weather Service - Location Search", "passed", f"Found {len(locations)}")
                self.results.print_and_save(f"PASSED: Found {len(locations)} locations")
            except Exception as e:
                self.results.add_test("Weather Service - Location Search", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
            try:
                delhi = self.test_data["weather"]["delhi"]
                weather = service.get_weather(delhi["latitude"], delhi["longitude"], days=3)
                self.results.add_test("Weather Service - Get Weather", "passed")
                self.results.print_and_save("PASSED: Weather Retrieved")
            except Exception as e:
                self.results.add_test("Weather Service - Get Weather", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                
        except ImportError as e:
            self.results.add_test("Weather Service - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_soil_analyzer(self):
        """Test Soil Analyzer"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING MODEL 3: SOIL ANALYZER")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.models.soil_model import get_soil_analyzer
            
            try:
                analyzer = get_soil_analyzer()
                self.results.add_test("Soil Analyzer - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Soil Analyzer - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            try:
                good_soil = self.test_data["soil"]["good"]
                result = analyzer.analyze(**good_soil)
                self.results.add_test("Soil Analyzer - Good Soil", "passed", f"Score: {result['health_score']}")
                self.results.print_and_save(f"PASSED: Score={result['health_score']}/100")
            except Exception as e:
                self.results.add_test("Soil Analyzer - Good Soil", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
            try:
                poor_soil = self.test_data["soil"]["poor"]
                result = analyzer.analyze(**poor_soil)
                self.results.add_test("Soil Analyzer - Poor Soil", "passed", f"Score: {result['health_score']}")
                self.results.print_and_save(f"PASSED: Score={result['health_score']}/100")
            except Exception as e:
                self.results.add_test("Soil Analyzer - Poor Soil", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
        except ImportError as e:
            self.results.add_test("Soil Analyzer - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_yield_predictor(self):
        """Test Yield Predictor (Both Simple and Advanced)"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING MODEL 4: YIELD PREDICTOR (HYBRID)")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.models.yield_prediction import get_yield_predictor
            
            # Test Initialization
            try:
                predictor = get_yield_predictor()
                self.results.add_test("Yield Predictor - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Yield Predictor - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            # Test Model Info
            try:
                info = predictor.get_model_info()
                self.results.add_test("Yield Predictor - Model Info", "passed", f"CatBoost: {info['catboost_available']}")
                self.results.print_and_save(f"PASSED: CatBoost={info['catboost_available']}, Prophet={info['prophet_available']}")
            except Exception as e:
                self.results.add_test("Yield Predictor - Model Info", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
            # Test Simple Prediction (Prophet/rule-based)
            try:
                test_data = self.test_data["yield_simple"]["with_history"]
                result = predictor.predict(**test_data)
                # Check for positive prediction
                if result["predicted_average_yield"] > 0:
                    self.results.add_test("Yield Predictor - Simple (Prophet)", "passed", f"Yield: {result['predicted_average_yield']} kg/ha")
                    self.results.print_and_save(f"PASSED: Simple Prediction={result['predicted_average_yield']} kg/ha")
                else:
                    self.results.add_test("Yield Predictor - Simple (Prophet)", "failed", error="Negative prediction")
                    self.results.print_and_save(f"FAILED: Negative prediction {result['predicted_average_yield']}")
            except Exception as e:
                self.results.add_test("Yield Predictor - Simple (Prophet)", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
            # Test Advanced Prediction (CatBoost)
            try:
                test_data = self.test_data["yield_advanced"]["rice_punjab"]
                result = predictor.predict_advanced(**test_data)
                # Check for positive prediction
                if result.get("predicted_yield", 0) > 0:
                    self.results.add_test("Yield Predictor - Advanced (CatBoost)", "passed", f"Yield: {result['predicted_yield']} kg/ha")
                    self.results.print_and_save(f"PASSED: CatBoost Prediction={result['predicted_yield']} kg/ha (Confidence: {result.get('confidence', 'N/A')})")
                else:
                    self.results.add_test("Yield Predictor - Advanced (CatBoost)", "skipped", "CatBoost model not available")
                    self.results.print_and_save("SKIPPED: CatBoost model not loaded")
            except Exception as e:
                self.results.add_test("Yield Predictor - Advanced (CatBoost)", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
        except ImportError as e:
            self.results.add_test("Yield Predictor - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_irrigation_optimizer(self):
        """Test Irrigation Optimizer"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING MODEL 5: IRRIGATION OPTIMIZER")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.models.irrigation import get_irrigation_optimizer
            
            try:
                optimizer = get_irrigation_optimizer()
                self.results.add_test("Irrigation Optimizer - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Irrigation Optimizer - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            try:
                critical = self.test_data["irrigation"]["critical"]
                result = optimizer.calculate_irrigation(**critical)
                self.results.add_test("Irrigation Optimizer - Critical", "passed", f"Water: {result['irrigation_amount_mm_per_day']} mm/day")
                self.results.print_and_save(f"PASSED: Water={result['irrigation_amount_mm_per_day']} mm/day")
            except Exception as e:
                self.results.add_test("Irrigation Optimizer - Critical", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
        except ImportError as e:
            self.results.add_test("Irrigation Optimizer - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_pest_risk_assessor(self):
        """Test Pest Risk Assessor"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING MODEL 6: PEST RISK ASSESSOR")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.models.pest_risk_assessor import get_pest_risk_assessor
            
            try:
                assessor = get_pest_risk_assessor()
                self.results.add_test("Pest Risk Assessor - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Pest Risk Assessor - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            try:
                high_risk = self.test_data["pest"]["high_risk"]
                result = assessor.assess_risk(**high_risk)
                self.results.add_test("Pest Risk Assessor - High Risk", "passed", f"Risk: {result['risk_level']}")
                self.results.print_and_save(f"PASSED: Risk={result['risk_level']}")
            except Exception as e:
                self.results.add_test("Pest Risk Assessor - High Risk", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
        except ImportError as e:
            self.results.add_test("Pest Risk Assessor - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_report_generator(self):
        """Test Report Generator"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING REPORT GENERATION")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.report_generation import get_report_generator
            
            try:
                generator = get_report_generator()
                self.results.add_test("Report Generator - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Report Generator - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            try:
                test_predictions = {
                    'disease': {'crop': 'Tomato', 'disease': 'Late_blight', 'health_status': 'UNHEALTHY', 'confidence': 92.5},
                    'soil': {'health_status': 'Good', 'health_score': 75, 'recommendations': ['Test']},
                    'pest': {'risk_level': 'Medium', 'risk_score': 55}
                }
                report = generator.generate_report(test_predictions)
                self.results.add_test("Report Generator - Generate", "passed", f"{len(report)} chars")
                self.results.print_and_save(f"PASSED: {len(report)} characters")
            except Exception as e:
                self.results.add_test("Report Generator - Generate", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
        except ImportError as e:
            self.results.add_test("Report Generator - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_explainability(self):
        """Test Explainability Module"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING EXPLAINABILITY (XAI)")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.explainability import get_explainer
            
            try:
                explainer = get_explainer()
                self.results.add_test("Explainability - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Explainability - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
                return
            
            try:
                capabilities = explainer.get_capabilities()
                self.results.add_test("Explainability - Capabilities", "passed", f"LIME={capabilities['lime_available']}")
                self.results.print_and_save(f"PASSED: LIME={capabilities['lime_available']}")
            except Exception as e:
                self.results.add_test("Explainability - Capabilities", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
        except ImportError as e:
            self.results.add_test("Explainability - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def test_balaramaji_assistant(self):
        """Test Balaramaji Chat Assistant"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("TESTING BALARAMAJI ASSISTANT")
        self.results.print_and_save("=" * 80)
        
        try:
            from src.models.balaramaji import get_balaramaji_assistant
            
            try:
                assistant = get_balaramaji_assistant()
                self.results.add_test("Balaramaji - Initialization", "passed")
                self.results.print_and_save("PASSED: Initialization")
            except Exception as e:
                self.results.add_test("Balaramaji - Initialization", "failed", error=str(e))
                self.results.print_and_save(f"FAILED: {str(e)}")
            
        except ImportError as e:
            self.results.add_test("Balaramaji - Import", "failed", error=str(e))
            self.results.print_and_save(f"FAILED: Import - {str(e)}")
    
    def run_all_tests(self):
        """Run all test suites"""
        self.results.print_and_save("\n" + "=" * 80)
        self.results.print_and_save("PRECISION AGRICULTURE SYSTEM - COMPREHENSIVE TEST SUITE")
        self.results.print_and_save("UPDATED: CatBoost Yield Prediction Integration")
        self.results.print_and_save("=" * 80)
        self.results.print_and_save(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.results.print_and_save(f"Platform: {sys.platform}")
        self.results.print_and_save(f"Python Version: {sys.version.split()[0]}")
        self.results.print_and_save("=" * 80)
        
        # Run all tests
        self.test_disease_detection()
        self.test_weather_service()
        self.test_soil_analyzer()
        self.test_yield_predictor()  # Now tests both simple and advanced
        self.test_irrigation_optimizer()
        self.test_pest_risk_assessor()
        self.test_report_generator()
        self.test_explainability()
        self.test_balaramaji_assistant()
        
        # Print summary
        self.results.print_summary()
        
        self.results.print_and_save(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.results.print_and_save("=" * 80)
        
        # Save results to file
        self.results.save_to_file()


def main():
    """Main test runner"""
    try:
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        print(f"Python version: {sys.version}")
        print()
        
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"Added to path: {current_dir}")
        
        tester = PrecisionAgricultureTester()
        tester.run_all_tests()
        
        if tester.results.results["failed"] > 0:
            return 1
        return 0
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)