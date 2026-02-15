"""
Model 4: Yield Prediction - UPDATED WITH CATBOOST
Hybrid approach: CatBoost (primary) + Prophet (fallback)
CatBoost: 96.48% accuracy for detailed predictions
Prophet: Simple trend-based predictions

UPDATED: Model path set to src/models/catboost_crop_yield_clean.cbm
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# CatBoost import with fallback
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("⚠️ CatBoost not installed. Install with: pip install catboost")

# Prophet import with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("⚠️ Prophet not installed. Install with: pip install prophet")


class YieldPredictor:
    """
    Crop yield prediction using hybrid approach:
    - CatBoost: High-accuracy ML model (96.48% R²) for detailed predictions
    - Prophet: Simple trend analysis for historical data
    """
    
    def __init__(self, model_path: str = None):
        """Initialize Yield Predictor"""
        logger.info("📈 Initializing Yield Predictor (Hybrid: CatBoost + Prophet)")
        
        self.catboost_model = None
        self.prophet_model = None
        self.model_path = model_path or self._find_model_path()
        
        # Try to load CatBoost model
        if CATBOOST_AVAILABLE:
            self._load_catboost_model()
        else:
            logger.warning("⚠️ CatBoost not available - will use Prophet or rule-based fallback")
        
        if not PROPHET_AVAILABLE:
            logger.warning("⚠️ Prophet not available - using CatBoost or rule-based only")
    
    def _find_model_path(self) -> Optional[str]:
        """Find CatBoost model file - UPDATED FOR src/models/ path"""
        possible_paths = [
            'src/models/catboost_crop_yield_clean.cbm',  # PRIMARY PATH
            'catboost_crop_yield_clean.cbm',
            'models/catboost_crop_yield_clean.cbm',
            '../models/catboost_crop_yield_clean.cbm',
            './src/models/catboost_crop_yield_clean.cbm',
            os.path.join('src', 'models', 'catboost_crop_yield_clean.cbm')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ Found CatBoost model at: {path}")
                return path
        
        logger.warning("⚠️ CatBoost model file not found in any of these locations:")
        for path in possible_paths:
            logger.warning(f"   - {path}")
        logger.warning("   Using fallback methods (Prophet or rule-based)")
        return None
    
    def _load_catboost_model(self):
        """Load pre-trained CatBoost model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.catboost_model = CatBoostRegressor()
                self.catboost_model.load_model(self.model_path)
                logger.info(f"✅ CatBoost model loaded from {self.model_path}")
                logger.info("   Expected accuracy: 96.48% (R² score)")
                logger.info("   Training data: 19,689 records (1997-2020)")
            else:
                logger.warning("⚠️ CatBoost model file not found. Will use Prophet or rules.")
                if self.model_path:
                    logger.warning(f"   Checked path: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load CatBoost model: {str(e)}")
            self.catboost_model = None
    
    # ========================================================================
    # PRIMARY METHOD: CATBOOST PREDICTION (Recommended)
    # ========================================================================
    
    def predict_advanced(self,
                        crop: str,
                        season: str,
                        state: str,
                        crop_year: int,
                        area: float,
                        annual_rainfall: float,
                        fertilizer: float,
                        pesticide: float) -> Dict:
        """
        Advanced yield prediction using CatBoost model (96.48% accuracy)
        
        Args:
            crop: Type of crop (e.g., 'Rice', 'Wheat', 'Maize', 'Cotton')
            season: Growing season (e.g., 'Kharif', 'Rabi', 'Whole Year')
            state: Indian state name (e.g., 'Punjab', 'Karnataka', 'Tamil Nadu')
            crop_year: Year of cultivation (e.g., 2024)
            area: Area under cultivation in hectares
            annual_rainfall: Annual rainfall in mm
            fertilizer: Fertilizer used in kg
            pesticide: Pesticide used in kg
        
        Returns:
            Dictionary with prediction results
        """
        try:
            if not CATBOOST_AVAILABLE or self.catboost_model is None:
                return self._fallback_advanced_prediction(crop, area, crop_year)
            
            # Calculate engineered features (same as training)
            fertilizer_per_area = fertilizer / (area + 1)
            pesticide_per_area = pesticide / (area + 1)
            rainfall_per_area = annual_rainfall / (area + 1)
            total_input = fertilizer + pesticide
            total_input_per_area = total_input / (area + 1)
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Crop': [crop],
                'Season': [season],
                'State': [state],
                'Crop_Year': [crop_year],
                'Area': [area],
                'Annual_Rainfall': [annual_rainfall],
                'Fertilizer': [fertilizer],
                'Pesticide': [pesticide],
                'Fertilizer_per_Area': [fertilizer_per_area],
                'Pesticide_per_Area': [pesticide_per_area],
                'Rainfall_per_Area': [rainfall_per_area],
                'Total_Input_per_Area': [total_input_per_area]
            })
            
            # Make prediction
            prediction = self.catboost_model.predict(input_data)[0]
            
            # Calculate confidence interval (approximately ±RMSE)
            rmse = 167.92  # From training results
            lower_bound = max(0, prediction - rmse)
            upper_bound = prediction + rmse
            
            result = {
                "crop_type": crop,
                "season": season,
                "state": state,
                "crop_year": crop_year,
                "prediction_method": "CatBoost ML Model",
                "predicted_yield": round(prediction, 2),
                "predicted_yield_range": {
                    "lower": round(lower_bound, 2),
                    "upper": round(upper_bound, 2)
                },
                "confidence": "96.48%",
                "r_squared": 0.9648,
                "area_hectares": area,
                "inputs": {
                    "fertilizer_kg": fertilizer,
                    "pesticide_kg": pesticide,
                    "rainfall_mm": annual_rainfall
                },
                "recommendations": self._generate_catboost_recommendations(
                    prediction, fertilizer_per_area, pesticide_per_area, rainfall_per_area
                ),
                "model_version": "CatBoost v1.0 (2026-02-01)",
                "model_path": self.model_path
            }
            
            logger.info(f"✅ CatBoost prediction: {prediction:.2f} kg/ha for {crop}")
            return result
            
        except Exception as e:
            logger.error(f"❌ CatBoost prediction error: {str(e)}")
            return self._fallback_advanced_prediction(crop, area, crop_year)
    
    def _generate_catboost_recommendations(self, predicted_yield, fert_per_area, 
                                           pest_per_area, rain_per_area) -> List[str]:
        """Generate recommendations based on CatBoost prediction"""
        recommendations = []
        
        # Yield-based recommendations
        if predicted_yield > 3000:
            recommendations.append("✅ Excellent predicted yield. Maintain current practices.")
        elif predicted_yield > 2000:
            recommendations.append("✅ Good predicted yield. Consider optimization for better results.")
        elif predicted_yield > 1000:
            recommendations.append("⚠️ Moderate yield predicted. Review fertilizer and pest management.")
        else:
            recommendations.append("🔴 Low yield predicted. Urgent intervention needed.")
        
        # Fertilizer recommendations
        if fert_per_area < 50:
            recommendations.append("🔴 Low fertilizer usage. Consider increasing fertilizer application.")
        elif fert_per_area > 200:
            recommendations.append("⚠️ High fertilizer usage. Monitor for over-fertilization.")
        
        # Pesticide recommendations
        if pest_per_area < 0.5:
            recommendations.append("⚠️ Low pesticide usage. Ensure adequate pest protection.")
        elif pest_per_area > 5:
            recommendations.append("⚠️ High pesticide usage. Consider integrated pest management.")
        
        # Rainfall recommendations
        if rain_per_area < 10:
            recommendations.append("💧 Low rainfall. Ensure adequate irrigation.")
        elif rain_per_area > 50:
            recommendations.append("💧 High rainfall. Monitor for waterlogging and drainage.")
        
        recommendations.append("📊 Based on CatBoost ML model trained on 19,689 records (1997-2020)")
        
        return recommendations
    
    def _fallback_advanced_prediction(self, crop: str, area: float, year: int) -> Dict:
        """Fallback when CatBoost is not available"""
        baseline_yields = {
            "rice": 3500,
            "wheat": 3000,
            "maize": 2800,
            "corn": 2800,
            "cotton": 1800,
            "soybean": 2500,
            "sugarcane": 70000,
            "default": 2500
        }
        
        baseline = baseline_yields.get(crop.lower(), baseline_yields["default"])
        predicted_yield = baseline * 1.02  # Slight growth assumption
        
        return {
            "crop_type": crop,
            "prediction_method": "Rule-Based Baseline",
            "predicted_yield": round(predicted_yield, 2),
            "predicted_yield_range": {
                "lower": round(predicted_yield * 0.85, 2),
                "upper": round(predicted_yield * 1.15, 2)
            },
            "confidence": "Estimated (baseline)",
            "note": "CatBoost model not available. Install with: pip install catboost and place model at src/models/catboost_crop_yield_clean.cbm",
            "recommendations": [
                "⚠️ Using baseline estimates. For accurate predictions, install CatBoost model.",
                "📥 Place model file at: src/models/catboost_crop_yield_clean.cbm",
                "📦 Install CatBoost: pip install catboost"
            ]
        }
    
    # ========================================================================
    # LEGACY METHOD: PROPHET PREDICTION (Historical trend analysis)
    # ========================================================================
    
    def predict(self, 
                historical_yields: List[Dict],
                crop_type: str,
                forecast_days: int = 120) -> Dict:
        """
        Simple yield prediction based on historical data (Prophet-based)
        
        This is kept for backward compatibility and simple trend analysis.
        For accurate predictions, use predict_advanced() instead.
        
        Args:
            historical_yields: List of dicts with 'date' and 'yield' keys
            crop_type: Type of crop
            forecast_days: Number of days to forecast
        
        Returns:
            Dictionary with prediction results
        """
        try:
            if PROPHET_AVAILABLE and historical_yields and len(historical_yields) >= 5:
                return self._predict_with_prophet(historical_yields, crop_type, forecast_days)
            else:
                return self._predict_with_rules(historical_yields, crop_type, forecast_days)
                
        except Exception as e:
            logger.error(f"❌ Yield prediction error: {str(e)}")
            raise
    
    def _predict_with_prophet(self, historical_yields, crop_type, forecast_days) -> Dict:
        """Predict using Prophet time-series model"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_yields)
            
            # Ensure proper column names for Prophet
            if 'date' in df.columns and 'yield' in df.columns:
                df = df.rename(columns={'date': 'ds', 'yield': 'y'})
            
            # Convert date to datetime
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Sort by date
            df = df.sort_values('ds')
            
            # Require at least 5 data points
            if len(df) < 5:
                logger.warning("⚠️ Less than 5 data points. Using rule-based prediction.")
                return self._predict_with_rules(historical_yields, crop_type, forecast_days)
            
            logger.info(f"📊 Training Prophet model with {len(df)} historical data points")
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days, freq='D')
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract relevant predictions
            last_historical_date = df['ds'].max()
            future_predictions = forecast[forecast['ds'] > last_historical_date]
            
            # Calculate statistics
            avg_yield = df['y'].mean()
            predicted_yield = future_predictions['yhat'].mean()
            growth_rate = ((predicted_yield - avg_yield) / avg_yield) * 100 if avg_yield > 0 else 0
            
            result = {
                "crop_type": crop_type,
                "prediction_method": "Prophet Time-Series",
                "historical_data_points": len(df),
                "forecast_period_days": forecast_days,
                "average_historical_yield": round(avg_yield, 2),
                "predicted_average_yield": round(predicted_yield, 2),
                "predicted_yield_range": {
                    "lower": round(future_predictions['yhat_lower'].mean(), 2),
                    "upper": round(future_predictions['yhat_upper'].mean(), 2)
                },
                "growth_rate_percent": round(growth_rate, 2),
                "confidence_interval": "95%",
                "trend": "increasing" if growth_rate > 0 else "decreasing",
                "recommendations": self._generate_yield_recommendations(growth_rate, crop_type),
                "note": "For more accurate predictions, use predict_advanced() with detailed farm data"
            }
            
            logger.info(f"✅ Prophet prediction: {predicted_yield:.2f} kg/ha")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prophet prediction failed: {str(e)}")
            return self._predict_with_rules(historical_yields, crop_type, forecast_days)
    
    def _predict_with_rules(self, historical_yields, crop_type, forecast_days) -> Dict:
        """Rule-based yield prediction (fallback)"""
        logger.info("Using rule-based yield prediction")
        
        # Calculate average from historical data
        if historical_yields:
            yields = [item['yield'] for item in historical_yields if 'yield' in item]
            avg_yield = sum(yields) / len(yields) if yields else 0
            
            # Apply growth factor based on crop type
            growth_factors = {
                "wheat": 1.02,
                "rice": 1.03,
                "corn": 1.04,
                "maize": 1.04,
                "cotton": 1.02,
                "soybean": 1.03,
                "sugarcane": 1.01
            }
            
            growth_factor = growth_factors.get(crop_type.lower(), 1.02)
            predicted_yield = avg_yield * growth_factor
        else:
            # Use crop-specific baseline yields (kg/ha)
            baseline_yields = {
                "wheat": 3000,
                "rice": 4000,
                "corn": 5000,
                "maize": 5000,
                "cotton": 2000,
                "soybean": 2500,
                "sugarcane": 70000,
                "default": 3000
            }
            avg_yield = baseline_yields.get(crop_type.lower(), baseline_yields["default"])
            predicted_yield = avg_yield * 1.02
        
        growth_rate = ((predicted_yield - avg_yield) / avg_yield) * 100 if avg_yield > 0 else 0
        
        result = {
            "crop_type": crop_type,
            "prediction_method": "Rule-Based Estimation",
            "historical_data_points": len(historical_yields) if historical_yields else 0,
            "forecast_period_days": forecast_days,
            "average_historical_yield": round(avg_yield, 2) if historical_yields else "N/A",
            "predicted_average_yield": round(predicted_yield, 2),
            "predicted_yield_range": {
                "lower": round(predicted_yield * 0.85, 2),
                "upper": round(predicted_yield * 1.15, 2)
            },
            "growth_rate_percent": round(growth_rate, 2),
            "confidence_interval": "Estimated",
            "trend": "increasing" if growth_rate > 0 else "stable",
            "recommendations": self._generate_yield_recommendations(growth_rate, crop_type),
            "note": "Using baseline estimates. For accurate predictions, use predict_advanced() or provide historical data."
        }
        
        logger.info(f"✅ Rule-based prediction: {predicted_yield:.2f} kg/ha")
        return result
    
    def _generate_yield_recommendations(self, growth_rate, crop_type) -> List[str]:
        """Generate yield improvement recommendations"""
        recommendations = []
        
        if growth_rate < 0:
            recommendations.append("🔴 Declining yield trend detected. Review soil health and pest management.")
            recommendations.append("Consider crop rotation or fallow period to restore soil fertility.")
        elif growth_rate < 2:
            recommendations.append("⚠️ Slow growth trend. Optimize fertilizer application and irrigation.")
        else:
            recommendations.append("✅ Positive growth trend. Maintain current practices.")
        
        # Crop-specific recommendations
        crop_tips = {
            "wheat": [
                "Ensure adequate nitrogen during tillering stage",
                "Monitor for rust diseases"
            ],
            "rice": [
                "Maintain consistent water levels during grain filling",
                "Apply potassium for better grain quality"
            ],
            "corn": [
                "Side-dress nitrogen at V6 stage",
                "Control weeds in early growth stages"
            ],
            "maize": [
                "Side-dress nitrogen at V6 stage",
                "Control weeds in early growth stages"
            ],
            "cotton": [
                "Monitor boll weevil activity",
                "Ensure adequate potassium for fiber quality"
            ]
        }
        
        if crop_type.lower() in crop_tips:
            recommendations.extend(crop_tips[crop_type.lower()])
        
        recommendations.append("📊 For high-accuracy predictions, use predict_advanced() with CatBoost model")
        
        return recommendations
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_model_info(self) -> Dict:
        """Get information about available models"""
        return {
            "catboost_available": CATBOOST_AVAILABLE and self.catboost_model is not None,
            "catboost_accuracy": "96.48% (R² score)" if (CATBOOST_AVAILABLE and self.catboost_model is not None) else "Not loaded",
            "prophet_available": PROPHET_AVAILABLE,
            "model_path": self.model_path if self.model_path else "Not found",
            "model_location": "src/models/catboost_crop_yield_clean.cbm",
            "recommended_method": "predict_advanced() for CatBoost (best accuracy)",
            "legacy_method": "predict() for Prophet/rule-based (simpler, less accurate)",
            "features_required": {
                "catboost": ["crop", "season", "state", "year", "area", "rainfall", "fertilizer", "pesticide"],
                "prophet": ["historical_yields (date, yield pairs)"]
            }
        }


# Singleton instance
_yield_predictor_instance = None


def get_yield_predictor() -> YieldPredictor:
    """Get singleton instance of YieldPredictor"""
    global _yield_predictor_instance
    if _yield_predictor_instance is None:
        _yield_predictor_instance = YieldPredictor()
    return _yield_predictor_instance


if __name__ == "__main__":
    # Test the predictor
    print("Testing Yield Predictor...")
    
    predictor = YieldPredictor()
    
    print(f"\n📊 Model Info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test CatBoost prediction (if available)
    if info["catboost_available"]:
        print("\n🌾 Test Case 1: CatBoost Advanced Prediction")
        result1 = predictor.predict_advanced(
            crop='Rice',
            season='Kharif',
            state='Punjab',
            crop_year=2024,
            area=50000,
            annual_rainfall=1200,
            fertilizer=5000000,
            pesticide=15000
        )
        print(f"Predicted Yield: {result1['predicted_yield']} kg/ha")
        print(f"Method: {result1['prediction_method']}")
        print(f"Confidence: {result1['confidence']}")
    else:
        print("\n⚠️ CatBoost model not available")
        print(f"   Expected location: {info['model_location']}")
    
    # Test Prophet prediction
    print("\n📈 Test Case 2: Prophet Historical Prediction")
    historical_data = [
        {'date': '2020-01-01', 'yield': 2800},
        {'date': '2021-01-01', 'yield': 2900},
        {'date': '2022-01-01', 'yield': 3000},
        {'date': '2023-01-01', 'yield': 3100},
        {'date': '2024-01-01', 'yield': 3200},
    ]
    
    result2 = predictor.predict(historical_data, "wheat", forecast_days=120)
    print(f"Predicted Yield: {result2['predicted_average_yield']} kg/ha")
    print(f"Method: {result2['prediction_method']}")
    
    print("\n✅ Yield Predictor test completed!")