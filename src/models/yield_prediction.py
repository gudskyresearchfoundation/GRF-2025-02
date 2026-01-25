"""
Model 4: Yield Prediction
Uses Prophet for time-series forecasting
NO TRAINING NEEDED - Prophet auto-fits on historical data
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Prophet import with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("⚠️ Prophet not installed. Install with: pip install prophet")


class YieldPredictor:
    """
    Crop yield prediction using time-series forecasting
    Uses Prophet for trend analysis and prediction
    """
    
    def __init__(self):
        """Initialize Yield Predictor"""
        logger.info("📈 Initializing Yield Predictor")
        self.model = None
        
        if not PROPHET_AVAILABLE:
            logger.warning("⚠️ Prophet not available - using rule-based fallback")
    
    def predict(self, 
                historical_yields: List[Dict],
                crop_type: str,
                forecast_days: int = 120) -> Dict:
        """
        Predict crop yield based on historical data
        
        Args:
            historical_yields: List of dicts with 'date' and 'yield' keys
                              Example: [{'date': '2023-01-01', 'yield': 2500}, ...]
            crop_type: Type of crop (wheat, rice, corn, etc.)
            forecast_days: Number of days to forecast
        
        Returns:
            Dictionary with yield predictions
        """
        try:
            if PROPHET_AVAILABLE and historical_yields:
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
            
            logger.info(f"📊 Training Prophet model with {len(df)} historical data points")
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05  # Flexibility of trend
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
            growth_rate = ((predicted_yield - avg_yield) / avg_yield) * 100
            
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
                "recommendations": self._generate_yield_recommendations(growth_rate, crop_type)
            }
            
            logger.info(f"✅ Yield prediction complete: {predicted_yield:.2f} kg/ha")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prophet prediction failed: {str(e)}")
            # Fallback to rule-based
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
                "wheat": 1.02,  # 2% annual growth
                "rice": 1.03,
                "corn": 1.04,
                "cotton": 1.02,
                "soybean": 1.03
            }
            
            growth_factor = growth_factors.get(crop_type.lower(), 1.02)
            predicted_yield = avg_yield * growth_factor
        else:
            # Use crop-specific baseline yields (kg/ha)
            baseline_yields = {
                "wheat": 3000,
                "rice": 4000,
                "corn": 5000,
                "cotton": 2000,
                "soybean": 2500,
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
            "note": "Using baseline estimates. Provide historical data for accurate predictions."
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
            "cotton": [
                "Monitor boll weevil activity",
                "Ensure adequate potassium for fiber quality"
            ]
        }
        
        if crop_type.lower() in crop_tips:
            recommendations.extend(crop_tips[crop_type.lower()])
        
        recommendations.append("Regular soil testing recommended for optimal nutrient management")
        
        return recommendations


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
    
    # Test with historical data
    print("\n📊 Test Case 1: With Historical Data")
    historical_data = [
        {'date': '2021-01-01', 'yield': 2800},
        {'date': '2021-06-01', 'yield': 2900},
        {'date': '2022-01-01', 'yield': 3000},
        {'date': '2022-06-01', 'yield': 3100},
        {'date': '2023-01-01', 'yield': 3200},
    ]
    
    result1 = predictor.predict(historical_data, "wheat", forecast_days=120)
    print(f"Predicted Yield: {result1['predicted_average_yield']} kg/ha")
    print(f"Growth Rate: {result1['growth_rate_percent']}%")
    
    # Test without historical data
    print("\n📊 Test Case 2: Without Historical Data")
    result2 = predictor.predict([], "rice", forecast_days=120)
    print(f"Predicted Yield: {result2['predicted_average_yield']} kg/ha")
    print(f"Method: {result2['prediction_method']}")
    
    print("\n✅ Yield Predictor test completed!")
