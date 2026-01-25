"""
Model 5: Irrigation Optimizer
Rule-based irrigation scheduling
NO TRAINING NEEDED - Uses FAO guidelines
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class IrrigationOptimizer:
    """Rule-based irrigation scheduling system"""
    
    def __init__(self):
        logger.info("💧 Initializing Irrigation Optimizer")
        
        # Crop water requirements (mm/day during peak season)
        self.crop_water_needs = {
            "rice": 7.5,
            "wheat": 5.0,
            "corn": 6.0,
            "cotton": 6.5,
            "tomato": 5.5,
            "potato": 5.0,
            "sugarcane": 8.0,
            "soybean": 5.5
        }
    
    def calculate_irrigation(self, 
                           crop_type: str,
                           soil_moisture: float,
                           growth_stage: str,
                           rainfall_forecast_mm: float = 0,
                           temperature_celsius: float = 25) -> Dict:
        """
        Calculate irrigation requirements
        
        Args:
            crop_type: Type of crop
            soil_moisture: Current soil moisture (0-1)
            growth_stage: Growth stage (vegetative, flowering, maturity)
            rainfall_forecast_mm: Expected rainfall in next 7 days
            temperature_celsius: Average temperature
        
        Returns:
            Irrigation schedule and recommendations
        """
        try:
            # Get base water requirement
            base_need = self.crop_water_needs.get(crop_type.lower(), 5.5)
            
            # Adjust for growth stage
            stage_factors = {
                "germination": 0.7,
                "vegetative": 1.0,
                "flowering": 1.3,
                "fruiting": 1.2,
                "maturity": 0.6
            }
            stage_factor = stage_factors.get(growth_stage.lower(), 1.0)
            adjusted_need = base_need * stage_factor
            
            # Adjust for temperature
            if temperature > 30:
                adjusted_need *= 1.2
            elif temperature < 20:
                adjusted_need *= 0.9
            
            # Adjust for soil moisture
            if soil_moisture < 0.3:
                irrigation_amount = adjusted_need * 1.5
            elif soil_moisture < 0.5:
                irrigation_amount = adjusted_need
            else:
                irrigation_amount = adjusted_need * 0.5
            
            # Account for rainfall
            effective_irrigation = max(0, irrigation_amount - (rainfall_forecast_mm / 7))
            
            # Calculate schedule
            if effective_irrigation > 0:
                next_irrigation_days = max(1, int(3 - (soil_moisture * 4)))
            else:
                next_irrigation_days = 7
            
            result = {
                "crop_type": crop_type,
                "irrigation_amount_mm_per_day": round(effective_irrigation, 2),
                "next_irrigation_days": next_irrigation_days,
                "soil_moisture_percent": round(soil_moisture * 100, 1),
                "growth_stage": growth_stage,
                "recommendations": self._generate_recommendations(
                    soil_moisture, effective_irrigation, crop_type, growth_stage
                )
            }
            
            logger.info(f"✅ Irrigation calculated: {effective_irrigation:.2f} mm/day")
            return result
            
        except Exception as e:
            logger.error(f"❌ Irrigation calculation error: {str(e)}")
            raise
    
    def _generate_recommendations(self, soil_moisture, irrigation_amount, crop_type, stage):
        """Generate irrigation recommendations"""
        recommendations = []
        
        if soil_moisture < 0.3:
            recommendations.append("🔴 URGENT: Low soil moisture. Irrigate immediately.")
        elif soil_moisture < 0.5:
            recommendations.append("⚠️ Moderate moisture. Schedule irrigation within 2 days.")
        else:
            recommendations.append("✅ Adequate soil moisture. No immediate irrigation needed.")
        
        if stage.lower() in ["flowering", "fruiting"]:
            recommendations.append(f"⚠️ Critical stage: Maintain consistent moisture for {crop_type}.")
        
        if irrigation_amount > 10:
            recommendations.append("💧 High water requirement. Consider drip irrigation for efficiency.")
        
        recommendations.append("Use mulching to reduce water evaporation.")
        
        return recommendations


def get_irrigation_optimizer():
    global _irrigation_optimizer_instance
    if _irrigation_optimizer_instance is None:
        _irrigation_optimizer_instance = IrrigationOptimizer()
    return _irrigation_optimizer_instance


_irrigation_optimizer_instance = None
