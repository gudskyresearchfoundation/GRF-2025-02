"""
Model 6: Pest Risk Assessor
Rule-based pest risk prediction
NO TRAINING NEEDED - Uses weather patterns and entomology rules
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PestRiskAssessor:
    """Rule-based pest risk assessment system"""
    
    def __init__(self):
        logger.info("🐛 Initializing Pest Risk Assessor")
        
        # Pest risk factors by crop
        self.pest_profiles = {
            "rice": ["Brown planthopper", "Rice stem borer", "Blast disease"],
            "wheat": ["Aphids", "Rust diseases", "Army worm"],
            "corn": ["Corn borer", "Fall armyworm", "Cutworm"],
            "cotton": ["Bollworm", "Whitefly", "Aphids"],
            "tomato": ["Whitefly", "Aphids", "Fruit borer"],
            "potato": ["Colorado beetle", "Late blight", "Aphids"]
        }
    
    def assess_risk(self,
                   crop_type: str,
                   temperature: float,
                   humidity: float,
                   rainfall_7days: float,
                   previous_infestation: bool = False) -> Dict:
        """
        Assess pest risk based on environmental conditions
        
        Args:
            crop_type: Type of crop
            temperature: Average temperature (°C)
            humidity: Average humidity (%)
            rainfall_7days: Total rainfall in last 7 days (mm)
            previous_infestation: Whether pests were detected before
        
        Returns:
            Risk assessment and recommendations
        """
        try:
            risk_score = 0
            factors = []
            
            # Temperature-based risk
            if 25 <= temperature <= 32:
                risk_score += 30
                factors.append(f"Optimal temperature for pest activity ({temperature}°C)")
            elif temperature > 35:
                risk_score += 10
                factors.append(f"High temperature may reduce some pest activity")
            
            # Humidity-based risk
            if humidity > 80:
                risk_score += 35
                factors.append(f"High humidity ({humidity}%) favors fungal diseases and pests")
            elif humidity > 60:
                risk_score += 20
                factors.append(f"Moderate humidity supports pest development")
            
            # Rainfall-based risk
            if rainfall_7days > 50:
                risk_score += 25
                factors.append(f"Heavy rainfall ({rainfall_7days}mm) increases disease risk")
            elif rainfall_7days == 0:
                risk_score += 10
                factors.append("Dry conditions may stress plants, attracting pests")
            
            # Previous infestation
            if previous_infestation:
                risk_score += 20
                factors.append("Previous pest infestation detected")
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = "High"
                color = "🔴"
            elif risk_score >= 40:
                risk_level = "Medium"
                color = "🟡"
            else:
                risk_level = "Low"
                color = "🟢"
            
            # Get likely pests for crop
            likely_pests = self.pest_profiles.get(crop_type.lower(), ["Common agricultural pests"])
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_level, crop_type, likely_pests)
            
            result = {
                "crop_type": crop_type,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "risk_indicator": color,
                "contributing_factors": factors,
                "likely_pests": likely_pests,
                "preventive_measures": recommendations,
                "monitoring_frequency": "Daily" if risk_level == "High" else "Every 3 days"
            }
            
            logger.info(f"✅ Pest risk assessed: {risk_level} ({risk_score}/100)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pest risk assessment error: {str(e)}")
            raise
    
    def _generate_recommendations(self, risk_level, crop_type, likely_pests):
        """Generate pest management recommendations"""
        recommendations = []
        
        if risk_level == "High":
            recommendations.append("🔴 HIGH RISK: Inspect crops daily for early pest detection")
            recommendations.append("Consider prophylactic treatment with approved pesticides")
            recommendations.append("Use pheromone traps to monitor pest population")
            recommendations.append("Remove affected plant parts immediately")
        elif risk_level == "Medium":
            recommendations.append("🟡 MODERATE RISK: Monitor crops every 2-3 days")
            recommendations.append("Keep biological pesticides ready")
            recommendations.append("Maintain field hygiene and remove weeds")
        else:
            recommendations.append("🟢 LOW RISK: Continue regular monitoring")
            recommendations.append("Focus on preventive measures")
        
        # Crop-specific recommendations
        recommendations.append(f"For {crop_type}: Scout for {', '.join(likely_pests[:2])}")
        recommendations.append("Use integrated pest management (IPM) practices")
        
        return recommendations


def get_pest_risk_assessor():
    global _pest_risk_assessor_instance
    if _pest_risk_assessor_instance is None:
        _pest_risk_assessor_instance = PestRiskAssessor()
    return _pest_risk_assessor_instance


_pest_risk_assessor_instance = None
