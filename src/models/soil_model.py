"""
Model 3: Soil Health Analyzer
Rule-based soil analysis with NPK recommendations
NO TRAINING NEEDED - Uses agricultural science guidelines
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class SoilAnalyzer:
    """
    Rule-based soil health analyzer
    Analyzes soil parameters and provides recommendations
    """
    
    def __init__(self):
        """Initialize Soil Analyzer"""
        logger.info("🌱 Initializing Soil Analyzer")
        
        # Optimal ranges for soil parameters
        self.optimal_ranges = {
            "ph": {"min": 6.0, "max": 7.5, "ideal": 6.5},
            "nitrogen": {"min": 40, "max": 100, "ideal": 70},  # kg/ha
            "phosphorus": {"min": 15, "max": 40, "ideal": 25},  # kg/ha
            "potassium": {"min": 100, "max": 250, "ideal": 150},  # kg/ha
            "organic_carbon": {"min": 0.5, "max": 2.0, "ideal": 1.0},  # %
            "moisture": {"min": 0.3, "max": 0.7, "ideal": 0.5}  # fraction
        }
    
    def analyze(self, 
                ph: float,
                nitrogen: float,
                phosphorus: float,
                potassium: float,
                organic_carbon: float = None,
                moisture: float = None) -> Dict:
        """
        Analyze soil health and generate recommendations
        
        Args:
            ph: Soil pH (4-9 range)
            nitrogen: Nitrogen content (kg/ha)
            phosphorus: Phosphorus content (kg/ha)
            potassium: Potassium content (kg/ha)
            organic_carbon: Organic carbon percentage (optional)
            moisture: Soil moisture fraction 0-1 (optional)
        
        Returns:
            Dictionary with analysis results and recommendations
        """
        try:
            logger.info(f"🧪 Analyzing soil: pH={ph}, N={nitrogen}, P={phosphorus}, K={potassium}")
            
            # Calculate individual scores
            ph_score = self._calculate_ph_score(ph)
            nitrogen_score = self._calculate_nutrient_score(nitrogen, "nitrogen")
            phosphorus_score = self._calculate_nutrient_score(phosphorus, "phosphorus")
            potassium_score = self._calculate_nutrient_score(potassium, "potassium")
            
            # Calculate overall health score
            health_score = (ph_score + nitrogen_score + phosphorus_score + potassium_score) / 4
            
            # Add optional parameters
            if organic_carbon is not None:
                oc_score = self._calculate_nutrient_score(organic_carbon, "organic_carbon")
                health_score = (health_score * 4 + oc_score) / 5
            
            if moisture is not None:
                moisture_score = self._calculate_nutrient_score(moisture, "moisture")
                health_score = (health_score * 5 + moisture_score) / 6
            
            # Determine health status
            if health_score >= 80:
                health_status = "Excellent"
            elif health_score >= 60:
                health_status = "Good"
            elif health_score >= 40:
                health_status = "Fair"
            else:
                health_status = "Poor"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                ph, nitrogen, phosphorus, potassium, organic_carbon, moisture
            )
            
            # Generate NPK fertilizer recommendation
            npk_recommendation = self._calculate_npk_requirement(nitrogen, phosphorus, potassium)
            
            result = {
                "health_score": round(health_score, 2),
                "health_status": health_status,
                "parameter_scores": {
                    "ph": round(ph_score, 2),
                    "nitrogen": round(nitrogen_score, 2),
                    "phosphorus": round(phosphorus_score, 2),
                    "potassium": round(potassium_score, 2)
                },
                "recommendations": recommendations,
                "npk_fertilizer": npk_recommendation,
                "summary": self._generate_summary(health_status, recommendations)
            }
            
            logger.info(f"✅ Soil analysis complete: {health_status} (Score: {health_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Soil analysis error: {str(e)}")
            raise
    
    def _calculate_ph_score(self, ph: float) -> float:
        """Calculate pH score (0-100)"""
        optimal = self.optimal_ranges["ph"]
        
        if optimal["min"] <= ph <= optimal["max"]:
            # Within optimal range
            if ph == optimal["ideal"]:
                return 100
            # Distance from ideal
            distance = abs(ph - optimal["ideal"])
            max_distance = max(abs(optimal["max"] - optimal["ideal"]), 
                             abs(optimal["min"] - optimal["ideal"]))
            return 100 - (distance / max_distance) * 20
        else:
            # Outside optimal range
            if ph < optimal["min"]:
                # Too acidic
                distance = optimal["min"] - ph
                return max(0, 60 - distance * 15)
            else:
                # Too alkaline
                distance = ph - optimal["max"]
                return max(0, 60 - distance * 15)
    
    def _calculate_nutrient_score(self, value: float, nutrient: str) -> float:
        """Calculate nutrient score (0-100)"""
        optimal = self.optimal_ranges[nutrient]
        
        if optimal["min"] <= value <= optimal["max"]:
            # Within optimal range
            if value == optimal["ideal"]:
                return 100
            # Distance from ideal
            distance = abs(value - optimal["ideal"])
            max_distance = max(abs(optimal["max"] - optimal["ideal"]), 
                             abs(optimal["min"] - optimal["ideal"]))
            return 100 - (distance / max_distance) * 20
        elif value < optimal["min"]:
            # Deficiency
            ratio = value / optimal["min"]
            return max(0, ratio * 60)
        else:
            # Excess
            excess = (value - optimal["max"]) / optimal["max"]
            return max(0, 60 - excess * 40)
    
    def _generate_recommendations(self, ph, nitrogen, phosphorus, potassium, 
                                   organic_carbon, moisture) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # pH recommendations
        if ph < 6.0:
            recommendations.append(f"🔴 Soil is acidic (pH {ph:.1f}). Apply lime at 2-3 tons/hectare to raise pH.")
        elif ph > 7.5:
            recommendations.append(f"🔴 Soil is alkaline (pH {ph:.1f}). Add sulfur or organic matter to lower pH.")
        else:
            recommendations.append(f"✅ Soil pH is optimal ({ph:.1f}).")
        
        # Nitrogen recommendations
        if nitrogen < 40:
            recommendations.append(f"🔴 Nitrogen deficiency ({nitrogen:.1f} kg/ha). Apply urea or compost.")
        elif nitrogen > 100:
            recommendations.append(f"⚠️ Excess nitrogen ({nitrogen:.1f} kg/ha). Reduce fertilizer application.")
        else:
            recommendations.append(f"✅ Nitrogen levels are adequate ({nitrogen:.1f} kg/ha).")
        
        # Phosphorus recommendations
        if phosphorus < 15:
            recommendations.append(f"🔴 Phosphorus deficiency ({phosphorus:.1f} kg/ha). Apply DAP or rock phosphate.")
        elif phosphorus > 40:
            recommendations.append(f"⚠️ Excess phosphorus ({phosphorus:.1f} kg/ha). May cause nutrient imbalance.")
        else:
            recommendations.append(f"✅ Phosphorus levels are good ({phosphorus:.1f} kg/ha).")
        
        # Potassium recommendations
        if potassium < 100:
            recommendations.append(f"🔴 Potassium deficiency ({potassium:.1f} kg/ha). Apply MOP (Muriate of Potash).")
        elif potassium > 250:
            recommendations.append(f"⚠️ Excess potassium ({potassium:.1f} kg/ha). Reduce potash application.")
        else:
            recommendations.append(f"✅ Potassium levels are sufficient ({potassium:.1f} kg/ha).")
        
        # Organic carbon
        if organic_carbon is not None:
            if organic_carbon < 0.5:
                recommendations.append(f"🔴 Low organic matter ({organic_carbon:.2f}%). Add compost or green manure.")
            else:
                recommendations.append(f"✅ Organic matter is adequate ({organic_carbon:.2f}%).")
        
        # Moisture
        if moisture is not None:
            if moisture < 0.3:
                recommendations.append(f"💧 Low soil moisture ({moisture*100:.0f}%). Increase irrigation frequency.")
            elif moisture > 0.7:
                recommendations.append(f"💧 High soil moisture ({moisture*100:.0f}%). Improve drainage.")
            else:
                recommendations.append(f"✅ Soil moisture is optimal ({moisture*100:.0f}%).")
        
        return recommendations
    
    def _calculate_npk_requirement(self, nitrogen, phosphorus, potassium) -> Dict:
        """Calculate NPK fertilizer requirement"""
        optimal = self.optimal_ranges
        
        # Calculate deficiency (if any)
        n_deficit = max(0, optimal["nitrogen"]["ideal"] - nitrogen)
        p_deficit = max(0, optimal["phosphorus"]["ideal"] - phosphorus)
        k_deficit = max(0, optimal["potassium"]["ideal"] - potassium)
        
        # Conversion factors (approximate)
        # Urea = 46% N, DAP = 18% N + 46% P2O5, MOP = 60% K2O
        urea_kg = (n_deficit / 0.46) if n_deficit > 0 else 0
        dap_kg = (p_deficit / 0.46) if p_deficit > 0 else 0
        mop_kg = (k_deficit / 0.60) if k_deficit > 0 else 0
        
        return {
            "nitrogen_deficit_kg_ha": round(n_deficit, 2),
            "phosphorus_deficit_kg_ha": round(p_deficit, 2),
            "potassium_deficit_kg_ha": round(k_deficit, 2),
            "recommended_fertilizers": {
                "urea_kg_ha": round(urea_kg, 2),
                "dap_kg_ha": round(dap_kg, 2),
                "mop_kg_ha": round(mop_kg, 2)
            },
            "application_timing": "Split application: 50% at sowing, 50% at 30 days after sowing"
        }
    
    def _generate_summary(self, health_status, recommendations) -> str:
        """Generate concise summary"""
        critical_issues = [r for r in recommendations if r.startswith("🔴")]
        
        if health_status in ["Excellent", "Good"]:
            return f"Soil health is {health_status.lower()}. Continue current management practices."
        elif critical_issues:
            return f"Soil health is {health_status.lower()}. Address {len(critical_issues)} critical issue(s) immediately."
        else:
            return f"Soil health is {health_status.lower()}. Minor adjustments recommended."


# Singleton instance
_soil_analyzer_instance = None


def get_soil_analyzer() -> SoilAnalyzer:
    """Get singleton instance of SoilAnalyzer"""
    global _soil_analyzer_instance
    if _soil_analyzer_instance is None:
        _soil_analyzer_instance = SoilAnalyzer()
    return _soil_analyzer_instance


if __name__ == "__main__":
    # Test the analyzer
    print("Testing Soil Analyzer...")
    
    analyzer = SoilAnalyzer()
    
    # Test case 1: Good soil
    print("\n📊 Test Case 1: Good Soil")
    result1 = analyzer.analyze(
        ph=6.5,
        nitrogen=70,
        phosphorus=25,
        potassium=150,
        organic_carbon=1.0,
        moisture=0.5
    )
    print(f"Health Status: {result1['health_status']}")
    print(f"Health Score: {result1['health_score']}")
    print(f"Recommendations: {len(result1['recommendations'])} items")
    
    # Test case 2: Poor soil
    print("\n📊 Test Case 2: Poor Soil")
    result2 = analyzer.analyze(
        ph=5.0,
        nitrogen=20,
        phosphorus=8,
        potassium=50
    )
    print(f"Health Status: {result2['health_status']}")
    print(f"Health Score: {result2['health_score']}")
    print(f"NPK Requirement: {result2['npk_fertilizer']}")
    
    print("\n✅ Soil Analyzer test completed!")
