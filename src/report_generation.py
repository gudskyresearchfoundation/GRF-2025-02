"""
AI Report Generation
Uses FLAN-T5 from Hugging Face for natural language reports
NO TRAINING NEEDED - Pre-trained model
"""

import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ Transformers not installed. Install with: pip install transformers")


class ReportGenerator:
    """
    AI-powered agricultural report generator
    Uses FLAN-T5 for natural language generation
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize report generator
        
        Args:
            model_name: Hugging Face model name
                      Options: 
                      - google/flan-t5-base (250M params, faster)
                      - google/flan-t5-large (780M params, better quality)
        """
        logger.info("📄 Initializing AI Report Generator...")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("⚠️ Using fallback template-based reports")
    
    def _load_model(self):
        """Load FLAN-T5 model from Hugging Face"""
        try:
            logger.info(f"📥 Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Report Generator loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            logger.warning("⚠️ Falling back to template-based reports")
            self.model = None
    
    def generate_report(self, predictions: Dict) -> str:
        """
        Generate comprehensive agricultural report
        
        Args:
            predictions: Dictionary with all model predictions
                        {
                            'disease': {...},
                            'soil': {...},
                            'weather': {...},
                            'yield': {...},
                            'irrigation': {...},
                            'pest': {...}
                        }
        
        Returns:
            Natural language report
        """
        try:
            if self.model is not None and TRANSFORMERS_AVAILABLE:
                return self._generate_with_ai(predictions)
            else:
                return self._generate_with_template(predictions)
                
        except Exception as e:
            logger.error(f"❌ Report generation error: {str(e)}")
            return self._generate_fallback_report(predictions)
    
    def _generate_with_ai(self, predictions: Dict) -> str:
        """Generate report using FLAN-T5"""
        try:
            # Create comprehensive prompt
            prompt = self._create_prompt(predictions)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode
            report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process
            report = self._format_report(report, predictions)
            
            logger.info("✅ AI report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ AI generation failed: {str(e)}")
            return self._generate_with_template(predictions)
    
    def _create_prompt(self, predictions: Dict) -> str:
        """Create prompt for FLAN-T5"""
        
        # Extract key information
        disease_info = predictions.get('disease', {})
        soil_info = predictions.get('soil', {})
        weather_info = predictions.get('weather', {})
        yield_info = predictions.get('yield', {})
        irrigation_info = predictions.get('irrigation', {})
        pest_info = predictions.get('pest', {})
        
        prompt = f"""Generate a professional agricultural advisory report based on this farm analysis:

CROP HEALTH:
- Disease Detected: {disease_info.get('disease', 'Not analyzed')}
- Health Status: {disease_info.get('health_status', 'Unknown')}
- Confidence: {disease_info.get('confidence', 0)}%

SOIL ANALYSIS:
- Health Status: {soil_info.get('health_status', 'Not analyzed')}
- Health Score: {soil_info.get('health_score', 0)}/100
- Key Issues: {len(soil_info.get('recommendations', []))} recommendations provided

WEATHER CONDITIONS:
- Temperature: {weather_info.get('data', {}).get('current_weather', {}).get('temperature', 'N/A')}°C
- Alerts: {len(weather_info.get('alerts', []))} weather alerts

YIELD FORECAST:
- Predicted Yield: {yield_info.get('predicted_average_yield', 'Not analyzed')} kg/ha
- Trend: {yield_info.get('trend', 'Unknown')}

IRRIGATION:
- Water Needed: {irrigation_info.get('irrigation_amount_mm_per_day', 'Not analyzed')} mm/day
- Soil Moisture: {irrigation_info.get('soil_moisture_percent', 'N/A')}%

PEST RISK:
- Risk Level: {pest_info.get('risk_level', 'Not assessed')}
- Risk Score: {pest_info.get('risk_score', 0)}/100

Please provide:
1. Executive Summary
2. Key Findings
3. Critical Action Items
4. Recommendations by Priority
5. Risk Alerts

Format as a professional agricultural advisory report."""

        return prompt
    
    def _generate_with_template(self, predictions: Dict) -> str:
        """Generate report using templates (fallback)"""
        
        # Extract information
        disease = predictions.get('disease', {})
        soil = predictions.get('soil', {})
        weather = predictions.get('weather', {})
        yield_pred = predictions.get('yield', {})
        irrigation = predictions.get('irrigation', {})
        pest = predictions.get('pest', {})
        
        # Build report sections
        report_sections = []
        
        # Header
        report_sections.append("=" * 70)
        report_sections.append("🌱 PRECISION AGRICULTURE COMPREHENSIVE REPORT")
        report_sections.append("=" * 70)
        report_sections.append(f"Generated: {self.get_timestamp()}")
        report_sections.append("")
        
        # Executive Summary
        report_sections.append("📊 EXECUTIVE SUMMARY")
        report_sections.append("-" * 70)
        
        summary_points = []
        
        if disease:
            status = disease.get('health_status', 'Unknown')
            summary_points.append(f"• Crop Health: {status}")
            if status == "UNHEALTHY":
                summary_points.append(f"  ⚠️ Disease Detected: {disease.get('disease', 'Unknown')}")
        
        if soil:
            summary_points.append(f"• Soil Health: {soil.get('health_status', 'Unknown')} (Score: {soil.get('health_score', 0)}/100)")
        
        if yield_pred:
            summary_points.append(f"• Predicted Yield: {yield_pred.get('predicted_average_yield', 'N/A')} kg/ha")
        
        if pest:
            risk = pest.get('risk_level', 'Unknown')
            summary_points.append(f"• Pest Risk: {risk} ({pest.get('risk_score', 0)}/100)")
        
        report_sections.extend(summary_points)
        report_sections.append("")
        
        # Crop Health Analysis
        if disease:
            report_sections.append("🔬 CROP HEALTH ANALYSIS")
            report_sections.append("-" * 70)
            report_sections.append(f"Crop Type: {disease.get('crop', 'Unknown')}")
            report_sections.append(f"Disease: {disease.get('disease', 'Unknown')}")
            report_sections.append(f"Health Status: {disease.get('health_status', 'Unknown')}")
            report_sections.append(f"Confidence: {disease.get('confidence', 0)}%")
            
            if disease.get('warning'):
                report_sections.append(f"⚠️ Warning: {disease.get('warning')}")
            
            report_sections.append("")
        
        # Soil Analysis
        if soil:
            report_sections.append("🌱 SOIL HEALTH ANALYSIS")
            report_sections.append("-" * 70)
            report_sections.append(f"Overall Health: {soil.get('health_status', 'Unknown')}")
            report_sections.append(f"Health Score: {soil.get('health_score', 0)}/100")
            report_sections.append("")
            report_sections.append("Parameter Scores:")
            
            params = soil.get('parameter_scores', {})
            for param, score in params.items():
                report_sections.append(f"  • {param.title()}: {score}/100")
            
            report_sections.append("")
            report_sections.append("Recommendations:")
            for rec in soil.get('recommendations', [])[:5]:
                report_sections.append(f"  {rec}")
            
            report_sections.append("")
        
        # Weather Conditions
        if weather:
            report_sections.append("🌤️ WEATHER CONDITIONS & ALERTS")
            report_sections.append("-" * 70)
            
            current = weather.get('data', {}).get('current_weather', {})
            if current:
                report_sections.append(f"Current Temperature: {current.get('temperature', 'N/A')}°C")
                report_sections.append(f"Wind Speed: {current.get('windspeed', 'N/A')} km/h")
            
            alerts = weather.get('alerts', [])
            if alerts:
                report_sections.append("")
                report_sections.append("Agricultural Alerts:")
                for alert in alerts[:5]:
                    report_sections.append(f"  {alert}")
            
            report_sections.append("")
        
        # Yield Forecast
        if yield_pred:
            report_sections.append("📈 YIELD FORECAST")
            report_sections.append("-" * 70)
            report_sections.append(f"Crop: {yield_pred.get('crop_type', 'Unknown')}")
            report_sections.append(f"Predicted Yield: {yield_pred.get('predicted_average_yield', 'N/A')} kg/ha")
            report_sections.append(f"Growth Trend: {yield_pred.get('trend', 'Unknown')}")
            report_sections.append(f"Confidence Interval: {yield_pred.get('confidence_interval', 'N/A')}")
            
            yield_range = yield_pred.get('predicted_yield_range', {})
            if yield_range:
                report_sections.append(f"Expected Range: {yield_range.get('lower', 0):.2f} - {yield_range.get('upper', 0):.2f} kg/ha")
            
            report_sections.append("")
        
        # Irrigation Schedule
        if irrigation:
            report_sections.append("💧 IRRIGATION SCHEDULE")
            report_sections.append("-" * 70)
            report_sections.append(f"Crop: {irrigation.get('crop_type', 'Unknown')}")
            report_sections.append(f"Growth Stage: {irrigation.get('growth_stage', 'Unknown')}")
            report_sections.append(f"Current Soil Moisture: {irrigation.get('soil_moisture_percent', 'N/A')}%")
            report_sections.append(f"Required Irrigation: {irrigation.get('irrigation_amount_mm_per_day', 0)} mm/day")
            report_sections.append(f"Next Irrigation: {irrigation.get('next_irrigation_days', 'N/A')} days")
            report_sections.append("")
            report_sections.append("Recommendations:")
            for rec in irrigation.get('recommendations', [])[:3]:
                report_sections.append(f"  {rec}")
            
            report_sections.append("")
        
        # Pest Risk Assessment
        if pest:
            report_sections.append("🐛 PEST RISK ASSESSMENT")
            report_sections.append("-" * 70)
            report_sections.append(f"Risk Level: {pest.get('risk_indicator', '')} {pest.get('risk_level', 'Unknown')}")
            report_sections.append(f"Risk Score: {pest.get('risk_score', 0)}/100")
            report_sections.append(f"Monitoring Frequency: {pest.get('monitoring_frequency', 'Regular')}")
            report_sections.append("")
            
            likely_pests = pest.get('likely_pests', [])
            if likely_pests:
                report_sections.append("Likely Pests:")
                for p in likely_pests[:3]:
                    report_sections.append(f"  • {p}")
            
            report_sections.append("")
            report_sections.append("Preventive Measures:")
            for measure in pest.get('preventive_measures', [])[:4]:
                report_sections.append(f"  {measure}")
            
            report_sections.append("")
        
        # Critical Action Items
        report_sections.append("🎯 CRITICAL ACTION ITEMS")
        report_sections.append("-" * 70)
        
        action_items = self._extract_action_items(predictions)
        for priority, items in action_items.items():
            if items:
                report_sections.append(f"\n{priority} PRIORITY:")
                for item in items:
                    report_sections.append(f"  {item}")
        
        report_sections.append("")
        
        # Footer
        report_sections.append("=" * 70)
        report_sections.append("📝 Report generated by AI-Powered Precision Agriculture System")
        report_sections.append(f"🔬 Models used: Disease Detection, Soil Analysis, Weather, Yield Prediction, Irrigation, Pest Risk")
        report_sections.append("=" * 70)
        
        return "\n".join(report_sections)
    
    def _extract_action_items(self, predictions: Dict) -> Dict:
        """Extract prioritized action items"""
        
        high_priority = []
        medium_priority = []
        low_priority = []
        
        # Disease
        disease = predictions.get('disease', {})
        if disease.get('health_status') == 'UNHEALTHY':
            high_priority.append(f"🔴 Address {disease.get('disease', 'disease')} immediately")
        
        # Soil
        soil = predictions.get('soil', {})
        if soil.get('health_score', 100) < 40:
            high_priority.append("🔴 Critical soil health issues - implement soil remediation")
        elif soil.get('health_score', 100) < 60:
            medium_priority.append("🟡 Improve soil health with recommended fertilizers")
        
        # Pest Risk
        pest = predictions.get('pest', {})
        if pest.get('risk_level') == 'High':
            high_priority.append("🔴 High pest risk - implement preventive measures immediately")
        elif pest.get('risk_level') == 'Medium':
            medium_priority.append("🟡 Monitor crops regularly for pest activity")
        
        # Irrigation
        irrigation = predictions.get('irrigation', {})
        soil_moisture = irrigation.get('soil_moisture_percent', 100)
        if soil_moisture < 30:
            high_priority.append(f"🔴 Low soil moisture ({soil_moisture}%) - irrigate immediately")
        
        # Weather Alerts
        weather = predictions.get('weather', {})
        alerts = weather.get('alerts', [])
        critical_alerts = [a for a in alerts if '🔴' in a or 'warning' in a.lower()]
        if critical_alerts:
            high_priority.extend(critical_alerts[:2])
        
        return {
            "HIGH": high_priority,
            "MEDIUM": medium_priority,
            "LOW": low_priority
        }
    
    def _format_report(self, ai_text: str, predictions: Dict) -> str:
        """Format AI-generated text with additional structure"""
        
        # Add header
        formatted = [
            "=" * 70,
            "🌱 AI-GENERATED AGRICULTURAL ADVISORY REPORT",
            "=" * 70,
            f"Generated: {self.get_timestamp()}",
            "",
            ai_text,
            "",
            "=" * 70,
            "📝 This report was generated using AI (FLAN-T5)",
            "=" * 70
        ]
        
        return "\n".join(formatted)
    
    def _generate_fallback_report(self, predictions: Dict) -> str:
        """Minimal fallback report if all else fails"""
        return f"""
AGRICULTURAL REPORT
Generated: {self.get_timestamp()}

Analysis Summary:
- {len(predictions)} models analyzed
- Review individual model outputs for details

For detailed analysis, please check each model's output separately.
        """
    
    def get_timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Singleton instance
_report_generator_instance = None


def get_report_generator() -> ReportGenerator:
    """Get singleton instance of ReportGenerator"""
    global _report_generator_instance
    if _report_generator_instance is None:
        _report_generator_instance = ReportGenerator()
    return _report_generator_instance


if __name__ == "__main__":
    # Test report generation
    print("Testing Report Generator...")
    
    generator = ReportGenerator()
    
    # Sample predictions
    test_predictions = {
        'disease': {
            'crop': 'Tomato',
            'disease': 'Late_blight',
            'health_status': 'UNHEALTHY',
            'confidence': 92.5
        },
        'soil': {
            'health_status': 'Good',
            'health_score': 75,
            'recommendations': ['Apply nitrogen fertilizer', 'Maintain pH']
        },
        'pest': {
            'risk_level': 'Medium',
            'risk_score': 55
        }
    }
    
    report = generator.generate_report(test_predictions)
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print("\n✅ Report Generator test completed!")
