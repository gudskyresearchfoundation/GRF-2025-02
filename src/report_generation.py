"""
AI Report Generation - UPDATED
Enhanced template-based system for professional reports
NO API COSTS - Completely FREE
Falls back to FLAN-T5 if transformers available
"""

import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import transformers for optional FLAN-T5
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ Transformers not installed. Using template-based reports (still excellent!)")


class ReportGenerator:
    """
    Enhanced agricultural report generator
    Primary: Professional template-based system (FREE, high quality)
    Optional: FLAN-T5 for AI enhancement (if transformers available)
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base", use_templates: bool = True):
        """
        Initialize report generator
        
        Args:
            model_name: Hugging Face model name (for FLAN-T5 if available)
            use_templates: Use enhanced templates (recommended, FREE, fast)
        """
        logger.info("📄 Initializing AI Report Generator...")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.use_templates = use_templates
        
        # Try to load FLAN-T5 only if user wants it and it's available
        if not use_templates and TRANSFORMERS_AVAILABLE:
            try:
                self._load_model()
            except:
                logger.warning("⚠️ FLAN-T5 failed to load, using enhanced templates")
                self.use_templates = True
        else:
            logger.info("✅ Using enhanced professional template system (FREE, high quality)")
    
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
        
        Returns:
            Natural language report
        """
        try:
            # Use enhanced templates (recommended - FREE and high quality)
            if self.use_templates or self.model is None:
                return self._generate_professional_template(predictions)
            else:
                # Use FLAN-T5 if available
                return self._generate_with_ai(predictions)
                
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
            return self._generate_professional_template(predictions)
    
    def _create_prompt(self, predictions: Dict) -> str:
        """Create prompt for FLAN-T5"""
        disease_info = predictions.get('disease', {})
        soil_info = predictions.get('soil', {})
        
        prompt = f"""Generate a farm advisory report.

Disease: {disease_info.get('disease', 'Not analyzed')}
Health: {disease_info.get('health_status', 'Unknown')}
Soil: {soil_info.get('health_status', 'Not analyzed')}

Provide summary and top 3 actions."""

        return prompt
    
    def _generate_professional_template(self, predictions: Dict) -> str:
        """
        Generate professional report using enhanced templates
        FREE - High quality - Recommended method
        """
        
        disease = predictions.get('disease', {})
        soil = predictions.get('soil', {})
        weather = predictions.get('weather', {})
        yield_pred = predictions.get('yield', {})
        irrigation = predictions.get('irrigation', {})
        pest = predictions.get('pest', {})
        
        # Build report sections
        sections = []
        
        # Header
        sections.append("=" * 70)
        sections.append("🌾 PRECISION AGRICULTURE ADVISORY REPORT")
        sections.append("=" * 70)
        sections.append(f"Generated: {self.get_timestamp()}")
        sections.append(f"Report ID: AGR-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        sections.append("")
        
        # Executive Summary
        sections.append("📊 EXECUTIVE SUMMARY")
        sections.append("-" * 70)
        exec_summary = self._generate_executive_summary(predictions)
        sections.append(exec_summary)
        sections.append("")
        
        # Critical Status
        sections.append("🎯 CRITICAL STATUS OVERVIEW")
        sections.append("-" * 70)
        
        if disease:
            status = disease.get('health_status', 'Unknown')
            emoji = "🔴" if status == "UNHEALTHY" else "✅"
            sections.append(f"{emoji} Crop Health: {status}")
            if status == "UNHEALTHY":
                sections.append(f"   Disease: {disease.get('disease', 'Unknown')} ({disease.get('confidence', 0)}% confidence)")
        
        if soil:
            score = soil.get('health_score', 0)
            emoji = "🔴" if score < 40 else "🟡" if score < 70 else "✅"
            sections.append(f"{emoji} Soil Health: {soil.get('health_status', 'Unknown')} (Score: {score}/100)")
        
        if pest:
            risk = pest.get('risk_level', 'Unknown')
            emoji = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "✅"
            sections.append(f"{emoji} Pest Risk: {risk} ({pest.get('risk_score', 0)}/100)")
        
        sections.append("")
        
        # Immediate Actions
        sections.append("⚡ IMMEDIATE ACTIONS (Next 24-48 Hours)")
        sections.append("-" * 70)
        
        immediate = self._get_immediate_actions(predictions)
        for action in immediate:
            sections.append(f"  {action}")
        
        if not immediate:
            sections.append("  ✅ No urgent actions required")
        sections.append("")
        
        # Detailed Analysis
        sections.append("📋 DETAILED ANALYSIS")
        sections.append("-" * 70)
        sections.append("")
        
        # Disease Section
        if disease:
            sections.append("🔬 Crop Disease Analysis")
            sections.append(f"  Crop: {disease.get('crop', 'Unknown')}")
            sections.append(f"  Disease: {disease.get('disease', 'None detected')}")
            sections.append(f"  Health Status: {disease.get('health_status', 'Unknown')}")
            sections.append(f"  Confidence: {disease.get('confidence', 0)}%")
            
            if disease.get('health_status') == 'UNHEALTHY':
                sections.append("")
                sections.append("  🔴 TREATMENT PROTOCOL:")
                treatment = self._get_treatment_protocol(disease.get('disease', ''))
                for step in treatment:
                    sections.append(f"    • {step}")
            sections.append("")
        
        # Soil Section
        if soil:
            sections.append("🌱 Soil Health Analysis")
            sections.append(f"  Overall Status: {soil.get('health_status', 'Unknown')}")
            sections.append(f"  Health Score: {soil.get('health_score', 0)}/100")
            sections.append("")
            
            params = soil.get('parameter_scores', {})
            sections.append("  Parameter Scores:")
            sections.append(f"    pH: {params.get('ph', 0)}/100")
            sections.append(f"    Nitrogen: {params.get('nitrogen', 0)}/100")
            sections.append(f"    Phosphorus: {params.get('phosphorus', 0)}/100")
            sections.append(f"    Potassium: {params.get('potassium', 0)}/100")
            sections.append("")
            
            npk = soil.get('npk_fertilizer', {}).get('recommended_fertilizers', {})
            if npk and any(npk.values()):
                sections.append("  💊 Fertilizer Recommendations:")
                if npk.get('urea_kg_ha', 0) > 0:
                    sections.append(f"    • Urea: {npk.get('urea_kg_ha', 0)} kg/hectare")
                if npk.get('dap_kg_ha', 0) > 0:
                    sections.append(f"    • DAP: {npk.get('dap_kg_ha', 0)} kg/hectare")
                if npk.get('mop_kg_ha', 0) > 0:
                    sections.append(f"    • MOP: {npk.get('mop_kg_ha', 0)} kg/hectare")
                
                timing = soil.get('npk_fertilizer', {}).get('application_timing', '')
                if timing:
                    sections.append(f"    Timing: {timing}")
            sections.append("")
        
        # Weather Section
        if weather and weather.get('alerts'):
            sections.append("🌤️ Weather Alerts")
            for alert in weather['alerts'][:5]:
                sections.append(f"  • {alert}")
            sections.append("")
        
        # Yield Section
        if yield_pred:
            sections.append("📈 Yield Forecast")
            sections.append(f"  Crop: {yield_pred.get('crop_type', 'Unknown')}")
            sections.append(f"  Predicted Yield: {yield_pred.get('predicted_average_yield', 'N/A')} kg/hectare")
            sections.append(f"  Trend: {yield_pred.get('trend', 'Unknown')}")
            sections.append(f"  Growth Rate: {yield_pred.get('growth_rate_percent', 0)}%")
            
            yield_range = yield_pred.get('predicted_yield_range', {})
            if yield_range:
                sections.append(f"  Expected Range: {yield_range.get('lower', 0):.0f} - {yield_range.get('upper', 0):.0f} kg/ha")
            sections.append("")
        
        # Irrigation Section
        if irrigation:
            sections.append("💧 Irrigation Schedule")
            sections.append(f"  Crop: {irrigation.get('crop_type', 'Unknown')}")
            sections.append(f"  Growth Stage: {irrigation.get('growth_stage', 'Unknown')}")
            sections.append(f"  Current Soil Moisture: {irrigation.get('soil_moisture_percent', 'N/A')}%")
            sections.append(f"  Daily Water Requirement: {irrigation.get('irrigation_amount_mm_per_day', 0)} mm/day")
            sections.append(f"  Next Irrigation: {irrigation.get('next_irrigation_days', 'N/A')} days")
            
            recs = irrigation.get('recommendations', [])
            if recs:
                sections.append("  Recommendations:")
                for rec in recs[:3]:
                    sections.append(f"    • {rec}")
            sections.append("")
        
        # Pest Section
        if pest:
            sections.append("🐛 Pest Risk Assessment")
            sections.append(f"  Risk Level: {pest.get('risk_indicator', '')} {pest.get('risk_level', 'Unknown')}")
            sections.append(f"  Risk Score: {pest.get('risk_score', 0)}/100")
            sections.append(f"  Monitoring Frequency: {pest.get('monitoring_frequency', 'Regular')}")
            
            pests = pest.get('likely_pests', [])
            if pests:
                sections.append("  Likely Pests:")
                for p in pests[:3]:
                    sections.append(f"    • {p}")
            
            measures = pest.get('preventive_measures', [])
            if measures:
                sections.append("  Preventive Measures:")
                for m in measures[:4]:
                    sections.append(f"    • {m}")
            sections.append("")
        
        # Cost Analysis
        sections.append("💰 COST ESTIMATE & ROI")
        sections.append("-" * 70)
        cost_lines = self._calculate_costs(predictions)
        sections.extend(cost_lines)
        sections.append("")
        
        # Prioritized Recommendations
        sections.append("📝 ALL RECOMMENDATIONS (By Priority)")
        sections.append("-" * 70)
        
        all_recs = self._compile_all_recommendations(predictions)
        
        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            priority_items = [r for r in all_recs if r['priority'] == priority]
            if priority_items:
                sections.append(f"\n{priority} PRIORITY:")
                for item in priority_items:
                    sections.append(f"  • {item['text']}")
        
        sections.append("")
        
        # Footer
        sections.append("=" * 70)
        sections.append("📌 IMPORTANT DISCLAIMERS")
        sections.append("-" * 70)
        sections.append("• This report is AI-generated based on scientific agricultural practices")
        sections.append("• Always consult local agricultural extension officers for verification")
        sections.append("• Adjust recommendations based on local soil and climate conditions")
        sections.append("• Monitor crops regularly and adapt strategies as needed")
        sections.append("• Follow local regulations for pesticide and fertilizer application")
        sections.append("")
        sections.append("🌱 Precision Agriculture System - AI-Powered Farm Advisory")
        sections.append(f"📅 Report valid until: {self._get_expiry_date()}")
        sections.append("=" * 70)
        
        return "\n".join(sections)
    
    def _generate_executive_summary(self, predictions: Dict) -> str:
        """Generate executive summary"""
        disease = predictions.get('disease', {})
        soil = predictions.get('soil', {})
        pest = predictions.get('pest', {})
        
        parts = []
        
        if disease:
            status = disease.get('health_status', 'Unknown')
            if status == 'UNHEALTHY':
                parts.append(f"🔴 CRITICAL: {disease.get('disease', 'Disease')} detected - immediate treatment required")
            else:
                parts.append(f"✅ Crop health is good")
        
        if soil:
            score = soil.get('health_score', 0)
            if score < 40:
                parts.append(f"🔴 Soil health critical (score: {score}/100)")
            elif score < 70:
                parts.append(f"🟡 Soil needs improvement (score: {score}/100)")
            else:
                parts.append(f"✅ Soil health good (score: {score}/100)")
        
        if pest:
            risk = pest.get('risk_level', 'Unknown')
            if risk == 'High':
                parts.append(f"🔴 High pest risk - preventive action needed")
        
        return " | ".join(parts) if parts else "Farm analysis completed successfully."
    
    def _get_immediate_actions(self, predictions: Dict) -> list:
        """Extract immediate priority actions"""
        actions = []
        
        disease = predictions.get('disease', {})
        soil = predictions.get('soil', {})
        irrigation = predictions.get('irrigation', {})
        pest = predictions.get('pest', {})
        
        if disease.get('health_status') == 'UNHEALTHY':
            actions.append(f"🔴 Apply treatment for {disease.get('disease', 'disease')} within 24 hours")
        
        if soil.get('health_score', 100) < 40:
            actions.append("🔴 Emergency soil amendment required - apply recommended fertilizers")
        
        moisture = irrigation.get('soil_moisture_percent', 100)
        if moisture < 30:
            actions.append(f"🔴 Irrigate immediately - critical low moisture ({moisture}%)")
        
        if pest.get('risk_level') == 'High':
            actions.append("🔴 Implement pest control measures - high infestation risk")
        
        return actions
    
    def _get_treatment_protocol(self, disease: str) -> list:
        """Get treatment steps for disease"""
        disease_lower = disease.lower()
        
        if 'late blight' in disease_lower or 'late_blight' in disease_lower:
            return [
                "Apply copper-based fungicide (Bordeaux mixture) immediately",
                "Remove and destroy all affected plant parts",
                "Improve air circulation - space plants properly",
                "Avoid overhead watering - use drip irrigation",
                "Monitor daily for disease spread",
                "Apply preventive spray every 7 days"
            ]
        elif 'early blight' in disease_lower or 'early_blight' in disease_lower:
            return [
                "Apply chlorothalonil or mancozeb fungicide",
                "Remove lower infected leaves immediately",
                "Mulch around plants to prevent soil splash",
                "Ensure proper plant spacing for air flow",
                "Rotate crops next season"
            ]
        elif 'powdery mildew' in disease_lower:
            return [
                "Spray with sulfur-based fungicide",
                "Improve air circulation around plants",
                "Remove heavily infected plant parts",
                "Apply neem oil as organic alternative",
                "Avoid overhead watering"
            ]
        else:
            return [
                "Consult agricultural extension officer for specific treatment",
                "Remove and isolate affected plants to prevent spread",
                "Improve overall plant health with balanced nutrition",
                "Monitor closely for disease progression",
                "Consider biological control methods"
            ]
    
    def _calculate_costs(self, predictions: Dict) -> list:
        """Calculate estimated costs"""
        lines = []
        total = 0
        
        disease = predictions.get('disease', {})
        soil = predictions.get('soil', {})
        
        if disease.get('health_status') == 'UNHEALTHY':
            cost = 500
            lines.append(f"  Disease Treatment: ₹{cost}/hectare (fungicide + labor)")
            total += cost
        
        npk = soil.get('npk_fertilizer', {}).get('recommended_fertilizers', {})
        if npk:
            urea = npk.get('urea_kg_ha', 0) * 6
            dap = npk.get('dap_kg_ha', 0) * 25
            mop = npk.get('mop_kg_ha', 0) * 20
            fert_cost = urea + dap + mop
            
            if fert_cost > 0:
                lines.append(f"  Fertilizers: ₹{fert_cost:.0f}/hectare")
                total += fert_cost
        
        if total > 0:
            lines.append(f"\n  TOTAL COST: ₹{total:.0f} per hectare")
            benefit = total * 3
            lines.append(f"  EXPECTED BENEFIT: ₹{benefit:.0f}/hectare")
            lines.append(f"  ROI: ~{(benefit/total):.1f}x return on investment")
        else:
            lines.append("  No immediate costs required")
            lines.append("  Continue current management practices")
        
        return lines
    
    def _compile_all_recommendations(self, predictions: Dict) -> list:
        """Compile all recommendations"""
        recs = []
        
        # From soil
        for rec in predictions.get('soil', {}).get('recommendations', []):
            if '🔴' in rec:
                priority = 'HIGH'
            elif '⚠️' in rec or '🟡' in rec:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            text = rec.replace('🔴', '').replace('⚠️', '').replace('✅', '').replace('🟡', '').strip()
            recs.append({'priority': priority, 'text': text})
        
        # From irrigation
        for rec in predictions.get('irrigation', {}).get('recommendations', []):
            if '🔴' in rec:
                priority = 'HIGH'
            elif '⚠️' in rec:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            text = rec.replace('🔴', '').replace('⚠️', '').replace('✅', '').strip()
            recs.append({'priority': priority, 'text': text})
        
        # From pest
        if predictions.get('pest', {}).get('risk_level') == 'High':
            recs.append({'priority': 'HIGH', 'text': 'Implement integrated pest management immediately'})
        
        return recs
    
    def _format_report(self, ai_text: str, predictions: Dict) -> str:
        """Format AI-generated text"""
        header = f"""{'=' * 70}
🌱 AI-GENERATED AGRICULTURAL ADVISORY REPORT
{'=' * 70}
Generated: {self.get_timestamp()}

"""
        footer = f"""

{'=' * 70}
📝 Generated using AI (FLAN-T5)
{'=' * 70}"""
        
        return header + ai_text + footer
    
    def _generate_fallback_report(self, predictions: Dict) -> str:
        """Minimal fallback"""
        return f"""AGRICULTURAL REPORT
Generated: {self.get_timestamp()}

Analysis Summary:
- {len(predictions)} models analyzed
- Review individual outputs below

For detailed analysis, check each model's output separately.
"""
    
    def _get_expiry_date(self) -> str:
        """Report expiry (7 days)"""
        from datetime import timedelta
        expiry = datetime.now() + timedelta(days=7)
        return expiry.strftime("%B %d, %Y")
    
    def get_timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Singleton instance
_report_generator_instance = None


def get_report_generator() -> ReportGenerator:
    """Get singleton instance of ReportGenerator"""
    global _report_generator_instance
    if _report_generator_instance is None:
        # Use enhanced templates by default (FREE, high quality)
        _report_generator_instance = ReportGenerator(use_templates=True)
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