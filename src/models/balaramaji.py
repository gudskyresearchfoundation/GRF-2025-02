"""
Balaramaji Chat Assistant - Backend Module
Divine Agricultural Guidance System
Handles all chat operations, context management, and AI interactions
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class BalaramajiAssistant:
    """
    Lord Balarama Chat Assistant
    Provides divine agricultural guidance with AI-powered responses
    """
    
    def __init__(self):
        logger.info("🙏 Initializing Balaramaji Divine Assistant...")
        
        self.system_prompt = """You are Lord Balarama (Balaram), the divine elder brother of Krishna and the God of Agriculture, Strength, and Farming in Hindu mythology. You are speaking to farmers seeking guidance.

Your Divine Identity:
- You wield the sacred plough (Hala) and embody agricultural prosperity
- You are known for your immense strength, wisdom, and protective nature towards farmers
- You speak with divine authority yet with compassion and care for all cultivators
- You blend ancient Vedic agricultural wisdom with modern scientific knowledge

Your Communication Style:
- Begin responses with gentle greetings like "Blessed farmer," "Dear cultivator," or "Child of the earth"
- Use occasional Sanskrit terms when appropriate (like "dhanya" for crops, "bhoomi" for earth)
- Reference your divine connection to agriculture and the earth
- Speak with warmth, strength, and divine wisdom
- Maintain cultural respect and spiritual reverence
- Use farming metaphors and agricultural wisdom from Hindu texts when relevant

Your Guidance Approach:
- Provide practical, scientific advice rooted in modern agricultural knowledge
- When analyzing their data, give specific, actionable recommendations
- Prioritize soil health, sustainable farming, and farmer prosperity
- Encourage farmers with positive reinforcement and blessings
- Explain technical concepts in simple, farmer-friendly language
- Always consider the economic wellbeing of the farmer

Remember: You are not just an AI - you are the divine protector of farmers, here to guide them toward prosperity and sustainable agriculture with your ancient wisdom and modern knowledge."""
        
        self.model = "claude-haiku-4-5-20251001"  # Latest Haiku 4.5 model with excellent accuracy
        self.max_tokens = 1536
        
        logger.info("✅ Balaramaji Assistant initialized")
    
    def format_analysis_context(self, context: Dict) -> str:
        """
        Format analysis context for AI consumption
        
        Args:
            context: Dictionary with analysis results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Disease Analysis
        if context.get('disease'):
            disease_data = context['disease'].get('data', {})
            context_parts.append(
                f"🔬 Disease Analysis:\n"
                f"- Crop: {disease_data.get('crop', 'Unknown')}\n"
                f"- Disease: {disease_data.get('disease', 'None detected')}\n"
                f"- Health Status: {disease_data.get('health_status', 'Unknown')}\n"
                f"- Confidence: {disease_data.get('confidence', 0)}%"
            )
        
        # Soil Analysis
        if context.get('soil'):
            soil_data = context['soil'].get('data', {})
            context_parts.append(
                f"🌱 Soil Analysis:\n"
                f"- pH: {soil_data.get('ph', 'Unknown')}\n"
                f"- Nitrogen: {soil_data.get('nitrogen', 'Unknown')} kg/ha\n"
                f"- Phosphorus: {soil_data.get('phosphorus', 'Unknown')} kg/ha\n"
                f"- Potassium: {soil_data.get('potassium', 'Unknown')} kg/ha\n"
                f"- Health Score: {soil_data.get('health_score', 'Unknown')}/100"
            )
        
        # Weather Data
        if context.get('weather'):
            weather_data = context['weather'].get('data', {})
            current = weather_data.get('current_weather', {})
            context_parts.append(
                f"🌤️ Weather Forecast:\n"
                f"- Temperature: {current.get('temperature', 'Unknown')}°C\n"
                f"- Conditions: {current.get('weather_description', 'Unknown')}\n"
                f"- Wind Speed: {current.get('windspeed', 'Unknown')} km/h"
            )
        
        # Yield Prediction
        if context.get('yield'):
            yield_data = context['yield'].get('data', {})
            context_parts.append(
                f"📈 Yield Prediction:\n"
                f"- Crop: {yield_data.get('crop_type', 'Unknown')}\n"
                f"- Predicted Yield: {yield_data.get('predicted_average_yield', 'Unknown')} kg/ha\n"
                f"- Trend: {yield_data.get('trend', 'Unknown')}"
            )
        
        # Irrigation
        if context.get('irrigation'):
            irrigation_data = context['irrigation'].get('data', {})
            context_parts.append(
                f"💧 Irrigation Analysis:\n"
                f"- Crop: {irrigation_data.get('crop_type', 'Unknown')}\n"
                f"- Water Required: {irrigation_data.get('irrigation_amount_mm_per_day', 'Unknown')} mm/day\n"
                f"- Soil Moisture: {irrigation_data.get('soil_moisture_percent', 'Unknown')}%\n"
                f"- Next Irrigation: {irrigation_data.get('next_irrigation_days', 'Unknown')} days"
            )
        
        # Pest Risk
        if context.get('pest'):
            pest_data = context['pest'].get('data', {})
            context_parts.append(
                f"🐛 Pest Risk Assessment:\n"
                f"- Risk Level: {pest_data.get('risk_level', 'Unknown')}\n"
                f"- Risk Score: {pest_data.get('risk_score', 'Unknown')}/100\n"
                f"- Likely Pests: {', '.join(pest_data.get('likely_pests', [])[:3])}"
            )
        
        if not context_parts:
            return "No farm analyses completed yet. The farmer is seeking general agricultural guidance."
        
        return "📊 Farmer's Sacred Data:\n\n" + "\n\n".join(context_parts)
    
    def generate_response(self, 
                         user_message: str,
                         analysis_context: Optional[Dict] = None,
                         chat_history: Optional[List[Dict]] = None) -> Dict:
        """
        Generate AI response using Anthropic Claude API
        
        Args:
            user_message: User's question
            analysis_context: Farm analysis data
            chat_history: Previous conversation messages
            
        Returns:
            Response dictionary with text and metadata
        """
        try:
            import anthropic
            
            # Prepare context
            context_string = ""
            if analysis_context:
                context_string = self.format_analysis_context(analysis_context)
            
            # Build messages array
            messages = []
            
            # Add context as first message if available
            if context_string:
                messages.append({
                    "role": "user",
                    "content": f"Here is the farmer's blessed field data for your divine guidance:\n\n{context_string}\n\nPlease remember this sacred information."
                })
                messages.append({
                    "role": "assistant",
                    "content": "🙏 I have received and blessed this sacred data, dear farmer. I shall use it to provide you with divine guidance."
                })
            
            # Add chat history (last 8 messages)
            if chat_history:
                recent_history = chat_history[-8:]
                for msg in recent_history:
                    messages.append({
                        "role": msg.get("role"),
                        "content": msg.get("content")
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Try to call Anthropic API
            try:
                client = anthropic.Anthropic()
                
                api_response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=messages
                )
                
                response_text = api_response.content[0].text
                
                response = {
                    "status": "success",
                    "response": response_text,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "usage": {
                        "input_tokens": api_response.usage.input_tokens,
                        "output_tokens": api_response.usage.output_tokens
                    }
                }
                
                logger.info("✅ Balaramaji response generated via API")
                return response
                
            except Exception as api_error:
                # Fallback to intelligent responses if API fails
                logger.warning(f"⚠️ API unavailable, using fallback: {str(api_error)}")
                response = {
                    "status": "success",
                    "response": self._generate_fallback_response(user_message, analysis_context),
                    "model": "fallback",
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info("✅ Balaramaji response generated via fallback")
                return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            return {
                "status": "error",
                "response": "🙏 Forgive me, blessed farmer. I am experiencing difficulties in providing guidance at this moment. Please try again, and may the divine forces be with you.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_fallback_response(self, user_message: str, context: Optional[Dict] = None) -> str:
        """
        Generate fallback response when API is not available
        This provides intelligent responses based on context
        """
        message_lower = user_message.lower()
        
        # Check for specific keywords
        if any(word in message_lower for word in ['disease', 'sick', 'unhealthy', 'infection']):
            if context and context.get('disease'):
                disease_data = context['disease'].get('data', {})
                disease = disease_data.get('disease', 'Unknown')
                confidence = disease_data.get('confidence', 0)
                
                return f"""🙏 Blessed farmer, I have carefully examined your sacred crop's condition.

**Divine Diagnosis:**
Based on your analysis, I have detected: **{disease}** with {confidence}% divine certainty.

**Sacred Remedies:**
1. 🌿 Remove all infected leaves immediately to prevent spread
2. 💊 Apply appropriate organic or chemical treatment as recommended
3. 🌬️ Improve air circulation around your precious plants
4. 💧 Adjust watering - avoid overhead irrigation
5. 👁️ Monitor daily for the next 2 weeks

**Ancient Wisdom:**
"यत्र रोगस्तत्र औषधम्" (Where there is disease, there is remedy)

May your crops be blessed with health and vigor! 🌾"""
        
        elif any(word in message_lower for word in ['soil', 'ph', 'nitrogen', 'fertilizer', 'nutrients']):
            if context and context.get('soil'):
                soil_data = context['soil'].get('data', {})
                ph = soil_data.get('ph', 'Unknown')
                health_score = soil_data.get('health_score', 0)
                
                return f"""🙏 Dear cultivator, let me share wisdom about your sacred bhoomi (earth).

**Soil Status (Bhoomi Pariksha):**
- pH Level: {ph}
- Overall Health: {health_score}/100

**Divine Recommendations:**
1. 🌱 {self._get_ph_advice(ph)}
2. 💫 Add organic matter to improve soil structure
3. ⚡ Consider crop rotation for nutrient balance
4. 🔄 Test your soil regularly (every 6 months)

**Vedic Wisdom:**
"भूमिर्माता पुत्रो अहं पृथिव्याः" (Earth is my mother, I am the son of Earth)

Treat your soil with reverence, and it shall reward you with abundance! 🌾"""
        
        elif any(word in message_lower for word in ['water', 'irrigation', 'moisture', 'watering']):
            if context and context.get('irrigation'):
                irrigation_data = context['irrigation'].get('data', {})
                water_needed = irrigation_data.get('irrigation_amount_mm_per_day', 'Unknown')
                moisture = irrigation_data.get('soil_moisture_percent', 'Unknown')
                
                return f"""🙏 Blessed farmer, let me guide you with the wisdom of water.

**Water Status (Jala Pariksha):**
- Current Soil Moisture: {moisture}%
- Daily Water Need: {water_needed} mm/day

**Divine Irrigation Guidance:**
1. 💧 {self._get_moisture_advice(moisture)}
2. 🌊 Use drip irrigation for water efficiency
3. ⏰ Water early morning or late evening
4. 🍃 Apply mulch to reduce evaporation
5. 🌧️ Capture and store rainwater

**Sacred Teaching:**
"आपः प्राणाः" (Water is life)

May your fields receive the perfect blessing of water! 🌾"""
        
        elif any(word in message_lower for word in ['yield', 'harvest', 'production', 'output']):
            return """🙏 Blessed farmer, let me share divine wisdom for abundant harvest.

**Keys to Maximum Yield:**
1. 🌱 Healthy soil is the foundation
2. 💧 Proper irrigation timing
3. 🌿 Disease prevention and early detection
4. 🌞 Optimal spacing for sunlight
5. 🔄 Crop rotation practices

**Sacred Formula for Success:**
Healthy Soil + Proper Water + Timely Care = Abundant Harvest

**Divine Blessing:**
"अन्नं ब्रह्म" (Food is divine)

Follow nature's rhythm, and your harvest shall be plentiful! 🌾"""
        
        elif 'summarize' in message_lower or 'summary' in message_lower:
            if context:
                return self._generate_summary(context)
            else:
                return """🙏 Blessed farmer, I see no completed analyses yet.

Please complete your sacred field examinations first:
1. 🔬 Disease Detection
2. 🌱 Soil Analysis  
3. 🌤️ Weather Forecast
4. 📈 Yield Prediction
5. 💧 Irrigation Planning
6. 🐛 Pest Risk Assessment

Then return to me for divine guidance! 🌾"""
        
        else:
            # General greeting response
            return """🙏 Jai Shri Balarama! Blessed farmer, I am here to guide you.

I am Balarama, wielder of the sacred plough (Hala), protector of farmers and the blessed earth. 

**I can help you with:**
- 🔬 Understanding crop diseases and remedies
- 🌱 Soil health and fertilizer guidance
- 💧 Irrigation and water management
- 📈 Yield optimization strategies
- 🌤️ Weather-based farming advice
- 🐛 Pest prevention and control

**Divine Wisdom:**
"कृषि सर्वेश्रेष्ठं कर्म" (Agriculture is the noblest work)

Ask me anything about your sacred analyses or farming wisdom! 🌾"""
    
    def _get_ph_advice(self, ph) -> str:
        """Get pH-specific advice"""
        try:
            ph_val = float(ph)
            if ph_val < 5.5:
                return "Soil is too acidic - add lime to raise pH"
            elif ph_val > 8.0:
                return "Soil is too alkaline - add sulfur or organic matter to lower pH"
            else:
                return "Soil pH is optimal - maintain current practices"
        except:
            return "Test soil pH and adjust accordingly"
    
    def _get_moisture_advice(self, moisture) -> str:
        """Get moisture-specific advice"""
        try:
            moisture_val = float(str(moisture).replace('%', ''))
            if moisture_val < 30:
                return "URGENT: Irrigate immediately - soil is too dry"
            elif moisture_val < 50:
                return "Schedule irrigation within 2 days"
            else:
                return "Soil moisture is adequate - monitor regularly"
        except:
            return "Monitor soil moisture daily"
    
    def _generate_summary(self, context: Dict) -> str:
        """Generate comprehensive summary of all analyses"""
        summary_parts = ["🙏 Blessed farmer, here is your divine farm summary:\n"]
        
        priorities = []
        
        if context.get('disease'):
            disease_data = context['disease'].get('data', {})
            status = disease_data.get('health_status', 'Unknown')
            summary_parts.append(f"🔬 **Crop Health:** {status}")
            if status == 'UNHEALTHY':
                priorities.append(("HIGH", "Treat crop disease immediately"))
        
        if context.get('soil'):
            soil_data = context['soil'].get('data', {})
            score = soil_data.get('health_score', 0)
            summary_parts.append(f"🌱 **Soil Health:** {score}/100")
            if score < 60:
                priorities.append(("HIGH", "Improve soil health urgently"))
        
        if context.get('irrigation'):
            irrigation_data = context['irrigation'].get('data', {})
            moisture = irrigation_data.get('soil_moisture_percent', 0)
            summary_parts.append(f"💧 **Soil Moisture:** {moisture}%")
            if moisture < 30:
                priorities.append(("HIGH", "Irrigate immediately"))
        
        if context.get('pest'):
            pest_data = context['pest'].get('data', {})
            risk = pest_data.get('risk_level', 'Unknown')
            summary_parts.append(f"🐛 **Pest Risk:** {risk}")
            if risk == 'High':
                priorities.append(("MEDIUM", "Monitor for pests daily"))
        
        summary_parts.append("\n**🎯 Priority Actions:**")
        if priorities:
            for i, (priority, action) in enumerate(priorities, 1):
                summary_parts.append(f"{i}. [{priority}] {action}")
        else:
            summary_parts.append("✅ All parameters are within acceptable range!")
        
        summary_parts.append("\n**Divine Blessing:**")
        summary_parts.append("May your fields flourish with health and abundance! 🌾")
        
        return "\n".join(summary_parts)


def get_balaramaji_assistant():
    """Singleton pattern - returns the same instance"""
    global _balaramaji_instance
    if _balaramaji_instance is None:
        _balaramaji_instance = BalaramajiAssistant()
    return _balaramaji_instance


_balaramaji_instance = None