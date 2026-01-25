"""
Model 2: Weather Service
Uses Open-Meteo API for weather data
NO TRAINING NEEDED - API-based service
"""

import requests
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WeatherService:
    """
    Weather data service using Open-Meteo API
    Provides weather forecasts and agricultural alerts
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    
    def __init__(self):
        """Initialize Weather Service"""
        logger.info("🌤️ Initializing Weather Service")
        self.timeout = 10  # seconds
    
    def get_weather(self, latitude: float, longitude: float, days: int = 7) -> Dict:
        """
        Get weather forecast for given coordinates
        
        Args:
            latitude: Latitude (-90 to 90)
            longitude: Longitude (-180 to 180)
            days: Number of forecast days (1-16)
        
        Returns:
            Weather data dictionary
        """
        try:
            # Validate inputs
            days = max(1, min(16, days))  # Clamp to API limits
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
                "hourly": "temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,weathercode",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
                "forecast_days": days,
                "timezone": "auto"
            }
            
            logger.info(f"🌍 Fetching weather for lat={latitude}, lon={longitude}, days={days}")
            
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            logger.info("✅ Weather data fetched successfully")
            return data
            
        except requests.Timeout:
            logger.error("⏱️ Weather API timeout")
            raise Exception("Weather service timeout. Please try again.")
        except requests.RequestException as e:
            logger.error(f"❌ Weather API error: {str(e)}")
            raise Exception(f"Weather service error: {str(e)}")
    
    def get_current_weather(self, latitude: float, longitude: float) -> Dict:
        """
        Get current weather conditions
        
        Returns:
            Current weather data
        """
        data = self.get_weather(latitude, longitude, days=1)
        
        current = data.get("current_weather", {})
        hourly = data.get("hourly", {})
        
        return {
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed"),
            "weather_code": current.get("weathercode"),
            "weather_description": self._get_weather_description(current.get("weathercode")),
            "humidity": hourly.get("relativehumidity_2m", [None])[0],
            "timestamp": current.get("time"),
            "location": {
                "latitude": latitude,
                "longitude": longitude
            }
        }
    
    def get_agricultural_alerts(self, latitude: float, longitude: float, days: int = 7) -> List[str]:
        """
        Generate agricultural alerts based on weather data
        
        Returns:
            List of alert messages
        """
        try:
            weather = self.get_weather(latitude, longitude, days)
            alerts = []
            
            # Extract data
            current = weather.get("current_weather", {})
            daily = weather.get("daily", {})
            hourly = weather.get("hourly", {})
            
            temp = current.get("temperature", 20)
            windspeed = current.get("windspeed", 0)
            
            # Temperature alerts
            if temp > 35:
                alerts.append(f"🌡️ High temperature alert ({temp}°C) - Increase irrigation to prevent heat stress")
            elif temp < 5:
                alerts.append(f"❄️ Frost warning ({temp}°C) - Protect sensitive crops")
            
            # Wind alerts
            if windspeed > 40:
                alerts.append(f"💨 Strong wind warning ({windspeed} km/h) - Secure structures and equipment")
            
            # Rainfall analysis
            if "precipitation_sum" in daily:
                precip_data = daily["precipitation_sum"]
                total_rainfall = sum([p for p in precip_data if p is not None])
                
                if total_rainfall > 100:
                    alerts.append(f"🌧️ Heavy rainfall expected ({total_rainfall:.1f}mm over {days} days) - Ensure proper drainage")
                elif total_rainfall == 0:
                    alerts.append("☀️ No rain forecasted - Monitor soil moisture and plan irrigation")
            
            # Humidity analysis
            if "relativehumidity_2m" in hourly:
                humidity_data = hourly["relativehumidity_2m"]
                avg_humidity = sum(humidity_data[:24]) / min(len(humidity_data), 24)
                
                if avg_humidity > 85:
                    alerts.append(f"💧 High humidity ({avg_humidity:.0f}%) - Increased disease risk, monitor crops closely")
            
            # Optimal conditions
            if 18 <= temp <= 28 and windspeed < 20 and 40 <= avg_humidity <= 70:
                alerts.append("🌱 Optimal growing conditions - Good time for field activities")
            
            # Default message if no alerts
            if not alerts:
                alerts.append("✅ No weather alerts - Conditions are favorable for agriculture")
            
            logger.info(f"📊 Generated {len(alerts)} agricultural alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error generating alerts: {str(e)}")
            return ["⚠️ Unable to generate weather alerts"]
    
    def search_location(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for location coordinates by name
        
        Args:
            query: Location name (city, region, country)
            max_results: Maximum number of results
        
        Returns:
            List of location dictionaries
        """
        try:
            params = {
                "name": query,
                "count": max_results,
                "language": "en",
                "format": "json"
            }
            
            logger.info(f"🔍 Searching for location: {query}")
            
            response = requests.get(self.GEOCODING_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if "results" not in data:
                logger.info(f"No locations found for: {query}")
                return []
            
            locations = []
            for result in data["results"]:
                locations.append({
                    "name": result.get("name", ""),
                    "country": result.get("country", ""),
                    "region": result.get("admin1", ""),
                    "latitude": result.get("latitude"),
                    "longitude": result.get("longitude"),
                    "timezone": result.get("timezone", ""),
                    "population": result.get("population", 0)
                })
            
            logger.info(f"✅ Found {len(locations)} location(s)")
            return locations
            
        except Exception as e:
            logger.error(f"❌ Location search error: {str(e)}")
            raise
    
    def _get_weather_description(self, weather_code: Optional[int]) -> str:
        """
        Convert WMO weather code to description
        
        Args:
            weather_code: WMO weather code
        
        Returns:
            Weather description
        """
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        
        return weather_codes.get(weather_code, "Unknown")


# Singleton instance
_weather_service_instance = None


def get_weather_service() -> WeatherService:
    """Get singleton instance of WeatherService"""
    global _weather_service_instance
    if _weather_service_instance is None:
        _weather_service_instance = WeatherService()
    return _weather_service_instance


if __name__ == "__main__":
    # Test the service
    print("Testing Weather Service...")
    
    service = WeatherService()
    
    # Test with a sample location (New Delhi, India)
    lat, lon = 28.6139, 77.2090
    
    print(f"\n🌍 Testing weather for Delhi (lat={lat}, lon={lon})")
    
    current = service.get_current_weather(lat, lon)
    print(f"\n📊 Current Weather: {current}")
    
    alerts = service.get_agricultural_alerts(lat, lon, days=7)
    print(f"\n⚠️ Agricultural Alerts:")
    for alert in alerts:
        print(f"   {alert}")
    
    print("\n✅ Weather Service test completed!")
