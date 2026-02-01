"""
Model 1: Crop Disease Detection
Uses pre-trained models from Hugging Face - FIXED
NO TRAINING NEEDED - FREE models
"""

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor
from PIL import Image
import logging
from typing import Dict, Union
import io

logger = logging.getLogger(__name__)


class DiseaseDetector:
    """
    Pre-trained crop disease detection model
    Detects diseases from plant leaf images
    FIXED: Proper image processor configuration
    """
    
    def __init__(self, model_name: str = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"):
        """
        Initialize disease detection model
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.labels = None
        
        logger.info(f"🔬 Initializing Disease Detector with {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model from Hugging Face"""
        try:
            logger.info("📥 Loading model and processor...")
            
            # Load model first
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Try to load processor with proper configuration
            try:
                # Try ViTImageProcessor first (works for most models)
                self.processor = ViTImageProcessor.from_pretrained(
                    self.model_name,
                    size={'height': 224, 'width': 224},  # Explicit size format
                    do_resize=True,
                    do_normalize=True
                )
            except Exception as e:
                logger.warning(f"⚠️ ViTImageProcessor with explicit size failed: {e}")
                try:
                    # Fallback: Try without size specification
                    self.processor = ViTImageProcessor.from_pretrained(self.model_name)
                except Exception as e2:
                    logger.warning(f"⚠️ ViTImageProcessor failed: {e2}")
                    # Final fallback: Try AutoImageProcessor
                    self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Extract label mapping
            self.labels = self.model.config.id2label
            
            logger.info(f"✅ Disease Detector loaded successfully!")
            logger.info(f"📊 Model can detect {len(self.labels)} disease classes")
            logger.info(f"💻 Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load disease detection model: {str(e)}")
            logger.info("🔄 Attempting to load fallback model...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Fallback to alternative working model"""
        try:
            fallback_model = "DEVIKAPRASAD/plant-disease-detection"
            logger.info(f"📥 Loading fallback model: {fallback_model}")
            
            self.model_name = fallback_model
            
            # Load model
            self.model = AutoModelForImageClassification.from_pretrained(fallback_model)
            self.model.to(self.device)
            self.model.eval()
            
            # Load processor
            self.processor = ViTImageProcessor.from_pretrained(
                fallback_model,
                size={'height': 224, 'width': 224},
                do_resize=True
            )
            
            self.labels = self.model.config.id2label
            
            logger.info(f"✅ Fallback model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Fallback model also failed: {str(e)}")
            raise Exception("Unable to load any disease detection model. Please check transformers version: pip install --upgrade transformers")
    
    def predict(self, image: Union[Image.Image, bytes, str]) -> Dict:
        """
        Predict disease from image
        
        Args:
            image: PIL Image, bytes, or file path
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, str):
                image = Image.open(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to expected size (224x224) before processing
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Preprocess image with proper parameters
            try:
                # Try with explicit parameters
                inputs = self.processor(
                    images=image,
                    return_tensors="pt",
                    do_resize=False  # Already resized above
                )
            except Exception as e:
                logger.warning(f"⚠️ Preprocessing with do_resize=False failed: {e}")
                # Fallback: Let processor handle resizing
                inputs = self.processor(images=image, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, predicted_idx = torch.max(probs, dim=-1)
                
                # Convert to Python types
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
            
            # Get predicted label
            predicted_label = self.labels[predicted_idx]
            
            # Parse label (format: Crop___Disease)
            crop, disease = self._parse_label(predicted_label)
            
            # Determine health status
            health_status = "HEALTHY" if "healthy" in disease.lower() else "UNHEALTHY"
            
            # Generate warning if confidence is low
            warning = None
            if confidence < 0.7:
                warning = f"Low confidence ({confidence*100:.1f}%). Consider using a clearer image."
            
            result = {
                "crop": crop,
                "disease": disease,
                "health_status": health_status,
                "confidence": round(confidence * 100, 2),
                "confidence_score": round(confidence, 4),
                "warning": warning,
                "raw_label": predicted_label,
                "model": self.model_name
            }
            
            logger.info(f"✅ Prediction: {crop} - {disease} ({confidence*100:.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, images: list) -> list:
        """
        Predict diseases for multiple images
        
        Args:
            images: List of PIL Images
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for idx, image in enumerate(images):
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Error processing image {idx}: {str(e)}")
                results.append({
                    "error": str(e),
                    "image_index": idx
                })
        
        return results
    
    def _parse_label(self, label: str) -> tuple:
        """
        Parse model label into crop and disease
        
        Args:
            label: Raw label from model (e.g., "Tomato___Late_blight")
        
        Returns:
            Tuple of (crop, disease)
        """
        if "___" in label:
            parts = label.split("___")
            crop = parts[0].replace("_", " ").title()
            disease = parts[1].replace("_", " ").title()
            return crop, disease
        elif "_" in label:
            # Handle labels with single underscore
            parts = label.split("_")
            # Common crop names
            potential_crops = ["tomato", "potato", "corn", "wheat", "rice", "apple", 
                             "grape", "pepper", "strawberry", "peach", "cherry", "bell"]
            
            if parts[0].lower() in potential_crops:
                crop = parts[0].title()
                disease = "_".join(parts[1:]).replace("_", " ").title()
                return crop, disease
            
            # Check if second part is a crop
            if len(parts) > 1 and parts[1].lower() in potential_crops:
                crop = parts[1].title()
                disease = "_".join([parts[0]] + parts[2:]).replace("_", " ").title()
                return crop, disease
            
            # If can't determine crop, use "Unknown"
            return "Unknown", label.replace("_", " ").title()
        else:
            return "Unknown", label.replace("_", " ").title()
    
    def get_supported_crops(self) -> list:
        """Get list of crops the model can detect"""
        crops = set()
        for label in self.labels.values():
            crop, _ = self._parse_label(label)
            if crop != "Unknown":
                crops.add(crop)
        
        # If no crops extracted, return default list
        if not crops:
            crops = {"Tomato", "Potato", "Corn", "Wheat", "Apple", "Grape", "Pepper", "Rice"}
        
        return sorted(list(crops))
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "num_classes": len(self.labels),
            "device": str(self.device),
            "supported_crops": self.get_supported_crops(),
            "model_type": "MobileNetV2" if "mobilenet" in self.model_name.lower() else "Vision Transformer",
            "source": "Hugging Face"
        }


# Singleton instance
_disease_detector_instance = None


def get_disease_detector() -> DiseaseDetector:
    """Get singleton instance of DiseaseDetector"""
    global _disease_detector_instance
    if _disease_detector_instance is None:
        _disease_detector_instance = DiseaseDetector()
    return _disease_detector_instance


if __name__ == "__main__":
    # Test the model
    print("Testing Disease Detection Model...")
    
    detector = DiseaseDetector()
    print(f"\n✅ Model loaded successfully!")
    print(f"📊 Model Info: {detector.get_model_info()}")
    print(f"🌱 Supported crops: {detector.get_supported_crops()}")