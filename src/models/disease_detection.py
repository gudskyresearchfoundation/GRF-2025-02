"""
Model 1: Crop Disease Detection
Uses pre-trained ResNet50 from Hugging Face
NO TRAINING NEEDED - Uses mesabo/agri-plant-disease-resnet50
"""

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import logging
from typing import Dict, Union
import io

logger = logging.getLogger(__name__)


class DiseaseDetector:
    """
    Pre-trained crop disease detection model
    Detects diseases from plant leaf images
    """
    
    def __init__(self, model_name: str = "mesabo/agri-plant-disease-resnet50"):
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
            logger.info("📥 Loading image processor...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            logger.info("📥 Loading model...")
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Extract label mapping
            self.labels = self.model.config.id2label
            
            logger.info(f"✅ Disease Detector loaded successfully!")
            logger.info(f"📊 Model can detect {len(self.labels)} disease classes")
            logger.info(f"💻 Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load disease detection model: {str(e)}")
            raise
    
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
            
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
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
        else:
            # Fallback for unexpected format
            return "Unknown", label.replace("_", " ").title()
    
    def get_supported_crops(self) -> list:
        """Get list of crops the model can detect"""
        crops = set()
        for label in self.labels.values():
            crop, _ = self._parse_label(label)
            crops.add(crop)
        return sorted(list(crops))
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "num_classes": len(self.labels),
            "device": str(self.device),
            "supported_crops": self.get_supported_crops(),
            "model_type": "ResNet50",
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
