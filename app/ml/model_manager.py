import joblib # pyre-ignore-all-errors
import os
import numpy as np # pyre-ignore-all-errors
from app.ml.processor import extract_glcm_features # pyre-ignore-all-errors

MODEL_DIR = "models"
MODELS = ["blood_group", "sex", "age", "diabetic_status"]

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        for name in MODELS:
            path = os.path.join(MODEL_DIR, f"{name}_model.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
            else:
                print(f"Warning: Model for {name} not found at {path}")

    def predict(self, image_path):
        """
        Processes image and predicts all attributes.
        """
        try:
            features = extract_glcm_features(image_path)
            # Reshape for single sample
            features = features.reshape(1, -1)
            
            results = {}
            for name, model in self.models.items():
                prediction = model.predict(features)[0]
                results[name] = prediction
            
            return results
        except Exception as e:
            return {"error": str(e)}

# Singleton instance
manager = ModelManager()
