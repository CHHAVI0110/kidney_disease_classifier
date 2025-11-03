import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # 🟩 Update these according to your model’s output order
        self.class_names = ["Cyst", "Normal", "Stone", "Tumor"]

    def predict(self):
        model_path = os.path.join("artifacts", "training", "model.keras")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = load_model(model_path)
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.class_names[predicted_index]
        confidence = float(np.max(predictions))  # convert np.float to float

        # 🟩 Return consistent structured output
        return {
            "class": predicted_class,
            "confidence": round(confidence, 4)
        }
