from transformers import pipeline
from PIL import Image
import numpy as np

def check_model_labels():
    print("Loading model Ateeqq/ai-vs-human-image-detector...")
    try:
        pipe = pipeline("image-classification", model="Ateeqq/ai-vs-human-image-detector")
        print("Model loaded successfully.")
        
        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        print("Running inference...")
        results = pipe(img)
        print("Raw Results:", results)
        
        # Print labels
        labels = [res['label'] for res in results]
        print("Labels found:", labels)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_model_labels()
