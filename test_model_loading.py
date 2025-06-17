from tensorflow.keras.models import load_model
import numpy as np
import os

    # Model and labels paths
model_path = "model.h5"
labels_path = "labels.npy"
new_model_path = "new_model.h5"  # Path for re-saved model

    # Test loading model and labels
if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
else:
        try:
            model = load_model(model_path)
            print("Model loaded successfully!")
            
            # Re-save the model streamlitto a new file
            model.save(new_model_path)
            print(f"Model re-saved as '{new_model_path}'")
        except Exception as e:
            print(f"Error loading model: {e}")

if not os.path.exists(labels_path):
        print(f"Labels file '{labels_path}' not found.")
else:
        try:
            label = np.load(labels_path)
            print("Labels loaded successfully!")
        except Exception as e:
            print(f"Error loading labels: {e}")