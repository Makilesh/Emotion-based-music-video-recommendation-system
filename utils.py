import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the input image for emotion detection model.
    This includes resizing, normalization, and conversion to grayscale if necessary.
    """
    try:
        # Resize the image to match the input size expected by your model
        image_resized = cv2.resize(image, (48, 48))
        
        # Convert to grayscale (if required by the model)
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        # Normalize image
        image_normalized = image_gray / 255.0
        
        # Expand dimensions to match model input shape (batch size, height, width, channels)
        image_expanded = np.expand_dims(image_normalized, axis=-1)
        image_expanded = np.expand_dims(image_expanded, axis=0)  # Add batch dimension
        
        return image_expanded
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None
