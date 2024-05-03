# predict_class.py

from keras.models import load_model
import numpy as np
import cv2

def preprocess_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (120, 120))
    image = image / 255.0  # Normalize the pixel values
    return image

def predict_class(image_path, model_path="C:/Users/DELL/OneDrive/Desktop/Project/models/trained_model.h5"):
    # Load the saved model
    model = load_model(model_path)
    # Preprocess the image
    image = preprocess_image(image_path)
    # Reshape the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    # Perform prediction
    predictions = model.predict(image)
    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    return predicted_class

# Example usage
image_path = "C:/Users/DELL/OneDrive/Desktop/Project/data/train_images/000a4bcdd.jpg"
predicted_class = predict_class(image_path)
print("Predicted class:", predicted_class)
