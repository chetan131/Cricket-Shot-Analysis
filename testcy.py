import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("/Users/Chetan/Desktop/python/cricket_shot_classifier.h5")

# Load and preprocess the test image
img_path = "/Users/Chetan/Desktop/python/tes3.webp.jpeg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Map to class names
class_names = ['Drive', 'Legglance-flick', 'pullshot', 'sweep']  # Edit as per your model
predicted_label = class_names[predicted_class]

print(f"Predicted Shot: {predicted_label}")