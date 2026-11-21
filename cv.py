import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load dataset using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

train_data = train_datagen.flow_from_directory(
    "dataset", 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode="categorical", 
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset", 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode="categorical", 
    subset="validation"
)
# Import necessary libraries
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Load pre-trained MobileNetV2 without top layers
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Freeze base model layers
base_model.trainable = False

# Define the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(len(train_data.class_indices), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# Train the model
EPOCHS = 10
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save the trained model
model.save("cricket_shot_classifier.h5")